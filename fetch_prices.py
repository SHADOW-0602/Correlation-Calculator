from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - dependency is optional at runtime
    yf = None


ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
POLYGON_BASE_URL = "https://api.polygon.io"
DEFAULT_TIMEOUT_S = 30

# Free-tier routing assumptions verified from official docs in April 2026:
# - Polygon basic/free: 2 years of historical data.
# - Alpha Vantage free: 25 requests per day, and full daily history is premium.
POLYGON_FREE_MAX_HISTORY_DAYS = 365 * 2
ALPHAVANTAGE_FREE_DAILY_POINTS = 100
DEFAULT_POLYGON_LIMIT = 50_000

STANDARD_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "dividends",
    "stock_splits",
]


class PriceFetchError(RuntimeError):
    """Raised when all providers fail to satisfy the request."""


MULTI_SYMBOL_COLUMNS = ["symbol", *STANDARD_COLUMNS]


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    reason: str
    success: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class FetchMetadata:
    asset_type: Literal["equity", "option"]
    provider: str
    symbol: str
    from_date: date
    to_date: date
    adjusted: bool
    option_ticker: Optional[str] = None


@dataclass(frozen=True)
class OptionContract:
    underlying: str
    expiry: date
    right: Literal["C", "P"]
    strike: float

    @property
    def polygon_ticker(self) -> str:
        return build_polygon_option_ticker(
            underlying=self.underlying,
            expiry=self.expiry,
            right=self.right,
            strike=self.strike,
        )


def _parse_date(value: Any, *, field_name: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid {field_name}: {value!r}")
    return ts.date()


def _parse_iso_date(s: Any) -> Optional[date]:
    if s is None:
        return None
    try:
        ts = pd.to_datetime(str(s), errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.date()


def _coerce_date_window(
    from_date: Any,
    to_date: Any,
    *,
    allow_future_to: bool = False,
) -> Tuple[date, date]:
    start = _parse_date(from_date, field_name="from_date")
    end = _parse_date(to_date, field_name="to_date")

    today = date.today()
    if not allow_future_to and end > today:
        end = today
    if start > end:
        raise ValueError(f"from_date {start.isoformat()} is after to_date {end.isoformat()}.")
    return start, end


def _normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if not value:
        raise ValueError("symbol is required.")
    return value


def parse_symbols(symbols: Any) -> List[str]:
    if isinstance(symbols, (list, tuple, set, pd.Index)):
        items = [str(s) for s in symbols]
    else:
        raw = str(symbols or "").replace("\n", ",").replace(";", ",")
        items = raw.split(",")

    parts = [piece.strip().upper() for piece in items]

    expanded: List[str] = []
    for part in parts:
        if not part:
            continue
        expanded.extend(token for token in part.split() if token)

    deduped: List[str] = []
    seen: set[str] = set()
    for symbol in expanded:
        normalized = _normalize_symbol(symbol)
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)

    if not deduped:
        raise ValueError("At least one ticker symbol is required.")
    return deduped


def _normalize_right(value: str) -> Literal["C", "P"]:
    cleaned = str(value or "").strip().upper()
    mapping = {
        "C": "C",
        "CALL": "C",
        "CALLS": "C",
        "P": "P",
        "PUT": "P",
        "PUTS": "P",
    }
    if cleaned not in mapping:
        raise ValueError("right must be one of: C, CALL, P, PUT.")
    return mapping[cleaned]


def _sleep_s(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "fetch-prices-backend/1.0"})
    return sess


def _http_get(
    url: str,
    *,
    params: Dict[str, Any],
    timeout_s: int = DEFAULT_TIMEOUT_S,
    session: Optional[requests.Session] = None,
    max_retries: int = 5,
    retry_statuses: Sequence[int] = (429, 500, 502, 503, 504),
) -> requests.Response:
    sess = session or _session()
    last_error: Optional[BaseException] = None

    for k, v in params.items():
        if isinstance(v, str) and "your_polygon_api_key_here" in v:
            raise ValueError(
                f"Detected placeholder API key '{v}' in parameter '{k}'. "
                "Please ensure your .env file is correctly configured and the app has been refreshed."
            )

    for attempt in range(max_retries + 1):
        try:
            response = sess.get(url, params=params, timeout=timeout_s)
            if response.status_code in retry_statuses:
                if attempt >= max_retries:
                    response.raise_for_status()

                retry_after = response.headers.get("Retry-After")
                wait_s = 0.0
                if retry_after:
                    try:
                        wait_s = float(retry_after)
                    except ValueError:
                        wait_s = 0.0
                if wait_s <= 0:
                    wait_s = random.uniform(0.5, min(30.0, 2 ** (attempt + 1)))
                _sleep_s(wait_s)
                continue

            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            _sleep_s(random.uniform(0.5, min(30.0, 2 ** (attempt + 1))))
        except Exception as exc:
            last_error = exc
            raise

    raise RuntimeError(f"HTTP GET failed after retries: {last_error}")


def _safe_json(response: requests.Response) -> Dict[str, Any]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object, got {type(payload)}")
    return payload


def _standardize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
        out = out.set_index("date")

    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)
    out = out[~out.index.isna()]
    out.index.name = "date"

    for col in STANDARD_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col not in {"dividends", "stock_splits"} else 0.0

    ordered = out[STANDARD_COLUMNS].sort_index()

    numeric_columns = ["open", "high", "low", "close", "adjusted_close", "volume", "dividends", "stock_splits"]
    for col in numeric_columns:
        ordered[col] = pd.to_numeric(ordered[col], errors="coerce")

    return ordered[~ordered.index.duplicated(keep="last")]


def _estimate_daily_points(from_date: date, to_date: date) -> int:
    if from_date > to_date:
        return 0
    business_days = pd.bdate_range(from_date, to_date)
    return len(business_days)


def _polygon_free_covers(from_date: date, to_date: date) -> bool:
    max_lookback_start = date.today() - timedelta(days=POLYGON_FREE_MAX_HISTORY_DAYS)
    return from_date >= max_lookback_start and to_date <= date.today()


def _alphavantage_free_covers_daily(from_date: date, to_date: date) -> bool:
    return _estimate_daily_points(from_date, to_date) <= ALPHAVANTAGE_FREE_DAILY_POINTS


def build_polygon_option_ticker(
    *,
    underlying: str,
    expiry: Any,
    right: str,
    strike: float,
) -> str:
    normalized_underlying = _normalize_symbol(underlying)
    normalized_right = _normalize_right(right)
    normalized_expiry = _parse_date(expiry, field_name="expiry")

    if strike <= 0:
        raise ValueError("strike must be positive.")

    strike_millis = int(round(float(strike) * 1000))
    if strike_millis >= 100_000_000:
        raise ValueError("strike is too large to fit Polygon's 8-digit strike format.")

    return f"O:{normalized_underlying}{normalized_expiry.strftime('%y%m%d')}{normalized_right}{strike_millis:08d}"


def parse_polygon_option_ticker(option_ticker: str) -> OptionContract:
    raw = str(option_ticker or "").strip().upper()
    if raw.startswith("O:"):
        raw = raw[2:]

    if len(raw) < 16:
        raise ValueError(f"Invalid Polygon option ticker: {option_ticker!r}")

    try:
        right_index = max(raw.rfind("C"), raw.rfind("P"))
        if right_index <= 5:
            raise ValueError
        underlying = raw[: right_index - 6]
        expiry_part = raw[right_index - 6 : right_index]
        right = raw[right_index]
        strike_part = raw[right_index + 1 :]
        expiry = datetime.strptime(expiry_part, "%y%m%d").date()
        strike = int(strike_part) / 1000.0
    except Exception as exc:
        raise ValueError(f"Invalid Polygon option ticker: {option_ticker!r}") from exc

    if not underlying:
        raise ValueError(f"Invalid Polygon option ticker: {option_ticker!r}")

    return OptionContract(
        underlying=underlying,
        expiry=expiry,
        right=_normalize_right(right),
        strike=strike,
    )


def _build_option_contract(
    *,
    option_ticker: Optional[str] = None,
    underlying: Optional[str] = None,
    expiry: Optional[Any] = None,
    right: Optional[str] = None,
    strike: Optional[float] = None,
) -> OptionContract:
    if option_ticker:
        return parse_polygon_option_ticker(option_ticker)

    missing = [
        field_name
        for field_name, value in (
            ("underlying", underlying),
            ("expiry", expiry),
            ("right", right),
            ("strike", strike),
        )
        if value is None or value == ""
    ]
    if missing:
        missing_fields = ", ".join(missing)
        raise ValueError(
            "Either option_ticker must be supplied, or all of "
            f"underlying/expiry/right/strike must be supplied. Missing: {missing_fields}"
        )

    return OptionContract(
        underlying=_normalize_symbol(str(underlying)),
        expiry=_parse_date(expiry, field_name="expiry"),
        right=_normalize_right(str(right)),
        strike=float(strike),
    )


def _with_metadata(frame: pd.DataFrame, metadata: FetchMetadata) -> pd.DataFrame:
    out = frame.copy()
    out.attrs["metadata"] = metadata
    return out


def _fetch_equity_yfinance(
    symbol: str,
    *,
    from_date: date,
    to_date: date,
    adjusted: bool = True,
) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed. Run `pip install yfinance` to enable it.")

    ticker = yf.Ticker(symbol)
    # Yahoo uses end-exclusive windows for `end`, so add one day.
    end_exclusive = to_date + timedelta(days=1)
    raw = ticker.history(
        start=from_date.isoformat(),
        end=end_exclusive.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=True,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No yfinance price history returned for {symbol}.")

    frame = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
        }
    )

    for col in STANDARD_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0

    if adjusted:
        close = pd.to_numeric(frame["close"], errors="coerce")
        adj_close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
        factor = adj_close.divide(close).replace([math.inf, -math.inf], pd.NA).fillna(1.0)
        factor = factor.where(close.ne(0), 1.0)

        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").multiply(factor)

        # Volume should not be dividend-adjusted. We keep Yahoo's reported volume so
        # callers always receive a usable series instead of inventing a dividend-adjusted one.
        frame["adjusted_close"] = adj_close
    else:
        frame["adjusted_close"] = pd.to_numeric(frame["close"], errors="coerce")

    return _standardize_price_frame(frame)


def _fetch_polygon_dividends(symbol: str, api_key: str, from_date: date, to_date: date) -> pd.Series:
    url = f"{POLYGON_BASE_URL}/v3/reference/dividends"
    params = {"ticker": symbol, "limit": 1000, "apiKey": api_key}
    payload = _safe_json(_http_get(url, params=params))
    results = payload.get("results", [])
    if not results:
        return pd.Series(dtype=float)
    
    data = []
    for r in results:
        ex_date = _parse_iso_date(r.get("ex_dividend_date"))
        if ex_date and from_date <= ex_date <= to_date:
            data.append({"date": pd.Timestamp(ex_date), "dividend": float(r.get("cash_amount", 0))})
    if not data:
        return pd.Series(dtype=float)
    df = pd.DataFrame(data).set_index("date")
    return df.groupby("date")["dividend"].sum()

def _fetch_polygon_splits(symbol: str, api_key: str, from_date: date, to_date: date) -> pd.Series:
    url = f"{POLYGON_BASE_URL}/v3/reference/splits"
    params = {"ticker": symbol, "limit": 1000, "apiKey": api_key}
    payload = _safe_json(_http_get(url, params=params))
    results = payload.get("results", [])
    if not results:
        return pd.Series(dtype=float)
    
    data = []
    for r in results:
        ex_date = _parse_iso_date(r.get("execution_date"))
        if ex_date and from_date <= ex_date <= to_date:
            # We want the ratio (e.g. 1-for-4 split -> factor 0.25 is not what yfinance uses, 
            # yfinance uses the split ratio directly, e.g. 4.0 for 4-for-1).
            # Polygon gives 'split_from' and 'split_to'. 4-for-1 is to=4, from=1.
            to_val = float(r.get("split_to", 1))
            from_val = float(r.get("split_from", 1))
            ratio = to_val / from_val if from_val != 0 else 1.0
            data.append({"date": pd.Timestamp(ex_date), "split": ratio})
    if not data:
        return pd.Series(dtype=float)
    df = pd.DataFrame(data).set_index("date")
    return df.groupby("date")["split"].prod()

def _fetch_equity_polygon(
    symbol: str,
    *,
    from_date: date,
    to_date: date,
    api_key: str,
    adjusted: bool = True,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    if not api_key:
        raise ValueError("Polygon API key is required for Polygon requests.")

    url = (
        f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{from_date.isoformat()}/{to_date.isoformat()}"
    )
    params = {
        "adjusted": str(bool(adjusted)).lower(),
        "sort": "asc",
        "limit": str(DEFAULT_POLYGON_LIMIT),
        "apiKey": api_key,
    }
    payload = _safe_json(_http_get(url, params=params, timeout_s=timeout_s, session=session))

    if payload.get("status") == "ERROR":
        raise RuntimeError(payload.get("error") or payload.get("message") or str(payload))

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"No Polygon results returned for {symbol}.")

    divs = _fetch_polygon_dividends(symbol, api_key, from_date, to_date)
    splits = _fetch_polygon_splits(symbol, api_key, from_date, to_date)


    frame = pd.DataFrame(results)
    standardized = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize(),
            "open": frame.get("o"),
            "high": frame.get("h"),
            "low": frame.get("l"),
            "close": frame.get("c"),
            "adjusted_close": frame.get("c"),
            "volume": frame.get("v"),
            "dividends": 0.0,
            "stock_splits": 0.0,
        }
    ).set_index("date")

    if not divs.empty:
        standardized["dividends"] = standardized.index.map(divs).fillna(0.0)
    if not splits.empty:
        standardized["stock_splits"] = standardized.index.map(splits).fillna(0.0)

    return _standardize_price_frame(standardized.reset_index())


def _fetch_equity_alphavantage(
    symbol: str,
    *,
    from_date: date,
    to_date: date,
    api_key: str,
    adjusted: bool = True,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    if not api_key:
        raise ValueError("Alpha Vantage API key is required for Alpha Vantage requests.")

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "datatype": "json",
        "apikey": api_key,
    }
    payload = _safe_json(
        _http_get(ALPHAVANTAGE_BASE_URL, params=params, timeout_s=timeout_s, session=session)
    )

    if "Error Message" in payload:
        raise RuntimeError(payload["Error Message"])
    if "Note" in payload:
        raise RuntimeError(payload["Note"])
    if "Information" in payload:
        raise RuntimeError(payload["Information"])

    series_key = next((k for k in payload if "Time Series" in k), None)
    if not series_key:
        raise ValueError(f"Unexpected Alpha Vantage response for {symbol}: {payload}")

    rows = []
    for row_date, values in payload[series_key].items():
        rows.append(
            {
                "date": row_date,
                "open": values.get("1. open"),
                "high": values.get("2. high"),
                "low": values.get("3. low"),
                "close": values.get("4. close"),
                "adjusted_close": values.get("5. adjusted close", values.get("4. close")),
                "volume": values.get("6. volume", values.get("5. volume")),
                "dividends": values.get("7. dividend amount", 0.0),
                "stock_splits": values.get("8. split coefficient", 0.0),
            }
        )

    frame = _standardize_price_frame(pd.DataFrame(rows))
    sliced = frame.loc[(frame.index.date >= from_date) & (frame.index.date <= to_date)]
    if sliced.empty:
        raise ValueError(f"Alpha Vantage returned no rows for {symbol} in the requested window.")
    return sliced


def fetch_equity_prices(
    symbol: str,
    *,
    from_date: Any,
    to_date: Any,
    adjusted: bool = True,
    polygon_api_key: Optional[str] = None,
    alphavantage_api_key: Optional[str] = None,
    prefer_provider: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch adjusted daily OHLCV data and route automatically across providers.

    Free-tier-aware routing:
    - yfinance first for broad daily-history coverage and best hit rate.
    - Polygon next when a free Polygon key can cover the requested 2-year window.
    - Alpha Vantage last when the requested range fits within the latest 100 daily points.
    """
    normalized_symbol = _normalize_symbol(symbol)
    start, end = _coerce_date_window(from_date, to_date)
    polygon_key = polygon_api_key or os.getenv("POLYGON_API_KEY", "")
    alpha_key = alphavantage_api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")

    providers: List[Tuple[str, str]] = []
    if prefer_provider:
        providers.append((prefer_provider.lower(), "explicit provider preference"))
    else:
        providers.append(("yfinance", "best free hit rate for broad daily history"))
        if polygon_key and _polygon_free_covers(start, end):
            providers.append(("polygon", "requested window fits Polygon free historical coverage"))
        if alpha_key and _alphavantage_free_covers_daily(start, end):
            providers.append(("alphavantage", "requested window fits Alpha Vantage free compact history"))
        if polygon_key and "polygon" not in [name for name, _ in providers]:
            providers.append(("polygon", "fallback if provider plan permits the requested range"))
        if alpha_key and "alphavantage" not in [name for name, _ in providers]:
            providers.append(("alphavantage", "last-resort fallback if compact history is sufficient"))

    attempts: List[ProviderAttempt] = []
    errors: List[str] = []

    for provider_name, reason in providers:
        try:
            if provider_name == "yfinance":
                data = _fetch_equity_yfinance(normalized_symbol, from_date=start, to_date=end, adjusted=adjusted)
            elif provider_name == "polygon":
                data = _fetch_equity_polygon(
                    normalized_symbol,
                    from_date=start,
                    to_date=end,
                    api_key=polygon_key,
                    adjusted=adjusted,
                )
            elif provider_name in {"alphavantage", "alpha_vantage"}:
                data = _fetch_equity_alphavantage(
                    normalized_symbol,
                    from_date=start,
                    to_date=end,
                    api_key=alpha_key,
                    adjusted=adjusted,
                )
                provider_name = "alphavantage"
            else:
                raise ValueError(f"Unsupported provider: {provider_name}")

            if data.empty:
                raise ValueError(f"{provider_name} returned an empty dataframe.")

            attempts.append(ProviderAttempt(provider=provider_name, reason=reason, success=True))
            data.attrs["provider_attempts"] = attempts
            return _with_metadata(
                data,
                FetchMetadata(
                    asset_type="equity",
                    provider=provider_name,
                    symbol=normalized_symbol,
                    from_date=start,
                    to_date=end,
                    adjusted=adjusted,
                ),
            )
        except Exception as exc:
            attempts.append(
                ProviderAttempt(provider=provider_name, reason=reason, success=False, error=str(exc))
            )
            errors.append(f"{provider_name}: {exc}")

    message = (
        f"Unable to fetch equity prices for {normalized_symbol} from {start} to {end}. "
        f"Attempts: {' | '.join(errors) if errors else 'none'}"
    )
    raise PriceFetchError(message)


def fetch_multiple_equity_prices(
    symbols: Any,
    *,
    from_date: Any,
    to_date: Any,
    adjusted: bool = True,
    polygon_api_key: Optional[str] = None,
    alphavantage_api_key: Optional[str] = None,
    prefer_provider: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch daily equity OHLCV for multiple tickers.

    Returns a long-form dataframe indexed by date with a `symbol` column plus the
    standard OHLCV fields. Individual per-symbol failures are tracked in attrs.
    """
    tickers = parse_symbols(symbols)
    frames: List[pd.DataFrame] = []
    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for ticker in tickers:
        try:
            frame = fetch_equity_prices(
                symbol=ticker,
                from_date=from_date,
                to_date=to_date,
                adjusted=adjusted,
                polygon_api_key=polygon_api_key,
                alphavantage_api_key=alphavantage_api_key,
                prefer_provider=prefer_provider,
            ).copy()
            frame.insert(0, "symbol", ticker)
            frames.append(frame)

            metadata: Optional[FetchMetadata] = frame.attrs.get("metadata")
            attempts = frame.attrs.get("provider_attempts", [])
            successes.append(
                {
                    "symbol": ticker,
                    "provider": metadata.provider if metadata else "",
                    "rows": len(frame),
                    "attempts": attempts,
                }
            )
        except Exception as exc:
            failures.append({"symbol": ticker, "error": str(exc)})

    if not frames:
        formatted_failures = " | ".join(f"{item['symbol']}: {item['error']}" for item in failures)
        raise PriceFetchError(f"Unable to fetch data for any requested ticker. {formatted_failures}")

    combined = pd.concat(frames, axis=0).sort_index()
    combined.attrs["multi_fetch_successes"] = successes
    combined.attrs["multi_fetch_failures"] = failures
    combined.attrs["requested_symbols"] = tickers
    combined.attrs["metadata"] = {
        "asset_type": "equity_multi",
        "symbols_requested": tickers,
        "symbols_returned": [item["symbol"] for item in successes],
        "from_date": _parse_date(from_date, field_name="from_date").isoformat(),
        "to_date": _parse_date(to_date, field_name="to_date").isoformat(),
        "adjusted": adjusted,
    }
    return combined[MULTI_SYMBOL_COLUMNS]


def _fetch_option_polygon(
    *,
    contract: OptionContract,
    from_date: date,
    to_date: date,
    api_key: str,
    adjusted: bool = True,
    multiplier: int = 1,
    timespan: str = "day",
    timeout_s: int = DEFAULT_TIMEOUT_S,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    if not api_key:
        raise ValueError("Polygon API key is required for option requests.")

    option_ticker = contract.polygon_ticker
    url = (
        f"{POLYGON_BASE_URL}/v2/aggs/ticker/{option_ticker}/range/"
        f"{multiplier}/{timespan}/{from_date.isoformat()}/{to_date.isoformat()}"
    )
    params = {
        "adjusted": str(bool(adjusted)).lower(),
        "sort": "asc",
        "limit": str(DEFAULT_POLYGON_LIMIT),
        "apiKey": api_key,
    }
    payload = _safe_json(_http_get(url, params=params, timeout_s=timeout_s, session=session))

    if payload.get("status") == "ERROR":
        raise RuntimeError(payload.get("error") or payload.get("message") or str(payload))

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        msg = f"Polygon returned no option aggregate bars for {option_ticker}."
        if from_date > contract.expiry:
            msg += f"\n\nWARNING: The requested 'From' date ({from_date}) is after the option's expiry date ({contract.expiry})."
        raise ValueError(msg)


    frame = pd.DataFrame(results)
    standardized = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize(),
            "open": frame.get("o"),
            "high": frame.get("h"),
            "low": frame.get("l"),
            "close": frame.get("c"),
            "adjusted_close": frame.get("c"),
            "volume": frame.get("v"),
            "dividends": 0.0,
            "stock_splits": 0.0,
        }
    ).set_index("date")

    if not divs.empty:
        standardized["dividends"] = standardized.index.map(divs).fillna(0.0)
    if not splits.empty:
        standardized["stock_splits"] = standardized.index.map(splits).fillna(0.0)

    return _standardize_price_frame(standardized.reset_index())


def fetch_option_prices(
    *,
    from_date: Any,
    to_date: Any,
    option_ticker: Optional[str] = None,
    underlying: Optional[str] = None,
    expiry: Optional[Any] = None,
    right: Optional[str] = None,
    strike: Optional[float] = None,
    adjusted: bool = True,
    polygon_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch historical option OHLCV bars from Polygon.

    Users can provide either:
    - `option_ticker="O:SPY251219C00640000"`
    - or `underlying="SPY", expiry="2025-12-19", right="C", strike=640`
    """
    start, end = _coerce_date_window(from_date, to_date)
    contract = _build_option_contract(
        option_ticker=option_ticker,
        underlying=underlying,
        expiry=expiry,
        right=right,
        strike=strike,
    )
    polygon_key = polygon_api_key or os.getenv("POLYGON_API_KEY", "")

    data = _fetch_option_polygon(
        contract=contract,
        from_date=start,
        to_date=end,
        api_key=polygon_key,
        adjusted=adjusted,
    )
    return _with_metadata(
        data,
        FetchMetadata(
            asset_type="option",
            provider="polygon",
            symbol=contract.underlying,
            option_ticker=contract.polygon_ticker,
            from_date=start,
            to_date=end,
            adjusted=adjusted,
        ),
    )


def fetch_prices(
    *,
    asset_type: Literal["equity", "option"],
    symbol: Optional[str] = None,
    from_date: Any,
    to_date: Any,
    adjusted: bool = True,
    prefer_provider: Optional[str] = None,
    option_ticker: Optional[str] = None,
    expiry: Optional[Any] = None,
    right: Optional[str] = None,
    strike: Optional[float] = None,
    polygon_api_key: Optional[str] = None,
    alphavantage_api_key: Optional[str] = None,
) -> pd.DataFrame:
    if asset_type == "equity":
        if not symbol:
            raise ValueError("symbol is required for equity requests.")
        return fetch_equity_prices(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            adjusted=adjusted,
            polygon_api_key=polygon_api_key,
            alphavantage_api_key=alphavantage_api_key,
            prefer_provider=prefer_provider,
        )

    if asset_type == "option":
        return fetch_option_prices(
            from_date=from_date,
            to_date=to_date,
            option_ticker=option_ticker,
            underlying=symbol,
            expiry=expiry,
            right=right,
            strike=strike,
            adjusted=adjusted,
            polygon_api_key=polygon_api_key,
        )

    raise ValueError("asset_type must be 'equity' or 'option'.")


def fetch_symbol_overview_av(
    symbol: str,
    api_key: str,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Fetch Alpha Vantage company overview for a symbol.
    Returns metadata including Sector and Industry.
    """
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": api_key,
    }
    response = _http_get(ALPHAVANTAGE_BASE_URL, params=params, timeout_s=timeout_s, session=session)
    return _safe_json(response)




def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch adjusted daily OHLCV equity data or historical option bars."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    equity = subparsers.add_parser("equity", help="Fetch equity OHLCV data.")
    equity.add_argument("--symbol", required=True, help="Equity ticker symbol, e.g. AAPL")
    equity.add_argument("--from", dest="from_date", required=True, help="Start date YYYY-MM-DD")
    equity.add_argument("--to", dest="to_date", required=True, help="End date YYYY-MM-DD")
    equity.add_argument(
        "--provider",
        dest="prefer_provider",
        choices=["yfinance", "polygon", "alphavantage"],
        help="Force a provider instead of automatic routing.",
    )
    equity.add_argument(
        "--unadjusted",
        action="store_true",
        help="Return raw OHLC fields when the provider supports it.",
    )
    equity.add_argument("--csv", dest="csv_path", help="Optional CSV export path.")

    option = subparsers.add_parser("option", help="Fetch historical option OHLCV bars from Polygon.")
    option.add_argument("--from", dest="from_date", required=True, help="Start date YYYY-MM-DD")
    option.add_argument("--to", dest="to_date", required=True, help="End date YYYY-MM-DD")
    option.add_argument("--option-ticker", help="Polygon option ticker, e.g. O:SPY251219C00640000")
    option.add_argument("--symbol", help="Underlying ticker, e.g. SPY")
    option.add_argument("--expiry", help="Expiration date YYYY-MM-DD")
    option.add_argument("--right", help="C/CALL or P/PUT")
    option.add_argument("--strike", type=float, help="Option strike, e.g. 640")
    option.add_argument("--csv", dest="csv_path", help="Optional CSV export path.")

    return parser


def _print_summary(frame: pd.DataFrame) -> None:
    metadata = frame.attrs.get("metadata")
    if metadata:
        print(f"provider={metadata.provider}")
        print(f"asset_type={metadata.asset_type}")
        print(f"symbol={metadata.symbol}")
        if metadata.option_ticker:
            print(f"option_ticker={metadata.option_ticker}")
        print(f"from={metadata.from_date.isoformat()} to={metadata.to_date.isoformat()}")
    print(f"rows={len(frame)}")
    if not frame.empty:
        print(frame.head(3).to_string())
        if len(frame) > 3:
            print("...")
            print(frame.tail(3).to_string())


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "equity":
        frame = fetch_equity_prices(
            symbol=args.symbol,
            from_date=args.from_date,
            to_date=args.to_date,
            adjusted=not args.unadjusted,
            prefer_provider=args.prefer_provider,
        )
    else:
        frame = fetch_option_prices(
            from_date=args.from_date,
            to_date=args.to_date,
            option_ticker=args.option_ticker,
            underlying=args.symbol,
            expiry=args.expiry,
            right=args.right,
            strike=args.strike,
        )

    if getattr(args, "csv_path", None):
        frame.to_csv(args.csv_path, index=True)
        print(f"saved_csv={args.csv_path}")

    _print_summary(frame)


if __name__ == "__main__":
    main()
