from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from fetch_prices import (
    ALPHAVANTAGE_BASE_URL,
    fetch_multiple_equity_prices,
    fetch_symbol_overview_av,
    parse_symbols,
)


DEFAULT_BENCHMARK_SYMBOL = "SPY"
TRADING_DAYS_PER_YEAR = 252
PRICE_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "dividends",
    "stock_splits",
]


@dataclass(frozen=True)
class CorrelationConfig:
    remove_earnings: bool = False
    earnings_window: Tuple[int, int] = (3, 3)  # (pre_days, post_days)
    winsorize: bool = False
    winsorize_limits: Tuple[float, float] = (0.01, 0.01)
    decay_factor: Optional[float] = None  # e.g. 0.94
    regime_filter: Literal["all", "bull", "bear"] = "all"
    benchmark_symbol: str = "SPY"
    sector_relative: bool = False
    rolling_window: int = 63      # window for rolling correlation chart
    regression_lookback: int = 252  # window for PIT OLS regression


@dataclass(frozen=True)
class EarningsImpact:
    symbol: str
    fiscal_date_ending: Optional[date]
    reported_date: date
    report_time: str
    impact_date: date


SECTOR_ETF_MAP = {
    "TECHNOLOGY": "XLK",
    "FINANCIAL SERVICES": "XLF",
    "FINANCIALS": "XLF",
    "HEALTHCARE": "XLV",
    "CONSUMER CYCLICAL": "XLY",
    "CONSUMER DISCRETIONARY": "XLY",
    "CONSUMER STAPLES": "XLP",
    "CONSUMER DEFENSIVE": "XLP",
    "ENERGY": "XLE",
    "BASIC MATERIALS": "XLB",
    "MATERIALS": "XLB",
    "INDUSTRIALS": "XLI",
    "UTILITIES": "XLU",
    "REAL ESTATE": "XLRE",
    "COMMUNICATION SERVICES": "XLC",
}


def winsorize_series(series: pd.Series, limits: Tuple[float, float] = (0.01, 0.01)) -> pd.Series:
    if series.empty:
        return series
    lower, upper = series.quantile(limits[0]), series.quantile(1.0 - limits[1])
    return series.clip(lower=lower, upper=upper)


def apply_decay_weights(returns: pd.Series, decay_factor: float) -> pd.Series:
    """Multiplies returns by exponential decay weights. Latest data has weight 1.0."""
    n = len(returns)
    weights = decay_factor ** np.arange(n - 1, -1, -1)
    return returns * weights


def get_regime_mask(benchmark_returns: pd.Series, regime: str, window: int = 200) -> pd.Series:
    """Returns a boolean mask where True indicates the requested regime."""
    if regime == "all":
        return pd.Series(True, index=benchmark_returns.index)

    # Simple regime definition: Bull if 200d MA of prices is upward sloping or returns > 0
    # For this implementation, we use a simpler rolling average of returns > 0
    rolling_avg = benchmark_returns.rolling(window=window, min_periods=1).mean()
    if regime == "bull":
        return rolling_avg > 0
    if regime == "bear":
        return rolling_avg <= 0
    return pd.Series(True, index=benchmark_returns.index)


def compute_rolling_correlation(
    s1: pd.Series, s2: pd.Series, window: int = 63
) -> pd.Series:
    return s1.rolling(window=window).corr(s2)


def resolve_sector_etf(symbol: str, api_key: str) -> Optional[str]:
    try:
        overview = fetch_symbol_overview_av(symbol, api_key)
        sector = overview.get("Sector", "").upper()
        return SECTOR_ETF_MAP.get(sector)
    except Exception:
        return None


def _sleep_s(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _http_get_json_with_retries(
    url: str,
    *,
    params: Dict[str, Any],
    timeout_s: int = 30,
    max_retries: int = 6,
    backoff_base_s: float = 1.0,
    backoff_cap_s: float = 60.0,
    retry_statuses: Iterable[int] = (429, 500, 502, 503, 504),
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    sess = session or requests.Session()
    last_err: Optional[BaseException] = None

    for attempt in range(max_retries + 1):
        try:
            response = sess.get(url, params=params, timeout=timeout_s)
            if response.status_code in set(int(x) for x in retry_statuses):
                retry_after = response.headers.get("Retry-After")
                wait_s = 0.0
                if retry_after is not None:
                    try:
                        wait_s = float(retry_after)
                    except Exception:
                        wait_s = 0.0
                if wait_s <= 0.0:
                    expo = min(backoff_cap_s, backoff_base_s * (2.0**attempt))
                    wait_s = random.random() * expo
                if attempt >= max_retries:
                    response.raise_for_status()
                _sleep_s(wait_s)
                continue

            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object, got {type(payload)}")
            return payload
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_err = exc
            if attempt >= max_retries:
                raise
            expo = min(backoff_cap_s, backoff_base_s * (2.0**attempt))
            _sleep_s(random.random() * expo)
        except Exception as exc:
            last_err = exc
            raise

    raise RuntimeError(f"HTTP GET failed after retries: {last_err}")


def build_price_field_matrix(price_frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in PRICE_FIELDS:
        raise ValueError(f"Unsupported field: {field}. Choose from {PRICE_FIELDS}.")
    if price_frame.empty:
        return pd.DataFrame()

    frame = price_frame.reset_index().copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    matrix = frame.pivot_table(index="date", columns="symbol", values=field, aggfunc="last").sort_index()
    matrix.index.name = "date"
    matrix.columns.name = "symbol"
    return matrix


def fetch_price_panel(
    *,
    symbols: Sequence[str] | str,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    from_date: Any,
    to_date: Any,
    adjusted: bool = True,
    polygon_api_key: Optional[str] = None,
    alphavantage_api_key: Optional[str] = None,
    prefer_provider: Optional[str] = None,
    timespan: str = "day",
    multiplier: int = 1,
) -> Dict[str, Any]:
    tickers = parse_symbols(",".join(symbols) if not isinstance(symbols, str) else symbols)
    benchmark = benchmark_symbol.strip().upper()
    requested = list(dict.fromkeys([*tickers, benchmark]))

    raw = fetch_multiple_equity_prices(
        requested,
        from_date=from_date,
        to_date=to_date,
        adjusted=adjusted,
        timespan=timespan,
        multiplier=multiplier,
        polygon_api_key=polygon_api_key,
        alphavantage_api_key=alphavantage_api_key,
        prefer_provider=prefer_provider,
    )

    matrices = {field: build_price_field_matrix(raw, field) for field in PRICE_FIELDS}
    close_field = "adjusted_close" if adjusted else "close"

    return {
        "raw": raw,
        "matrices": matrices,
        "close_matrix": matrices[close_field],
        "open_matrix": matrices["open"],
        "benchmark_symbol": benchmark,
        "symbols": tickers,
        "requested_symbols": requested,
        "adjusted": adjusted,
    }


def compute_log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    numeric = prices.apply(pd.to_numeric, errors="coerce") if isinstance(prices, pd.DataFrame) else pd.to_numeric(prices, errors="coerce")
    log_prices = np.log(numeric)
    returns = log_prices.diff()
    return returns.dropna(how="all") if isinstance(returns, pd.DataFrame) else returns.dropna()


def compute_simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    numeric = prices.apply(pd.to_numeric, errors="coerce") if isinstance(prices, pd.DataFrame) else pd.to_numeric(prices, errors="coerce")
    returns = numeric.pct_change()
    return returns.dropna(how="all") if isinstance(returns, pd.DataFrame) else returns.dropna()


def compute_correlation_matrix(returns: pd.DataFrame, *, method: str = "pearson") -> pd.DataFrame:
    clean = returns.dropna(how="all")
    if clean.empty:
        return pd.DataFrame()
    return clean.corr(method=method)


def compute_drawdown_series(prices: pd.Series) -> pd.DataFrame:
    clean = pd.to_numeric(prices, errors="coerce").dropna()
    if clean.empty:
        return pd.DataFrame(columns=["price", "running_max", "drawdown"])
    running_max = clean.cummax()
    drawdown = clean.div(running_max) - 1.0
    return pd.DataFrame({"price": clean, "running_max": running_max, "drawdown": drawdown})


def summarize_drawdown(prices: pd.Series) -> Dict[str, Any]:
    dd = compute_drawdown_series(prices)
    if dd.empty:
        return {
            "max_drawdown": float("nan"),
            "max_drawdown_date": pd.NaT,
            "peak_date": pd.NaT,
            "recovery_date": pd.NaT,
            "days_to_trough": float("nan"),
            "days_to_recovery": float("nan"),
        }

    trough_date = dd["drawdown"].idxmin()
    max_drawdown = float(dd.loc[trough_date, "drawdown"])

    peak_window = dd.loc[:trough_date, "price"]
    peak_date = peak_window.idxmax()
    peak_price = float(dd.loc[peak_date, "price"])

    recovery_date = pd.NaT
    post_trough = dd.loc[trough_date:]
    recovered = post_trough.index[post_trough["price"] >= peak_price]
    if len(recovered) > 0:
        recovery_date = recovered[0]

    days_to_trough = float((trough_date - peak_date).days) if pd.notna(peak_date) else float("nan")
    days_to_recovery = (
        float((recovery_date - trough_date).days) if pd.notna(recovery_date) else float("nan")
    )

    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_date": trough_date,
        "peak_date": peak_date,
        "recovery_date": recovery_date,
        "days_to_trough": days_to_trough,
        "days_to_recovery": days_to_recovery,
    }


def compute_annualized_volatility(returns: pd.Series, *, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def compute_annualized_return(returns: pd.Series, *, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    compounded = float((1.0 + clean).prod())
    n = len(clean)
    if n == 0 or compounded <= 0:
        return float("nan")
    return float(compounded ** (periods_per_year / n) - 1.0)


def compute_sharpe_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    daily_rf = risk_free_rate / periods_per_year
    excess = clean - daily_rf
    denom = excess.std(ddof=1)
    if denom == 0 or pd.isna(denom):
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / denom)


def compute_information_ratio(
    active_returns: pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    clean = pd.to_numeric(active_returns, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    denom = clean.std(ddof=1)
    if denom == 0 or pd.isna(denom):
        return float("nan")
    return float(np.sqrt(periods_per_year) * clean.mean() / denom)


def run_simple_regression(y: pd.Series, x: pd.Series) -> Dict[str, float]:
    df = pd.concat([y, x], axis=1).dropna()
    if df.shape[0] < 3:
        raise ValueError("Not enough overlapping return observations to run regression (need >= 3).")

    yv = df.iloc[:, 0].to_numpy(dtype=float)
    xv = df.iloc[:, 1].to_numpy(dtype=float)
    n = float(len(xv))

    x_mean = float(xv.mean())
    y_mean = float(yv.mean())

    sxx = float(((xv - x_mean) ** 2).sum())
    if sxx == 0.0:
        raise ValueError("Benchmark returns have zero variance in the selected window.")

    sxy = float(((xv - x_mean) * (yv - y_mean)).sum())
    beta = sxy / sxx
    alpha = y_mean - beta * x_mean

    y_hat = alpha + beta * xv
    resid = yv - y_hat

    sse = float((resid**2).sum())
    stderr_reg = float(np.sqrt(sse / (n - 2.0)))
    stderr_beta = float(stderr_reg / np.sqrt(sxx))
    corr = float(np.corrcoef(xv, yv)[0, 1]) if len(xv) > 1 else float("nan")
    r2 = corr * corr

    return {
        "n_obs": float(len(xv)),
        "alpha": float(alpha),
        "beta": float(beta),
        "r": float(corr),
        "r_squared": float(r2),
        "std_error_regression": float(stderr_reg),
        "std_error_beta": float(stderr_beta),
    }


def rolling_ols_params_point_in_time(
    y: pd.Series,
    x: pd.Series,
    *,
    lookback: int,
) -> pd.DataFrame:
    if lookback < 3:
        raise ValueError("lookback must be >= 3 to estimate OLS with an intercept.")

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame(index=y.index, columns=["alpha", "beta", "std_error_regression", "n_obs"])

    n = float(lookback)
    sx = df["x"].rolling(window=lookback, min_periods=lookback).sum().shift(1)
    sy = df["y"].rolling(window=lookback, min_periods=lookback).sum().shift(1)
    sxx = (df["x"] ** 2).rolling(window=lookback, min_periods=lookback).sum().shift(1)
    syy = (df["y"] ** 2).rolling(window=lookback, min_periods=lookback).sum().shift(1)
    sxy = (df["x"] * df["y"]).rolling(window=lookback, min_periods=lookback).sum().shift(1)

    x_mean = sx / n
    y_mean = sy / n
    sxx_c = sxx - n * (x_mean**2)
    sxy_c = sxy - n * (x_mean * y_mean)

    beta = sxy_c / sxx_c
    alpha = y_mean - beta * x_mean

    sse = syy - 2.0 * alpha * sy - 2.0 * beta * sxy + 2.0 * alpha * beta * sx + n * (
        alpha**2
    ) + (beta**2) * sxx
    sse = sse.clip(lower=0.0)
    stderr_reg = np.sqrt(sse / (n - 2.0))

    params = pd.DataFrame(
        {
            "alpha": alpha,
            "beta": beta,
            "std_error_regression": stderr_reg,
            "n_obs": lookback,
        },
        index=df.index,
    )
    return params.reindex(y.index)


def _validate_no_cheating(
    *,
    reg_df: pd.DataFrame,
    params: pd.DataFrame,
    lookback: int,
    validate_date: Optional[pd.Timestamp] = None,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> None:
    if reg_df.empty or params.empty:
        return

    t: Optional[pd.Timestamp] = None
    if validate_date is not None and validate_date in params.index and pd.notna(params.loc[validate_date, "beta"]):
        t = validate_date

    if t is None:
        non_null = params["beta"].dropna()
        if non_null.empty:
            return
        t = pd.Timestamp(non_null.index[0])

    pos = int(reg_df.index.get_indexer([t])[0])
    if pos < 0:
        return
    end = pos
    start = end - int(lookback)
    if start < 0:
        return

    window = reg_df.iloc[start:end].dropna()
    if window.shape[0] != int(lookback):
        return

    expected = run_simple_regression(window["asset_logret"], window["benchmark_logret"])
    got_alpha = float(params.loc[t, "alpha"])
    got_beta = float(params.loc[t, "beta"])
    got_se = float(params.loc[t, "std_error_regression"])

    if not np.isfinite(got_alpha) or not np.isfinite(got_beta) or not np.isfinite(got_se):
        raise RuntimeError(f"PIT validation failed at {t.date().isoformat()}: non-finite parameters.")
    if not np.isclose(got_beta, float(expected["beta"]), atol=atol, rtol=rtol):
        raise RuntimeError(f"PIT validation failed at {t.date().isoformat()}: beta mismatch.")
    if not np.isclose(got_alpha, float(expected["alpha"]), atol=atol, rtol=rtol):
        raise RuntimeError(f"PIT validation failed at {t.date().isoformat()}: alpha mismatch.")
    if not np.isclose(got_se, float(expected["std_error_regression"]), atol=atol, rtol=rtol):
        raise RuntimeError(f"PIT validation failed at {t.date().isoformat()}: std error mismatch.")
    if window.index.max() >= t:
        raise RuntimeError(f"PIT validation failed at {t.date().isoformat()}: current/future data leaked.")


def _parse_iso_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _next_trading_day(trading_days: List[date], current_day: date) -> Optional[date]:
    lo, hi = 0, len(trading_days)
    while lo < hi:
        mid = (lo + hi) // 2
        if trading_days[mid] <= current_day:
            lo = mid + 1
        else:
            hi = mid
    return trading_days[lo] if lo < len(trading_days) else None


def _normalize_report_time(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def fetch_earnings_av(symbol: str, api_key: str, *, timeout_s: int = 30) -> Dict[str, Any]:
    params = {"function": "EARNINGS", "symbol": symbol, "apikey": api_key}
    for attempt in range(7):
        payload = _http_get_json_with_retries(
            ALPHAVANTAGE_BASE_URL,
            params=params,
            timeout_s=timeout_s,
            max_retries=6,
            backoff_base_s=1.0,
            backoff_cap_s=60.0,
        )
        if "Note" in payload and attempt < 6:
            _sleep_s(random.random() * min(60.0, 1.0 * (2.0**attempt)))
            continue
        break

    if "Error Message" in payload:
        raise ValueError(f"Alpha Vantage error for symbol '{symbol}': {payload['Error Message']}")
    if "Note" in payload:
        raise RuntimeError(f"Alpha Vantage note (often rate limit): {payload['Note']}")
    if "Information" in payload:
        raise RuntimeError(f"Alpha Vantage information message: {payload['Information']}")
    return payload


def compute_earnings_impacts(
    payload: Dict[str, Any],
    *,
    symbol: str,
    trading_days: Iterable[date],
    window_start: date,
    window_end: date,
    pre_window_buffer_days: int = 7,
    amc_fallback_calendar_plus_one: bool = False,
) -> List[EarningsImpact]:
    quarterly = payload.get("quarterlyEarnings")
    if not isinstance(quarterly, list):
        raise ValueError("Unexpected Alpha Vantage earnings payload: missing 'quarterlyEarnings'.")

    trading_list = sorted(set(trading_days))
    if not trading_list:
        return []

    earliest = window_start - timedelta(days=max(0, int(pre_window_buffer_days)))
    out: List[EarningsImpact] = []

    for entry in quarterly:
        if not isinstance(entry, dict):
            continue
        reported_date = _parse_iso_date(entry.get("reportedDate"))
        if reported_date is None or reported_date < earliest or reported_date > window_end:
            continue

        fiscal = _parse_iso_date(entry.get("fiscalDateEnding"))
        report_time = _normalize_report_time(entry.get("reportTime")) or "after_market_close"
        if report_time in {"before_market_open", "beforemarketopen", "bmo"}:
            impact = reported_date
        else:
            impact = _next_trading_day(trading_list, reported_date)
            if impact is None and amc_fallback_calendar_plus_one:
                impact = reported_date + timedelta(days=1)

        if impact is not None and window_start <= impact <= window_end:
            out.append(
                EarningsImpact(
                    symbol=symbol,
                    fiscal_date_ending=fiscal,
                    reported_date=reported_date,
                    report_time=report_time,
                    impact_date=impact,
                )
            )

    out.sort(key=lambda item: item.reported_date)
    return out


def impacts_to_frame(impacts: List[EarningsImpact]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": item.symbol,
                "fiscalDateEnding": item.fiscal_date_ending.isoformat() if item.fiscal_date_ending else None,
                "reportedDate": item.reported_date.isoformat(),
                "reportTime": item.report_time,
                "impactDate": item.impact_date.isoformat(),
            }
            for item in impacts
        ]
    )


def _earnings_removed_dates(
    close_matrix: pd.DataFrame,
    *,
    symbol: str,
    alpha_vantage_api_key: Optional[str],
    pre_window_buffer_days: int = 7,
    removal_window: Tuple[int, int] = (3, 3),
) -> tuple[list[pd.Timestamp], pd.DataFrame]:
    if not alpha_vantage_api_key:
        return [], pd.DataFrame()

    trading_days = [stamp.date() for stamp in close_matrix.index.to_pydatetime()]
    if not trading_days:
        return [], pd.DataFrame()

    payload = fetch_earnings_av(symbol, alpha_vantage_api_key, timeout_s=30)
    impacts = compute_earnings_impacts(
        payload,
        symbol=symbol,
        trading_days=trading_days,
        window_start=trading_days[0],
        window_end=trading_days[-1],
        pre_window_buffer_days=pre_window_buffer_days,
        amc_fallback_calendar_plus_one=False,
    )

    removed: set[pd.Timestamp] = set()
    pre_days, post_days = removal_window
    for impact in impacts:
        impact_ts = pd.Timestamp(impact.impact_date)
        # Remove a window of calendar days around the impact
        start_ts = impact_ts - pd.Timedelta(days=int(pre_days))
        end_ts = impact_ts + pd.Timedelta(days=int(post_days))
        
        mask = (close_matrix.index >= start_ts) & (close_matrix.index <= end_ts)
        removed.update(close_matrix.index[mask])

    return sorted(removed), impacts_to_frame(impacts)


def analyze_symbol_vs_benchmark(
    *,
    symbol: str,
    close_matrix: pd.DataFrame,
    open_matrix: pd.DataFrame,
    config: CorrelationConfig = CorrelationConfig(),
    alpha_vantage_api_key: Optional[str] = None,
    max_missing_vs_benchmark: float = 0.05,
    pre_window_buffer_days: int = 7,
    risk_free_rate: float = 0.0,
    sector_etf: Optional[str] = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, Any]:
    asset = symbol.strip().upper()
    benchmark = config.benchmark_symbol.strip().upper()

    if asset == benchmark:
        raise ValueError("symbol and benchmark_symbol must be different.")
    if asset not in close_matrix.columns:
        raise ValueError(f"{asset} not found in close matrix.")
    if benchmark not in close_matrix.columns:
        raise ValueError(f"{benchmark} not found in close matrix.")

    bench_index = close_matrix[benchmark].dropna().index.sort_values()
    asset_on_bench = close_matrix[asset].reindex(bench_index)
    missing_ratio = float(asset_on_bench.isna().mean()) if len(asset_on_bench) else 1.0
    if missing_ratio > float(max_missing_vs_benchmark):
        raise ValueError(
            f"Ticker '{asset}' is missing {missing_ratio:.2%} of benchmark '{benchmark}' days."
        )

    # 1. Align Main Pairs
    aligned = pd.concat(
        [
            open_matrix[asset].reindex(bench_index).rename(f"{asset}_open"),
            close_matrix[asset].reindex(bench_index).rename(f"{asset}_close"),
            open_matrix[benchmark].reindex(bench_index).rename(f"{benchmark}_open"),
            close_matrix[benchmark].reindex(bench_index).rename(f"{benchmark}_close"),
        ],
        axis=1,
    ).dropna(subset=[f"{asset}_close", f"{benchmark}_close"])

    if sector_etf and sector_etf in close_matrix.columns:
        aligned[f"{sector_etf}_close"] = close_matrix[sector_etf].reindex(aligned.index)
        aligned = aligned.dropna(subset=[f"{sector_etf}_close"])

    if aligned.empty:
        raise ValueError(f"No overlapping dates between {asset} and {benchmark}.")

    # 2. Earnings Removal
    removed_dates: list[pd.Timestamp] = []
    impacts_df = pd.DataFrame()
    if config.remove_earnings:
        removed_dates, impacts_df = _earnings_removed_dates(
            close_matrix,
            symbol=asset,
            alpha_vantage_api_key=alpha_vantage_api_key,
            pre_window_buffer_days=pre_window_buffer_days,
            removal_window=config.earnings_window,
        )

    # 3. Base Returns
    asset_logret = compute_log_returns(aligned[f"{asset}_close"]).rename("asset_logret")
    bench_logret = compute_log_returns(aligned[f"{benchmark}_close"]).rename("benchmark_logret")
    asset_simple = compute_simple_returns(aligned[f"{asset}_close"]).rename("asset_simple")
    bench_simple = compute_simple_returns(aligned[f"{benchmark}_close"]).rename("benchmark_simple")

    if removed_dates:
        asset_logret = asset_logret.drop(index=removed_dates, errors="ignore")
        bench_logret = bench_logret.drop(index=removed_dates, errors="ignore")
        asset_simple = asset_simple.drop(index=removed_dates, errors="ignore")
        bench_simple = bench_simple.drop(index=removed_dates, errors="ignore")

    reg_df = pd.concat([asset_logret, bench_logret, asset_simple, bench_simple], axis=1).dropna()

    # 4. Regime Filtering
    if config.regime_filter != "all":
        mask = get_regime_mask(bench_logret, config.regime_filter)
        reg_df = reg_df[mask.reindex(reg_df.index).fillna(False)]

    if reg_df.empty:
        raise ValueError("No observations remain after regime/earnings filtering.")

    # 5. Winsorization
    if config.winsorize:
        reg_df["asset_logret"] = winsorize_series(reg_df["asset_logret"], config.winsorize_limits)
        reg_df["benchmark_logret"] = winsorize_series(reg_df["benchmark_logret"], config.winsorize_limits)

    # 6. Decay Weighting (for summary stats, we'll use weighted correlation if requested)
    # Note: run_simple_regression doesn't support weights yet, we'll keep it simple for now
    # but we can apply decay to the returns series directly for the correlation call.
    final_asset_rets = reg_df["asset_logret"]
    final_bench_rets = reg_df["benchmark_logret"]
    if config.decay_factor is not None:
        final_asset_rets = apply_decay_weights(final_asset_rets, config.decay_factor)
        final_bench_rets = apply_decay_weights(final_bench_rets, config.decay_factor)

    # 7. Core Analytics
    params = rolling_ols_params_point_in_time(
        reg_df["asset_logret"], reg_df["benchmark_logret"], lookback=config.regression_lookback
    )
    model_return = (params["alpha"] + params["beta"] * reg_df["benchmark_logret"]).rename("model_return")
    residual = (reg_df["asset_logret"] - model_return).rename("residual")
    residual_z = (residual / params["std_error_regression"].clip(lower=1e-12)).rename("residual_z")
    outlier_mask = residual_z.abs() > 2.0

    regression = run_simple_regression(final_asset_rets, final_bench_rets)
    corr = float(final_asset_rets.corr(final_bench_rets))
    rolling_corr = compute_rolling_correlation(
        reg_df["asset_logret"], reg_df["benchmark_logret"], window=config.rolling_window
    )

    # 8. Sector Relative
    sector_corr = None
    if sector_etf and f"{sector_etf}_close" in aligned.columns:
        sector_rets = compute_log_returns(aligned[f"{sector_etf}_close"])
        if removed_dates:
            sector_rets = sector_rets.drop(index=removed_dates, errors="ignore")
        sector_corr = float(reg_df["asset_logret"].corr(sector_rets))

    # 9. Performance & Risk
    asset_dd = summarize_drawdown(aligned[f"{asset}_close"])
    bench_dd = summarize_drawdown(aligned[f"{benchmark}_close"])
    active_simple = (reg_df["asset_simple"] - reg_df["benchmark_simple"]).rename("active_return")

    summary = {
        "symbol": asset,
        "benchmark": benchmark,
        "sector_etf": sector_etf,
        "n_obs": int(regression["n_obs"]),
        "correlation": corr,
        "sector_correlation": sector_corr,
        "alpha": float(regression["alpha"]),
        "beta": float(regression["beta"]),
        "r_squared": float(regression["r_squared"]),
        "asset_annual_return": compute_annualized_return(reg_df["asset_simple"], periods_per_year=periods_per_year),
        "benchmark_annual_return": compute_annualized_return(reg_df["benchmark_simple"], periods_per_year=periods_per_year),
        "asset_sharpe": compute_sharpe_ratio(reg_df["asset_simple"], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
        "tracking_error": compute_annualized_volatility(active_simple, periods_per_year=periods_per_year),
        "information_ratio": compute_information_ratio(active_simple, periods_per_year=periods_per_year),
        "asset_max_drawdown": asset_dd["max_drawdown"],
        "outlier_count": int(outlier_mask.fillna(False).sum()),
        "rolling_correlation_latest": rolling_corr.iloc[-1] if not rolling_corr.dropna().empty else None,
    }

    # 10. Outliers Table (Keep institutional format)
    outlier_dates = residual_z.index[outlier_mask.fillna(False)]
    trade_date = aligned.index.to_series().shift(-1)
    
    outliers_table = pd.DataFrame(index=outlier_dates)
    outliers_table["symbol"] = asset
    outliers_table["Ticker Price"] = aligned.loc[outlier_dates, f"{asset}_close"]
    outliers_table["Beta"] = params.loc[outlier_dates, "beta"]
    outliers_table["residual_z"] = residual_z.loc[outlier_dates]
    outliers_table["trade date"] = trade_date.loc[outlier_dates]
    outliers_table = outliers_table.reset_index().sort_values("date").reset_index(drop=True)

    return {
        "summary": summary,
        "aligned_prices": aligned,
        "returns": reg_df,
        "rolling_params": params,
        "rolling_correlation": rolling_corr,
        "outliers_table": outliers_table,
        "removed_dates": removed_dates,
        "impacts_df": impacts_df,
    }




def analyze_symbols_vs_benchmark(
    *,
    symbols: Sequence[str] | str,
    config: CorrelationConfig = CorrelationConfig(),
    from_date: Any,
    to_date: Any,
    adjusted: bool = True,
    polygon_api_key: Optional[str] = None,
    alphavantage_api_key: Optional[str] = None,
    prefer_provider: Optional[str] = None,
    max_missing_vs_benchmark: float = 0.05,
    risk_free_rate: float = 0.0,
    manual_sector_etf: Optional[str] = None,
    timespan: str = "day",
    multiplier: int = 1,
) -> Dict[str, Any]:
    tickers = parse_symbols(",".join(symbols) if not isinstance(symbols, str) else symbols)
    benchmark = config.benchmark_symbol.strip().upper()
    
    # 1. Sector Resolution
    ticker_sectors = {}
    sector_etfs_to_fetch = set()
    if manual_sector_etf:
        # Manual override: apply the same sector ETF to all tickers
        for ticker in tickers:
            if ticker != benchmark:
                ticker_sectors[ticker] = manual_sector_etf
        sector_etfs_to_fetch.add(manual_sector_etf)
    elif config.sector_relative:
        for ticker in tickers:
            if ticker == benchmark:
                continue
            sector_etf = resolve_sector_etf(ticker, alphavantage_api_key or "")
            if sector_etf:
                ticker_sectors[ticker] = sector_etf
                sector_etfs_to_fetch.add(sector_etf)

    # 2. Fetch Price Panel
    all_symbols = list(dict.fromkeys([*tickers, benchmark, *sector_etfs_to_fetch]))
    panel = fetch_price_panel(
        symbols=all_symbols,
        benchmark_symbol=benchmark,
        from_date=from_date,
        to_date=to_date,
        adjusted=adjusted,
        timespan=timespan,
        multiplier=multiplier,
        polygon_api_key=polygon_api_key,
        alphavantage_api_key=alphavantage_api_key,
        prefer_provider=prefer_provider,
    )

    # Determine periods per year for annualization
    periods_per_year = TRADING_DAYS_PER_YEAR
    if timespan == "hour":
        periods_per_year = TRADING_DAYS_PER_YEAR * 7 # Assumption: 7 hour trading day
    elif timespan == "minute":
        periods_per_year = TRADING_DAYS_PER_YEAR * 7 * 60
    
    if multiplier > 1:
        periods_per_year = max(1, periods_per_year // multiplier)

    close_matrix = panel["close_matrix"]
    open_matrix = panel["open_matrix"]
    
    analyses: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    scenario_grid: List[Dict[str, Any]] = []

    # 3. Individual Analyses
    for ticker in tickers:
        if ticker == benchmark:
            continue
        try:
            analysis = analyze_symbol_vs_benchmark(
                symbol=ticker,
                close_matrix=close_matrix,
                open_matrix=open_matrix,
                config=config,
                alpha_vantage_api_key=alphavantage_api_key,
                max_missing_vs_benchmark=max_missing_vs_benchmark,
                risk_free_rate=risk_free_rate,
                sector_etf=ticker_sectors.get(ticker),
                periods_per_year=periods_per_year,
            )
            analyses[ticker] = analysis
            summary_rows.append(analysis["summary"])

            # 4. Scenario Grid (Bull vs Bear, Earnings Removed vs Included)
            # This is a bit expensive but very valuable for the user.
            for regime in ["bull", "bear"]:
                try:
                    scen_cfg = replace(config, regime_filter=regime)
                    scen_analysis = analyze_symbol_vs_benchmark(
                        symbol=ticker,
                        close_matrix=close_matrix,
                        open_matrix=open_matrix,
                        config=scen_cfg,
                        alpha_vantage_api_key=alphavantage_api_key,
                        max_missing_vs_benchmark=max_missing_vs_benchmark,
                        risk_free_rate=risk_free_rate,
                        sector_etf=ticker_sectors.get(ticker),
                        periods_per_year=periods_per_year,
                    )
                    scenario_grid.append({
                        "symbol": ticker,
                        "scenario": f"Regime: {regime.capitalize()}",
                        "correlation": scen_analysis["summary"]["correlation"],
                        "beta": scen_analysis["summary"]["beta"],
                    })
                except Exception:
                    pass

        except Exception as exc:
            failures.append({"symbol": ticker, "error": str(exc)})

    summary_table = pd.DataFrame(summary_rows).sort_values("symbol").reset_index(drop=True) if summary_rows else pd.DataFrame()
    scenario_table = pd.DataFrame(scenario_grid).sort_values(["symbol", "scenario"]) if scenario_grid else pd.DataFrame()
    
    close_returns = compute_log_returns(close_matrix)
    simple_returns = compute_simple_returns(close_matrix)

    return {
        "panel": panel,
        "summary_table": summary_table,
        "scenario_table": scenario_table,
        "analyses": analyses,
        "failures": failures,
        "correlation_matrix": compute_correlation_matrix(close_returns),
        "simple_return_correlation_matrix": compute_correlation_matrix(simple_returns),
    }





def generate_consolidated_stat_arb_signals(
    *,
    symbols: Sequence[str] | str,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    from_date: Any,
    to_date: Any,
    lookback: int = 252,
    adjusted: bool = True,
    polygon_api_key: Optional[str] = None,
    alphavantage_api_key: Optional[str] = None,
    prefer_provider: Optional[str] = None,
    max_missing_vs_benchmark: float = 0.05,
    timespan: str = "day",
    multiplier: int = 1,
) -> pd.DataFrame:
    config = CorrelationConfig(
        benchmark_symbol=benchmark_symbol,
        remove_earnings=False,
        rolling_window=lookback,
    )
    result = analyze_symbols_vs_benchmark(
        symbols=symbols,
        config=config,
        from_date=from_date,
        to_date=to_date,
        adjusted=adjusted,
        polygon_api_key=polygon_api_key,
        alphavantage_api_key=alphavantage_api_key,
        prefer_provider=prefer_provider,
        max_missing_vs_benchmark=max_missing_vs_benchmark,
        timespan=timespan,
        multiplier=multiplier,
    )

    rows: List[pd.DataFrame] = []
    for symbol, analysis in result["analyses"].items():
        table = analysis["outliers_table"].copy()
        if table.empty:
            continue
        formatted = pd.DataFrame(
            {
                "ticker": table["symbol"],
                "Anomaly Date": pd.to_datetime(table["date"], errors="coerce").dt.strftime("%m/%d/%Y"),
                "Ticker Price": pd.to_numeric(table["Ticker Price"], errors="coerce").round(6),
                "SPY Price": pd.to_numeric(table["Benchmark Price"], errors="coerce").round(6),
                "Beta": pd.to_numeric(table["Beta"], errors="coerce").round(6),
                "Standard error": pd.to_numeric(table["Standard error"], errors="coerce").round(6),
                "SPY return": pd.to_numeric(table["Benchmark return"], errors="coerce").round(6),
                "Model return": pd.to_numeric(table["Model return"], errors="coerce").round(6),
                "Actual return": pd.to_numeric(table["Actual return"], errors="coerce").round(6),
                "residual": pd.to_numeric(table["residual"], errors="coerce").round(6),
                "trade date": pd.to_datetime(table["trade date"], errors="coerce").dt.strftime("%m/%d/%Y"),
                "next day open price of ticker": pd.to_numeric(
                    table["next day open price of ticker"], errors="coerce"
                ).round(6),
                "next day open price of SPY": pd.to_numeric(
                    table["next day open price of benchmark"], errors="coerce"
                ).round(6),
                "future exit trade date": pd.to_datetime(
                    table["future exit trade date"], errors="coerce"
                ).dt.strftime("%m/%d/%Y"),
                "future exit date close price of ticker": pd.to_numeric(
                    table["future exit date close price of ticker"], errors="coerce"
                ).round(6),
                "future exit date close price of SPY": pd.to_numeric(
                    table["future exit date close price of benchmark"], errors="coerce"
                ).round(6),
            }
        )
        rows.append(formatted)

    final_columns = [
        "ticker",
        "Anomaly Date",
        "Ticker Price",
        "SPY Price",
        "Beta",
        "Standard error",
        "SPY return",
        "Model return",
        "Actual return",
        "residual",
        "trade date",
        "next day open price of ticker",
        "next day open price of SPY",
        "future exit trade date",
        "future exit date close price of ticker",
        "future exit date close price of SPY",
    ]
    if not rows:
        return pd.DataFrame(columns=final_columns)
    return pd.concat(rows, ignore_index=True)[final_columns]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark-aware quant analysis for one or more stock tickers."
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated ticker symbols, e.g. AAPL,MSFT,NVDA")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_SYMBOL, help="Benchmark ticker. Default SPY.")
    parser.add_argument("--from-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--lookback", type=int, default=252, help="Rolling PIT regression lookback.")
    parser.add_argument(
        "--provider",
        choices=["yfinance", "polygon", "alphavantage"],
        default=None,
        help="Force a fetch provider instead of automatic routing.",
    )
    parser.add_argument("--polygon-key", default=None, help="Polygon API key.")
    parser.add_argument("--alpha-key", default=None, help="Alpha Vantage API key.")
    parser.add_argument(
        "--remove-earnings-impact",
        action="store_true",
        help="Remove +/- 3 trading days around earnings impacts using Alpha Vantage earnings.",
    )
    parser.add_argument("--timespan", default="day", choices=["day", "hour", "minute"], help="Interval for prices.")
    parser.add_argument("--multiplier", type=int, default=1, help="Multiplier for timespan.")
    parser.add_argument("--summary-output", default=None, help="Optional CSV path for summary metrics.")
    parser.add_argument("--signals-output", default=None, help="Optional CSV path for stat-arb signals.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = CorrelationConfig(
        benchmark_symbol=args.benchmark,
        remove_earnings=bool(args.remove_earnings_impact),
        rolling_window=int(args.lookback),
        sector_relative=True,  # Default to sector relative if we have an API key
    )

    analysis = analyze_symbols_vs_benchmark(
        symbols=args.symbols,
        config=config,
        from_date=args.from_date,
        to_date=args.to_date,
        adjusted=True,
        polygon_api_key=args.polygon_key or os.getenv("POLYGON_API_KEY"),
        alphavantage_api_key=args.alpha_key or os.getenv("ALPHAVANTAGE_API_KEY"),
        prefer_provider=args.provider,
        timespan=args.timespan,
        multiplier=args.multiplier,
    )

    summary_table = analysis["summary_table"]
    failures = analysis["failures"]
    correlation_matrix = analysis["correlation_matrix"]

    if summary_table.empty:
        print("No successful analyses.")
    else:
        with pd.option_context("display.max_rows", 200, "display.width", 200):
            print("Summary metrics:")
            print(summary_table.to_string(index=False))
            print()
            print("Log-return correlation matrix:")
            print(correlation_matrix.to_string())

    if failures:
        print()
        print("Failures:")
        print(pd.DataFrame(failures).to_string(index=False))

    if args.summary_output:
        summary_table.to_csv(args.summary_output, index=False)
        print()
        print(f"Saved summary CSV to {args.summary_output}")

    if args.signals_output:
        signals = generate_consolidated_stat_arb_signals(
            symbols=args.symbols,
            benchmark_symbol=args.benchmark,
            from_date=args.from_date,
            to_date=args.to_date,
            lookback=int(args.lookback),
            adjusted=True,
            polygon_api_key=args.polygon_key or os.getenv("POLYGON_API_KEY"),
            alphavantage_api_key=args.alpha_key or os.getenv("ALPHAVANTAGE_API_KEY"),
            prefer_provider=args.provider,
        )
        signals.to_csv(args.signals_output, index=False)
        print(f"Saved signals CSV to {args.signals_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
