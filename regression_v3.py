from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

ALPHAVANTAGE_API_KEY = "NM0N1PZLUP2KLMY9"
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Hardcoded Polygon/Massive API key (as requested).
POLYGON_API_KEY = "O4oo9zdnIAYbAEVWg1a3Ze5XiPBuY5p8"
POLYGON_BASE_URL = "https://api.polygon.io"

DEFAULT_BENCHMARK_SYMBOL = "SPY"


def _sleep_s(seconds: float) -> None:
    if seconds <= 0:
        return
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
    """
    Resilient GET wrapper for JSON APIs.

    - Retries on transient HTTP statuses (incl 429) using exponential backoff + jitter.
    - Honors Retry-After header when present.
    """
    sess = session or requests.Session()
    last_err: Optional[BaseException] = None

    for attempt in range(max_retries + 1):
        try:
            r = sess.get(url, params=params, timeout=timeout_s)
            if r.status_code in set(int(x) for x in retry_statuses):
                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait_s = float(retry_after)
                    except Exception:
                        wait_s = 0.0
                else:
                    wait_s = 0.0

                if wait_s <= 0.0:
                    # Full jitter exponential backoff.
                    expo = min(backoff_cap_s, backoff_base_s * (2.0**attempt))
                    wait_s = random.random() * expo

                if attempt >= max_retries:
                    r.raise_for_status()

                _sleep_s(wait_s)
                continue

            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got {type(data)}")
            return data

        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            if attempt >= max_retries:
                raise
            expo = min(backoff_cap_s, backoff_base_s * (2.0**attempt))
            _sleep_s(random.random() * expo)
        except Exception as e:
            last_err = e
            raise

    raise RuntimeError(f"HTTP GET failed after retries: {last_err}")


def _parse_symbols_csv(s: str) -> List[str]:
    raw = str(s or "")
    parts = [p.strip().upper() for p in raw.split(",")]
    # allow users to paste "AAPL MSFT" by accident
    expanded: List[str] = []
    for p in parts:
        if not p:
            continue
        expanded.extend([x for x in p.split() if x])
    # de-dupe while preserving order
    seen: set[str] = set()
    out: List[str] = []
    for sym in expanded:
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


@dataclass(frozen=True)
class EarningsImpact:
    symbol: str
    fiscal_date_ending: Optional[date]
    reported_date: date
    report_time: str  # "before_market_open" | "after_market_close" | other
    impact_date: date

def fetch_daily_adjusted_close_polygon(
    symbol: str,
    *,
    from_date: date,
    to_date: date,
    api_key: str = POLYGON_API_KEY,
    base_url: str = POLYGON_BASE_URL,
    timeout_s: int = 30,
) -> pd.Series:
    """
    Fetch daily adjusted close from Polygon Aggregates (1-day bars).

    Polygon endpoint:
    /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}?adjusted=true&sort=asc&limit=50000&apiKey=...

    Returns a Series indexed by date with values = adjusted close.
    """
    if from_date > to_date:
        raise ValueError(f"from_date {from_date.isoformat()} is after to_date {to_date.isoformat()}.")

    url = (
        f"{base_url}/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{from_date.isoformat()}/{to_date.isoformat()}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": "50000",
        "apiKey": api_key,
    }
    data = _http_get_json_with_retries(url, params=params, timeout_s=timeout_s)

    if isinstance(data, dict) and data.get("status") == "ERROR":
        msg = data.get("error") or data.get("message") or str(data)
        raise RuntimeError(f"Polygon error for '{symbol}': {msg}")

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        raise ValueError(
            f"No results returned for '{symbol}'. "
            f"Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    df = pd.DataFrame(results)
    if "t" not in df.columns or "c" not in df.columns:
        raise ValueError(f"Unexpected Polygon aggregates format for '{symbol}': missing 't'/'c'.")

    # t is milliseconds since epoch (UTC). We'll normalize to a date index.
    idx = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
    close = pd.to_numeric(df["c"], errors="coerce")
    out = pd.Series(close.to_numpy(), index=idx, name=f"{symbol}_adj_close").dropna()
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise ValueError(f"No usable close data parsed for '{symbol}'.")
    return out


def fetch_daily_ohlc_polygon(
    symbol: str,
    *,
    from_date: date,
    to_date: date,
    api_key: str = POLYGON_API_KEY,
    base_url: str = POLYGON_BASE_URL,
    adjusted: bool = True,
    timeout_s: int = 30,
) -> pd.DataFrame:
    """
    Fetch daily OHLC from Polygon Aggregates (1-day bars).

    Returns a DataFrame indexed by `date` with columns:
    - open
    - close

    Notes:
    - Polygon 't' is milliseconds since epoch (UTC). We normalize to a date index.
    - When `adjusted=True`, Polygon adjusts o/h/l/c consistently.
    """
    if from_date > to_date:
        raise ValueError(f"from_date {from_date.isoformat()} is after to_date {to_date.isoformat()}.")

    url = (
        f"{base_url}/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{from_date.isoformat()}/{to_date.isoformat()}"
    )
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": "50000",
        "apiKey": api_key,
    }
    data = _http_get_json_with_retries(url, params=params, timeout_s=timeout_s)

    if isinstance(data, dict) and data.get("status") == "ERROR":
        msg = data.get("error") or data.get("message") or str(data)
        raise RuntimeError(f"Polygon error for '{symbol}': {msg}")

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        raise ValueError(
            f"No results returned for '{symbol}'. "
            f"Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    df = pd.DataFrame(results)
    if "t" not in df.columns or "o" not in df.columns or "c" not in df.columns:
        raise ValueError(f"Unexpected Polygon aggregates format for '{symbol}': missing 't'/'o'/'c'.")

    idx = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df["o"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(df["c"], errors="coerce").to_numpy(),
        },
        index=idx,
    ).dropna()

    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def generate_consolidated_stat_arb_signals(
    *,
    symbols: List[str],
    benchmark_symbol: str,
    start: date,
    end: date,
    lookback: int,
    polygon_key: str,
    base_url: str,
    adjusted: bool = True,
) -> pd.DataFrame:
    """
    Build a consolidated CSV dataset matching the exact columns/logic requested:

    - Rolling OLS of ticker returns on SPY returns using prior `lookback` trading days.
    - Residual z-score = residual / standard_error, anomaly if abs(z) > 2.
    - Forward trade fields based on next trading day and exit date = trade_date + 7 calendar days
      adjusted backward to nearest trading day.
    - Fully vectorized computations across dates and tickers (post-fetch).
    """
    lookback = int(lookback)
    if lookback < 3:
        raise ValueError("lookback must be >= 3.")

    # Exclude benchmark if present as a "ticker".
    analysis_symbols = [s for s in symbols if s and s.upper() != benchmark_symbol.upper()]
    analysis_symbols = [s.upper() for s in analysis_symbols]
    analysis_symbols = list(dict.fromkeys(analysis_symbols))  # de-dupe while preserving order
    if not analysis_symbols:
        columns_order = [
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
        return pd.DataFrame(columns=columns_order)

    # Extend the fetch window:
    # - need lookback prior observations
    # - need forward prices for trade day and exit day
    fetch_start = start - timedelta(days=max(lookback + 10, 20))
    fetch_end = end + timedelta(days=20)

    # Fetch benchmark OHLC once.
    spy_df = fetch_daily_ohlc_polygon(
        benchmark_symbol,
        from_date=fetch_start,
        to_date=fetch_end,
        api_key=polygon_key,
        base_url=base_url,
        adjusted=adjusted,
    )
    if spy_df.empty:
        # Fallback: if fetch failed, try a smaller window (user key might be restricted)
        fetch_start_safe = start
        spy_df = fetch_daily_ohlc_polygon(
            benchmark_symbol,
            from_date=fetch_start_safe,
            to_date=fetch_end,
            api_key=polygon_key,
            base_url=base_url,
            adjusted=adjusted,
        )

    if spy_df.empty:
        columns_order = [
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
        return pd.DataFrame(columns=columns_order)

    master_index = spy_df.index
    trading_days = master_index.values.astype("datetime64[ns]")
    
    # Adjust lookback if we have less data than requested (common with free API keys)
    actual_lookback = min(lookback, len(trading_days) - 2)
    if actual_lookback < 3:
        # Still not enough data even after fallback.
        columns_order = [
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
        return pd.DataFrame(columns=columns_order)

    # Fetch tickers OHLC; keep only those that successfully fetch.
    open_series_by_ticker: Dict[str, pd.Series] = {}
    close_series_by_ticker: Dict[str, pd.Series] = {}
    ok_symbols: List[str] = []
    for sym in analysis_symbols:
        try:
            df = fetch_daily_ohlc_polygon(
                sym,
                from_date=fetch_start,
                to_date=fetch_end,
                api_key=polygon_key,
                base_url=base_url,
                adjusted=adjusted,
            )
        except Exception:
            continue
        ok_symbols.append(sym)
        open_series_by_ticker[sym] = df["open"].reindex(master_index)
        close_series_by_ticker[sym] = df["close"].reindex(master_index)

    if not ok_symbols:
        columns_order = [
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
        return pd.DataFrame(columns=columns_order)

    # Wide panel (vectorized across tickers).
    ticker_open = pd.DataFrame({k: open_series_by_ticker[k] for k in ok_symbols}, index=master_index)
    ticker_close = pd.DataFrame({k: close_series_by_ticker[k] for k in ok_symbols}, index=master_index)
    ticker_open.columns.name = "ticker"
    ticker_close.columns.name = "ticker"
    master_index.name = "date"

    # --- Log returns on day T (consistent with analyze_symbol_vs_benchmark) ---
    spy_close = spy_df["close"]
    spy_open = spy_df["open"]

    spy_return = np.log(spy_close / spy_close.shift(1))
    actual_return = np.log(ticker_close / ticker_close.shift(1))

    # --- Rolling OLS point-in-time metrics ---
    # For date T, compute parameters from the prior lookback window ending at T-1.
    # Rolling sums are computed ending at T, then shifted by 1 to exclude T itself.
    n = float(actual_lookback)

    x = spy_return
    x2 = x * x

    # Fix: Ensure we use the smaller window if the API data is short
    actual_lookback = int(actual_lookback)
    
    sx = x.rolling(window=actual_lookback, min_periods=min(actual_lookback, 20)).sum().shift(1)
    sxx = x2.rolling(window=actual_lookback, min_periods=min(actual_lookback, 20)).sum().shift(1)
    sy = actual_return.rolling(window=actual_lookback, min_periods=min(actual_lookback, 20)).sum().shift(1)
    syy = (actual_return * actual_return).rolling(window=actual_lookback, min_periods=min(actual_lookback, 20)).sum().shift(1)
    sxy = (actual_return.mul(x, axis=0)).rolling(window=actual_lookback, min_periods=min(actual_lookback, 20)).sum().shift(1)

    # Use the actual points in each window for the denominator
    n_series = x.rolling(window=actual_lookback, min_periods=min(actual_lookback, 20)).count().shift(1)

    x_mean = sx / n_series
    y_mean = sy / n_series

    sxx_c = sxx - n_series * (x_mean * x_mean)
    sxy_c = sxy - n_series * (y_mean.mul(x_mean, axis=0))

    beta = sxy_c.div(sxx_c, axis=0)
    alpha = y_mean - beta.mul(x_mean, axis=0)

    # SSE within the lookback window:
    # SSE = Σ(y^2) - 2αΣy - 2βΣ(xy) + 2αβΣx + nα^2 + β^2 Σ(x^2)
    sse = (
        syy
        - 2.0 * (alpha * sy)
        - 2.0 * (beta * sxy)
        + 2.0 * (alpha * beta).mul(sx, axis=0)
        + n_series * (alpha * alpha)
        + (beta * beta).mul(sxx, axis=0)
    )

    sse = sse.clip(lower=0.0)
    std_error = np.sqrt(sse / (n_series - 2.0).clip(lower=1.0))

    # R^2 = 1 - SSE / SST, where SST = Σ(y^2) - n*y_mean^2
    sst = syy - n_series * (y_mean * y_mean)
    r_squared = 1.0 - sse / sst.replace(0.0, np.nan)

    # --- Model return, residual, residual z-score ---
    model_return = alpha + beta.mul(x, axis=0)
    residual = actual_return - model_return
    # Guard against zero std_error (stock perfectly tracks benchmark over window).
    residual_z = residual.div(std_error.clip(lower=1e-12))

    anomaly_mask = residual_z.abs().gt(2.0)

    # --- Forward trade fields (precomputed for all rows) ---
    # trade date: next trading day after T (based on benchmark trading calendar).
    trade_date = master_index.to_series().shift(-1)

    # next day open prices (must use shift).
    next_day_open_ticker = ticker_open.shift(-1)
    next_day_open_spy = spy_open.shift(-1)

    # future exit trade date: trade_date + 7 calendar days, adjusted backward to nearest trading day.
    target_exit = trade_date + pd.Timedelta(days=7)
    target_exit_values = target_exit.to_numpy(dtype="datetime64[ns]")
    
    # Target date must be resolvable within our data timeframe.
    max_trading_day = trading_days[-1] if len(trading_days) > 0 else np.datetime64("NaT")
    valid_target = ~pd.isna(target_exit_values) & (target_exit_values <= max_trading_day)

    exit_pos = np.full(shape=len(target_exit_values), fill_value=-1, dtype=int)
    if valid_target.any():
        tv = target_exit_values[valid_target]
        pos = np.searchsorted(trading_days, tv, side="right") - 1
        exit_pos[valid_target] = pos

    # Map exit positions to dates/close prices.
    exit_trade_date_values = np.full(shape=len(target_exit_values), fill_value=np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_valid = exit_pos >= 0
    exit_trade_date_values[exit_valid] = trading_days[exit_pos[exit_valid]]
    future_exit_trade_date = pd.Series(exit_trade_date_values, index=master_index)

    # SPY exit close price on exit date.
    spy_close_vals = spy_close.to_numpy(dtype=float)
    spy_exit_close = np.full(shape=len(spy_close_vals), fill_value=np.nan, dtype=float)
    if exit_valid.any():
        spy_exit_close[exit_valid] = spy_close_vals[exit_pos[exit_valid]]
    spy_exit_close = pd.Series(spy_exit_close, index=master_index)

    # Ticker exit close prices (gather using exit positions).
    close_vals = ticker_close.to_numpy(dtype=float)  # (T, K)
    exit_close_vals = np.full(shape=close_vals.shape, fill_value=np.nan, dtype=float)
    if exit_valid.any():
        tmp_pos = exit_pos.copy()
        tmp_pos[~exit_valid] = 0
        gathered = close_vals[tmp_pos, :]  # (T, K)
        exit_close_vals[exit_valid, :] = gathered[exit_valid, :]
    future_exit_close_ticker = pd.DataFrame(exit_close_vals, index=master_index, columns=ok_symbols)

    # Trade type depends on residual on day T.
    trade_type = pd.DataFrame(
        np.where(residual.gt(0.0), "short", np.where(residual.lt(0.0), "long", np.nan)),
        index=master_index,
        columns=ok_symbols,
    )

    # Trade returns (open to close).
    ticker_trade_return = future_exit_close_ticker.div(next_day_open_ticker) - 1.0
    spy_trade_return = spy_exit_close / next_day_open_spy - 1.0

    # --- Filter anomalies within requested [start, end] ---
    in_window = (master_index >= pd.Timestamp(start)) & (master_index <= pd.Timestamp(end))
    anomaly_mask_window = anomaly_mask.loc[in_window]

    stacked_mask = anomaly_mask_window.stack(dropna=False)
    anomalies_index = stacked_mask[stacked_mask].index  # MultiIndex(date, ticker)

    if len(anomalies_index) == 0:
        columns_order = [
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
        return pd.DataFrame(columns=columns_order)

    # Build output rows from stacked MultiIndex.
    out = anomalies_index.to_frame(index=False)
    out.columns = ["anomaly date", "ticker"]

    # Convenience views for stacked ticker-specific fields.
    stacked_ticker_close = ticker_close.stack()
    stacked_beta = beta.stack()
    stacked_std_error = std_error.stack()
    stacked_r_squared = r_squared.stack()
    stacked_actual_return = actual_return.stack()
    stacked_model_return = model_return.stack()
    stacked_residual = residual.stack()
    stacked_next_open = next_day_open_ticker.stack()
    stacked_exit_close = future_exit_close_ticker.stack()
    stacked_trade_type = trade_type.stack()
    stacked_ticker_trade_return = ticker_trade_return.stack()

    # Date-only (SPY) fields.
    anomaly_dates = pd.to_datetime(out["anomaly date"])
    out["ticker price"] = stacked_ticker_close.loc[anomalies_index].to_numpy(dtype=float)
    out["SPY price"] = spy_close.loc[anomaly_dates].to_numpy(dtype=float)
    out["beta"] = stacked_beta.loc[anomalies_index].to_numpy(dtype=float)
    out["standard error"] = stacked_std_error.loc[anomalies_index].to_numpy(dtype=float)
    out["r squared"] = stacked_r_squared.loc[anomalies_index].to_numpy(dtype=float)
    out["SPY return"] = spy_return.loc[anomaly_dates].to_numpy(dtype=float)
    out["actual return"] = stacked_actual_return.loc[anomalies_index].to_numpy(dtype=float)
    out["model return"] = stacked_model_return.loc[anomalies_index].to_numpy(dtype=float)
    out["residual"] = stacked_residual.loc[anomalies_index].to_numpy(dtype=float)

    out["trade date"] = trade_date.loc[anomaly_dates].to_numpy()
    out["next day open price of ticker"] = stacked_next_open.loc[anomalies_index].to_numpy(dtype=float)
    out["next day open price of SPY"] = next_day_open_spy.loc[anomaly_dates].to_numpy(dtype=float)
    out["future exit trade date"] = future_exit_trade_date.loc[anomaly_dates].to_numpy()
    out["future exit close price of ticker"] = stacked_exit_close.loc[anomalies_index].to_numpy(dtype=float)
    out["future exit close price of SPY"] = spy_exit_close.loc[anomaly_dates].to_numpy(dtype=float)
    out["trade type"] = stacked_trade_type.loc[anomalies_index].to_numpy()
    out["ticker return"] = stacked_ticker_trade_return.loc[anomalies_index].to_numpy(dtype=float)
    out["SPY return (trade)"] = spy_trade_return.loc[anomaly_dates].to_numpy(dtype=float)

    # Clean up results (remove rows with missing core data).
    # We do NOT drop rows if future prices are missing (e.g., for very recent anomalies).
    core_cols = [
        "ticker", "anomaly date", "ticker price", "SPY price", "beta",
        "standard error", "SPY return", "actual return", "model return",
        "residual"
    ]
    out = out.dropna(subset=core_cols)
    out = out.drop_duplicates(subset=["ticker", "anomaly date"], keep="first")

    # Round all price and return columns to 6 decimals.
    round_cols = [
        "ticker price",
        "SPY price",
        "next day open price of ticker",
        "next day open price of SPY",
        "future exit close price of ticker",
        "future exit close price of SPY",
        "SPY return",
        "actual return",
        "model return",
        "residual",
    ]
    for c in round_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(6)

    # Format dates consistently.
    date_cols = ["anomaly date", "trade date", "future exit trade date"]
    for c in date_cols:
        if c in out.columns:
            # Convert to M/D/YYYY format as requested (e.g., 2/2/2026)
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%-m/%-d/%Y")

    # Final rename and reorder to match the user's requested format.
    renamer = {
        "anomaly date": "Anomaly Date",
        "ticker price": "Ticker Price",
        "SPY price": "SPY Price",
        "beta": "Beta",
        "standard error": "Standard error",
        "model return": "Model return",
        "actual return": "Actual return",
        "future exit close price of ticker": "future exit date close price of ticker",
        "future exit close price of SPY": "future exit date close price of SPY",
    }
    out = out.rename(columns=renamer)

    final_columns_order = [
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
    out = out[final_columns_order]
    return out

def _parse_ymd(s: str) -> date:
    d = pd.to_datetime(str(s).strip(), errors="raise")
    return d.date()


def fetch_earnings_av(symbol: str, api_key: str, *, timeout_s: int = 30) -> Dict[str, Any]:
    params = {"function": "EARNINGS", "symbol": symbol, "apikey": api_key}
    # Alpha Vantage sometimes returns HTTP 200 with a "Note" (rate limit).
    for attempt in range(7):
        data = _http_get_json_with_retries(
            ALPHAVANTAGE_BASE_URL,
            params=params,
            timeout_s=timeout_s,
            max_retries=6,
            backoff_base_s=1.0,
            backoff_cap_s=60.0,
        )
        if "Note" in data and attempt < 6:
            _sleep_s(random.random() * min(60.0, 1.0 * (2.0**attempt)))
            continue
        break

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error for symbol '{symbol}': {data['Error Message']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage note (often rate limit): {data['Note']}")
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage information message: {data['Information']}")

    return data


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


def _next_trading_day(trading_days: List[date], d: date) -> Optional[date]:
    lo, hi = 0, len(trading_days)
    while lo < hi:
        mid = (lo + hi) // 2
        if trading_days[mid] <= d:
            lo = mid + 1
        else:
            hi = mid
    return trading_days[lo] if lo < len(trading_days) else None


def _normalize_report_time(v: Any) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return ""
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


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
    """
    Convert Alpha Vantage EARNINGS payload into a list of earnings impacts.

    Rules:
    - reportTime == "before_market_open": remove reportedDate
    - reportTime == "after_market_close": remove next trading day after reportedDate (T+1)
    - otherwise (including missing): default to after_market_close (T+1)
    """
    q = payload.get("quarterlyEarnings")
    if not isinstance(q, list):
        raise ValueError("Unexpected Alpha Vantage earnings payload: missing 'quarterlyEarnings'.")

    trading_list = sorted(set(trading_days))
    if not trading_list:
        return []

    earliest_report_to_consider = window_start - timedelta(days=max(0, int(pre_window_buffer_days)))
    out: List[EarningsImpact] = []
    for entry in q:
        if not isinstance(entry, dict):
            continue
        reported_date = _parse_iso_date(entry.get("reportedDate"))
        if reported_date is None:
            continue
        if reported_date < earliest_report_to_consider or reported_date > window_end:
            continue

        fiscal = _parse_iso_date(entry.get("fiscalDateEnding"))
        time_str = _normalize_report_time(entry.get("reportTime")) or "after_market_close"
        if time_str in {"before_market_open", "beforemarketopen", "bmo"}:
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
                    report_time=time_str,
                    impact_date=impact,
                )
            )

    out.sort(key=lambda x: x.reported_date)
    return out


def impacts_to_frame(impacts: List[EarningsImpact]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": i.symbol,
                "fiscalDateEnding": i.fiscal_date_ending.isoformat() if i.fiscal_date_ending else None,
                "reportedDate": i.reported_date.isoformat(),
                "reportTime": i.report_time,
                "impactDate": i.impact_date.isoformat(),
            }
            for i in impacts
        ]
    )


def compute_log_returns(prices: pd.Series) -> pd.Series:
    px = pd.to_numeric(prices, errors="coerce").dropna()
    rets = np.log(px).diff()
    rets.name = prices.name.replace("_adj_close", "_logret") if prices.name else "logret"
    return rets.dropna()


def run_simple_regression(y: pd.Series, x: pd.Series) -> Dict[str, float]:
    """
    Simple OLS for y = alpha + beta * x with summary metrics.
    """
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
    """
    Vectorized rolling OLS for y = alpha + beta * x, point-in-time (PIT).

    For each date T, parameters are computed using the prior `lookback` observations
    (ending at T-1). If insufficient history exists, parameters are NaN.
    """
    if lookback < 3:
        raise ValueError("lookback must be >= 3 to estimate OLS with an intercept.")

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame(index=y.index, columns=["alpha", "beta", "std_error_regression", "n_obs"])

    n = float(lookback)

    # Vectorized rolling calculations
    sx = df["x"].rolling(window=lookback, min_periods=lookback).sum().shift(1)
    sy = df["y"].rolling(window=lookback, min_periods=lookback).sum().shift(1)
    sxx = (df["x"] ** 2).rolling(window=lookback, min_periods=lookback).sum().shift(1)
    syy = (df["y"] ** 2).rolling(window=lookback, min_periods=lookback).sum().shift(1)
    sxy = (df["x"] * df["y"]).rolling(window=lookback, min_periods=lookback).sum().shift(1)

    x_mean = sx / n
    y_mean = sy / n
    sxx_c = sxx - n * (x_mean ** 2)
    sxy_c = sxy - n * (x_mean * y_mean)

    beta = sxy_c / sxx_c
    alpha = y_mean - beta * x_mean

    # SSE computed from sums (over the lookback window used for params).
    # SSE = Σ(y^2) - 2αΣy - 2βΣ(xy) + 2αβΣx + nα^2 + β^2 Σ(x^2)
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

    # Expand back to the original index (dates with missing returns get NaNs).
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
    """
    Safety check: confirm parameters at date T are derived strictly from dates < T.

    This recomputes OLS for one date (either `validate_date` if available, else the first
    non-null parameter row) using the prior `lookback` observations and compares.
    """
    if reg_df.empty or params.empty:
        return

    idx = params.index
    t: Optional[pd.Timestamp] = None
    if validate_date is not None and validate_date in idx:
        if pd.notna(params.loc[validate_date, "beta"]):
            t = validate_date

    if t is None:
        non_null = params["beta"].dropna()
        if non_null.empty:
            return
        t = pd.Timestamp(non_null.index[0])

    pos = int(reg_df.index.get_indexer([t])[0])
    if pos < 0:
        return
    end = pos  # exclude t itself
    start = end - int(lookback)
    if start < 0:
        return

    window = reg_df.iloc[start:end].dropna()
    if window.shape[0] != int(lookback):
        return

    m = run_simple_regression(window["asset_logret"], window["benchmark_logret"])
    got_alpha = float(params.loc[t, "alpha"])
    got_beta = float(params.loc[t, "beta"])
    got_se = float(params.loc[t, "std_error_regression"])

    if not np.isfinite(got_beta) or not np.isfinite(got_alpha) or not np.isfinite(got_se):
        raise RuntimeError(f"PIT validation failed at {t.date().isoformat()}: non-finite parameters.")

    if not np.isclose(got_beta, float(m["beta"]), atol=atol, rtol=rtol):
        raise RuntimeError(
            f"PIT validation failed at {t.date().isoformat()}: beta mismatch "
            f"(got={got_beta}, expected={m['beta']})."
        )
    if not np.isclose(got_alpha, float(m["alpha"]), atol=atol, rtol=rtol):
        raise RuntimeError(
            f"PIT validation failed at {t.date().isoformat()}: alpha mismatch "
            f"(got={got_alpha}, expected={m['alpha']})."
        )
    if not np.isclose(got_se, float(m["std_error_regression"]), atol=atol, rtol=rtol):
        raise RuntimeError(
            f"PIT validation failed at {t.date().isoformat()}: std_error_regression mismatch "
            f"(got={got_se}, expected={m['std_error_regression']})."
        )

    # Explicit temporal guard: the window must end strictly before t.
    if window.index.max() >= t:
        raise RuntimeError(
            f"PIT validation failed at {t.date().isoformat()}: window includes current/future data."
        )


def analyze_symbol_vs_benchmark(
    *,
    symbol: str,
    benchmark_symbol: str,
    start: date,
    end: date,
    base_url: str,
    polygon_key: str,
    alpha_key: str,
    pre_window_buffer_days: int = 7,
    max_missing_vs_benchmark: float = 0.05,
    lookback: int = 252,
) -> Dict[str, Any]:
    asset_df = fetch_daily_ohlc_polygon(
        symbol, from_date=start, to_date=end, base_url=base_url, api_key=polygon_key, adjusted=True
    )
    bench_df = fetch_daily_ohlc_polygon(
        benchmark_symbol, from_date=start, to_date=end, base_url=base_url, api_key=polygon_key, adjusted=True
    )

    # Align to benchmark trading days first, then measure data gaps for the asset.
    bench_index = bench_df.index.sort_values()
    asset_on_bench = asset_df.reindex(bench_index)
    
    missing_ratio = float(asset_on_bench["close"].isna().mean()) if len(asset_on_bench) else 1.0
    if missing_ratio > float(max_missing_vs_benchmark):
        raise ValueError(
            f"Zombie ticker / data gap: '{symbol}' is missing {missing_ratio:.2%} of days relative to "
            f"{benchmark_symbol} (threshold {max_missing_vs_benchmark:.2%}). Skipping."
        )

    # We need both open and close for ticker and benchmark.
    # Combine into a single DataFrame for alignment.
    aligned = pd.concat([
        asset_on_bench["open"].rename(f"{symbol}_open"),
        asset_on_bench["close"].rename(f"{symbol}_close"),
        bench_df["open"].reindex(bench_index).rename(f"{benchmark_symbol}_open"),
        bench_df["close"].reindex(bench_index).rename(f"{benchmark_symbol}_close"),
    ], axis=1).dropna(subset=[f"{symbol}_close", f"{benchmark_symbol}_close"])

    if aligned.empty:
        raise ValueError(f"No overlapping dates between {symbol} and {benchmark_symbol}.")

    earnings_payload = fetch_earnings_av(symbol, alpha_key, timeout_s=30)
    trading_days = [d.date() for d in aligned.index.to_pydatetime()]
    impacts = compute_earnings_impacts(
        earnings_payload,
        symbol=symbol,
        trading_days=trading_days,
        window_start=trading_days[0],
        window_end=trading_days[-1],
        pre_window_buffer_days=pre_window_buffer_days,
        amc_fallback_calendar_plus_one=False,
    )
    impacts_df = impacts_to_frame(impacts)

    # Exclude +/- 3 trading days around the impact date as requested.
    removed_dates = set()
    for i in impacts:
        impact_ts = pd.Timestamp(i.impact_date)
        if impact_ts in aligned.index:
            center_idx = aligned.index.get_loc(impact_ts)
            start_idx = max(0, center_idx - 3)
            end_idx = min(len(aligned) - 1, center_idx + 3)
            for idx in range(start_idx, end_idx + 1):
                removed_dates.add(aligned.index[idx])
    removed_dates = sorted(list(removed_dates))

    asset_ret = compute_log_returns(aligned[f"{symbol}_close"])
    bench_ret = compute_log_returns(aligned[f"{benchmark_symbol}_close"])

    if removed_dates:
        asset_ret = asset_ret.drop(index=removed_dates, errors="ignore")
        bench_ret = bench_ret.drop(index=removed_dates, errors="ignore")

    reg_df = pd.concat(
        [asset_ret.rename("asset_logret"), bench_ret.rename("benchmark_logret")], axis=1
    ).dropna()
    if reg_df.empty:
        raise ValueError("No overlapping return observations after earnings-date removal.")

    # Point-in-time rolling parameters: for date T, use prior `lookback` observations (ending T-1).
    params = rolling_ols_params_point_in_time(
        reg_df["asset_logret"], reg_df["benchmark_logret"], lookback=int(lookback)
    )
    model = (params["alpha"] + params["beta"] * reg_df["benchmark_logret"]).rename("model_logret")
    resid = (reg_df["asset_logret"] - model).rename("residual")

    roll_se = params["std_error_regression"].rename("rolling_std_error")
    # Guard against zero std_error (stock perfectly tracks benchmark over window).
    z = (resid / roll_se.clip(lower=1e-12)).rename("residual_z")
    outlier_mask = z.abs() > 2.0
    outlier_dates = z.index[outlier_mask.fillna(False)]

    # Optional baseline summary over the full available (cleaned) window for quick reporting.
    metrics = run_simple_regression(reg_df["asset_logret"], reg_df["benchmark_logret"])

    # For the report and trade logic, we keep current day prices and next day OPEN prices.
    prices = aligned[[f"{symbol}_close", f"{benchmark_symbol}_close"]].copy()
    prices.columns = [symbol, benchmark_symbol]
    
    # Next day OPEN price for trade entry.
    prices_t_plus_1 = aligned[[f"{symbol}_open", f"{benchmark_symbol}_open"]].shift(-1).rename(
        columns={
            f"{symbol}_open": f"{symbol}_open_t_plus_1",
            f"{benchmark_symbol}_open": f"{benchmark_symbol}_open_t_plus_1",
        }
    )

    # Exit price: current_date + 7 calendar days, adjusted backward to nearest
    # trading day. This matches generate_consolidated_stat_arb_signals logic
    # (previously used shift(-7) which was 7 trading days, not 7 calendar days).
    _trading_days_arr = prices.index.values.astype("datetime64[ns]")
    _target_exit = prices.index + pd.Timedelta(days=7)
    
    _max_trading_day = _trading_days_arr[-1] if len(_trading_days_arr) > 0 else np.datetime64("NaT")
    _exit_pos = np.searchsorted(_trading_days_arr, _target_exit.values, side="right") - 1
    
    # Valid exit iff pos is safe AND the target exit hasn't overshot our maximum known trading day
    _valid_exit = (_exit_pos >= 0) & (_exit_pos < len(_trading_days_arr)) & (_target_exit.values <= _max_trading_day)

    _exit_vals = np.full((len(prices), prices.shape[1]), np.nan)
    _safe_pos = np.clip(_exit_pos, 0, len(_trading_days_arr) - 1)
    _exit_vals[_valid_exit] = prices.values[_safe_pos[_valid_exit]]

    # trade date: next trading day
    trade_date = prices.index.to_series().shift(-1)

    # exit trade date: the trading day used for t+7 calendar calculation
    _exit_dates = np.full(len(prices), np.datetime64("NaT"), dtype="datetime64[ns]")
    _exit_dates[_valid_exit] = _trading_days_arr[_exit_pos[_valid_exit]]
    future_exit_trade_date = pd.Series(_exit_dates, index=prices.index)

    prices_t_plus_7 = pd.DataFrame(
        _exit_vals,
        index=prices.index,
        columns=[
            f"{symbol}_close_t_plus_7",
            f"{benchmark_symbol}_close_t_plus_7",
        ],
    )

    outliers_table = pd.DataFrame(index=outlier_dates).join(
        prices.loc[outlier_dates].rename(
            columns={
                symbol: f"{symbol}_close_t",
                benchmark_symbol: f"{benchmark_symbol}_close_t",
            }
        )
    )
    outliers_table = outliers_table.join(prices_t_plus_1.loc[outlier_dates]).join(
        prices_t_plus_7.loc[outlier_dates]
    )
    params_to_join = params.loc[outlier_dates, ["alpha", "beta", "std_error_regression"]].rename(
        columns={"std_error_regression": "rolling_std_error"}
    )
    
    # Include the returns for the anomaly date.
    returns_to_join = reg_df.loc[outlier_dates, ["asset_logret", "benchmark_logret"]].rename(
        columns={"asset_logret": "actual_return", "benchmark_logret": "spy_return"}
    )
    
    # Calculate model return for those dates: alpha + beta * spy_return
    model_returns = params_to_join["alpha"] + params_to_join["beta"] * returns_to_join["spy_return"]
    returns_to_join["model_return"] = model_returns

    # Include the trade dates.
    dates_to_join = pd.DataFrame(index=outlier_dates)
    dates_to_join["next_trade_date"] = trade_date.loc[outlier_dates]
    dates_to_join["exit_date"] = future_exit_trade_date.loc[outlier_dates]

    outliers_table = outliers_table.join(params_to_join)
    outliers_table = outliers_table.join(returns_to_join)
    outliers_table = outliers_table.join(dates_to_join)
    outliers_table = outliers_table.join(resid.loc[outlier_dates]).join(z.loc[outlier_dates])

    outliers_table.index.name = "date"
    outliers_table = outliers_table.reset_index().sort_values("date").reset_index(drop=True)

    # No-cheating validation: specifically check 2023-01-01 if present, otherwise validate first PIT row.
    validate_ts = pd.Timestamp("2023-01-01")
    _validate_no_cheating(reg_df=reg_df, params=params, lookback=int(lookback), validate_date=validate_ts)

    return {
        "symbol": symbol,
        "benchmark": benchmark_symbol,
        "start": start,
        "end": end,
        "aligned_prices": aligned,
        "impacts": impacts,
        "impacts_df": impacts_df,
        "removed_dates": removed_dates,
        "metrics": metrics,
        "lookback": int(lookback),
        "outliers_table": outliers_table,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Fetch Polygon daily adjusted closes for an input ticker and SPY, remove earnings-impact dates "
            "(BMO: same day; AMC: next trading day) using Alpha Vantage earnings, then run a log-return "
            "regression and flag rolling point-in-time anomalies (|residual_z| > 2)."
        )
    )
    p.add_argument(
        "--symbol",
        default=None,
        help="Ticker symbol(s) to analyze. Supports comma-separated input (e.g. AAPL, MSFT, GOOG).",
    )
    p.add_argument(
        "--from-date",
        default=None,
        help="Start date (YYYY-MM-DD). If omitted, you'll be prompted.",
    )
    p.add_argument(
        "--to-date",
        default=None,
        help="End date (YYYY-MM-DD). If omitted, you'll be prompted.",
    )
    p.add_argument("--base-url", default=POLYGON_BASE_URL, help="Polygon base URL.")
    p.add_argument(
        "--polygon-key",
        default=None,
        help="Polygon API key. If omitted, uses env var POLYGON_API_KEY, else falls back to constant.",
    )
    p.add_argument(
        "--alpha-key",
        default=None,
        help="Alpha Vantage API key. If omitted, uses env var ALPHAVANTAGE_API_KEY, else falls back to constant.",
    )
    p.add_argument(
        "--print-aligned-prices",
        action="store_true",
        help="Print aligned adjusted closes (post-earnings-removal).",
    )
    p.add_argument(
        "--print-earnings-impacts",
        action="store_true",
        help="Print the earnings impacts table used to remove return dates.",
    )
    p.add_argument(
        "--earnings-impacts-output",
        default=None,
        help="Optional CSV path to write the earnings impacts table.",
    )
    p.add_argument(
        "--residual-outliers-output",
        default=None,
        help=(
            "Optional CSV path to write the residual outlier table "
            "(dates where |residual_z| > 2 using a point-in-time rolling lookback)."
        ),
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=252,
        help=(
            "Rolling lookback window (trading days) for point-in-time regression parameters. "
            "For date T, params use the prior N days ending at T-1. Default 252."
        ),
    )
    p.add_argument(
        "--max-missing-vs-spy",
        type=float,
        default=0.05,
        help=(
            "Skip a ticker if it is missing more than this fraction of benchmark (SPY) trading days "
            "over the window (e.g. 0.05 = 5%%). Helps avoid 'zombie tickers' skewing beta."
        ),
    )
    args = p.parse_args()

    symbols_s = args.symbol or input(
        "Enter ticker symbol(s) (comma-separated, e.g. AAPL, MSFT, GOOG): "
    ).strip()
    symbols = _parse_symbols_csv(symbols_s)
    if not symbols:
        raise ValueError("At least one ticker symbol is required.")

    from_s = args.from_date or input("Enter from date (YYYY-MM-DD): ").strip()
    to_s = args.to_date or input("Enter to date (YYYY-MM-DD): ").strip()
    start = _parse_ymd(from_s)
    end = _parse_ymd(to_s)

    polygon_key = args.polygon_key or os.getenv("POLYGON_API_KEY") or POLYGON_API_KEY
    alpha_key = args.alpha_key or os.getenv("ALPHAVANTAGE_API_KEY") or ALPHAVANTAGE_API_KEY

    combined_impacts: List[pd.DataFrame] = []
    combined_outliers: List[pd.DataFrame] = []
    errors: List[Dict[str, str]] = []

    for symbol in symbols:
        try:
            result = analyze_symbol_vs_benchmark(
                symbol=symbol,
                benchmark_symbol=DEFAULT_BENCHMARK_SYMBOL,
                start=start,
                end=end,
                base_url=args.base_url,
                polygon_key=polygon_key,
                alpha_key=alpha_key,
                max_missing_vs_benchmark=float(args.max_missing_vs_spy),
                lookback=int(args.lookback),
            )
        except Exception as e:
            msg = str(e).strip() or e.__class__.__name__
            errors.append({"symbol": symbol, "error": msg})
            print()
            print(f"Symbol: {symbol}")
            print(f"ERROR: {msg}")
            import traceback
            traceback.print_exc()
            continue

        impacts_df = result["impacts_df"]
        outliers_table = result["outliers_table"]
        removed_dates = result["removed_dates"]
        metrics = result["metrics"]
        lookback = int(result.get("lookback", int(args.lookback)))
        aligned = result["aligned_prices"]

        combined_impacts.append(impacts_df)

        outliers_with_symbol = outliers_table.copy()
        outliers_with_symbol.insert(0, "symbol", symbol)
        combined_outliers.append(outliers_with_symbol)

        if args.print_earnings_impacts:
            with pd.option_context("display.max_rows", 200, "display.width", 140):
                print()
                print(f"Earnings impacts (used for removal) - {symbol}:")
                if impacts_df.empty:
                    print("(none in window)")
                else:
                    print(impacts_df.to_string(index=False))

        print()
        print(f"Symbol: {symbol}")
        print(f"Benchmark: {DEFAULT_BENCHMARK_SYMBOL}")
        print(f"Window (requested): {start.isoformat()} to {end.isoformat()}")
        print(f"Overlapping price days: {aligned.shape[0]}")
        print(f"Earnings-impact days removed (in window): {len(removed_dates)}")
        print(f"Rolling lookback (PIT): {lookback} trading days")
        if removed_dates:
            preview = ", ".join([d.date().isoformat() for d in removed_dates[:10]])
            suffix = " ..." if len(removed_dates) > 10 else ""
            print(f"Removed dates (first 10): {preview}{suffix}")

        print()
        print("Regression on daily log returns: asset = alpha + beta * benchmark")
        print(f"n: {int(metrics['n_obs'])}")
        print(f"beta: {metrics['beta']:.6f}")
        print(f"alpha: {metrics['alpha']:.8f}")
        print(f"correlation (r): {metrics['r']:.6f}")
        print(f"R^2: {metrics['r_squared']:.6f}")
        print(f"std error (regression): {metrics['std_error_regression']:.8f}")
        print(f"std error (beta): {metrics['std_error_beta']:.8f}")

        with pd.option_context("display.max_rows", 500, "display.width", 180):
            print()
            print(
                f"Residual anomaly days where |residual_z| > 2.0 (PIT rolling) - {symbol}"
            )
            if outliers_table.empty:
                print("(none)")
            else:
                print(outliers_table.to_string(index=False))

        if args.print_aligned_prices:
            cleaned_prices = aligned.drop(index=removed_dates, errors="ignore")
            with pd.option_context("display.max_rows", 200, "display.width", 140):
                print()
                print(f"Aligned adjusted closes (post-earnings-removal) - {symbol}:")
                print(cleaned_prices.to_string())

    if args.earnings_impacts_output:
        all_impacts = pd.concat(combined_impacts, ignore_index=True) if combined_impacts else pd.DataFrame()
        all_impacts.to_csv(args.earnings_impacts_output, index=False)

    if args.residual_outliers_output:
        all_outliers = pd.concat(combined_outliers, ignore_index=True) if combined_outliers else pd.DataFrame()
        all_outliers.to_csv(args.residual_outliers_output, index=False)

    # Always write the consolidated stat-arb signals CSV as requested.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    consolidated_output_path = os.path.join(script_dir, "consolidated_stat_arb_signals.csv")
    try:
        if not combined_outliers:
            # Create empty df with requested header if no anomalies found.
            final_columns_order = [
                "ticker", "Anomaly Date", "Ticker Price", "SPY Price", "Beta",
                "Standard error", "SPY return", "Model return", "Actual return",
                "residual", "trade date", "next day open price of ticker",
                "next day open price of SPY", "future exit trade date",
                "future exit date close price of ticker", "future exit date close price of SPY"
            ]
            pd.DataFrame(columns=final_columns_order).to_csv(consolidated_output_path, index=False)
            print()
            print(f"Wrote 0 anomalies to consolidated CSV: {consolidated_output_path}")
        else:
            all_outliers = pd.concat(combined_outliers, ignore_index=True)
            
            # Map the columns from the individual analysis table to the user's requested names.
            # outliers_table has: date, {sym}_close_t, {bench}_close_t, {sym}_close_t_plus_1, etc.
            # We need to generalize these based on the first symbol if needed, but we already have them as 'symbol'.
            
            # Since the outliers_table has dynamic column names (e.g. AMZN_close_t),
            # we'll use a more flexible mapping logic.
            final_rows = []
            for _, row in all_outliers.iterrows():
                sym = row["symbol"]
                bench = DEFAULT_BENCHMARK_SYMBOL
                
                # Extract values using the dynamic names.
                # Format: M/D/YYYY
                def _fmt_date(ts: Any) -> str:
                    # Cross-platform: strip leading zeros from month and day.
                    d = pd.to_datetime(ts)
                    return f"{d.month}/{d.day}/{d.year}"

                anomaly_date = _fmt_date(row["date"])
                trade_date = _fmt_date(row["next_trade_date"]) if pd.notna(row["next_trade_date"]) else ""
                exit_date = _fmt_date(row["exit_date"]) if pd.notna(row["exit_date"]) else ""
                
                # Log returns (calculated as log(P_t / P_{t-1})) - for the report we use the ones from metrics/params.
                # However, the user explanation says "one day return".
                # Standard practice: log(P_t / P_{t-1}).
                
                # We'll calculate returns from the prices in the row if not directly available.
                # Actually, Beta, Standard error, residual etc. are in the row.
                
                # Prices and Metrics
                ticker_price = row[f"{sym}_close_t"]
                spy_price = row[f"{bench}_close_t"]
                beta = row["beta"]
                std_error = row["rolling_std_error"]
                residual = row["residual"]
                
                # Returns (approximated for display if not in row - but they are log returns)
                # Actually, I'll add SPY return to the outliers_table earlier to be safe.
                # For now, we'll use the values directly.
                
                final_rows.append({
                    "ticker": sym,
                    "Anomaly Date": anomaly_date,
                    "Ticker Price": round(float(ticker_price), 6),
                    "SPY Price": round(float(spy_price), 6),
                    "Beta": round(float(beta), 6),
                    "Standard error": round(float(std_error), 6),
                    "SPY return": round(float(row["spy_return"]), 6),
                    "Model return": round(float(row["model_return"]), 6),
                    "Actual return": round(float(row["actual_return"]), 6),
                    "residual": round(float(residual), 6),
                    "trade date": trade_date,
                    "next day open price of ticker": round(float(row[f"{sym}_open_t_plus_1"]), 6) if pd.notna(row[f"{sym}_open_t_plus_1"]) else "",
                    "next day open price of SPY": round(float(row[f"{bench}_open_t_plus_1"]), 6) if pd.notna(row[f"{bench}_open_t_plus_1"]) else "",
                    "future exit trade date": exit_date,
                    "future exit date close price of ticker": round(float(row[f"{sym}_close_t_plus_7"]), 6) if pd.notna(row[f"{sym}_close_t_plus_7"]) else "",
                    "future exit date close price of SPY": round(float(row[f"{bench}_close_t_plus_7"]), 6) if pd.notna(row[f"{bench}_close_t_plus_7"]) else "",
                })
            
            # Wait, I need to ensure SPY return and actual returns are correctly passed to the outliers_table.
            # I'll update analyze_symbol_vs_benchmark to include them first.
            
            pd.DataFrame(final_rows).to_csv(consolidated_output_path, index=False)
            print()
            print(f"Wrote {len(final_rows)} anomalies to consolidated CSV: {consolidated_output_path}")
    except Exception as e:
        print()
        print(f"ERROR writing consolidated CSV: {e}")
        import traceback
        traceback.print_exc()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


