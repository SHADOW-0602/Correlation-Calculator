from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

# Note: On Streamlit Cloud, set POLYGON_API_KEY and ALPHAVANTAGE_API_KEY 
# in the Secrets or Environment Variables section of your app settings.

from fetch_prices import (
    FetchMetadata,
    PriceFetchError,
    build_polygon_option_ticker,
    fetch_equity_prices,
    fetch_multiple_equity_prices,
    fetch_option_prices,
)
from quant_analysis import analyze_symbols_vs_benchmark, CorrelationConfig


st.set_page_config(
    page_title="Price Fetcher",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

FIELD_OPTIONS = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "dividends",
    "stock_splits",
]

SUMMARY_COLUMNS = [
    "symbol",
    "benchmark",
    "n_obs",
    "correlation",
    "alpha",
    "beta",
    "r_squared",
    "std_error_regression",
    "asset_annual_return",
    "benchmark_annual_return",
    "asset_annual_volatility",
    "benchmark_annual_volatility",
    "asset_sharpe",
    "benchmark_sharpe",
    "tracking_error",
    "information_ratio",
    "asset_max_drawdown",
    "benchmark_max_drawdown",
    "outlier_count",
    "missing_vs_benchmark",
    "earnings_removed_days",
]

SECTOR_ETF_OPTIONS = [
    "Auto detect / not set",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

WINSOR_OPTIONS = {
    "None": None,
    "1%": 0.01,
    "2%": 0.02,
    "5%": 0.05,
    "10%": 0.10,
}


def _init_state() -> None:
    import os
    # Force reload from .env to ensure session state is in sync with file
    poly_key = os.getenv("POLYGON_API_KEY", "")
    alpha_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    
    if "your_polygon_api_key_here" in poly_key or not poly_key:
        st.warning("Polygon API key is missing or set to a placeholder in .env")
    if "your_alpha_vantage_key_here" in alpha_key or not alpha_key:
        st.warning("Alpha Vantage API key is missing or set to a placeholder in .env")

    st.session_state["polygon_api_key"] = poly_key
    st.session_state["alphavantage_api_key"] = alpha_key


def _inject_correlation_styles() -> None:
    st.markdown(
        """
        <style>
        .corr-shell {
            background:
                radial-gradient(circle at top left, rgba(25,118,210,0.16), transparent 30%),
                radial-gradient(circle at bottom right, rgba(0,150,136,0.14), transparent 28%),
                linear-gradient(135deg, #f7fafc 0%, #edf4f7 100%);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 24px;
            padding: 1.2rem 1.2rem 0.8rem 1.2rem;
            margin-bottom: 1rem;
        }
        .corr-kicker {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            color: #0f766e;
            font-weight: 700;
        }
        .corr-title {
            font-size: 2rem;
            line-height: 1.1;
            color: #0f172a;
            font-weight: 800;
            margin: 0.25rem 0 0.35rem 0;
        }
        .corr-copy {
            color: #334155;
            font-size: 0.98rem;
            max-width: 60rem;
        }
        .corr-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.85rem;
        }
        .corr-chip {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 999px;
            padding: 0.4rem 0.75rem;
            color: #0f172a;
            font-size: 0.83rem;
            font-weight: 600;
        }
        .corr-section {
            margin-top: 0.5rem;
            margin-bottom: 0.35rem;
            font-size: 1.15rem;
            font-weight: 700;
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _winsorize_series(series: pd.Series, tail_pct: Optional[float]) -> pd.Series:
    if tail_pct is None or series.dropna().empty:
        return series
    lower = series.quantile(tail_pct)
    upper = series.quantile(1 - tail_pct)
    return series.clip(lower=lower, upper=upper)


def _weighted_corr(left: pd.Series, right: pd.Series, halflife: float) -> float:
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 2:
        return float("nan")

    positions = np.arange(len(frame) - 1, -1, -1, dtype=float)
    weights = np.power(0.5, positions / halflife)
    weights = weights / weights.sum()

    x = frame.iloc[:, 0].to_numpy(dtype=float)
    y = frame.iloc[:, 1].to_numpy(dtype=float)
    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)
    covariance = np.average((x - x_mean) * (y - y_mean), weights=weights)
    x_var = np.average((x - x_mean) ** 2, weights=weights)
    y_var = np.average((y - y_mean) ** 2, weights=weights)
    denom = np.sqrt(x_var * y_var)
    if denom == 0:
        return float("nan")
    return float(covariance / denom)


def _build_correlation_view(
    selected_symbol: str,
    benchmark_symbol: str,
    analysis: Dict[str, Any],
    *,
    regime: str,
    winsorize_pct: Optional[float],
    use_decay: bool,
    decay_halflife: float,
    rolling_window: int,
) -> Dict[str, Any]:
    aligned_prices = analysis.get("aligned_prices", pd.DataFrame()).copy()
    if aligned_prices.empty:
        return {}

    asset_col = f"{selected_symbol}_close"
    benchmark_col = f"{benchmark_symbol}_close"
    if asset_col not in aligned_prices.columns or benchmark_col not in aligned_prices.columns:
        return {}

    returns = pd.DataFrame(
        {
            selected_symbol: aligned_prices[asset_col].pct_change(),
            benchmark_symbol: aligned_prices[benchmark_col].pct_change(),
        }
    ).dropna()

    if winsorize_pct is not None:
        returns[selected_symbol] = _winsorize_series(returns[selected_symbol], winsorize_pct)
        returns[benchmark_symbol] = _winsorize_series(returns[benchmark_symbol], winsorize_pct)

    if regime == "Bull":
        filtered = returns[returns[benchmark_symbol] > 0].copy()
    elif regime == "Bear":
        filtered = returns[returns[benchmark_symbol] < 0].copy()
    else:
        filtered = returns.copy()

    if filtered.empty:
        return {}

    rolling = filtered[selected_symbol].rolling(rolling_window, min_periods=max(5, rolling_window // 3)).corr(
        filtered[benchmark_symbol]
    )
    raw_corr = filtered[selected_symbol].corr(filtered[benchmark_symbol])
    displayed_corr = (
        _weighted_corr(filtered[selected_symbol], filtered[benchmark_symbol], decay_halflife)
        if use_decay
        else float(raw_corr)
    )

    full_range = {
        "all": returns[selected_symbol].corr(returns[benchmark_symbol]) if not returns.empty else float("nan"),
        "bull": returns.loc[returns[benchmark_symbol] > 0, selected_symbol].corr(
            returns.loc[returns[benchmark_symbol] > 0, benchmark_symbol]
        ),
        "bear": returns.loc[returns[benchmark_symbol] < 0, selected_symbol].corr(
            returns.loc[returns[benchmark_symbol] < 0, benchmark_symbol]
        ),
    }

    range_values = pd.Series(list(full_range.values()), dtype=float).dropna()
    rolling_clean = rolling.dropna()
    earnings_impacts = analysis.get("impacts_df", pd.DataFrame())

    return {
        "returns": filtered,
        "rolling": rolling,
        "correlation": float(displayed_corr) if pd.notna(displayed_corr) else float("nan"),
        "base_correlation": float(raw_corr) if pd.notna(raw_corr) else float("nan"),
        "range_min": float(range_values.min()) if not range_values.empty else float("nan"),
        "range_max": float(range_values.max()) if not range_values.empty else float("nan"),
        "range_spread": float(range_values.max() - range_values.min()) if len(range_values) > 1 else float("nan"),
        "rolling_min": float(rolling_clean.min()) if not rolling_clean.empty else float("nan"),
        "rolling_max": float(rolling_clean.max()) if not rolling_clean.empty else float("nan"),
        "rolling_latest": float(rolling_clean.iloc[-1]) if not rolling_clean.empty else float("nan"),
        "observations": int(len(filtered)),
        "earnings_removed_days": int(len(earnings_impacts)) if not earnings_impacts.empty else 0,
        "regime_map": full_range,
    }


def _render_correlation_results(
    result: Dict[str, Any],
    benchmark_symbol: str,
    *,
    regime: str,
    sector_proxy: str,
    remove_earnings: bool,
    winsor_label: str,
    use_decay: bool,
    decay_halflife: float,
    rolling_window: int,
) -> None:
    analyses = result.get("analyses", {})
    corr_matrix = result.get("correlation_matrix", pd.DataFrame())
    simple_corr_matrix = result.get("simple_return_correlation_matrix", pd.DataFrame())
    failures = result.get("failures", [])

    if not analyses:
        st.warning("No successful symbol analyses were returned.")
        if failures:
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
        return

    top = st.columns(4)
    with top[0]:
        st.metric("Benchmark", benchmark_symbol)
    with top[1]:
        st.metric("Symbols", len(analyses))
    with top[2]:
        st.metric("Regime", regime)
    with top[3]:
        st.metric("Rolling Window", rolling_window)

    selected_symbol = st.selectbox(
        "Correlation Focus",
        options=list(analyses.keys()),
        index=0,
        help="Switch the single-name view without rerunning the full analysis.",
    )
    selected = analyses[selected_symbol]
    winsorize_pct = WINSOR_OPTIONS[winsor_label]
    view = _build_correlation_view(
        selected_symbol,
        benchmark_symbol,
        selected,
        regime=regime,
        winsorize_pct=winsorize_pct,
        use_decay=use_decay,
        decay_halflife=decay_halflife,
        rolling_window=rolling_window,
    )

    selected_sector = None if sector_proxy == "Auto detect / not set" else sector_proxy
    st.caption(
        f"Proxy stack: market `{benchmark_symbol}`"
        + (f" | sector `{selected_sector}`" if selected_sector else "")
        + f" | earnings {'removed' if remove_earnings else 'included'}"
        + f" | winsorization `{winsor_label}`"
        + (f" | decay half-life `{decay_halflife:g}`" if use_decay else "")
    )

    if view:
        metric_cols = st.columns(6)
        with metric_cols[0]:
            st.metric("Displayed Corr", f"{view['correlation']:.4f}" if pd.notna(view["correlation"]) else "N/A")
        with metric_cols[1]:
            st.metric("All-Regime Corr", f"{view['regime_map']['all']:.4f}" if pd.notna(view["regime_map"]["all"]) else "N/A")
        with metric_cols[2]:
            st.metric("Bull Corr", f"{view['regime_map']['bull']:.4f}" if pd.notna(view["regime_map"]["bull"]) else "N/A")
        with metric_cols[3]:
            st.metric("Bear Corr", f"{view['regime_map']['bear']:.4f}" if pd.notna(view["regime_map"]["bear"]) else "N/A")
        with metric_cols[4]:
            st.metric("Corr Range", f"{view['range_min']:.3f} to {view['range_max']:.3f}" if pd.notna(view["range_min"]) and pd.notna(view["range_max"]) else "N/A")
        with metric_cols[5]:
            st.metric("Obs", view["observations"])

        chart_left, chart_right = st.columns([1.45, 1])
        with chart_left:
            st.markdown("**Rolling correlation**")
            rolling_df = pd.DataFrame({"rolling_correlation": view["rolling"]}).dropna()
            if not rolling_df.empty:
                st.line_chart(rolling_df)
            else:
                st.info("Not enough observations for the selected rolling window.")
        with chart_right:
            st.markdown("**Return scatter**")
            scatter_df = view["returns"].rename(
                columns={selected_symbol: f"{selected_symbol} return", benchmark_symbol: f"{benchmark_symbol} return"}
            )
            st.scatter_chart(scatter_df)

        table_left, table_right = st.columns(2)
        with table_left:
            regime_table = pd.DataFrame(
                [
                    {"slice": "All", "correlation": view["regime_map"]["all"]},
                    {"slice": "Bull", "correlation": view["regime_map"]["bull"]},
                    {"slice": "Bear", "correlation": view["regime_map"]["bear"]},
                ]
            )
            st.markdown("**Regime map**")
            st.dataframe(regime_table, use_container_width=True, hide_index=True)
        with table_right:
            rolling_summary = pd.DataFrame(
                [
                    {"metric": "Rolling min", "value": view["rolling_min"]},
                    {"metric": "Rolling max", "value": view["rolling_max"]},
                    {"metric": "Rolling latest", "value": view["rolling_latest"]},
                    {"metric": "Range spread", "value": view["range_spread"]},
                    {"metric": "Earnings removed days", "value": view["earnings_removed_days"]},
                ]
            )
            st.markdown("**Range summary**")
            st.dataframe(rolling_summary, use_container_width=True, hide_index=True)

    with st.expander("Correlation matrices", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown("**Log-return correlation**")
            st.dataframe(corr_matrix, use_container_width=True)
        with right:
            st.markdown("**Simple-return correlation**")
            st.dataframe(simple_corr_matrix, use_container_width=True)

    with st.expander("Underlying aligned prices", expanded=False):
        aligned_prices = selected.get("aligned_prices", pd.DataFrame())
        st.dataframe(aligned_prices.reset_index(), use_container_width=True, hide_index=True)

    impacts_df = selected.get("impacts_df", pd.DataFrame())
    if not impacts_df.empty:
        with st.expander("Earnings impacts used by backend", expanded=False):
            st.dataframe(impacts_df, use_container_width=True, hide_index=True)

    if failures:
        with st.expander("Failed tickers", expanded=False):
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)


def _metadata_dict(metadata: Optional[FetchMetadata]) -> Dict[str, Any]:
    if metadata is None:
        return {}
    return {
        "asset_type": metadata.asset_type,
        "provider": metadata.provider,
        "symbol": metadata.symbol,
        "option_ticker": metadata.option_ticker or "",
        "from_date": metadata.from_date.isoformat(),
        "to_date": metadata.to_date.isoformat(),
        "adjusted": metadata.adjusted,
    }


def _render_frame(frame: pd.DataFrame) -> None:
    metadata = frame.attrs.get("metadata")
    attempts = frame.attrs.get("provider_attempts", [])

    if metadata:
        meta_cols = st.columns(6)
        with meta_cols[0]:
            st.metric("Provider", metadata.provider)
        with meta_cols[1]:
            st.metric("Asset", metadata.asset_type)
        with meta_cols[2]:
            st.metric("Symbol", metadata.symbol)
        with meta_cols[3]:
            st.metric("Interval", f"{metadata.multiplier} {metadata.timespan}")
        with meta_cols[4]:
            st.metric("Rows", len(frame))
        with meta_cols[5]:
            st.metric("Adjusted", "Yes" if metadata.adjusted else "No")

        if metadata.option_ticker:
            st.caption(f"Option ticker: `{metadata.option_ticker}`")

    if attempts:
        with st.expander("Routing attempts", expanded=False):
            attempts_frame = pd.DataFrame(
                [
                    {
                        "provider": attempt.provider,
                        "reason": attempt.reason,
                        "success": attempt.success,
                        "error": attempt.error or "",
                    }
                    for attempt in attempts
                ]
            )
            st.dataframe(attempts_frame, use_container_width=True, hide_index=True)

    chart_col, info_col = st.columns([2.2, 1])
    with chart_col:
        plot_frame = frame[["close", "adjusted_close"]].copy()
        st.line_chart(plot_frame)

    with info_col:
        summary = {
            "start": frame.index.min().date().isoformat() if not frame.empty else "",
            "end": frame.index.max().date().isoformat() if not frame.empty else "",
            "columns": ", ".join(frame.columns.tolist()),
        }
        st.json(summary)

    with st.expander("Preview data", expanded=True):
        st.dataframe(frame.reset_index(), use_container_width=True, hide_index=True)

    csv_data = frame.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="price_data.csv",
        mime="text/csv",
        use_container_width=False,
    )

    with st.expander("Metadata JSON", expanded=False):
        st.json(_metadata_dict(metadata))


def _download_csv_button(label: str, frame: pd.DataFrame, file_name: str, *, use_container_width: bool = True) -> None:
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=use_container_width,
    )


def _build_filtered_multiticker_view(
    frame: pd.DataFrame,
    selected_fields: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if not selected_fields:
        raise ValueError("Select at least one field for the filtered view.")

    indexed = frame.reset_index().copy()
    indexed["date"] = pd.to_datetime(indexed["date"], errors="coerce")
    pieces: list[pd.DataFrame] = []

    for field in selected_fields:
        pivoted = indexed.pivot_table(index="date", columns="symbol", values=field, aggfunc="last")
        pivoted = pivoted.sort_index()
        if len(selected_fields) == 1:
            pivoted.columns = [str(col) for col in pivoted.columns]
        else:
            pivoted.columns = [f"{str(col)}__{field}" for col in pivoted.columns]
        pieces.append(pivoted)

    filtered = pd.concat(pieces, axis=1).sort_index(axis=1)
    filtered.index.name = "date"
    return filtered


def _render_multi_ticker_results(frame: pd.DataFrame, selected_fields: list[str]) -> None:
    metadata = frame.attrs.get("metadata", {})
    successes = frame.attrs.get("multi_fetch_successes", [])
    failures = frame.attrs.get("multi_fetch_failures", [])
    filtered_frame = _build_filtered_multiticker_view(frame, selected_fields)

    meta_cols = st.columns(5)
    with meta_cols[0]:
        st.metric("Requested", len(metadata.get("symbols_requested", [])))
    with meta_cols[1]:
        st.metric("Fetched", len(metadata.get("symbols_returned", [])))
    with meta_cols[2]:
        st.metric("Failed", len(failures))
    with meta_cols[3]:
        st.metric("Rows", len(frame))
    with meta_cols[4]:
        st.metric("Fields in CSV", len(selected_fields))

    if successes:
        with st.expander("Per-ticker fetch status", expanded=False):
            success_frame = pd.DataFrame(
                [
                    {
                        "symbol": item["symbol"],
                        "provider": item["provider"],
                        "rows": item["rows"],
                    }
                    for item in successes
                ]
            )
            st.dataframe(success_frame, use_container_width=True, hide_index=True)

    if failures:
        with st.expander("Failed tickers", expanded=True):
            failure_frame = pd.DataFrame(failures)
            st.dataframe(failure_frame, use_container_width=True, hide_index=True)

    chart_source = filtered_frame.copy()
    numeric_candidates = [col for col in chart_source.columns if pd.api.types.is_numeric_dtype(chart_source[col])]
    if numeric_candidates:
        st.line_chart(chart_source[numeric_candidates])

    st.markdown("**Filtered view for CSV/export**")
    st.dataframe(filtered_frame.reset_index(), use_container_width=True, hide_index=True)

    download_cols = st.columns(2)
    with download_cols[0]:
        _download_csv_button(
            "Download filtered CSV",
            filtered_frame.reset_index(),
            "multi_ticker_filtered.csv",
        )
    with download_cols[1]:
        _download_csv_button(
            "Download full raw CSV",
            frame.reset_index(),
            "multi_ticker_full.csv",
        )

    with st.expander("Full raw dataset preview", expanded=False):
        st.dataframe(frame.reset_index(), use_container_width=True, hide_index=True)


def _multi_ticker_panel() -> None:
    st.subheader("Multi-Ticker Prices")
    st.write(
        "Fetch full daily OHLCV for multiple tickers in one run, then export only the columns you care about."
    )

    with st.form("multi_ticker_form", clear_on_submit=False):
        row1, row2, row3 = st.columns([2.2, 1, 1])
        with row1:
            symbols = st.text_area(
                "Tickers",
                value="AAPL, MSFT, NVDA, SPY",
                help="Comma, space, newline, or semicolon separated tickers are supported.",
                height=100,
            )
        with row2:
            from_date = st.date_input("From", value=date.today() - timedelta(days=180), key="multi_from")
        with row3:
            to_date = st.date_input("To", value=date.today(), key="multi_to")

        row4, row5, row6 = st.columns(3)
        with row4:
            provider = st.selectbox(
                "Provider routing",
                options=["auto", "yfinance", "polygon", "alphavantage"],
                index=0,
                key="multi_provider",
            )
        with row5:
            adjusted = st.toggle("Adjusted OHLC", value=True, key="multi_adjusted")
        with row6:
            interval_label = st.radio("Interval", options=["Daily", "Hourly"], horizontal=True, key="multi_interval")
            timespan = "hour" if interval_label == "Hourly" else "day"

        selected_fields = st.multiselect(
                "Fields to show in CSV",
                options=FIELD_OPTIONS,
                default=["close"],
                help="All fields are fetched in the backend. This controls the filtered export view.",
            )


        submitted = st.form_submit_button("Fetch multiple tickers", use_container_width=True)

    if submitted:

        if not symbols.strip():
            st.error("Please enter at least one ticker.")
            return
        if not selected_fields:
            st.error("Please select at least one field for the export filter.")
            return

        with st.spinner("Fetching multiple ticker histories..."):
            try:
                frame = fetch_multiple_equity_prices(
                    symbols=symbols,
                    from_date=from_date,
                    to_date=to_date,
                    adjusted=adjusted,
                    timespan=timespan,
                    multiplier=1,
                    polygon_api_key=st.session_state["polygon_api_key"] or None,
                    alphavantage_api_key=st.session_state["alphavantage_api_key"] or None,
                    prefer_provider=None if provider == "auto" else provider,
                )
                st.success("Multi-ticker data fetched successfully.")
                _render_multi_ticker_results(frame, selected_fields)
            except (PriceFetchError, ValueError, RuntimeError, ImportError) as exc:
                st.error(str(exc))


def _equity_panel() -> None:
    st.subheader("Equity Prices")
    st.write(
        "Fetch daily OHLCV with automatic routing across free-friendly providers. "
        "The backend prioritizes hit rate first and then falls back intelligently."
    )

    with st.form("equity_form", clear_on_submit=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            symbol = st.text_input("Ticker", value="AAPL").strip().upper()
        with col2:
            from_date = st.date_input("From", value=date.today() - timedelta(days=180))
        with col3:
            to_date = st.date_input("To", value=date.today())
        with col4:
            provider = st.selectbox(
                "Provider routing",
                options=["auto", "yfinance", "polygon", "alphavantage"],
                index=0,
            )

        col5, col6 = st.columns(2)
        with col5:
            adjusted = st.toggle("Adjusted OHLC", value=True)
        with col6:
            interval_label = st.radio("Interval", options=["Daily", "Hourly"], horizontal=True, key="equity_interval")
            timespan = "hour" if interval_label == "Hourly" else "day"


        submitted = st.form_submit_button("Fetch equity data", use_container_width=True)

    if submitted:

        if not symbol:
            st.error("Ticker is required.")
            return

        with st.spinner("Fetching equity price history..."):
            try:
                frame = fetch_equity_prices(
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date,
                    adjusted=adjusted,
                    timespan=timespan,
                    multiplier=1,
                    polygon_api_key=st.session_state["polygon_api_key"] or None,
                    alphavantage_api_key=st.session_state["alphavantage_api_key"] or None,
                    prefer_provider=None if provider == "auto" else provider,
                )
                st.success("Equity data fetched successfully.")
                _render_frame(frame)
            except (PriceFetchError, ValueError, RuntimeError, ImportError) as exc:
                st.error(str(exc))


def _option_panel() -> None:
    st.subheader("Option Prices")
    st.write(
        "Fetch historical option aggregate bars from Polygon. "
        "You can pass either the full Polygon option ticker or build one from underlying inputs."
    )

    ticker_mode = st.radio(
        "Input mode",
        options=["Build from symbol/expiry/strike/right", "Use complete option ticker"],
        horizontal=True,
    )

    with st.form("option_form", clear_on_submit=False):
        top1, top2 = st.columns(2)
        with top1:
            from_date = st.date_input("From ", value=date.today() - timedelta(days=45), key="opt_from")
        with top2:
            to_date = st.date_input("To ", value=date.today(), key="opt_to")

        interval_label = st.radio("Interval ", options=["Daily", "Hourly"], horizontal=True, key="opt_interval")
        timespan = "hour" if interval_label == "Hourly" else "day"


        option_ticker = None
        symbol = None
        expiry = None
        strike = None
        right = None

        if ticker_mode == "Use complete option ticker":
            option_ticker = st.text_input(
                "Option ticker",
                value="O:SPY261218C00640000",
                help="Example: O:SPY251219C00640000",
            ).strip().upper()
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                symbol = st.text_input("Underlying ticker", value="SPY").strip().upper()
            with c2:
                expiry = st.date_input("Expiry", value=date(2026, 12, 18))
            with c3:
                strike = st.number_input("Strike", min_value=0.01, value=640.0, step=1.0)
            with c4:
                right = st.selectbox("Call / Put", options=["C", "P"], index=0)

            if symbol and expiry and strike and right:
                built_ticker = build_polygon_option_ticker(
                    underlying=symbol,
                    expiry=expiry,
                    right=right,
                    strike=float(strike),
                )
                st.caption(f"Built Polygon ticker: `{built_ticker}`")

        submitted = st.form_submit_button("Fetch option data", use_container_width=True)

    if submitted:

        with st.spinner("Fetching option price history..."):
            try:
                frame = fetch_option_prices(
                    from_date=from_date,
                    to_date=to_date,
                    option_ticker=option_ticker or None,
                    underlying=symbol or None,
                    expiry=expiry,
                    right=right,
                    strike=float(strike) if strike is not None else None,
                    adjusted=True,
                    timespan=timespan,
                    multiplier=1,
                    polygon_api_key=st.session_state["polygon_api_key"] or None,
                )
                st.success("Option data fetched successfully.")
                _render_frame(frame)
            except (PriceFetchError, ValueError, RuntimeError, ImportError) as exc:
                st.error(str(exc))


def _render_quant_results(result: Dict[str, Any], benchmark_symbol: str) -> None:
    summary_table = result.get("summary_table", pd.DataFrame())
    failures = result.get("failures", [])
    analyses = result.get("analyses", {})
    corr_matrix = result.get("correlation_matrix", pd.DataFrame())
    simple_corr_matrix = result.get("simple_return_correlation_matrix", pd.DataFrame())
    signals = result.get("signals_table", pd.DataFrame())

    top_cols = st.columns(5)
    with top_cols[0]:
        st.metric("Benchmark", benchmark_symbol)
    with top_cols[1]:
        st.metric("Analyzed", len(summary_table))
    with top_cols[2]:
        st.metric("Failures", len(failures))
    with top_cols[3]:
        st.metric("Signals", len(signals))
    with top_cols[4]:
        st.metric("Corr Size", f"{corr_matrix.shape[0]}x{corr_matrix.shape[1]}")

    if not summary_table.empty:
        st.markdown("**Summary metrics**")
        ordered_cols = [col for col in SUMMARY_COLUMNS if col in summary_table.columns]
        summary_display = summary_table[ordered_cols].copy()
        st.dataframe(summary_display, use_container_width=True, hide_index=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            _download_csv_button("Download summary CSV", summary_display, "quant_summary.csv")
        with dl2:
            _download_csv_button("Download signals CSV", signals, "quant_signals.csv")

    if not corr_matrix.empty:
        left, right = st.columns(2)
        with left:
            st.markdown("**Log-return correlation**")
            st.dataframe(corr_matrix, use_container_width=True)
        with right:
            st.markdown("**Simple-return correlation**")
            st.dataframe(simple_corr_matrix, use_container_width=True)

    if failures:
        with st.expander("Failed tickers", expanded=True):
            failure_frame = pd.DataFrame(failures)
            st.dataframe(failure_frame, use_container_width=True, hide_index=True)

    if not signals.empty:
        with st.expander("Stat-arb signals", expanded=True):
            st.dataframe(signals, use_container_width=True, hide_index=True)

    if analyses:
        symbol_options = list(analyses.keys())
        selected_symbol = st.selectbox("Inspect symbol analysis", options=symbol_options, index=0)
        selected = analyses[selected_symbol]

        st.markdown(f"**{selected_symbol} vs {benchmark_symbol}**")
        chart_cols = st.columns(2)
        aligned_prices = selected.get("aligned_prices", pd.DataFrame())
        residuals = selected.get("residuals", pd.DataFrame())

        with chart_cols[0]:
            if not aligned_prices.empty:
                st.line_chart(
                    aligned_prices[[f"{selected_symbol}_close", f"{benchmark_symbol}_close"]]
                )
        with chart_cols[1]:
            if not residuals.empty:
                st.line_chart(residuals[["residual", "residual_z"]])

        exp1, exp2, exp3, exp4 = st.columns(4)
        with exp1:
            st.metric("Outliers", int(selected["summary"]["outlier_count"]))
        with exp2:
            st.metric("Beta", f"{selected['summary']['beta']:.4f}")
        with exp3:
            st.metric("Correlation", f"{selected['summary']['correlation']:.4f}")
        with exp4:
            st.metric("Max DD", f"{selected['summary']['asset_max_drawdown']:.2%}")

        with st.expander("Outlier table", expanded=False):
            outliers = selected.get("outliers_table", pd.DataFrame())
            st.dataframe(outliers, use_container_width=True, hide_index=True)
            if not outliers.empty:
                _download_csv_button(
                    f"Download {selected_symbol} outliers CSV",
                    outliers,
                    f"{selected_symbol.lower()}_outliers.csv",
                    use_container_width=False,
                )

        with st.expander("Aligned price matrix", expanded=False):
            st.dataframe(aligned_prices.reset_index(), use_container_width=True, hide_index=True)

        impacts_df = selected.get("impacts_df", pd.DataFrame())
        if not impacts_df.empty:
            with st.expander("Earnings impacts used for filtering", expanded=False):
                st.dataframe(impacts_df, use_container_width=True, hide_index=True)


def _correlation_panel() -> None:
    _inject_correlation_styles()
    st.markdown(
        """
        <div class="corr-shell">
            <div class="corr-kicker">Correlation Studio</div>
            <div class="corr-title">Structural, Regime-Aware Correlation</div>
            <div class="corr-copy">
                Explore how a single equity's relationship to the market changes across time,
                regimes, earnings windows, and tail handling choices. This frontend keeps the
                workflow correlation-first and lets you compare the clean view against the raw one.
            </div>
            <div class="corr-chip-row">
                <div class="corr-chip">SPY systematic beta</div>
                <div class="corr-chip">Sector ETF proxy</div>
                <div class="corr-chip">Bull vs bear regimes</div>
                <div class="corr-chip">Earnings-date scrubbing</div>
                <div class="corr-chip">Winsorized tails</div>
                <div class="corr-chip">Optional decay weighting</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="corr-section">Inputs</div>', unsafe_allow_html=True)
    with st.form("correlation_form", clear_on_submit=False):
        row1, row2, row3, row4 = st.columns([1.85, 1, 1, 1])
        with row1:
            symbols = st.text_area(
                "Equities to analyze",
                value="AAPL, MSFT, NVDA",
                help="Enter one or more symbols. The detailed single-name view updates after the run.",
                height=96,
            )
        with row2:
            benchmark = st.text_input("Market proxy", value="SPY").strip().upper()
        with row3:
            from_date = st.date_input("From", value=date.today() - timedelta(days=365), key="corr_from")
        with row4:
            to_date = st.date_input("To", value=date.today(), key="corr_to")

        row5, row6, row7, row8 = st.columns(4)
        with row5:
            provider = st.selectbox(
                "Provider routing",
                options=["auto", "yfinance", "polygon", "alphavantage"],
                index=0,
                key="corr_provider",
            )
        with row6:
            adjusted = st.toggle("Adjusted prices", value=True, key="corr_adjusted")
        with row7:
            regime = st.selectbox("Regime view", options=["All", "Bull", "Bear"], index=0)
        with row8:
            sector_proxy = st.selectbox("Sector ETF proxy", options=SECTOR_ETF_OPTIONS, index=0)

        row9, row10, row11, row12 = st.columns(4)
        with row9:
            remove_earnings = st.toggle("Remove earnings windows", value=False, key="corr_earnings")
        with row10:
            winsor_label = st.selectbox("Tail handling", options=list(WINSOR_OPTIONS.keys()), index=0)
        with row11:
            use_decay = st.toggle("Use decay weighting", value=False, key="corr_decay")
        with row12:
            decay_halflife = st.number_input(
                "Decay half-life",
                min_value=2.0,
                max_value=252.0,
                value=20.0,
                step=1.0,
                format="%.1f",
                disabled=not use_decay,
            )

        row13, row14, row15, row16 = st.columns(4)
        with row13:
            lookback = st.number_input("Regression lookback", min_value=20, max_value=1000, value=252, step=10)
        with row14:
            rolling_window = st.number_input("Rolling corr window", min_value=10, max_value=252, value=60, step=5)
        with row15:
            max_missing = st.number_input(
                "Max missing vs benchmark",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                format="%.2f",
            )
        with row16:
            risk_free_rate = st.number_input(
                "Risk-free rate",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                format="%.2f",
            )


        submitted = st.form_submit_button("Run correlation analysis", use_container_width=True)

    if submitted:

        winsorize_pct = WINSOR_OPTIONS[winsor_label]

        if not symbols.strip():
            st.error("Please enter at least one ticker to analyze.")
            return
        if not benchmark:
            st.error("Market proxy is required.")
            return
        if remove_earnings and not st.session_state["alphavantage_api_key"]:
            st.warning("Removing earnings windows requires an Alpha Vantage API key. This feature will be skipped.")

        with st.spinner("Building correlation views..."):
            try:
                # Initialize the new configuration object
                # Convert half-life to per-period decay factor: factor = 0.5^(1/halflife)
                decay_factor = 0.5 ** (1.0 / float(decay_halflife)) if use_decay else None

                config = CorrelationConfig(
                    benchmark_symbol=benchmark,
                    remove_earnings=remove_earnings,
                    winsorize=winsorize_pct is not None,
                    winsorize_limits=(winsorize_pct, winsorize_pct) if winsorize_pct else (0.01, 0.01),
                    decay_factor=decay_factor,
                    regime_filter=regime.lower(),
                    rolling_window=int(rolling_window),
                    regression_lookback=int(lookback),
                    sector_relative=sector_proxy != "Auto detect / not set",
                )

                result = analyze_symbols_vs_benchmark(
                    symbols=symbols,
                    config=config,
                    from_date=from_date,
                    to_date=to_date,
                    adjusted=adjusted,
                    polygon_api_key=st.session_state["polygon_api_key"] or None,
                    alphavantage_api_key=st.session_state["alphavantage_api_key"] or None,
                    prefer_provider=None if provider == "auto" else provider,
                    max_missing_vs_benchmark=float(max_missing),
                    risk_free_rate=float(risk_free_rate),
                    manual_sector_etf=None if sector_proxy == "Auto detect / not set" else sector_proxy,
                )
                st.success("Correlation analysis completed successfully.")
                _render_correlation_results(
                    result,
                    benchmark,
                    regime=regime,
                    sector_proxy=sector_proxy,
                    remove_earnings=remove_earnings,
                    winsor_label=winsor_label,
                    use_decay=use_decay,
                    decay_halflife=float(decay_halflife),
                    rolling_window=int(rolling_window),
                )
            except (PriceFetchError, ValueError, RuntimeError, ImportError) as exc:
                st.error(str(exc))


def main() -> None:
    _init_state()

    st.title("Market Data And Quant Workbench")
    st.caption(
        "Modular frontend over `fetch_prices.py` and `quant_analysis.py` for market data, options, and benchmark-aware analytics."
    )

    intro_left, intro_right = st.columns([1.35, 1])
    with intro_left:
        st.markdown(
            """
            This UI sits on top of modular backend modules:

            - `fetch_prices.py` handles provider-aware stock and option data retrieval.
            - `quant_analysis.py` handles benchmark-aware correlation, regression, volatility, drawdown, and signal generation.
            - The frontend stays intentionally thin so the backend logic remains reusable and testable.
            """
        )
    with intro_right:
        try:
            import yfinance as yf
        except ImportError:
            st.info(
                "Install `yfinance` if you want the best equity fallback path:\n\n"
                "`pip install yfinance`"
            )

    tab_corr, tab_multi, tab_equity, tab_option = st.tabs(
        ["Correlation", "Multi Tickers", "Equities", "Options"]
    )
    with tab_corr:
        _correlation_panel()
    with tab_multi:
        _multi_ticker_panel()
    with tab_equity:
        _equity_panel()
    with tab_option:
        _option_panel()


if __name__ == "__main__":
    main()
