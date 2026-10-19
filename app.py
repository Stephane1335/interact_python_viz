from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    layout="wide",
    page_title="CPS*5745 Stock Comparison Dashboard",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F6F9;
        color: #1E293B;
    }
    h1, h2, h3, [data-testid="stMetricLabel"] {
        color: #1E293B;
        font-family: "Segoe UI", Arial, sans-serif;
    }
    h1 {
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        padding: 16px;
    }
    .kpi-card {
        box-sizing: border-box;
        min-height: 142px;
        height: 100%;
        padding: 16px;
        background: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        font-family: "Segoe UI", Arial, sans-serif;
    }
    .kpi-label {
        margin: 0 0 12px;
        color: #64748B;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .kpi-value {
        margin: 0;
        color: #1E293B;
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.25;
    }
    .kpi-detail {
        margin: 8px 0 0;
        color: #64748B;
        font-size: 0.85rem;
    }
    .insights-container {
        margin-top: 24px;
        padding: 22px 24px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #0083B0;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        color: #334155;
        font-family: "Segoe UI", Arial, sans-serif;
        line-height: 1.65;
    }
    .insights-container h3 {
        margin: 0 0 12px;
        color: #1E293B;
        font-size: 1.2rem;
    }
    .insights-container ul {
        margin: 0;
        padding-left: 22px;
    }
    .insights-container li {
        margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Interactive Stock Comparison Dashboard")
st.subheader("Developed by Jean Stephane Ruben YAPO")
st.markdown("**Developed for CPS*5745**")

available_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]

with st.sidebar:
    st.header("Dashboard Controls")
    selected_tickers = st.multiselect(
        "Select stock tickers",
        options=available_tickers,
        default=["AAPL", "MSFT", "GOOGL"],
    )
    today = date.today()
    selected_dates = st.date_input(
        "Select date range",
        value=(today - timedelta(days=365), today),
        max_value=today,
    )


@st.cache_data
def fetch_stock_data(
    tickers: tuple[str, ...], start_date: date, end_date: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted closing prices and trading volumes."""
    data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date + timedelta(days=1),
        progress=False,
        auto_adjust=False,
    )

    def extract_field(field: str) -> pd.DataFrame:
        if isinstance(data.columns, pd.MultiIndex):
            matching_level = next(
                (
                    level
                    for level in range(data.columns.nlevels)
                    if field in data.columns.get_level_values(level)
                ),
                None,
            )
            if matching_level is None:
                return pd.DataFrame(index=data.index, columns=tickers)
            values = data.xs(field, axis=1, level=matching_level)
        elif field in data.columns:
            values = data[field]
        else:
            return pd.DataFrame(index=data.index, columns=tickers)

        if isinstance(values, pd.Series):
            values = values.to_frame(name=tickers[0])
        return values.reindex(columns=tickers)

    return extract_field("Adj Close"), extract_field("Volume")


if not selected_tickers:
    st.info("Select at least one ticker from the sidebar to view stock data.")
elif len(selected_dates) != 2:
    st.info("Select both a start date and an end date.")
else:
    start_date, end_date = selected_dates
    if start_date > end_date:
        st.warning("The start date must be on or before the end date.")
    else:
        try:
            close_prices, volumes = fetch_stock_data(
                tuple(selected_tickers), start_date, end_date
            )
        except Exception as error:
            st.error(f"Unable to download stock data: {error}")
            st.stop()

        close_prices = close_prices.dropna(how="all")
        volumes = volumes.reindex(close_prices.index)

        if close_prices.empty:
            st.warning("No stock data was returned for the selected tickers and dates.")
        else:
            chart_tickers = [
                ticker
                for ticker in selected_tickers
                if ticker in close_prices.columns
                and close_prices[ticker].notna().any()
            ]
            returns = close_prices[chart_tickers].pct_change(fill_method=None) * 100
            returns = returns.replace([np.inf, -np.inf], np.nan)
            performance = {}
            average_volumes = {}
            volatility = {}

            for ticker in chart_tickers:
                prices = close_prices[ticker].dropna()
                if not prices.empty and prices.iloc[0] != 0:
                    performance[ticker] = (
                        prices.iloc[-1] / prices.iloc[0] - 1
                    ) * 100

                ticker_volumes = volumes[ticker].dropna()
                if not ticker_volumes.empty:
                    average_volumes[ticker] = ticker_volumes.mean()

                ticker_returns = returns[ticker].dropna()
                if not ticker_returns.empty:
                    volatility[ticker] = ticker_returns.std()

            best_performer = max(performance, key=performance.get) if performance else None
            highest_volume = max(average_volumes, key=average_volumes.get) if average_volumes else None
            most_volatile = max(volatility, key=volatility.get) if volatility else None

            def format_volume(value: float) -> str:
                if value >= 1_000_000:
                    return f"{value / 1_000_000:,.2f}M shares/day"
                if value >= 1_000:
                    return f"{value / 1_000:,.2f}K shares/day"
                return f"{value:,.0f} shares/day"

            def render_kpi(
                label: str, value: str, detail: str
            ) -> str:
                return (
                    '<div class="kpi-card">'
                    f'<p class="kpi-label">{label}</p>'
                    f'<p class="kpi-value">{value}</p>'
                    f'<p class="kpi-detail">{detail}</p>'
                    "</div>"
                )

            st.subheader("Executive Summary")
            kpi_columns = st.columns(3)
            kpi_columns[0].markdown(
                render_kpi(
                    "Best Performing Stock",
                    f"{performance[best_performer]:+.2f}%"
                    if best_performer is not None
                    else "N/A",
                    best_performer or "No return data available",
                ),
                unsafe_allow_html=True,
            )
            kpi_columns[1].markdown(
                render_kpi(
                    "Highest Average Daily Volume",
                    format_volume(average_volumes[highest_volume])
                    if highest_volume is not None
                    else "N/A",
                    highest_volume or "No volume data available",
                ),
                unsafe_allow_html=True,
            )
            kpi_columns[2].markdown(
                render_kpi(
                    "Most Volatile Stock",
                    f"{volatility[most_volatile]:.2f}%"
                    if most_volatile is not None
                    else "N/A",
                    f"{most_volatile} · daily return standard deviation"
                    if most_volatile is not None
                    else "No return data available",
                ),
                unsafe_allow_html=True,
            )

            plotly_palette = [
                "#0083B0",
                "#00B4DB",
                "#2E3192",
                "#11998E",
                "#38EF7D",
            ]

            def apply_dashboard_style(figure: go.Figure) -> go.Figure:
                figure.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    colorway=plotly_palette,
                    font={"color": "#1E293B"},
                    legend={
                        "title": {"text": ""},
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.02,
                        "xanchor": "right",
                        "x": 1,
                    },
                    margin={"t": 55, "r": 20, "b": 40, "l": 55},
                )
                return figure

            st.subheader("Adjusted Closing Price")
            price_chart = go.Figure()
            for index, ticker in enumerate(chart_tickers):
                price_chart.add_trace(
                    go.Scatter(
                        x=close_prices.index,
                        y=close_prices[ticker],
                        mode="lines",
                        name=ticker,
                        line={"color": plotly_palette[index % len(plotly_palette)]},
                        hovertemplate=(
                            "%{x|%b %d, %Y}<br>"
                            f"{ticker}: $%{{y:,.2f}}<extra></extra>"
                        ),
                    )
                )
            price_chart.update_layout(
                title="Adjusted Closing Price Over Time",
                xaxis_title="Date",
                yaxis_title="Adjusted Close ($)",
                hovermode="x unified",
            )
            st.plotly_chart(
                apply_dashboard_style(price_chart),
                use_container_width=True,
            )

            cumulative_returns = close_prices[chart_tickers].apply(
                lambda prices: (prices / prices.dropna().iloc[0] - 1) * 100
            )
            daily_returns = returns.dropna(how="all")
            chart_columns = st.columns(2)

            with chart_columns[0]:
                st.subheader("Cumulative Returns")
                cumulative_chart = px.line(
                    cumulative_returns.rename_axis("Date").reset_index(),
                    x="Date",
                    y=chart_tickers,
                    color_discrete_sequence=plotly_palette,
                    title="Cumulative Returns Since Start Date",
                    labels={"value": "Cumulative Return (%)", "variable": "Ticker"},
                )
                cumulative_chart.update_traces(
                    hovertemplate=(
                        "%{x|%b %d, %Y}<br>%{fullData.name}: "
                        "%{y:+.2f}%<extra></extra>"
                    )
                )
                cumulative_chart.update_layout(
                    yaxis={"ticksuffix": "%"},
                    hovermode="x unified",
                )
                st.plotly_chart(
                    apply_dashboard_style(cumulative_chart),
                    use_container_width=True,
                )

            with chart_columns[1]:
                st.subheader("Daily Return Distribution")
                returns_long = (
                    daily_returns.rename_axis("Date")
                    .reset_index()
                    .melt(
                        id_vars="Date",
                        var_name="Ticker",
                        value_name="Daily Return (%)",
                    )
                    .dropna(subset=["Daily Return (%)"])
                )
                returns_box = px.box(
                    returns_long,
                    x="Ticker",
                    y="Daily Return (%)",
                    color="Ticker",
                    color_discrete_sequence=plotly_palette,
                    title="Distribution of Daily Percentage Returns",
                    points="outliers",
                )
                returns_box.update_traces(
                    hovertemplate=(
                        "%{x}<br>Daily return: %{y:+.2f}%<extra></extra>"
                    )
                )
                returns_box.update_layout(showlegend=False)
                st.plotly_chart(
                    apply_dashboard_style(returns_box),
                    use_container_width=True,
                )

            chart_columns = st.columns(2)
            with chart_columns[0]:
                st.subheader("Daily Trading Volume")
                volume_long = (
                    volumes[chart_tickers]
                    .rename_axis("Date")
                    .reset_index()
                    .melt(
                        id_vars="Date",
                        var_name="Ticker",
                        value_name="Shares Traded",
                    )
                    .dropna(subset=["Shares Traded"])
                )
                volume_chart = px.bar(
                    volume_long,
                    x="Date",
                    y="Shares Traded",
                    color="Ticker",
                    barmode="group",
                    color_discrete_sequence=plotly_palette,
                    title="Daily Trading Volume by Stock",
                )
                volume_chart.update_traces(
                    hovertemplate=(
                        "%{x|%b %d, %Y}<br>%{fullData.name}: "
                        "%{y:,.0f} shares<extra></extra>"
                    )
                )
                volume_chart.update_layout(
                    yaxis={"title": "Shares Traded"},
                    hovermode="x",
                )
                st.plotly_chart(
                    apply_dashboard_style(volume_chart),
                    use_container_width=True,
                )

            with chart_columns[1]:
                st.subheader("Return Correlation")
                correlation = daily_returns.corr(method="pearson")
                correlation_chart = go.Figure(
                    data=go.Heatmap(
                        z=correlation.values,
                        x=correlation.columns,
                        y=correlation.index,
                        zmin=-1,
                        zmax=1,
                        colorscale=[
                            [0, plotly_palette[2]],
                            [0.5, "#FFFFFF"],
                            [1, plotly_palette[0]],
                        ],
                        colorbar={"title": "Pearson r"},
                        hovertemplate=(
                            "%{y} vs %{x}<br>Pearson r: "
                            "%{z:.2f}<extra></extra>"
                        ),
                    )
                )
                correlation_chart.update_layout(
                    title="Pearson Correlation of Daily Returns",
                    xaxis_title="Ticker",
                    yaxis_title="Ticker",
                    yaxis={"autorange": "reversed"},
                )
                st.plotly_chart(
                    apply_dashboard_style(correlation_chart),
                    use_container_width=True,
                )

            insight_bullets = []
            if len(performance) >= 2:
                best_ticker = max(performance, key=performance.get)
                worst_ticker = min(performance, key=performance.get)
                performance_spread = (
                    performance[best_ticker] - performance[worst_ticker]
                )
                insight_bullets.append(
                    f"Price performance diverged by {performance_spread:.2f} "
                    f"percentage points: {best_ticker} led at "
                    f"{performance[best_ticker]:+.2f}%, while {worst_ticker} "
                    f"returned {performance[worst_ticker]:+.2f}% over the period."
                )
            elif performance:
                only_ticker = next(iter(performance))
                insight_bullets.append(
                    f"{only_ticker} returned {performance[only_ticker]:+.2f}% "
                    "over the selected period; select another stock to compare "
                    "performance divergence."
                )
            else:
                insight_bullets.append(
                    "There is not enough adjusted-price data to compare "
                    "performance over this period."
                )

            return_volatility = returns.std().dropna()
            if len(return_volatility) >= 2:
                riskiest_ticker = return_volatility.idxmax()
                steadiest_ticker = return_volatility.idxmin()
                riskiest_iqr = (
                    returns[riskiest_ticker].quantile(0.75)
                    - returns[riskiest_ticker].quantile(0.25)
                )
                insight_bullets.append(
                    f"The daily-return box plots show {riskiest_ticker} has the "
                    f"highest return volatility ({return_volatility[riskiest_ticker]:.2f}% "
                    f"standard deviation; {riskiest_iqr:.2f}% interquartile range), "
                    f"while {steadiest_ticker} is the steadiest at "
                    f"{return_volatility[steadiest_ticker]:.2f}% standard deviation. "
                    "Wider boxes and more extreme points indicate greater "
                    "day-to-day risk."
                )
            elif len(return_volatility) == 1:
                ticker = return_volatility.index[0]
                insight_bullets.append(
                    f"The daily-return standard deviation for {ticker} is "
                    f"{return_volatility[ticker]:.2f}%; add another stock to "
                    "compare risk distributions."
                )
            else:
                insight_bullets.append(
                    "There is not enough daily-return data to assess volatility "
                    "and risk."
                )

            correlation = daily_returns.corr(method="pearson")
            correlation_pairs = [
                (correlation.iloc[row, column], correlation.index[row], correlation.columns[column])
                for row in range(len(correlation.columns))
                for column in range(row + 1, len(correlation.columns))
                if pd.notna(correlation.iloc[row, column])
            ]
            if correlation_pairs:
                lowest_correlation, lower_ticker, upper_ticker = min(
                    correlation_pairs, key=lambda pair: pair[0]
                )
                highest_correlation, high_ticker_a, high_ticker_b = max(
                    correlation_pairs, key=lambda pair: pair[0]
                )
                if lowest_correlation < 0:
                    diversification_text = (
                        f"{lower_ticker} and {upper_ticker} are the most "
                        f"negatively correlated pair (Pearson r = "
                        f"{lowest_correlation:.2f}), suggesting the strongest "
                        "diversification potential among these selections."
                    )
                elif lowest_correlation < 0.5:
                    diversification_text = (
                        f"{lower_ticker} and {upper_ticker} have the lowest "
                        f"return correlation (Pearson r = {lowest_correlation:.2f}), "
                        "offering relatively better diversification than the "
                        "other selected pairs."
                    )
                else:
                    diversification_text = (
                        f"All selected pairs are positively correlated; even "
                        f"the lowest pair, {lower_ticker} and {upper_ticker} "
                        f"(Pearson r = {lowest_correlation:.2f}), may provide "
                        "limited diversification."
                    )
                insight_bullets.append(
                    diversification_text
                    + f" The strongest co-movement is between {high_ticker_a} "
                    f"and {high_ticker_b} (r = {highest_correlation:.2f})."
                )
            else:
                insight_bullets.append(
                    "The correlation heatmap needs at least two stocks with "
                    "overlapping daily returns to provide diversification insights."
                )

            insights_html = "".join(
                f"<li>{bullet}</li>" for bullet in insight_bullets
            )
            st.markdown(
                '<div class="insights-container">'
                "<h3>Key Insights &amp; Differences Revealed</h3>"
                f"<ul>{insights_html}</ul>"
                "</div>",
                unsafe_allow_html=True,
            )
