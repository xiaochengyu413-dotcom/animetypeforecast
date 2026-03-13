from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from evaluate_forecasts import (
    build_display_axis,
    build_prophet_model,
    build_xticks,
    resolve_font,
    safe_name,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "generated" / "theme_quarterly_model_ready.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated" / "future_forecast"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Prophet on each theme and forecast two years beyond the last observed quarter."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", default="popularity_index")
    parser.add_argument("--forecast-years", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--theme", action="append", dest="themes")
    parser.add_argument("--forecast-stretch", type=float, default=3.0)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-html-plots", action="store_true")
    return parser.parse_args()


def seasonal_naive_future(
    theme_frame: pd.DataFrame,
    future_dates: pd.Series,
    target: str,
) -> np.ndarray:
    ordered = theme_frame.sort_values("ds").copy()
    ordered["quarter_num"] = ordered["ds"].dt.quarter
    last_same_quarter = (
        ordered.groupby("quarter_num", as_index=False)
        .tail(1)
        .set_index("quarter_num")[target]
        .to_dict()
    )
    fallback = float(ordered[target].iloc[-1])

    predictions: list[float] = []
    for ds in pd.Series(future_dates).tolist():
        timestamp = pd.Timestamp(ds)
        predictions.append(float(last_same_quarter.get(timestamp.quarter, fallback)))
    return np.asarray(predictions, dtype=float)


def prophet_series_forecast(
    theme_frame: pd.DataFrame,
    target: str,
    future_dates: pd.Series,
    nonnegative: bool = False,
) -> pd.DataFrame:
    prophet_train = theme_frame[["ds", target]].rename(columns={target: "y"})
    model = build_prophet_model()
    model.fit(prophet_train)

    prediction_frame = pd.DataFrame(
        {"ds": pd.Series(future_dates).drop_duplicates().sort_values().reset_index(drop=True)}
    )
    forecast = model.predict(prediction_frame)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    if nonnegative:
        forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[
            ["yhat", "yhat_lower", "yhat_upper"]
        ].clip(lower=0.0)
    return forecast


def forecast_theme(
    theme_frame: pd.DataFrame,
    target: str,
    forecast_years: int,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    ordered = theme_frame.sort_values("ds").reset_index(drop=True).copy()
    last_observed_ds = pd.Timestamp(ordered["ds"].max())
    forecast_periods = max(int(forecast_years) * 4, 1)

    popularity_train = ordered[["ds", target]].rename(columns={target: "y"})
    popularity_model = build_prophet_model()
    popularity_model.fit(popularity_train)

    future = popularity_model.make_future_dataframe(periods=forecast_periods, freq="QS")
    forecast = popularity_model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    title_forecast = prophet_series_forecast(
        theme_frame=ordered,
        target="title_count",
        future_dates=future["ds"],
        nonnegative=True,
    ).rename(
        columns={
            "yhat": "title_count_yhat",
            "yhat_lower": "title_count_yhat_lower",
            "yhat_upper": "title_count_yhat_upper",
        }
    )

    merged = forecast.merge(
        ordered[["ds", target, "title_count"]],
        on="ds",
        how="left",
    )
    merged = merged.merge(title_forecast, on="ds", how="left")
    merged["theme"] = str(ordered["theme"].iloc[0])
    merged["is_future"] = merged["ds"] > last_observed_ds
    merged["forecast_years"] = forecast_years

    future_only = merged.loc[merged["is_future"]].copy()
    future_only["seasonal_naive"] = seasonal_naive_future(
        theme_frame=ordered,
        future_dates=future_only["ds"],
        target=target,
    )
    future_only["quarter"] = future_only["ds"].dt.to_period("Q").astype(str)
    merged["seasonal_naive"] = np.nan
    merged.loc[future_only.index, "seasonal_naive"] = future_only["seasonal_naive"].to_numpy()

    summary = {
        "theme": str(ordered["theme"].iloc[0]),
        "target": target,
        "last_observed_ds": last_observed_ds.strftime("%Y-%m-%d"),
        "last_observed_quarter": last_observed_ds.to_period("Q").strftime("%YQ%q"),
        "forecast_start_ds": future_only["ds"].min().strftime("%Y-%m-%d"),
        "forecast_end_ds": future_only["ds"].max().strftime("%Y-%m-%d"),
        "forecast_periods": int(len(future_only)),
        "last_actual_value": float(ordered[target].iloc[-1]),
        "forecast_mean_future": float(future_only["yhat"].mean()),
        "forecast_final_value": float(future_only["yhat"].iloc[-1]),
        "baseline_final_value": float(future_only["seasonal_naive"].iloc[-1]),
        "last_actual_title_count": float(ordered["title_count"].iloc[-1]),
        "forecast_mean_future_title_count": float(future_only["title_count_yhat"].mean()),
        "forecast_final_title_count": float(future_only["title_count_yhat"].iloc[-1]),
        "forecast_peak_value": float(future_only["yhat"].max()),
        "forecast_low_value": float(future_only["yhat"].min()),
        "forecast_delta_from_last_actual": float(future_only["yhat"].iloc[-1] - ordered[target].iloc[-1]),
        "forecast_delta_vs_baseline_final": float(
            future_only["yhat"].iloc[-1] - future_only["seasonal_naive"].iloc[-1]
        ),
    }
    return merged, summary


def plot_future_forecast(
    theme_frame: pd.DataFrame,
    merged_forecast: pd.DataFrame,
    target: str,
    output_path: Path,
    font,
    forecast_stretch: float,
    forecast_years: int,
) -> None:
    history = theme_frame.sort_values("ds").copy()
    future_only = merged_forecast.loc[merged_forecast["is_future"]].copy()
    combined_dates = pd.concat([history["ds"], future_only["ds"]]).drop_duplicates().sort_values()

    display_positions, split_position = build_display_axis(
        history_dates=combined_dates,
        test_dates=future_only["ds"],
        forecast_stretch=forecast_stretch,
    )

    history["x"] = history["ds"].map(display_positions)
    forecast_plot = merged_forecast.copy()
    forecast_plot["x"] = forecast_plot["ds"].map(display_positions)
    future_plot = forecast_plot.loc[forecast_plot["is_future"]].copy()

    tick_positions, tick_labels = build_xticks(
        history_dates=combined_dates,
        test_dates=future_only["ds"],
        positions=display_positions,
    )

    forecast_end = float(forecast_plot["x"].max()) if not forecast_plot.empty else split_position
    historical_bar_width = 0.72
    future_bar_width = max(float(forecast_stretch), 1.0) * 0.72

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    ax1.axvspan(split_position, forecast_end, color="#f5efe2", alpha=0.45, zorder=0)
    ax1.plot(history["x"], history[target], color="#1f77b4", linewidth=2, label="Actual")
    ax1.plot(
        forecast_plot["x"],
        forecast_plot["yhat"],
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label="Prophet fit / forecast",
    )
    ax1.plot(
        future_plot["x"],
        future_plot["seasonal_naive"],
        color="#2ca02c",
        linestyle=":",
        linewidth=1.8,
        label="Seasonal naive baseline",
    )
    ax1.fill_between(
        forecast_plot["x"],
        forecast_plot["yhat_lower"],
        forecast_plot["yhat_upper"],
        color="#ff7f0e",
        alpha=0.16,
        label="Prophet interval",
    )
    ax1.axvline(split_position, color="#666666", linestyle="--", linewidth=1)
    ax1.set_ylabel(target)
    ax1.legend(prop=font)
    ax1.grid(alpha=0.25)

    ax2.axvspan(split_position, forecast_end, color="#f5efe2", alpha=0.45, zorder=0)
    ax2.fill_between(
        future_plot["x"],
        future_plot["title_count_yhat_lower"],
        future_plot["title_count_yhat_upper"],
        color="#ffd79b",
        alpha=0.28,
        label="Title forecast interval",
        zorder=1,
    )
    ax2.bar(
        history["x"],
        history["title_count"],
        width=historical_bar_width,
        color="#7f7f7f",
        alpha=0.88,
        label="Historical titles",
        zorder=2,
    )
    ax2.bar(
        future_plot["x"],
        future_plot["title_count_yhat"],
        width=future_bar_width,
        color="#ffcf8a",
        alpha=0.65,
        label="Forecast titles",
        zorder=3,
    )
    ax2.plot(
        future_plot["x"],
        future_plot["title_count_yhat"],
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.6,
        zorder=4,
    )
    ax2.plot(
        future_plot["x"],
        future_plot["title_count_yhat_lower"],
        color="#c26a00",
        linestyle=":",
        linewidth=1.0,
        zorder=4,
    )
    ax2.plot(
        future_plot["x"],
        future_plot["title_count_yhat_upper"],
        color="#c26a00",
        linestyle=":",
        linewidth=1.0,
        zorder=4,
    )
    ax2.set_ylabel("Titles")
    ax2.set_xlabel("Quarter")
    ax2.grid(alpha=0.25)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontproperties=font)
    ax2.legend(prop=font, loc="upper left")

    title = f"{history['theme'].iloc[0]}: {target} forecast (+{forecast_years} years)"
    ax1.set_title(title, fontproperties=font)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def write_interactive_future_plot(
    theme_frame: pd.DataFrame,
    merged_forecast: pd.DataFrame,
    target: str,
    output_path: Path,
    forecast_years: int,
) -> None:
    history = theme_frame.sort_values("ds").copy()
    forecast_plot = merged_forecast.sort_values("ds").copy()
    future_plot = forecast_plot.loc[forecast_plot["is_future"]].copy()
    split_start = pd.Timestamp(future_plot["ds"].min())

    history["quarter_label"] = history["ds"].dt.to_period("Q").astype(str)
    forecast_plot["quarter_label"] = forecast_plot["ds"].dt.to_period("Q").astype(str)
    future_plot["quarter_label"] = future_plot["ds"].dt.to_period("Q").astype(str)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"{history['theme'].iloc[0]}: {target} forecast (+{forecast_years} years)",
            "",
        ),
    )

    fig.add_vrect(
        x0=split_start,
        x1=forecast_plot["ds"].max(),
        fillcolor="rgba(245, 239, 226, 0.5)",
        line_width=0,
        row=1,
        col=1,
    )
    fig.add_vrect(
        x0=split_start,
        x1=forecast_plot["ds"].max(),
        fillcolor="rgba(245, 239, 226, 0.5)",
        line_width=0,
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=history["ds"],
            y=history[target],
            mode="lines",
            name="Actual",
            line=dict(color="#1f77b4", width=2),
            customdata=history["quarter_label"],
            hovertemplate="Quarter=%{customdata}<br>Actual=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_plot["ds"],
            y=forecast_plot["yhat_upper"],
            mode="lines",
            line=dict(color="rgba(255,127,14,0)"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_plot["ds"],
            y=forecast_plot["yhat_lower"],
            mode="lines",
            line=dict(color="rgba(255,127,14,0)"),
            fill="tonexty",
            fillcolor="rgba(255,127,14,0.18)",
            name="Prophet interval",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_plot["ds"],
            y=forecast_plot["yhat"],
            mode="lines",
            name="Prophet fit / forecast",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            customdata=forecast_plot["quarter_label"],
            hovertemplate="Quarter=%{customdata}<br>Prophet=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=future_plot["ds"],
            y=future_plot["seasonal_naive"],
            mode="lines",
            name="Seasonal naive baseline",
            line=dict(color="#2ca02c", width=1.8, dash="dot"),
            customdata=future_plot["quarter_label"],
            hovertemplate="Quarter=%{customdata}<br>Naive=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=future_plot["ds"],
            y=future_plot["title_count_yhat_upper"],
            mode="lines",
            line=dict(color="rgba(194,106,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=future_plot["ds"],
            y=future_plot["title_count_yhat_lower"],
            mode="lines",
            line=dict(color="rgba(194,106,0,0)"),
            fill="tonexty",
            fillcolor="rgba(255,215,155,0.28)",
            name="Title forecast interval",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=history["ds"],
            y=history["title_count"],
            name="Historical titles",
            marker_color="rgba(127,127,127,0.88)",
            customdata=history["quarter_label"],
            hovertemplate="Quarter=%{customdata}<br>Historical titles=%{y:.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=future_plot["ds"],
            y=future_plot["title_count_yhat"],
            name="Forecast titles",
            marker_color="rgba(255,207,138,0.72)",
            customdata=future_plot["quarter_label"],
            hovertemplate="Quarter=%{customdata}<br>Forecast titles=%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=future_plot["ds"],
            y=future_plot["title_count_yhat"],
            mode="lines",
            name="Title forecast line",
            line=dict(color="#ff7f0e", width=1.6, dash="dash"),
            customdata=future_plot["quarter_label"],
            hovertemplate="Quarter=%{customdata}<br>Title forecast=%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=future_plot["ds"],
            y=future_plot["title_count_yhat_lower"],
            mode="lines",
            name="Title lower bound",
            line=dict(color="#c26a00", width=1, dash="dot"),
            customdata=future_plot["quarter_label"],
            hovertemplate="Quarter=%{customdata}<br>Lower=%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=future_plot["ds"],
            y=future_plot["title_count_yhat_upper"],
            mode="lines",
            name="Title upper bound",
            line=dict(color="#c26a00", width=1, dash="dot"),
            customdata=future_plot["quarter_label"],
            hovertemplate="Quarter=%{customdata}<br>Upper=%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_vline(x=split_start, line_dash="dash", line_color="#666666", line_width=1)
    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        height=760,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=90, b=60),
        barmode="overlay",
    )
    fig.update_yaxes(title_text=target, row=1, col=1)
    fig.update_yaxes(title_text="Titles", row=2, col=1)
    fig.update_xaxes(title_text="Quarter", row=2, col=1)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def forecast_dataset(
    input_path: Path,
    output_dir: Path,
    target: str,
    forecast_years: int,
    top_n: int,
    selected_themes: list[str] | None,
    forecast_stretch: float,
    skip_plots: bool,
    skip_html_plots: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(input_path, encoding="utf-8-sig", parse_dates=["ds"])
    required_columns = {"theme", "ds", target, "title_count"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if selected_themes:
        themes = selected_themes
    else:
        theme_order = (
            dataset.groupby("theme")
            .agg(total_votes=("total_votes", "sum"), observed_quarters=("ds", "size"))
            .sort_values(["total_votes", "observed_quarters"], ascending=[False, False])
            .index.tolist()
        )
        themes = theme_order[:top_n] if top_n > 0 else theme_order

    forecast_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str]] = []
    font = resolve_font()

    for theme in themes:
        theme_frame = dataset.loc[dataset["theme"] == theme].copy()
        if theme_frame.empty:
            continue

        merged_forecast, summary = forecast_theme(
            theme_frame=theme_frame,
            target=target,
            forecast_years=forecast_years,
        )
        forecast_rows.append(merged_forecast)
        summary_rows.append(summary)

        if not skip_plots:
            plot_future_forecast(
                theme_frame=theme_frame,
                merged_forecast=merged_forecast,
                target=target,
                output_path=plots_dir / f"{safe_name(theme)}_{target}_future.png",
                font=font,
                forecast_stretch=forecast_stretch,
                forecast_years=forecast_years,
            )
        if not skip_html_plots:
            write_interactive_future_plot(
                theme_frame=theme_frame,
                merged_forecast=merged_forecast,
                target=target,
                output_path=html_dir / f"{safe_name(theme)}_{target}_future.html",
                forecast_years=forecast_years,
            )

    if not forecast_rows:
        raise RuntimeError("No themes produced forecast output.")

    full_forecast_df = pd.concat(forecast_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["forecast_final_value", "forecast_mean_future"],
        ascending=[False, False],
    ).reset_index(drop=True)

    forecast_path = output_dir / f"future_forecasts_{target}.csv"
    summary_path = output_dir / f"future_summary_{target}.csv"
    full_forecast_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"forecasted {len(summary_df):,} themes for target {target}")
    print(f"saved future forecast rows to {forecast_path}")
    print(f"saved future summary to {summary_path}")
    if not skip_plots:
        print(f"saved future plots to {plots_dir}")
    if not skip_html_plots:
        print(f"saved interactive html plots to {html_dir}")

    return {
        "forecasts": forecast_path,
        "summary": summary_path,
        "plots_dir": plots_dir,
        "html_dir": html_dir,
    }


def main() -> None:
    args = parse_args()
    forecast_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        target=args.target,
        forecast_years=args.forecast_years,
        top_n=args.top_n,
        selected_themes=args.themes,
        forecast_stretch=args.forecast_stretch,
        skip_plots=args.skip_plots,
        skip_html_plots=args.skip_html_plots,
    )


if __name__ == "__main__":
    main()
