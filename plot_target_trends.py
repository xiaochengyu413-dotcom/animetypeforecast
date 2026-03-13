from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from evaluate_forecasts import build_display_axis, build_xticks, resolve_font, safe_name


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "generated" / "future_forecast"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated" / "target_comparison"
RATING_TARGET = "avg_weighted_rating"
POPULARITY_TARGET = "popularity_index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dual-target future trend plots with avg_weighted_rating on the "
            "upper panel and popularity_index on the lower panel."
        )
    )
    parser.add_argument(
        "--rating-forecasts",
        type=Path,
        default=DEFAULT_INPUT_DIR / "future_forecasts_avg_weighted_rating.csv",
    )
    parser.add_argument(
        "--popularity-forecasts",
        type=Path,
        default=DEFAULT_INPUT_DIR / "future_forecasts_popularity_index.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--theme", action="append", dest="themes")
    parser.add_argument("--forecast-stretch", type=float, default=3.0)
    parser.add_argument("--skip-html-plots", action="store_true")
    return parser.parse_args()


def load_forecasts(path: Path, target: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["ds"])
    required = {
        "theme",
        "ds",
        "is_future",
        "yhat",
        "yhat_lower",
        "yhat_upper",
        "seasonal_naive",
        target,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["is_future"] = frame["is_future"].astype(str).str.lower().eq("true")
    return frame.sort_values(["theme", "ds"]).reset_index(drop=True)


def infer_forecast_years(frame: pd.DataFrame) -> int:
    if "forecast_years" in frame.columns and frame["forecast_years"].notna().any():
        return int(float(frame["forecast_years"].dropna().iloc[0]))
    future_points = int(frame["is_future"].sum())
    return max(future_points // 4, 1)


def resolve_themes(
    rating_frame: pd.DataFrame,
    popularity_frame: pd.DataFrame,
    selected_themes: list[str] | None,
    top_n: int,
) -> list[str]:
    available = sorted(set(rating_frame["theme"]) & set(popularity_frame["theme"]))
    if selected_themes:
        return [theme for theme in selected_themes if theme in available]

    ranking = (
        popularity_frame.loc[popularity_frame["is_future"]]
        .groupby("theme", as_index=False)
        .agg(forecast_final=("yhat", "last"))
        .sort_values("forecast_final", ascending=False)["theme"]
        .tolist()
    )
    ranking = [theme for theme in ranking if theme in available]
    return ranking[:top_n] if top_n > 0 else ranking


def align_theme_frames(
    rating_frame: pd.DataFrame,
    popularity_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_dates = sorted(set(rating_frame["ds"]) & set(popularity_frame["ds"]))
    if not common_dates:
        raise ValueError("No overlapping forecast dates between rating and popularity frames.")

    rating_aligned = rating_frame.loc[rating_frame["ds"].isin(common_dates)].copy()
    popularity_aligned = popularity_frame.loc[popularity_frame["ds"].isin(common_dates)].copy()
    return (
        rating_aligned.sort_values("ds").reset_index(drop=True),
        popularity_aligned.sort_values("ds").reset_index(drop=True),
    )


def plot_dual_target_png(
    rating_frame: pd.DataFrame,
    popularity_frame: pd.DataFrame,
    output_path: Path,
    forecast_stretch: float,
) -> None:
    font = resolve_font()
    history_dates = rating_frame["ds"]
    future_dates = rating_frame.loc[rating_frame["is_future"], "ds"]
    display_positions, split_position = build_display_axis(
        history_dates=history_dates,
        test_dates=future_dates,
        forecast_stretch=forecast_stretch,
    )
    tick_positions, tick_labels = build_xticks(
        history_dates=history_dates,
        test_dates=future_dates,
        positions=display_positions,
    )
    forecast_end = max(display_positions.values())

    rating_plot = rating_frame.copy()
    popularity_plot = popularity_frame.copy()
    rating_plot["x"] = rating_plot["ds"].map(display_positions)
    popularity_plot["x"] = popularity_plot["ds"].map(display_positions)

    rating_history = rating_plot.loc[~rating_plot["is_future"]].copy()
    popularity_history = popularity_plot.loc[~popularity_plot["is_future"]].copy()
    rating_future = rating_plot.loc[rating_plot["is_future"]].copy()
    popularity_future = popularity_plot.loc[popularity_plot["is_future"]].copy()

    forecast_years = infer_forecast_years(rating_plot)
    theme = str(rating_plot["theme"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    for axis in (ax1, ax2):
        axis.axvspan(split_position, forecast_end, color="#f5efe2", alpha=0.45, zorder=0)
        axis.axvline(split_position, color="#666666", linestyle="--", linewidth=1)
        axis.grid(alpha=0.25)

    ax1.plot(rating_history["x"], rating_history[RATING_TARGET], color="#1f77b4", linewidth=2, label="Actual")
    ax1.plot(
        rating_plot["x"],
        rating_plot["yhat"],
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label="Prophet fit / forecast",
    )
    ax1.plot(
        rating_future["x"],
        rating_future["seasonal_naive"],
        color="#2ca02c",
        linestyle=":",
        linewidth=1.8,
        label="Seasonal naive baseline",
    )
    ax1.fill_between(
        rating_plot["x"],
        rating_plot["yhat_lower"],
        rating_plot["yhat_upper"],
        color="#ff7f0e",
        alpha=0.16,
        label="Prophet interval",
    )
    ax1.set_ylabel("Avg weighted rating")
    ax1.legend(prop=font, loc="upper left")

    ax2.plot(
        popularity_history["x"],
        popularity_history[POPULARITY_TARGET],
        color="#1f77b4",
        linewidth=2,
        label="Actual",
    )
    ax2.plot(
        popularity_plot["x"],
        popularity_plot["yhat"],
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label="Prophet fit / forecast",
    )
    ax2.plot(
        popularity_future["x"],
        popularity_future["seasonal_naive"],
        color="#2ca02c",
        linestyle=":",
        linewidth=1.8,
        label="Seasonal naive baseline",
    )
    ax2.fill_between(
        popularity_plot["x"],
        popularity_plot["yhat_lower"],
        popularity_plot["yhat_upper"],
        color="#ff7f0e",
        alpha=0.16,
        label="Prophet interval",
    )
    ax2.set_ylabel("Popularity index")
    ax2.set_xlabel("Quarter")
    ax2.legend(prop=font, loc="upper left")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontproperties=font)

    ax1.set_title(
        f"{theme}: rating vs popularity trend (+{forecast_years} years)",
        fontproperties=font,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def add_target_traces(
    fig,
    row: int,
    frame: pd.DataFrame,
    target: str,
    axis_label: str,
) -> None:
    history = frame.loc[~frame["is_future"]].copy()
    future_only = frame.loc[frame["is_future"]].copy()
    history["quarter_label"] = history["ds"].dt.to_period("Q").astype(str)
    frame = frame.copy()
    frame["quarter_label"] = frame["ds"].dt.to_period("Q").astype(str)
    future_only["quarter_label"] = future_only["ds"].dt.to_period("Q").astype(str)

    fig.add_trace(
        go.Scatter(
            x=history["ds"],
            y=history[target],
            mode="lines",
            name="Actual",
            line=dict(color="#1f77b4", width=2),
            customdata=history["quarter_label"],
            hovertemplate=f"Quarter=%{{customdata}}<br>{axis_label}=%{{y:.2f}}<extra></extra>",
            showlegend=row == 1,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["ds"],
            y=frame["yhat_upper"],
            mode="lines",
            line=dict(color="rgba(255,127,14,0)"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["ds"],
            y=frame["yhat_lower"],
            mode="lines",
            line=dict(color="rgba(255,127,14,0)"),
            fill="tonexty",
            fillcolor="rgba(255,127,14,0.18)",
            name="Prophet interval",
            hoverinfo="skip",
            showlegend=row == 1,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["ds"],
            y=frame["yhat"],
            mode="lines",
            name="Prophet fit / forecast",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            customdata=frame["quarter_label"],
            hovertemplate="Quarter=%{customdata}<br>Prophet=%{y:.2f}<extra></extra>",
            showlegend=row == 1,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=future_only["ds"],
            y=future_only["seasonal_naive"],
            mode="lines",
            name="Seasonal naive baseline",
            line=dict(color="#2ca02c", width=1.8, dash="dot"),
            customdata=future_only["quarter_label"],
            hovertemplate=f"Quarter=%{{customdata}}<br>Naive=%{{y:.2f}}<extra></extra>",
            showlegend=row == 1,
        ),
        row=row,
        col=1,
    )


def plot_dual_target_html(
    rating_frame: pd.DataFrame,
    popularity_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    theme = str(rating_frame["theme"].iloc[0])
    forecast_years = infer_forecast_years(rating_frame)
    future_only = rating_frame.loc[rating_frame["is_future"]].copy()
    split_start = pd.Timestamp(future_only["ds"].min())
    forecast_end = rating_frame["ds"].max()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
        subplot_titles=(
            f"{theme}: rating vs popularity trend (+{forecast_years} years)",
            "",
        ),
    )

    for row in (1, 2):
        fig.add_vrect(
            x0=split_start,
            x1=forecast_end,
            fillcolor="rgba(245, 239, 226, 0.5)",
            line_width=0,
            row=row,
            col=1,
        )

    add_target_traces(fig, row=1, frame=rating_frame, target=RATING_TARGET, axis_label="Rating")
    add_target_traces(
        fig,
        row=2,
        frame=popularity_frame,
        target=POPULARITY_TARGET,
        axis_label="Popularity index",
    )

    fig.add_vline(x=split_start, line_dash="dash", line_color="#666666", line_width=1)
    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        height=760,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=90, b=60),
    )
    fig.update_yaxes(title_text="Avg weighted rating", row=1, col=1)
    fig.update_yaxes(title_text="Popularity index", row=2, col=1)
    fig.update_xaxes(title_text="Quarter", row=2, col=1)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def generate_trend_plots(
    rating_forecasts_path: Path,
    popularity_forecasts_path: Path,
    output_dir: Path,
    selected_themes: list[str] | None,
    top_n: int,
    forecast_stretch: float,
    skip_html_plots: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "trend_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    html_dir = output_dir / "trend_html"
    if not skip_html_plots:
        html_dir.mkdir(parents=True, exist_ok=True)

    rating = load_forecasts(rating_forecasts_path, RATING_TARGET)
    popularity = load_forecasts(popularity_forecasts_path, POPULARITY_TARGET)
    themes = resolve_themes(
        rating_frame=rating,
        popularity_frame=popularity,
        selected_themes=selected_themes,
        top_n=top_n,
    )
    if not themes:
        raise RuntimeError("No overlapping themes found for target trend plotting.")

    generated = 0
    for theme in themes:
        rating_theme = rating.loc[rating["theme"] == theme].copy()
        popularity_theme = popularity.loc[popularity["theme"] == theme].copy()
        if rating_theme.empty or popularity_theme.empty:
            continue

        rating_theme, popularity_theme = align_theme_frames(rating_theme, popularity_theme)
        png_path = plots_dir / f"{safe_name(theme)}_rating_vs_popularity_future.png"
        plot_dual_target_png(
            rating_frame=rating_theme,
            popularity_frame=popularity_theme,
            output_path=png_path,
            forecast_stretch=forecast_stretch,
        )

        if not skip_html_plots:
            html_path = html_dir / f"{safe_name(theme)}_rating_vs_popularity_future.html"
            plot_dual_target_html(
                rating_frame=rating_theme,
                popularity_frame=popularity_theme,
                output_path=html_path,
            )
        generated += 1

    print(f"saved {generated:,} dual-target trend plots to {plots_dir}")
    if not skip_html_plots:
        print(f"saved dual-target interactive plots to {html_dir}")

    result = {"plots_dir": plots_dir}
    if not skip_html_plots:
        result["html_dir"] = html_dir
    return result


def main() -> None:
    args = parse_args()
    generate_trend_plots(
        rating_forecasts_path=args.rating_forecasts,
        popularity_forecasts_path=args.popularity_forecasts,
        output_dir=args.output_dir,
        selected_themes=args.themes,
        top_n=args.top_n,
        forecast_stretch=args.forecast_stretch,
        skip_html_plots=args.skip_html_plots,
    )


if __name__ == "__main__":
    main()
