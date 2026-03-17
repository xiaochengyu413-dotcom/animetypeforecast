from __future__ import annotations

import argparse
import math
import re
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from prophet import Prophet


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "generated" / "theme_quarterly_model_ready.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Prophet forecasts against a seasonal naive baseline."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", default="popularity_index")
    parser.add_argument("--test-periods", type=int, default=4)
    parser.add_argument("--min-train-points", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--theme", action="append", dest="themes")
    parser.add_argument("--forecast-stretch", type=float, default=3.0)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denominator = y_true.abs().clip(lower=1e-6)
    return float(((y_true - y_pred).abs() / denominator).mean() * 100.0)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def build_prophet_model() -> Prophet:
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.8,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0,
    )
    model.add_seasonality(name="yearly_cycle", period=365.25, fourier_order=3)
    return model


def seasonal_naive_predict(train_frame: pd.DataFrame, test_ds: pd.Series, target: str) -> np.ndarray:
    history = train_frame.set_index("ds")[target].sort_index()
    predictions: list[float] = []
    for ds in test_ds:
        lag_ds = ds - pd.DateOffset(months=12)
        if lag_ds in history.index:
            predictions.append(float(history.loc[lag_ds]))
        else:
            predictions.append(float(history.iloc[-1]))
    return np.asarray(predictions, dtype=float)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", value, flags=re.UNICODE).strip("_")
    return cleaned or "theme"


@lru_cache(maxsize=1)
def resolve_font() -> font_manager.FontProperties | None:
    candidate_paths = [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            return font_manager.FontProperties(fname=str(candidate))
    return None


def build_display_axis(
    history_dates: pd.Series,
    test_dates: pd.Series,
    forecast_stretch: float,
) -> tuple[dict[pd.Timestamp, float], float]:
    ordered_dates = pd.Series(history_dates).drop_duplicates().sort_values().tolist()
    if not ordered_dates:
        return {}, 0.0

    split_start = pd.Timestamp(test_dates.min())
    step_train = 1.0
    step_test = max(float(forecast_stretch), 1.0)

    positions: dict[pd.Timestamp, float] = {pd.Timestamp(ordered_dates[0]): 0.0}
    current_position = 0.0

    for previous_date, current_date in zip(ordered_dates[:-1], ordered_dates[1:]):
        current_timestamp = pd.Timestamp(current_date)
        increment = step_test if current_timestamp >= split_start else step_train
        current_position += increment
        positions[current_timestamp] = current_position

    prior_dates = [date for date in ordered_dates if pd.Timestamp(date) < split_start]
    if not prior_dates:
        split_position = positions[pd.Timestamp(ordered_dates[0])]
    else:
        last_train = pd.Timestamp(prior_dates[-1])
        first_test = pd.Timestamp(split_start)
        split_position = (positions[last_train] + positions[first_test]) / 2.0

    return positions, split_position


def build_xticks(
    history_dates: pd.Series,
    test_dates: pd.Series,
    positions: dict[pd.Timestamp, float],
) -> tuple[list[float], list[str]]:
    ordered_dates = pd.Series(history_dates).drop_duplicates().sort_values().tolist()
    if not ordered_dates:
        return [], []

    yearly_dates = [pd.Timestamp(date) for date in ordered_dates if pd.Timestamp(date).quarter == 1]
    if not yearly_dates:
        yearly_dates = [pd.Timestamp(ordered_dates[0])]

    max_training_labels = 8
    stride = max(1, math.ceil(len(yearly_dates) / max_training_labels))
    tick_dates = yearly_dates[::stride]

    for test_date in pd.Series(test_dates).drop_duplicates().sort_values().tolist():
        timestamp = pd.Timestamp(test_date)
        if timestamp not in tick_dates:
            tick_dates.append(timestamp)

    final_date = pd.Timestamp(ordered_dates[-1])
    if final_date not in tick_dates:
        tick_dates.append(final_date)

    tick_dates = sorted(set(tick_dates))
    tick_positions = [positions[timestamp] for timestamp in tick_dates if timestamp in positions]

    tick_labels: list[str] = []
    test_set = {pd.Timestamp(date) for date in pd.Series(test_dates).drop_duplicates().tolist()}
    for timestamp in tick_dates:
        if timestamp in test_set:
            tick_labels.append(timestamp.to_period("Q").strftime("%YQ%q"))
        else:
            tick_labels.append(str(timestamp.year))

    return tick_positions, tick_labels


def evaluate_theme(
    theme_frame: pd.DataFrame,
    target: str,
    test_periods: int,
    min_train_points: int,
) -> tuple[dict[str, float | str], pd.DataFrame] | None:
    ordered = theme_frame.sort_values("ds").reset_index(drop=True).copy()
    if len(ordered) < min_train_points + test_periods:
        return None

    train = ordered.iloc[:-test_periods].copy()
    test = ordered.iloc[-test_periods:].copy()
    if len(train) < min_train_points:
        return None

    prophet_train = train[["ds", target]].rename(columns={target: "y"})
    model = build_prophet_model()
    model.fit(prophet_train)

    prediction_frame = pd.DataFrame({"ds": pd.concat([train["ds"], test["ds"]]).drop_duplicates().sort_values()})
    forecast = model.predict(prediction_frame)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    prophet_test = test[["ds", target]].merge(forecast, on="ds", how="left")
    prophet_test["seasonal_naive"] = seasonal_naive_predict(train, test["ds"], target)

    actual = prophet_test[target]
    prophet_pred = prophet_test["yhat"]
    naive_pred = prophet_test["seasonal_naive"]

    metrics = {
        "theme": str(ordered["theme"].iloc[0]),
        "target": target,
        "train_points": int(len(train)),
        "test_points": int(len(test)),
        "prophet_mae": float((actual - prophet_pred).abs().mean()),
        "prophet_rmse": rmse(actual, prophet_pred),
        "prophet_mape": mape(actual, prophet_pred),
        "naive_mae": float((actual - naive_pred).abs().mean()),
        "naive_rmse": rmse(actual, naive_pred),
        "naive_mape": mape(actual, naive_pred),
        "train_end": train["ds"].max().strftime("%Y-%m-%d"),
        "test_start": test["ds"].min().strftime("%Y-%m-%d"),
        "test_end": test["ds"].max().strftime("%Y-%m-%d"),
    }

    prophet_test["theme"] = metrics["theme"]
    prophet_test["split"] = "test"
    return metrics, prophet_test


def plot_theme_result(
    theme_frame: pd.DataFrame,
    result_frame: pd.DataFrame,
    target: str,
    output_path: Path,
    font: font_manager.FontProperties | None,
    forecast_stretch: float,
) -> None:
    history = theme_frame.sort_values("ds").copy()
    display_positions, split_position = build_display_axis(
        history_dates=history["ds"],
        test_dates=result_frame["ds"],
        forecast_stretch=forecast_stretch,
    )
    history["x"] = history["ds"].map(display_positions)
    result_plot = result_frame.copy()
    result_plot["x"] = result_plot["ds"].map(display_positions)
    tick_positions, tick_labels = build_xticks(
        history_dates=history["ds"],
        test_dates=result_plot["ds"],
        positions=display_positions,
    )

    bar_widths = np.where(
        history["ds"] >= result_plot["ds"].min(),
        max(float(forecast_stretch), 1.0) * 0.72,
        0.72,
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    ax1.plot(history["x"], history[target], color="#1f77b4", linewidth=2, label="Actual")
    forecast_end = float(history["x"].max()) if not history["x"].empty else split_position
    ax1.axvspan(split_position, forecast_end, color="#f5efe2", alpha=0.45, zorder=0)
    ax1.plot(result_plot["x"], result_plot["yhat"], color="#ff7f0e", linestyle="--", linewidth=2, label="Prophet")
    ax1.plot(
        result_plot["x"],
        result_plot["seasonal_naive"],
        color="#2ca02c",
        linestyle=":",
        linewidth=2,
        label="Seasonal naive",
    )
    ax1.fill_between(
        result_plot["x"],
        result_plot["yhat_lower"],
        result_plot["yhat_upper"],
        color="#ff7f0e",
        alpha=0.18,
        label="Prophet interval",
    )
    ax1.axvline(split_position, color="#666666", linestyle="--", linewidth=1)
    ax1.set_ylabel(target)
    ax1.legend(prop=font)
    ax1.grid(alpha=0.25)

    ax2.axvspan(split_position, forecast_end, color="#f5efe2", alpha=0.45, zorder=0)
    ax2.bar(history["x"], history["title_count"], width=bar_widths, color="#7f7f7f", alpha=0.85)
    ax2.set_ylabel("Titles")
    ax2.set_xlabel("Quarter")
    ax2.grid(alpha=0.25)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontproperties=font)

    title = f"{history['theme'].iloc[0]}: {target} validation"
    ax1.set_title(title, fontproperties=font)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_dataset(
    input_path: Path,
    output_dir: Path,
    target: str,
    test_periods: int,
    min_train_points: int,
    top_n: int,
    selected_themes: list[str] | None,
    forecast_stretch: float,
    skip_plots: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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
        if top_n > 0:
            themes = theme_order[:top_n]
        else:
            themes = theme_order

    metrics_rows: list[dict[str, float | str]] = []
    forecast_frames: list[pd.DataFrame] = []
    font = resolve_font()

    for theme in themes:
        theme_frame = dataset.loc[dataset["theme"] == theme].copy()
        if theme_frame.empty:
            continue

        result = evaluate_theme(
            theme_frame=theme_frame,
            target=target,
            test_periods=test_periods,
            min_train_points=min_train_points,
        )
        if result is None:
            continue

        metrics, forecast_frame = result
        metrics_rows.append(metrics)
        forecast_frames.append(forecast_frame)

        if not skip_plots:
            plot_theme_result(
                theme_frame=theme_frame,
                result_frame=forecast_frame,
                target=target,
                output_path=plots_dir / f"{safe_name(theme)}_{target}.png",
                font=font,
                forecast_stretch=forecast_stretch,
            )

    if not metrics_rows:
        raise RuntimeError("No themes produced a valid evaluation split.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("prophet_rmse").reset_index(drop=True)
    forecast_df = pd.concat(forecast_frames, ignore_index=True)
    metrics_path = output_dir / f"metrics_{target}.csv"
    forecasts_path = output_dir / f"forecast_comparison_{target}.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    forecast_df.to_csv(forecasts_path, index=False, encoding="utf-8-sig")

    better_count = int((metrics_df["prophet_rmse"] < metrics_df["naive_rmse"]).sum())
    print(f"evaluated {len(metrics_df):,} themes for target {target}")
    print(f"Prophet beat seasonal naive on RMSE for {better_count:,} themes")
    print(f"saved metrics to {metrics_path}")
    print(f"saved forecast comparison rows to {forecasts_path}")
    if not skip_plots:
        print(f"saved plots to {plots_dir}")

    return {
        "metrics": metrics_path,
        "forecasts": forecasts_path,
        "plots_dir": plots_dir,
    }


def main() -> None:
    args = parse_args()
    evaluate_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        target=args.target,
        test_periods=args.test_periods,
        min_train_points=args.min_train_points,
        top_n=args.top_n,
        selected_themes=args.themes,
        forecast_stretch=args.forecast_stretch,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
