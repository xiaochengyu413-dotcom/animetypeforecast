from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from forecast_future import forecast_theme
from popularity_weights import (
    DEFAULT_POPULARITY_WEIGHTS,
    compute_popularity_index,
    weight_record,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "generated" / "theme_quarterly_model_ready.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated" / "sensitivity_analysis"

SCENARIO_WEIGHTS: dict[str, dict[str, float]] = {
    "baseline": DEFAULT_POPULARITY_WEIGHTS,
    "rating_heavy": {
        "rating_component": 0.60,
        "votes_component": 0.20,
        "favorites_component": 0.10,
        "titles_component": 0.10,
    },
    "votes_heavy": {
        "rating_component": 0.30,
        "votes_component": 0.45,
        "favorites_component": 0.15,
        "titles_component": 0.10,
    },
    "engagement_heavy": {
        "rating_component": 0.30,
        "votes_component": 0.35,
        "favorites_component": 0.25,
        "titles_component": 0.10,
    },
    "balanced": {
        "rating_component": 0.25,
        "votes_component": 0.25,
        "favorites_component": 0.25,
        "titles_component": 0.25,
    },
    "supply_heavy": {
        "rating_component": 0.30,
        "votes_component": 0.25,
        "favorites_component": 0.10,
        "titles_component": 0.35,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sensitivity analysis over multiple popularity-index weighting schemes."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--forecast-years", type=int, default=2)
    parser.add_argument("--skip-forecast", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def rank_frame(frame: pd.DataFrame, value_column: str, scenario: str, rank_column: str) -> pd.DataFrame:
    ordered = frame.sort_values(value_column, ascending=False).reset_index(drop=True).copy()
    ordered[rank_column] = ordered[value_column].rank(method="dense", ascending=False).astype(int)
    ordered["scenario"] = scenario
    return ordered


def top_k_overlap(base_frame: pd.DataFrame, compare_frame: pd.DataFrame, theme_column: str, top_k: int) -> int:
    base_top = set(base_frame.sort_values("rank").head(top_k)[theme_column])
    compare_top = set(compare_frame.sort_values("rank").head(top_k)[theme_column])
    return len(base_top & compare_top)


def spearman_corr(left: pd.Series, right: pd.Series) -> float:
    left_rank = pd.Series(left).rank(method="average")
    right_rank = pd.Series(right).rank(method="average")
    return float(left_rank.corr(right_rank, method="pearson"))


def scenario_dataset(base_dataset: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    frame = base_dataset.copy()
    frame["popularity_index"] = compute_popularity_index(frame, weights)
    return frame


def plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.loc[summary_df["scenario"] != "baseline"].copy()
    if plot_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    x = list(range(len(plot_df)))
    width = 0.36

    axes[0].bar(
        [value - width / 2 for value in x],
        plot_df["history_pearson_corr"],
        width=width,
        color="#4e79a7",
        alpha=0.85,
        label="All-quarter Pearson",
    )
    axes[0].bar(
        [value + width / 2 for value in x],
        plot_df["history_rank_spearman"],
        width=width,
        color="#f28e2b",
        alpha=0.75,
        label="Last-quarter Spearman",
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Correlation")
    axes[0].set_title("Historical Stability vs Baseline")
    axes[0].legend(loc="upper right")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["scenario"], rotation=25)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(alpha=0.25, axis="y")

    if "future_rank_spearman" in plot_df.columns:
        axes[1].bar(
            [value - width / 2 for value in x],
            plot_df["future_rank_spearman"],
            width=width,
            color="#59a14f",
            alpha=0.85,
            label="Future rank Spearman",
        )
        axes[1].bar(
            [value + width / 2 for value in x],
            plot_df["future_topk_overlap_ratio"],
            width=width,
            color="#e15759",
            alpha=0.75,
            label="Future top-k overlap ratio",
        )
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("Agreement")
        axes[1].set_title("Future Forecast Stability vs Baseline")
        axes[1].legend(loc="upper right")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(plot_df["scenario"], rotation=25)
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].grid(alpha=0.25, axis="y")

    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def run_sensitivity_analysis(
    input_path: Path,
    output_dir: Path,
    top_k: int,
    forecast_years: int,
    skip_forecast: bool,
    skip_plots: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(input_path, encoding="utf-8-sig", parse_dates=["ds"])
    required_columns = {
        "theme",
        "ds",
        "rating_component",
        "votes_component",
        "favorites_component",
        "titles_component",
        "title_count",
    }
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    scenario_defs = pd.DataFrame.from_records(
        [weight_record(name, weights) for name, weights in SCENARIO_WEIGHTS.items()]
    )

    baseline_dataset = scenario_dataset(dataset, SCENARIO_WEIGHTS["baseline"])
    last_ds = baseline_dataset["ds"].max()
    baseline_last = rank_frame(
        baseline_dataset.loc[baseline_dataset["ds"] == last_ds, ["theme", "ds", "popularity_index"]].copy(),
        value_column="popularity_index",
        scenario="baseline",
        rank_column="rank",
    )

    historical_rank_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str]] = []
    future_summaries: list[pd.DataFrame] = []
    future_rank_frames: list[pd.DataFrame] = []
    baseline_future_rank: pd.DataFrame | None = None
    baseline_future_summary: pd.DataFrame | None = None

    for scenario, weights in SCENARIO_WEIGHTS.items():
        scenario_frame = scenario_dataset(dataset, weights)
        history_corr = float(
            baseline_dataset["popularity_index"].corr(scenario_frame["popularity_index"], method="pearson")
        )
        history_mae = float((baseline_dataset["popularity_index"] - scenario_frame["popularity_index"]).abs().mean())

        scenario_last = rank_frame(
            scenario_frame.loc[scenario_frame["ds"] == last_ds, ["theme", "ds", "popularity_index"]].copy(),
            value_column="popularity_index",
            scenario=scenario,
            rank_column="rank",
        )
        historical_rank_frames.append(scenario_last if scenario != "baseline" else baseline_last)

        common_last = baseline_last.merge(
            scenario_last[["theme", "rank", "popularity_index"]],
            on="theme",
            suffixes=("_baseline", "_scenario"),
        )
        history_rank_spearman = spearman_corr(common_last["rank_baseline"], common_last["rank_scenario"])
        history_topk_overlap = top_k_overlap(
            baseline_last,
            scenario_last,
            theme_column="theme",
            top_k=top_k,
        )

        summary_row: dict[str, float | str] = {
            "scenario": scenario,
            "history_pearson_corr": history_corr,
            "history_mae": history_mae,
            "history_rank_spearman": history_rank_spearman,
            "history_topk_overlap": history_topk_overlap,
            "history_topk_overlap_ratio": history_topk_overlap / max(top_k, 1),
            **weight_record(scenario, weights),
        }

        if not skip_forecast:
            scenario_future_rows: list[dict[str, float | str]] = []
            for theme, theme_frame in scenario_frame.groupby("theme"):
                merged_forecast, forecast_summary = forecast_theme(
                    theme_frame=theme_frame.copy(),
                    target="popularity_index",
                    forecast_years=forecast_years,
                )
                scenario_future_rows.append(forecast_summary)

            future_summary_df = pd.DataFrame(scenario_future_rows)
            future_summary_df["scenario"] = scenario
            future_summary_df["forecast_rank"] = future_summary_df["forecast_final_value"].rank(
                method="dense",
                ascending=False,
            ).astype(int)
            future_summaries.append(future_summary_df)

            scenario_future_rank = future_summary_df[
                ["theme", "forecast_final_value", "forecast_rank", "scenario"]
            ].rename(columns={"forecast_rank": "rank"})
            future_rank_frames.append(scenario_future_rank)

            if scenario == "baseline":
                baseline_future_summary = future_summary_df.copy()
                baseline_future_rank = scenario_future_rank.copy()

            if baseline_future_summary is not None and baseline_future_rank is not None:
                common_future = baseline_future_rank.merge(
                    scenario_future_rank[["theme", "rank", "forecast_final_value"]],
                    on="theme",
                    suffixes=("_baseline", "_scenario"),
                )
                summary_row["future_final_pearson_corr"] = float(
                    common_future["forecast_final_value_baseline"].corr(
                        common_future["forecast_final_value_scenario"],
                        method="pearson",
                    )
                )
                summary_row["future_rank_spearman"] = float(
                    spearman_corr(common_future["rank_baseline"], common_future["rank_scenario"])
                )
                future_topk_overlap = top_k_overlap(
                    baseline_future_rank,
                    scenario_future_rank,
                    theme_column="theme",
                    top_k=top_k,
                )
                summary_row["future_topk_overlap"] = future_topk_overlap
                summary_row["future_topk_overlap_ratio"] = future_topk_overlap / max(top_k, 1)
                summary_row["future_mean_abs_diff"] = float(
                    (
                        common_future["forecast_final_value_baseline"]
                        - common_future["forecast_final_value_scenario"]
                    )
                    .abs()
                    .mean()
                )

        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("scenario").reset_index(drop=True)
    historical_ranks_df = pd.concat(historical_rank_frames, ignore_index=True).sort_values(
        ["scenario", "rank", "theme"]
    )

    scenario_defs_path = output_dir / "scenario_definitions.csv"
    summary_path = output_dir / "scenario_metrics.csv"
    historical_ranks_path = output_dir / "historical_last_quarter_ranks.csv"
    scenario_defs.to_csv(scenario_defs_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    historical_ranks_df.to_csv(historical_ranks_path, index=False, encoding="utf-8-sig")

    outputs: dict[str, Path] = {
        "scenario_definitions": scenario_defs_path,
        "summary": summary_path,
        "historical_ranks": historical_ranks_path,
    }

    if not skip_forecast and future_summaries and future_rank_frames:
        future_summary_path = output_dir / "future_forecast_summary_by_scenario.csv"
        future_ranks_path = output_dir / "future_final_ranks_by_scenario.csv"
        pd.concat(future_summaries, ignore_index=True).sort_values(
            ["scenario", "forecast_rank", "theme"]
        ).to_csv(future_summary_path, index=False, encoding="utf-8-sig")
        pd.concat(future_rank_frames, ignore_index=True).sort_values(
            ["scenario", "rank", "theme"]
        ).to_csv(future_ranks_path, index=False, encoding="utf-8-sig")
        outputs["future_summary"] = future_summary_path
        outputs["future_ranks"] = future_ranks_path

    if not skip_plots:
        summary_plot_path = output_dir / "sensitivity_summary.png"
        plot_summary(summary_df, summary_plot_path)
        outputs["summary_plot"] = summary_plot_path

    print(f"ran {len(SCENARIO_WEIGHTS):,} weighting scenarios")
    print(f"saved scenario definitions to {scenario_defs_path}")
    print(f"saved scenario metrics to {summary_path}")
    print(f"saved historical rank comparison to {historical_ranks_path}")
    if "future_summary" in outputs:
        print(f"saved future forecast comparison to {outputs['future_summary']}")
        print(f"saved future rank comparison to {outputs['future_ranks']}")
    if "summary_plot" in outputs:
        print(f"saved summary plot to {outputs['summary_plot']}")

    return outputs


def main() -> None:
    args = parse_args()
    run_sensitivity_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        top_k=args.top_k,
        forecast_years=args.forecast_years,
        skip_forecast=args.skip_forecast,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
