from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from evaluate_forecasts import resolve_font


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "generated" / "future_forecast"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated" / "target_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare future forecasts under popularity_index and avg_weighted_rating."
    )
    parser.add_argument(
        "--popularity-summary",
        type=Path,
        default=DEFAULT_INPUT_DIR / "future_summary_popularity_index.csv",
    )
    parser.add_argument(
        "--rating-summary",
        type=Path,
        default=DEFAULT_INPUT_DIR / "future_summary_avg_weighted_rating.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rank-gap-threshold", type=int, default=3)
    return parser.parse_args()


def classify_gap(pop_rank: int, rating_rank: int, threshold: int) -> str:
    if rating_rank + threshold <= pop_rank:
        return "rating_stronger_than_heat"
    if pop_rank + threshold <= rating_rank:
        return "heat_stronger_than_rating"
    return "roughly_aligned"


def write_summary_markdown(comparison: pd.DataFrame, output_path: Path, threshold: int) -> None:
    pop_top5 = comparison.sort_values("popularity_rank").head(5)
    rating_top5 = comparison.sort_values("rating_rank").head(5)
    rating_stronger = comparison.loc[
        comparison["comparison_label"] == "rating_stronger_than_heat"
    ].sort_values("rank_gap", ascending=False)
    heat_stronger = comparison.loc[
        comparison["comparison_label"] == "heat_stronger_than_rating"
    ].sort_values("rank_gap")

    lines: list[str] = []
    lines.append("# Forecast Target Comparison")
    lines.append("")
    lines.append(
        "This file compares the two-year future forecast ranking under "
        "`popularity_index` and `avg_weighted_rating`."
    )
    lines.append("")
    lines.append(f"Rank-gap threshold for labeling: {threshold}")
    lines.append("")
    lines.append("## Popularity Top 5")
    for _, row in pop_top5.iterrows():
        lines.append(
            f"- {row['theme']}: popularity rank {row['popularity_rank']}, "
            f"rating rank {row['rating_rank']}"
        )
    lines.append("")
    lines.append("## Rating Top 5")
    for _, row in rating_top5.iterrows():
        lines.append(
            f"- {row['theme']}: rating rank {row['rating_rank']}, "
            f"popularity rank {row['popularity_rank']}"
        )
    lines.append("")
    lines.append("## Rating Stronger Than Heat")
    if rating_stronger.empty:
        lines.append("- None")
    else:
        for _, row in rating_stronger.head(5).iterrows():
            lines.append(
                f"- {row['theme']}: rating rank {row['rating_rank']}, "
                f"popularity rank {row['popularity_rank']}, gap {row['rank_gap']}"
            )
    lines.append("")
    lines.append("## Heat Stronger Than Rating")
    if heat_stronger.empty:
        lines.append("- None")
    else:
        for _, row in heat_stronger.head(5).iterrows():
            lines.append(
                f"- {row['theme']}: popularity rank {row['popularity_rank']}, "
                f"rating rank {row['rating_rank']}, gap {row['rank_gap']}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_rank_gap(comparison: pd.DataFrame, output_path: Path) -> None:
    ordered = comparison.sort_values("rank_gap").reset_index(drop=True)
    font = resolve_font()

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    colors = [
        "#4e79a7" if gap < 0 else "#e15759" if gap > 0 else "#9d9da1"
        for gap in ordered["rank_gap"]
    ]
    ax.barh(ordered["theme"], ordered["rank_gap"], color=colors, alpha=0.88)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_xlabel("Popularity Rank - Rating Rank")
    ax.set_title("Future Forecast Rank Gap Between Heat and Rating", fontproperties=font)
    ax.grid(axis="x", alpha=0.25)
    if font is not None:
        for label in ax.get_yticklabels():
            label.set_fontproperties(font)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def compare_targets(
    popularity_summary_path: Path,
    rating_summary_path: Path,
    output_dir: Path,
    rank_gap_threshold: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    popularity = pd.read_csv(popularity_summary_path, encoding="utf-8-sig")
    rating = pd.read_csv(rating_summary_path, encoding="utf-8-sig")

    pop = popularity[["theme", "forecast_final_value", "forecast_delta_from_last_actual", "forecast_final_title_count"]].copy()
    pop = pop.rename(
        columns={
            "forecast_final_value": "popularity_forecast_final",
            "forecast_delta_from_last_actual": "popularity_delta_from_last_actual",
            "forecast_final_title_count": "popularity_forecast_final_title_count",
        }
    )
    pop["popularity_rank"] = pop["popularity_forecast_final"].rank(method="dense", ascending=False).astype(int)

    rate = rating[["theme", "forecast_final_value", "forecast_delta_from_last_actual", "forecast_final_title_count"]].copy()
    rate = rate.rename(
        columns={
            "forecast_final_value": "rating_forecast_final",
            "forecast_delta_from_last_actual": "rating_delta_from_last_actual",
            "forecast_final_title_count": "rating_forecast_final_title_count",
        }
    )
    rate["rating_rank"] = rate["rating_forecast_final"].rank(method="dense", ascending=False).astype(int)

    comparison = pop.merge(rate, on="theme", how="inner")
    comparison["popularity_rank_pct"] = comparison["popularity_rank"].rank(method="dense", pct=True)
    comparison["rating_rank_pct"] = comparison["rating_rank"].rank(method="dense", pct=True)
    comparison["rank_gap"] = comparison["popularity_rank"] - comparison["rating_rank"]
    comparison["comparison_label"] = comparison.apply(
        lambda row: classify_gap(
            int(row["popularity_rank"]),
            int(row["rating_rank"]),
            rank_gap_threshold,
        ),
        axis=1,
    )
    comparison = comparison.sort_values(["popularity_rank", "rating_rank"]).reset_index(drop=True)

    comparison_path = output_dir / "forecast_target_comparison.csv"
    summary_path = output_dir / "forecast_target_comparison.md"
    plot_path = output_dir / "forecast_target_rank_gap.png"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    write_summary_markdown(comparison, summary_path, rank_gap_threshold)
    plot_rank_gap(comparison, plot_path)

    print(f"saved target comparison table to {comparison_path}")
    print(f"saved target comparison summary to {summary_path}")
    print(f"saved target rank-gap plot to {plot_path}")

    return {
        "comparison": comparison_path,
        "summary": summary_path,
        "plot": plot_path,
    }


def main() -> None:
    args = parse_args()
    compare_targets(
        popularity_summary_path=args.popularity_summary,
        rating_summary_path=args.rating_summary,
        output_dir=args.output_dir,
        rank_gap_threshold=args.rank_gap_threshold,
    )


if __name__ == "__main__":
    main()
