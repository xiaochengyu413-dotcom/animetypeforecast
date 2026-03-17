from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from statistics import mean


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SITE_DIR = SCRIPT_DIR / "web_dashboard"

POPULARITY_SUMMARY = SCRIPT_DIR / "generated" / "future_forecast" / "future_summary_popularity_index.csv"
RATING_SUMMARY = SCRIPT_DIR / "generated" / "future_forecast" / "future_summary_avg_weighted_rating.csv"
COMPARISON_SUMMARY = SCRIPT_DIR / "generated" / "target_comparison" / "forecast_target_comparison.csv"
EVALUATION_METRICS = SCRIPT_DIR / "generated" / "evaluation" / "metrics_popularity_index.csv"
EVALUATION_WINDOW_METRICS = (
    SCRIPT_DIR / "generated" / "evaluation_2024_2025" / "metrics_avg_weighted_rating.csv"
)
THEME_READINESS = SCRIPT_DIR / "generated" / "theme_readiness.csv"
SCENARIO_METRICS = SCRIPT_DIR / "generated" / "sensitivity_analysis" / "scenario_metrics.csv"
SCENARIO_FORECASTS = (
    SCRIPT_DIR / "generated" / "sensitivity_analysis" / "future_forecast_summary_by_scenario.csv"
)
SENSITIVITY_SUMMARY_PLOT = (
    SCRIPT_DIR / "generated" / "sensitivity_analysis" / "sensitivity_summary.png"
)
ARCHIVE_METADATA = SCRIPT_DIR / "data" / "bangumi_archive" / "latest_metadata.json"

COMPARISON_LABELS = {
    "rating_stronger_than_heat": "评分明显强于热度",
    "heat_stronger_than_rating": "热度明显强于评分",
    "roughly_aligned": "热度与评分大体一致",
}

SCENARIO_LABELS = {
    "baseline": "基线权重",
    "balanced": "均衡权重",
    "engagement_heavy": "互动偏重",
    "rating_heavy": "评分偏重",
    "supply_heavy": "供给偏重",
    "votes_heavy": "投票偏重",
}

SCENARIO_ORDER = [
    "baseline",
    "balanced",
    "engagement_heavy",
    "rating_heavy",
    "supply_heavy",
    "votes_heavy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static web dashboard from the generated anime theme forecast outputs."
    )
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        message = "Missing required generated files:\n" + "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(message)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def to_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(round(float(value)))


def quarter_from_date(date_str: str) -> str:
    year, month, _ = (int(part) for part in date_str.split("-"))
    quarter = ((month - 1) // 3) + 1
    return f"{year}Q{quarter}"


def stage_asset(site_dir: Path, source: Path, *relative_parts: str) -> str:
    destination = site_dir / "assets" / Path(*relative_parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, destination)
    return Path("assets", *relative_parts).as_posix()


def rank_rows(rows: list[dict[str, object]], value_key: str, rank_key: str) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row[value_key]), reverse=True)
    ranked: list[dict[str, object]] = []
    for index, row in enumerate(ordered, start=1):
        ranked.append({**row, rank_key: index})
    return ranked


def label_count(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = {key: 0 for key in COMPARISON_LABELS}
    for row in rows:
        label = str(row["comparisonLabel"])
        counts[label] = counts.get(label, 0) + 1
    return counts


def rank_position_map(
    rows: list[dict[str, object]], value_key: str, *, reverse: bool = False
) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: float(row[value_key]), reverse=reverse)
    return {str(row["theme"]): index for index, row in enumerate(ordered, start=1)}


def classify_trend_direction(delta: float | None, metric: str) -> str:
    if delta is None:
        return "趋势暂缺"

    threshold = {
        "popularity": 1.0,
        "rating": 0.05,
    }.get(metric, 0.0)

    if delta >= threshold * 2:
        return "明显上升"
    if delta >= threshold:
        return "上升"
    if delta <= -threshold * 2:
        return "明显下降"
    if delta <= -threshold:
        return "下降"
    return "基本持平"


def describe_delta(delta: float | None, rank_position: int, total: int, metric: str) -> str:
    if delta is None:
        return "趋势暂缺"

    positive_labels = {
        "popularity": ("小幅上升", "温和上升", "明显上升"),
        "rating": ("评分小幅上升", "评分温和上升", "评分明显上升"),
    }
    negative_labels = {
        "popularity": ("小幅下降", "温和下降", "明显下降"),
        "rating": ("评分小幅下降", "评分温和下降", "评分明显下降"),
    }
    stable_labels = {
        "popularity": "热度基本持平",
        "rating": "评分基本持平",
    }

    percentile = rank_position / max(total, 1)
    if delta > 0:
        mild, medium, strong = positive_labels.get(metric, positive_labels["popularity"])
        if percentile >= 0.8:
            return strong
        if percentile >= 0.55:
            return medium
        return mild
    if delta < 0:
        mild, medium, strong = negative_labels.get(metric, negative_labels["popularity"])
        if percentile <= 0.2:
            return strong
        if percentile <= 0.45:
            return medium
        return mild
    return stable_labels.get(metric, "基本持平")


def describe_supply(last_count: float | None, final_count: float | None) -> tuple[str, str]:
    if last_count is None or final_count is None:
        return "供给信息暂缺", "缺少季度供给预测，暂不单独判断供给变化。"

    delta = final_count - last_count
    ratio = delta / max(last_count, 1.0)

    if ratio >= 0.25 or delta >= 6:
        label = "供给明显扩张"
    elif ratio >= 0.1 or delta >= 2:
        label = "供给温和扩张"
    elif ratio <= -0.25 or delta <= -6:
        label = "供给明显收缩"
    elif ratio <= -0.1 or delta <= -2:
        label = "供给温和收缩"
    else:
        label = "供给基本平稳"

    detail = (
        f"未来末期预测标题数约为 {final_count:.1f}，相对最近实测季度 "
        f"{last_count:.1f} 变动 {delta:+.1f}。"
    )
    return label, detail


def describe_alignment(comparison: dict[str, object]) -> tuple[str, str]:
    rank_gap = int(comparison["rankGap"])
    gap = abs(rank_gap)
    theme = str(comparison["theme"])
    popularity_rank = int(comparison["popularityRank"])
    rating_rank = int(comparison["ratingRank"])

    if rank_gap <= -6:
        return (
            "热度显著强于评分",
            f"{theme} 的热度排名第 {popularity_rank}，比评分排名第 {rating_rank} 高出 {gap} 位，说明讨论度和受众规模明显走在口碑前面。",
        )
    if rank_gap < 0:
        return (
            "热度略强于评分",
            f"{theme} 的热度排名第 {popularity_rank}，评分排名第 {rating_rank}，热度侧领先 {gap} 位，呈现先被看见、再等口碑兑现的结构。",
        )
    if rank_gap >= 6:
        return (
            "评分显著强于热度",
            f"{theme} 的评分排名第 {rating_rank}，比热度排名第 {popularity_rank} 高出 {gap} 位，说明更接近口碑驱动而不是流量驱动。",
        )
    if rank_gap > 0:
        return (
            "评分略强于热度",
            f"{theme} 的评分排名第 {rating_rank}，热度排名第 {popularity_rank}，评分侧领先 {gap} 位，说明质量感知强于外部热度。",
        )
    return (
        "热度与评分基本一致",
        f"{theme} 的热度和评分排名都接近前后同一位置，当前没有明显的流量偏差或口碑偏差。",
    )


def describe_scenario_stability(
    scenario_rows: list[dict[str, object]],
) -> tuple[str, str, int | None, float | None]:
    if not scenario_rows:
        return "情景稳健性暂缺", "缺少不同权重场景下的结果，暂不判断稳健性。", None, None

    ranks = [int(row["forecastRank"]) for row in scenario_rows if row["forecastRank"] is not None]
    values = [
        float(row["forecastFinalValue"])
        for row in scenario_rows
        if row["forecastFinalValue"] is not None
    ]
    if not ranks or not values:
        return "情景稳健性暂缺", "情景结果不完整，暂不判断稳健性。", None, None

    rank_span = max(ranks) - min(ranks)
    value_span = max(values) - min(values)

    if rank_span <= 1:
        label = "结论较稳健"
    elif rank_span <= 3:
        label = "结论有一定波动"
    else:
        label = "结论对权重较敏感"

    detail = (
        f"在 {len(ranks)} 个权重场景里，最终热度排名落在 #{min(ranks)} 到 #{max(ranks)} 之间，"
        f"跨度 {rank_span} 名；末期热度值区间跨度约 {value_span:.2f}。"
    )
    return label, detail, rank_span, value_span


def describe_validation(
    evaluation: dict[str, object], mae_rank: int | None, total: int
) -> tuple[str, str, str]:
    prophet_mae = evaluation.get("prophetMae")
    naive_mae = evaluation.get("naiveMae")
    prophet_beats_naive = evaluation.get("prophetBeatsNaive")

    if prophet_mae is None or naive_mae is None or mae_rank is None:
        return (
            "参考度暂缺",
            "缺少历史回测指标，当前结论主要依赖预测曲线本身。",
            "谨慎参考",
        )

    if prophet_beats_naive and mae_rank <= max(1, total // 3):
        level = "较高置信度"
    elif prophet_beats_naive or mae_rank <= max(1, (total * 2) // 3):
        level = "中等置信度"
    else:
        level = "谨慎参考"

    baseline_text = (
        "Prophet 在回测中优于 seasonal naive"
        if prophet_beats_naive
        else "seasonal naive 在回测中仍不弱于 Prophet"
    )
    detail = (
        f"{baseline_text}，当前 Prophet MAE 为 {float(prophet_mae):.2f}，"
        f"在 {total} 个题材里属于第 {mae_rank} 低误差水平。"
    )
    return "模型参考度", detail, level


def build_plain_language_summary(
    *,
    theme: str,
    forecast_end_quarter: str,
    popularity_direction: str,
    rating_direction: str,
    confidence_label: str,
    rank_span: int | None,
) -> str:
    if "上升" in popularity_direction and "上升" in rating_direction:
        opening = (
            f"一句话总结：到 {forecast_end_quarter}，{theme} 大概率还是往上走的类型。"
            " 热度在升，评分也在升，更像会继续变强。"
        )
    elif "上升" in popularity_direction and "下降" in rating_direction:
        opening = (
            f"一句话总结：到 {forecast_end_quarter}，{theme} 看起来会更火，"
            "但不一定更受好评。它更像靠讨论度和受众扩张往上冲的类型。"
        )
    elif "下降" in popularity_direction and "上升" in rating_direction:
        opening = (
            f"一句话总结：到 {forecast_end_quarter}，{theme} 未必会更火，"
            "但口碑有机会稳住甚至变好。它更接近口碑撑住热度的小众类型。"
        )
    elif "下降" in popularity_direction and "下降" in rating_direction:
        opening = (
            f"一句话总结：到 {forecast_end_quarter}，{theme} 目前更像在降温。"
            " 热度和评分都没有给出转强信号，短期不属于最值得押注的方向。"
        )
    else:
        opening = (
            f"一句话总结：到 {forecast_end_quarter}，{theme} 整体变化不算激烈。"
            " 它更像维持现状、边走边看的类型。"
        )

    confidence_note = {
        "较高置信度": "这条判断相对更稳，可以把它当成优先参考。",
        "中等置信度": "这条判断可以参考，但最好结合后续季度更新继续看。",
        "谨慎参考": "这条判断要保守看，适合当作趋势提示，不适合当成绝对结论。",
    }.get(confidence_label, "这条判断更适合做方向参考。")

    stability_note = ""
    if rank_span is not None:
        if rank_span <= 1:
            stability_note = " 不同权重设定下结论变化也不大。"
        elif rank_span >= 4:
            stability_note = " 不过不同权重设定下名次波动偏大，说明这类判断还有弹性。"

    return f"{opening} {confidence_note}{stability_note}"


def build_theme_analysis(
    *,
    theme: str,
    popularity: dict[str, object],
    rating: dict[str, object],
    comparison: dict[str, object],
    readiness: dict[str, object],
    evaluation: dict[str, object],
    scenario_rows: list[dict[str, object]],
    popularity_delta_rank: int,
    rating_delta_rank: int,
    mae_rank: int | None,
    theme_count: int,
) -> dict[str, object]:
    popularity_delta = float(popularity["forecastDeltaFromLastActual"])
    rating_delta = float(rating["forecastDeltaFromLastActual"])
    popularity_direction = classify_trend_direction(popularity_delta, "popularity")
    rating_direction = classify_trend_direction(rating_delta, "rating")

    popularity_trend = describe_delta(
        popularity_delta, popularity_delta_rank, theme_count, "popularity"
    )
    rating_trend = describe_delta(rating_delta, rating_delta_rank, theme_count, "rating")
    supply_label, supply_detail = describe_supply(
        popularity.get("lastActualTitleCount"), popularity.get("forecastFinalTitleCount")
    )
    alignment_label, alignment_detail = describe_alignment(comparison)
    stability_label, stability_detail, rank_span, _ = describe_scenario_stability(scenario_rows)
    validation_label, validation_detail, confidence_label = describe_validation(
        evaluation, mae_rank, theme_count
    )

    if popularity_delta > 0 and rating_delta >= 0:
        headline = "热度与口碑同步改善"
        conclusion = (
            f"到 {popularity['forecastEndQuarter']}，{theme} 更像是景气继续抬升的强势题材，"
            "适合放在优先关注列表。"
        )
    elif popularity_delta > 0 and rating_delta < 0:
        headline = "热度先行上冲，口碑端仍待兑现"
        conclusion = (
            f"到 {popularity['forecastEndQuarter']}，{theme} 更像讨论度驱动的上行题材，"
            "适合持续跟踪，但解读时要把口碑回落风险一起考虑。"
        )
    elif popularity_delta <= 0 and rating_delta >= 0:
        headline = "市场热度转弱，但口碑韧性仍在"
        conclusion = (
            f"到 {popularity['forecastEndQuarter']}，{theme} 更接近口碑型题材，"
            "适合审慎观察是否会从高评分重新转化为更强热度。"
        )
    else:
        headline = "热度与评分同步承压"
        conclusion = (
            f"到 {popularity['forecastEndQuarter']}，{theme} 暂时不属于景气扩张型题材，"
            "更适合放在观察位而不是高优先级押注。"
        )

    if rank_span is not None and rank_span <= 1 and confidence_label == "较高置信度":
        conclusion += " 目前不同权重场景下的排序也比较稳定。"
    elif rank_span is not None and rank_span >= 4:
        conclusion += " 不过不同权重设定下名次波动较大，结论应保留弹性。"

    bullets = [
        {
            "title": "趋势判断",
            "body": (
                f"{theme} 的未来热度趋势明确为“{popularity_direction}”，评分趋势为“{rating_direction}”。"
                f" 进一步看强弱，热度属于“{popularity_trend}”，到 "
                f"{popularity['forecastEndQuarter']} 预计达到 {float(popularity['forecastFinalValue']):.2f}，"
                f"较最近实测季度变动 {popularity_delta:+.2f}；评分端则是“{rating_trend}”，"
                f"末期预测值为 {float(rating['forecastFinalValue']):.2f}，变动 {rating_delta:+.2f}。"
            ),
        },
        {
            "title": "热度与口碑结构",
            "body": alignment_detail,
        },
        {
            "title": "供给与稳健性",
            "body": f"{supply_detail} {stability_detail}",
        },
        {
            "title": validation_label,
            "body": validation_detail,
        },
    ]

    return {
        "headline": headline,
        "confidenceLabel": confidence_label,
        "plainSummary": build_plain_language_summary(
            theme=theme,
            forecast_end_quarter=str(popularity["forecastEndQuarter"]),
            popularity_direction=popularity_direction,
            rating_direction=rating_direction,
            confidence_label=confidence_label,
            rank_span=rank_span,
        ),
        "summary": (
            f"{theme} 的未来热度趋势是“{popularity_direction}”，评分趋势是“{rating_direction}”。"
            f" 当前呈现“{popularity_trend} + {rating_trend}”的组合，"
            f"整体更接近“{alignment_label}”的题材结构。"
        ),
        "conclusion": conclusion,
        "supplyLabel": supply_label,
        "stabilityLabel": stability_label,
        "trendDirections": [
            {
                "label": "热度趋势",
                "value": popularity_direction,
                "delta": popularity_delta,
            },
            {
                "label": "评分趋势",
                "value": rating_direction,
                "delta": rating_delta,
            },
        ],
        "bullets": bullets,
    }


def normalize_future_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "theme": row["theme"],
        "target": row["target"],
        "lastObservedDate": row["last_observed_ds"],
        "lastObservedQuarter": row["last_observed_quarter"],
        "forecastStartDate": row["forecast_start_ds"],
        "forecastEndDate": row["forecast_end_ds"],
        "forecastStartQuarter": quarter_from_date(row["forecast_start_ds"]),
        "forecastEndQuarter": quarter_from_date(row["forecast_end_ds"]),
        "forecastPeriods": to_int(row["forecast_periods"]),
        "lastActualValue": to_float(row["last_actual_value"]),
        "forecastMeanFuture": to_float(row["forecast_mean_future"]),
        "forecastFinalValue": to_float(row["forecast_final_value"]),
        "baselineFinalValue": to_float(row["baseline_final_value"]),
        "lastActualTitleCount": to_float(row["last_actual_title_count"]),
        "forecastMeanFutureTitleCount": to_float(row["forecast_mean_future_title_count"]),
        "forecastFinalTitleCount": to_float(row["forecast_final_title_count"]),
        "forecastPeakValue": to_float(row["forecast_peak_value"]),
        "forecastLowValue": to_float(row["forecast_low_value"]),
        "forecastDeltaFromLastActual": to_float(row["forecast_delta_from_last_actual"]),
        "forecastDeltaVsBaselineFinal": to_float(row["forecast_delta_vs_baseline_final"]),
    }


def normalize_readiness_row(row: dict[str, str]) -> dict[str, object]:
    observed_first = row.get("observed_first_quarter") or row.get("first_quarter")
    observed_last = row.get("observed_last_quarter") or row.get("last_quarter")
    model_first = row.get("model_first_quarter") or row.get("first_quarter")
    model_last = row.get("model_last_quarter") or row.get("last_quarter")

    return {
        "theme": row["theme"],
        "observedFirstQuarter": quarter_from_date(observed_first) if observed_first else None,
        "observedLastQuarter": quarter_from_date(observed_last) if observed_last else None,
        "modelFirstQuarter": quarter_from_date(model_first) if model_first else None,
        "modelLastQuarter": quarter_from_date(model_last) if model_last else None,
        "firstQuarter": quarter_from_date(model_first) if model_first else None,
        "lastQuarter": quarter_from_date(model_last) if model_last else None,
        "observedQuarters": to_int(row["observed_quarters"]),
        "usableQuarters": to_int(row["usable_quarters"]),
        "totalTitles": to_int(row["total_titles"]),
        "totalVotes": to_int(row["total_votes"]),
        "meanTitlesPerQuarter": to_float(row["mean_titles_per_quarter"]),
        "medianTitlesPerQuarter": to_float(row["median_titles_per_quarter"]),
        "observedSpanQuarters": to_int(row.get("observed_span_quarters")) or to_int(row["span_quarters"]),
        "modelSpanQuarters": to_int(row.get("model_span_quarters")) or to_int(row["span_quarters"]),
        "spanQuarters": to_int(row.get("model_span_quarters")) or to_int(row["span_quarters"]),
        "observedCoverageRatio": to_float(row.get("observed_coverage_ratio")) or to_float(row["coverage_ratio"]),
        "modelCoverageRatio": to_float(row.get("model_coverage_ratio")) or to_float(row["coverage_ratio"]),
        "coverageRatio": to_float(row.get("model_coverage_ratio")) or to_float(row["coverage_ratio"]),
        "observedUsableCoverageRatio": to_float(row.get("observed_usable_coverage_ratio")) or to_float(row["usable_coverage_ratio"]),
        "usableCoverageRatio": to_float(row.get("model_coverage_ratio")) or to_float(row["usable_coverage_ratio"]),
        "readyForForecast": str(row["ready_for_forecast"]).lower() == "true",
    }


def normalize_evaluation_row(row: dict[str, str]) -> dict[str, object]:
    prophet_mae = to_float(row["prophet_mae"])
    naive_mae = to_float(row["naive_mae"])
    return {
        "theme": row["theme"],
        "target": row["target"],
        "trainPoints": to_int(row["train_points"]),
        "testPoints": to_int(row["test_points"]),
        "prophetMae": prophet_mae,
        "prophetRmse": to_float(row["prophet_rmse"]),
        "prophetMape": to_float(row["prophet_mape"]),
        "naiveMae": naive_mae,
        "naiveRmse": to_float(row["naive_rmse"]),
        "naiveMape": to_float(row["naive_mape"]),
        "trainEnd": row["train_end"],
        "testStart": row["test_start"],
        "testEnd": row["test_end"],
        "prophetBeatsNaive": (
            prophet_mae is not None and naive_mae is not None and prophet_mae < naive_mae
        ),
    }


def normalize_scenario_metric_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "scenario": row["scenario"],
        "label": SCENARIO_LABELS.get(row["scenario"], row["scenario"]),
        "historyPearsonCorr": to_float(row["history_pearson_corr"]),
        "historyMae": to_float(row["history_mae"]),
        "historyRankSpearman": to_float(row["history_rank_spearman"]),
        "historyTopkOverlap": to_int(row["history_topk_overlap"]),
        "historyTopkOverlapRatio": to_float(row["history_topk_overlap_ratio"]),
        "futureFinalPearsonCorr": to_float(row["future_final_pearson_corr"]),
        "futureRankSpearman": to_float(row["future_rank_spearman"]),
        "futureTopkOverlap": to_int(row["future_topk_overlap"]),
        "futureTopkOverlapRatio": to_float(row["future_topk_overlap_ratio"]),
        "futureMeanAbsDiff": to_float(row["future_mean_abs_diff"]),
        "weights": {
            "rating": to_float(row["rating_component"]),
            "votes": to_float(row["votes_component"]),
            "favorites": to_float(row["favorites_component"]),
            "titles": to_float(row["titles_component"]),
        },
    }


def normalize_scenario_forecast_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "theme": row["theme"],
        "scenario": row["scenario"],
        "scenarioLabel": SCENARIO_LABELS.get(row["scenario"], row["scenario"]),
        "target": row["target"],
        "lastObservedQuarter": row["last_observed_quarter"],
        "forecastEndQuarter": quarter_from_date(row["forecast_end_ds"]),
        "forecastFinalValue": to_float(row["forecast_final_value"]),
        "forecastDeltaFromLastActual": to_float(row["forecast_delta_from_last_actual"]),
        "forecastFinalTitleCount": to_float(row["forecast_final_title_count"]),
        "forecastRank": to_int(row["forecast_rank"]),
    }


def build_dashboard_data(site_dir: Path) -> dict[str, object]:
    require_files(
        [
            POPULARITY_SUMMARY,
            RATING_SUMMARY,
            COMPARISON_SUMMARY,
            EVALUATION_METRICS,
            EVALUATION_WINDOW_METRICS,
            THEME_READINESS,
            SCENARIO_METRICS,
            SCENARIO_FORECASTS,
        ]
    )
    assets_dir = site_dir / "assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)

    raw_popularity = [normalize_future_row(row) for row in read_csv_rows(POPULARITY_SUMMARY)]
    raw_rating = [normalize_future_row(row) for row in read_csv_rows(RATING_SUMMARY)]
    raw_readiness = [normalize_readiness_row(row) for row in read_csv_rows(THEME_READINESS)]
    raw_evaluation = [normalize_evaluation_row(row) for row in read_csv_rows(EVALUATION_METRICS)]
    raw_scenarios = [normalize_scenario_metric_row(row) for row in read_csv_rows(SCENARIO_METRICS)]
    raw_scenario_forecasts = [
        normalize_scenario_forecast_row(row) for row in read_csv_rows(SCENARIO_FORECASTS)
    ]

    popularity_ranked = rank_rows(raw_popularity, "forecastFinalValue", "rank")
    rating_ranked = rank_rows(raw_rating, "forecastFinalValue", "rank")
    popularity_delta_positions = rank_position_map(
        raw_popularity, "forecastDeltaFromLastActual"
    )
    rating_delta_positions = rank_position_map(raw_rating, "forecastDeltaFromLastActual")
    mae_positions = rank_position_map(raw_evaluation, "prophetMae")
    readiness_by_theme = {row["theme"]: row for row in raw_readiness}
    evaluation_by_theme = {row["theme"]: row for row in raw_evaluation}

    comparison_rows: list[dict[str, object]] = []
    for row in read_csv_rows(COMPARISON_SUMMARY):
        comparison_rows.append(
            {
                "theme": row["theme"],
                "popularityForecastFinal": to_float(row["popularity_forecast_final"]),
                "popularityDeltaFromLastActual": to_float(row["popularity_delta_from_last_actual"]),
                "popularityForecastFinalTitleCount": to_float(
                    row["popularity_forecast_final_title_count"]
                ),
                "popularityRank": to_int(row["popularity_rank"]),
                "ratingForecastFinal": to_float(row["rating_forecast_final"]),
                "ratingDeltaFromLastActual": to_float(row["rating_delta_from_last_actual"]),
                "ratingForecastFinalTitleCount": to_float(row["rating_forecast_final_title_count"]),
                "ratingRank": to_int(row["rating_rank"]),
                "popularityRankPct": to_float(row["popularity_rank_pct"]),
                "ratingRankPct": to_float(row["rating_rank_pct"]),
                "rankGap": to_int(row["rank_gap"]),
                "comparisonLabel": row["comparison_label"],
                "comparisonLabelZh": COMPARISON_LABELS.get(row["comparison_label"], row["comparison_label"]),
            }
        )
    comparison_rows.sort(key=lambda row: int(row["popularityRank"]))
    comparison_by_theme = {row["theme"]: row for row in comparison_rows}

    scenario_by_theme: dict[str, list[dict[str, object]]] = {}
    for row in raw_scenario_forecasts:
        if row["target"] != "popularity_index":
            continue
        scenario_by_theme.setdefault(str(row["theme"]), []).append(row)
    for theme_rows in scenario_by_theme.values():
        theme_rows.sort(key=lambda row: SCENARIO_ORDER.index(str(row["scenario"])))

    popularity_by_theme = {row["theme"]: row for row in popularity_ranked}
    rating_by_theme = {row["theme"]: row for row in rating_ranked}

    theme_order = [str(row["theme"]) for row in popularity_ranked]
    theme_count = len(theme_order)
    theme_cards: list[dict[str, object]] = []
    for theme in theme_order:
        popularity = popularity_by_theme[theme]
        rating = rating_by_theme[theme]
        comparison = comparison_by_theme[theme]
        readiness = readiness_by_theme.get(theme, {})
        evaluation = evaluation_by_theme.get(theme, {})
        scenario_rows = scenario_by_theme.get(theme, [])
        future_popularity_plot = SCRIPT_DIR / "generated" / "future_forecast" / "plots" / f"{theme}_popularity_index_future.png"
        future_rating_plot = SCRIPT_DIR / "generated" / "future_forecast" / "plots" / f"{theme}_avg_weighted_rating_future.png"
        comparison_plot = SCRIPT_DIR / "generated" / "target_comparison" / "trend_plots" / f"{theme}_rating_vs_popularity_future.png"
        evaluation_plot = SCRIPT_DIR / "generated" / "evaluation" / "plots" / f"{theme}_popularity_index.png"
        evaluation_window_plot = (
            SCRIPT_DIR
            / "generated"
            / "evaluation_2024_2025"
            / "window_plots"
            / f"{theme}_avg_weighted_rating_window.png"
        )
        theme_cards.append(
            {
                "theme": theme,
                "popularity": popularity,
                "rating": rating,
                "comparison": comparison,
                "readiness": readiness,
                "evaluation": evaluation,
                "scenarioRanks": scenario_rows,
                "analysis": build_theme_analysis(
                    theme=theme,
                    popularity=popularity,
                    rating=rating,
                    comparison=comparison,
                    readiness=readiness,
                    evaluation=evaluation,
                    scenario_rows=scenario_rows,
                    popularity_delta_rank=popularity_delta_positions[theme],
                    rating_delta_rank=rating_delta_positions[theme],
                    mae_rank=mae_positions.get(theme),
                    theme_count=theme_count,
                ),
                "assets": {
                    "futurePopularityPlot": stage_asset(site_dir, future_popularity_plot, "future", future_popularity_plot.name),
                    "futureRatingPlot": stage_asset(site_dir, future_rating_plot, "future", future_rating_plot.name),
                    "comparisonTrendPlot": stage_asset(site_dir, comparison_plot, "comparison", comparison_plot.name),
                    "evaluationPlot": stage_asset(site_dir, evaluation_plot, "evaluation", evaluation_plot.name),
                    "evaluationWindowPlot": stage_asset(site_dir, evaluation_window_plot, "evaluation_window", evaluation_window_plot.name),
                },
            }
        )

    prophet_better_count = sum(1 for row in raw_evaluation if bool(row["prophetBeatsNaive"]))
    naive_better_count = len(raw_evaluation) - prophet_better_count
    avg_prophet_mae = mean(float(row["prophetMae"]) for row in raw_evaluation)
    avg_naive_mae = mean(float(row["naiveMae"]) for row in raw_evaluation)
    avg_prophet_mape = mean(float(row["prophetMape"]) for row in raw_evaluation)
    avg_naive_mape = mean(float(row["naiveMape"]) for row in raw_evaluation)
    best_validation_theme = min(raw_evaluation, key=lambda row: float(row["prophetMae"]))
    hardest_validation_theme = max(raw_evaluation, key=lambda row: float(row["prophetMae"]))

    non_baseline_scenarios = [row for row in raw_scenarios if row["scenario"] != "baseline"]
    stable_non_baseline = sum(
        1 for row in non_baseline_scenarios if float(row["futureTopkOverlapRatio"]) >= 1.0
    )

    popularity_up_count = sum(
        1 for row in popularity_ranked if float(row["forecastDeltaFromLastActual"]) > 0
    )
    rating_up_count = sum(1 for row in rating_ranked if float(row["forecastDeltaFromLastActual"]) > 0)

    heat_leader = popularity_ranked[0]
    rating_leader = rating_ranked[0]
    strongest_heat_bias = min(comparison_rows, key=lambda row: int(row["rankGap"]))
    strongest_rating_bias = max(comparison_rows, key=lambda row: int(row["rankGap"]))

    all_ready = [row for row in raw_readiness if bool(row["readyForForecast"])]
    first_quarter = min(str(row["modelFirstQuarter"] or row["firstQuarter"]) for row in all_ready)
    last_quarter = max(str(row["modelLastQuarter"] or row["lastQuarter"]) for row in all_ready)
    forecast_end_quarter = max(str(row["forecastEndQuarter"]) for row in popularity_ranked)
    forecast_horizon = max(int(row["forecastPeriods"]) for row in popularity_ranked)
    archive_meta = read_json(ARCHIVE_METADATA) if ARCHIVE_METADATA.exists() else {}
    archive_name = str(archive_meta.get("name") or "")
    archive_created_at = str(archive_meta.get("created_at") or "")
    archive_published_at = str(archive_meta.get("published_at") or "")

    data = {
        "meta": {
            "projectName": "Anime Type Forecast",
            "subtitle": "追踪 Bangumi 15 类动画题材的季度表现、未来走势与口碑热度分歧。",
            "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "archiveName": archive_name,
            "archiveCreatedAt": archive_created_at,
            "archivePublishedAt": archive_published_at,
            "themeCount": len(theme_cards),
            "dataFirstQuarter": first_quarter,
            "dataLastQuarter": last_quarter,
            "latestObservedQuarter": max(str(row["lastObservedQuarter"]) for row in popularity_ranked),
            "latestObservedDate": max(str(row["lastObservedDate"]) for row in popularity_ranked),
            "forecastEndQuarter": forecast_end_quarter,
            "forecastHorizonQuarters": forecast_horizon,
            "scenarioCount": len(raw_scenarios),
        },
        "overview": {
            "heatLeader": {
                "theme": heat_leader["theme"],
                "rank": heat_leader["rank"],
                "forecastFinalValue": heat_leader["forecastFinalValue"],
                "delta": heat_leader["forecastDeltaFromLastActual"],
            },
            "ratingLeader": {
                "theme": rating_leader["theme"],
                "rank": rating_leader["rank"],
                "forecastFinalValue": rating_leader["forecastFinalValue"],
                "delta": rating_leader["forecastDeltaFromLastActual"],
            },
            "strongestHeatBias": strongest_heat_bias,
            "strongestRatingBias": strongest_rating_bias,
            "popularityUpCount": popularity_up_count,
            "ratingUpCount": rating_up_count,
            "stableNonBaselineTop5Count": stable_non_baseline,
            "nonBaselineScenarioCount": len(non_baseline_scenarios),
        },
        "leaderboards": {
            "popularity": popularity_ranked,
            "rating": rating_ranked,
        },
        "comparison": {
            "rows": comparison_rows,
            "counts": label_count(comparison_rows),
        },
        "validation": {
            "summary": {
                "prophetBetterCount": prophet_better_count,
                "naiveBetterCount": naive_better_count,
                "avgProphetMae": avg_prophet_mae,
                "avgNaiveMae": avg_naive_mae,
                "avgProphetMape": avg_prophet_mape,
                "avgNaiveMape": avg_naive_mape,
                "bestTheme": {
                    "theme": best_validation_theme["theme"],
                    "prophetMae": best_validation_theme["prophetMae"],
                },
                "hardestTheme": {
                    "theme": hardest_validation_theme["theme"],
                    "prophetMae": hardest_validation_theme["prophetMae"],
                },
            },
            "rows": raw_evaluation,
        },
        "sensitivity": {
            "scenarios": sorted(
                raw_scenarios, key=lambda row: SCENARIO_ORDER.index(str(row["scenario"]))
            ),
            "summaryPlot": stage_asset(site_dir, SENSITIVITY_SUMMARY_PLOT, "sensitivity", SENSITIVITY_SUMMARY_PLOT.name),
        },
        "themes": theme_cards,
        "themeOrder": theme_order,
        "roadmap": [
            {
                "title": "重建多标签主题体系",
                "detail": "把单一主标签替换为标准化多标签映射，降低分类偏差和样本流失。",
            },
            {
                "title": "扩充热度指标定义",
                "detail": "将评分、投票、收藏、供给量等信号联合建模，而不是把评分直接等价为热度。",
            },
            {
                "title": "补充跨平台数据源",
                "detail": "在 Bangumi 之外加入更多公开平台数据，检验结论是否能代表更广泛的动画市场。",
            },
            {
                "title": "比较更多时间序列模型",
                "detail": "把 Prophet 与 seasonal naive、ARIMA/SARIMA、指数平滑等基线放到同一验证框架中。",
            },
            {
                "title": "处理稀疏季度与不平衡样本",
                "detail": "在季度聚合中引入样本量门槛、加权聚合和不确定性修正，减少稀疏主题的波动噪声。",
            },
        ],
    }
    return data


def write_data_js(site_dir: Path, data: dict[str, object]) -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    output_path = site_dir / "data.js"
    output_path.write_text(
        "window.DASHBOARD_DATA = "
        + json.dumps(data, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    args = parse_args()
    data = build_dashboard_data(site_dir=args.site_dir)
    output_path = write_data_js(site_dir=args.site_dir, data=data)
    print(f"saved dashboard data to {output_path}")


if __name__ == "__main__":
    main()
