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
    return {
        "theme": row["theme"],
        "firstQuarter": quarter_from_date(row["first_quarter"]),
        "lastQuarter": quarter_from_date(row["last_quarter"]),
        "observedQuarters": to_int(row["observed_quarters"]),
        "usableQuarters": to_int(row["usable_quarters"]),
        "totalTitles": to_int(row["total_titles"]),
        "totalVotes": to_int(row["total_votes"]),
        "meanTitlesPerQuarter": to_float(row["mean_titles_per_quarter"]),
        "medianTitlesPerQuarter": to_float(row["median_titles_per_quarter"]),
        "spanQuarters": to_int(row["span_quarters"]),
        "coverageRatio": to_float(row["coverage_ratio"]),
        "usableCoverageRatio": to_float(row["usable_coverage_ratio"]),
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
    theme_cards: list[dict[str, object]] = []
    for theme in theme_order:
        popularity = popularity_by_theme[theme]
        rating = rating_by_theme[theme]
        comparison = comparison_by_theme[theme]
        readiness = readiness_by_theme.get(theme, {})
        evaluation = evaluation_by_theme.get(theme, {})
        future_popularity_plot = SCRIPT_DIR / "generated" / "future_forecast" / "plots" / f"{theme}_popularity_index_future.png"
        future_rating_plot = SCRIPT_DIR / "generated" / "future_forecast" / "plots" / f"{theme}_avg_weighted_rating_future.png"
        comparison_plot = SCRIPT_DIR / "generated" / "target_comparison" / "trend_plots" / f"{theme}_rating_vs_popularity_future.png"
        evaluation_plot = SCRIPT_DIR / "generated" / "evaluation" / "plots" / f"{theme}_popularity_index.png"
        theme_cards.append(
            {
                "theme": theme,
                "popularity": popularity,
                "rating": rating,
                "comparison": comparison,
                "readiness": readiness,
                "evaluation": evaluation,
                "scenarioRanks": scenario_by_theme.get(theme, []),
                "assets": {
                    "futurePopularityPlot": stage_asset(site_dir, future_popularity_plot, "future", future_popularity_plot.name),
                    "futureRatingPlot": stage_asset(site_dir, future_rating_plot, "future", future_rating_plot.name),
                    "comparisonTrendPlot": stage_asset(site_dir, comparison_plot, "comparison", comparison_plot.name),
                    "evaluationPlot": stage_asset(site_dir, evaluation_plot, "evaluation", evaluation_plot.name),
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
    first_quarter = min(str(row["firstQuarter"]) for row in all_ready)
    last_quarter = max(str(row["lastQuarter"]) for row in all_ready)
    forecast_end_quarter = max(str(row["forecastEndQuarter"]) for row in popularity_ranked)
    forecast_horizon = max(int(row["forecastPeriods"]) for row in popularity_ranked)
    archive_meta = read_json(ARCHIVE_METADATA) if ARCHIVE_METADATA.exists() else {}
    archive_name = str(archive_meta.get("name") or "")
    archive_created_at = str(archive_meta.get("created_at") or "")
    archive_published_at = str(archive_meta.get("published_at") or "")

    data = {
        "meta": {
            "projectName": "Bangumi 动漫主题趋势预测网页",
            "subtitle": "把季度主题数据、双目标预测、验证结果和敏感性分析集中到一个静态仪表盘里。",
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
