from __future__ import annotations

import argparse
from pathlib import Path

from build_theme_dataset import DEFAULT_OUTPUT_DIR, build_dataset
from build_web_dashboard import DEFAULT_SITE_DIR, build_dashboard_data, write_data_js
from compare_backtest_models import (
    LSTMTrainingConfig,
    compare_models_for_target,
    load_ready_theme_dataset,
)
from compare_forecast_targets import compare_targets
from download_bangumi_archive import DEFAULT_DATA_DIR, sync_latest_archive
from evaluate_forecasts import evaluate_dataset
from forecast_future import DEFAULT_OUTPUT_DIR as DEFAULT_FUTURE_DIR
from forecast_future import forecast_dataset
from plot_target_trends import generate_trend_plots
from sensitivity_analysis import run_sensitivity_analysis


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = DEFAULT_OUTPUT_DIR / "evaluation"
DEFAULT_RECENT_EVAL_DIR = DEFAULT_OUTPUT_DIR / "evaluation_2024_2025"
DEFAULT_MODEL_BACKTEST_DIR = DEFAULT_OUTPUT_DIR / "model_backtest_2024_2025"
DEFAULT_COMPARISON_DIR = DEFAULT_OUTPUT_DIR / "target_comparison"
DEFAULT_SENSITIVITY_DIR = DEFAULT_OUTPUT_DIR / "sensitivity_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the full dashboard website from the latest Bangumi Archive dump."
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--archive-data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluation-output-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--recent-evaluation-output-dir", type=Path, default=DEFAULT_RECENT_EVAL_DIR)
    parser.add_argument("--model-backtest-output-dir", type=Path, default=DEFAULT_MODEL_BACKTEST_DIR)
    parser.add_argument("--future-output-dir", type=Path, default=DEFAULT_FUTURE_DIR)
    parser.add_argument("--comparison-output-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--sensitivity-output-dir", type=Path, default=DEFAULT_SENSITIVITY_DIR)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    parser.add_argument("--vote-quantile", type=float, default=0.6)
    parser.add_argument("--min-titles-per-quarter", type=int, default=3)
    parser.add_argument("--min-quarters", type=int, default=12)
    parser.add_argument("--min-coverage", type=float, default=0.45)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--test-periods", type=int, default=4)
    parser.add_argument("--min-train-points", type=int, default=12)
    parser.add_argument("--forecast-years", type=int, default=2)
    parser.add_argument("--forecast-stretch", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-html-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.archive_data_dir / "subject.jsonlines"
    if not args.skip_download:
        sync_result = sync_latest_archive(
            data_dir=args.archive_data_dir,
            force_download=args.force_download,
        )
        input_path = sync_result["subject"]

    build_result = build_dataset(
        input_path=input_path,
        output_dir=args.dataset_output_dir,
        as_of_date=None,
        keep_incomplete_quarter=False,
        vote_quantile=args.vote_quantile,
        min_titles_per_quarter=args.min_titles_per_quarter,
        min_quarters=args.min_quarters,
        min_coverage=args.min_coverage,
        progress_every=args.progress_every,
    )

    evaluate_dataset(
        input_path=build_result["quarterly_ready"],
        output_dir=args.evaluation_output_dir,
        target="popularity_index",
        test_periods=args.test_periods,
        min_train_points=args.min_train_points,
        top_n=0,
        selected_themes=None,
        forecast_stretch=args.forecast_stretch,
        test_start_quarter=None,
        test_end_quarter=None,
        skip_plots=False,
    )

    evaluate_dataset(
        input_path=build_result["quarterly_ready"],
        output_dir=args.recent_evaluation_output_dir,
        target="avg_weighted_rating",
        test_periods=8,
        min_train_points=args.min_train_points,
        top_n=0,
        selected_themes=None,
        forecast_stretch=1.0,
        test_start_quarter="2024Q1",
        test_end_quarter="2025Q4",
        skip_plots=False,
    )

    model_backtest_dataset = load_ready_theme_dataset(
        input_path=build_result["quarterly_ready"],
        readiness_path=build_result["readiness"],
    )
    compare_models_for_target(
        dataset=model_backtest_dataset,
        output_dir=args.model_backtest_output_dir,
        prophet_evaluation_dir=args.recent_evaluation_output_dir,
        target="avg_weighted_rating",
        test_start_quarter="2024Q1",
        test_end_quarter="2025Q4",
        min_train_points=max(args.min_train_points, 16),
        lstm_config=LSTMTrainingConfig(
            lookback=8,
            hidden_size=12,
            learning_rate=0.02,
            epochs=80,
            seed=42,
            backend="auto",
        ),
        skip_plots=False,
    )

    popularity_forecast = forecast_dataset(
        input_path=build_result["quarterly_ready"],
        output_dir=args.future_output_dir,
        target="popularity_index",
        forecast_years=args.forecast_years,
        top_n=0,
        selected_themes=None,
        forecast_stretch=args.forecast_stretch,
        skip_plots=False,
        skip_html_plots=not args.include_html_plots,
    )
    rating_forecast = forecast_dataset(
        input_path=build_result["quarterly_ready"],
        output_dir=args.future_output_dir,
        target="avg_weighted_rating",
        forecast_years=args.forecast_years,
        top_n=0,
        selected_themes=None,
        forecast_stretch=args.forecast_stretch,
        skip_plots=False,
        skip_html_plots=not args.include_html_plots,
    )

    compare_targets(
        popularity_summary_path=popularity_forecast["summary"],
        rating_summary_path=rating_forecast["summary"],
        output_dir=args.comparison_output_dir,
        rank_gap_threshold=3,
    )
    generate_trend_plots(
        rating_forecasts_path=rating_forecast["forecasts"],
        popularity_forecasts_path=popularity_forecast["forecasts"],
        output_dir=args.comparison_output_dir,
        selected_themes=None,
        top_n=0,
        forecast_stretch=args.forecast_stretch,
        skip_html_plots=not args.include_html_plots,
    )
    run_sensitivity_analysis(
        input_path=build_result["quarterly_ready"],
        output_dir=args.sensitivity_output_dir,
        top_k=args.top_k,
        forecast_years=args.forecast_years,
        skip_forecast=False,
        skip_plots=False,
    )

    dashboard_data = build_dashboard_data(site_dir=args.site_dir)
    write_data_js(site_dir=args.site_dir, data=dashboard_data)
    print(f"refreshed site data in {args.site_dir}")


if __name__ == "__main__":
    main()
