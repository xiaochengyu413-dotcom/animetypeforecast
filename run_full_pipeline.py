from __future__ import annotations

import argparse
from pathlib import Path

from build_theme_dataset import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    build_dataset,
)
from download_bangumi_archive import DEFAULT_DATA_DIR, sync_latest_archive
from evaluate_forecasts import evaluate_dataset
from forecast_future import DEFAULT_OUTPUT_DIR as DEFAULT_FUTURE_DIR
from forecast_future import forecast_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = DEFAULT_OUTPUT_DIR / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full improved anime theme forecasting pipeline."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--download-latest", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--archive-data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluation-output-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--future-output-dir", type=Path, default=DEFAULT_FUTURE_DIR)
    parser.add_argument("--as-of-date", type=str, default=None)
    parser.add_argument("--keep-incomplete-quarter", action="store_true")
    parser.add_argument("--vote-quantile", type=float, default=0.6)
    parser.add_argument("--min-titles-per-quarter", type=int, default=3)
    parser.add_argument("--min-quarters", type=int, default=12)
    parser.add_argument("--min-coverage", type=float, default=0.45)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--target", default="popularity_index")
    parser.add_argument("--test-periods", type=int, default=4)
    parser.add_argument("--min-train-points", type=int, default=12)
    parser.add_argument("--forecast-years", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--theme", action="append", dest="themes")
    parser.add_argument("--forecast-stretch", type=float, default=3.0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-future-forecast", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-html-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input

    if args.download_latest:
        sync_result = sync_latest_archive(
            data_dir=args.archive_data_dir,
            force_download=args.force_download,
        )
        input_path = sync_result["subject"]

    build_result = build_dataset(
        input_path=input_path,
        output_dir=args.dataset_output_dir,
        as_of_date=args.as_of_date,
        keep_incomplete_quarter=args.keep_incomplete_quarter,
        vote_quantile=args.vote_quantile,
        min_titles_per_quarter=args.min_titles_per_quarter,
        min_quarters=args.min_quarters,
        min_coverage=args.min_coverage,
        progress_every=args.progress_every,
    )

    if not args.skip_validation:
        evaluate_dataset(
            input_path=build_result["quarterly_ready"],
            output_dir=args.evaluation_output_dir,
            target=args.target,
            test_periods=args.test_periods,
            min_train_points=args.min_train_points,
            top_n=args.top_n,
            selected_themes=args.themes,
            forecast_stretch=args.forecast_stretch,
            skip_plots=args.skip_plots,
            skip_html_plots=args.skip_html_plots,
        )

    if not args.skip_future_forecast:
        forecast_dataset(
            input_path=build_result["quarterly_ready"],
            output_dir=args.future_output_dir,
            target=args.target,
            forecast_years=args.forecast_years,
            top_n=args.top_n,
            selected_themes=args.themes,
            forecast_stretch=args.forecast_stretch,
            skip_plots=args.skip_plots,
        )


if __name__ == "__main__":
    main()
