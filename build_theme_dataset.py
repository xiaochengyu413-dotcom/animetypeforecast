from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from popularity_weights import (
    DEFAULT_POPULARITY_WEIGHTS,
    compute_popularity_index,
)
from theme_taxonomy import assign_themes, taxonomy_records


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LATEST_LOCAL_INPUT = SCRIPT_DIR / "data" / "bangumi_archive" / "subject.jsonlines"
DEFAULT_INPUT = LATEST_LOCAL_INPUT if LATEST_LOCAL_INPUT.exists() else PROJECT_ROOT / "subject.jsonlines"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a quarterly anime theme dataset with multi-label themes."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--as-of-date", type=str, default=None)
    parser.add_argument("--keep-incomplete-quarter", action="store_true")
    parser.add_argument("--vote-quantile", type=float, default=0.6)
    parser.add_argument("--min-titles-per-quarter", type=int, default=3)
    parser.add_argument("--min-quarters", type=int, default=12)
    parser.add_argument("--min-coverage", type=float, default=0.45)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_votes(score_details: Any) -> int:
    if not isinstance(score_details, dict):
        return 0
    total = 0
    for value in score_details.values():
        total += safe_int(value, default=0)
    return total


def extract_tag_names(raw_tags: Any) -> list[str]:
    if not isinstance(raw_tags, list):
        return []

    names: list[str] = []
    for item in raw_tags:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        names.append(str(name))
    return names


def iter_anime_records(input_path: Path, progress_every: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    kept = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if progress_every > 0 and line_number % progress_every == 0:
                print(f"scanned {line_number:,} lines, kept {kept:,} anime records")

            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            if row.get("type") != 2:
                continue

            score = safe_float(row.get("score"))
            if score is None or score <= 0:
                continue

            votes = extract_votes(row.get("score_details"))
            if votes <= 0:
                continue

            date_value = row.get("date")
            if not date_value:
                continue

            tags = extract_tag_names(row.get("tags"))
            themes = assign_themes(tags)
            if not themes:
                continue

            records.append(
                {
                    "subject_id": safe_int(row.get("id"), default=-1),
                    "date": str(date_value),
                    "score": score,
                    "votes": votes,
                    "favorite_count": safe_int(row.get("favorite"), default=0),
                    "themes": themes,
                }
            )
            kept += 1

    return records


def minmax_scale(series: pd.Series, fill_value: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    min_value = numeric.min()
    max_value = numeric.max()
    if pd.isna(min_value) or pd.isna(max_value) or np.isclose(min_value, max_value):
        return pd.Series(fill_value, index=series.index, dtype=float)
    return (numeric - min_value) / (max_value - min_value)


def quarter_span(first_ds: pd.Timestamp, last_ds: pd.Timestamp) -> int:
    first_period = first_ds.to_period("Q")
    last_period = last_ds.to_period("Q")
    return int(last_period.ordinal - first_period.ordinal + 1)


def infer_archive_as_of_date(input_path: Path) -> pd.Timestamp | None:
    metadata_path = input_path.with_name("latest_metadata.json")
    if not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    archive_name = str(metadata.get("name") or "")
    match = re.search(r"dump-(\d{4}-\d{2}-\d{2})", archive_name)
    if match:
        return pd.Timestamp(match.group(1))

    created_at = metadata.get("created_at")
    if created_at:
        return pd.Timestamp(created_at).normalize()
    return None


def resolve_as_of_date(input_path: Path, as_of_date: str | None) -> pd.Timestamp | None:
    if as_of_date:
        return pd.Timestamp(as_of_date).normalize()
    return infer_archive_as_of_date(input_path)


def build_dataset(
    input_path: Path,
    output_dir: Path,
    as_of_date: str | None,
    keep_incomplete_quarter: bool,
    vote_quantile: float,
    min_titles_per_quarter: int,
    min_quarters: int,
    min_coverage: float,
    progress_every: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading source data from {input_path}")
    records = iter_anime_records(input_path=input_path, progress_every=progress_every)
    if not records:
        raise RuntimeError("No anime records matched the current filtering rules.")

    titles = pd.DataFrame.from_records(records)
    titles["date"] = pd.to_datetime(titles["date"], errors="coerce")
    titles = titles.dropna(subset=["date"]).copy()
    effective_as_of_date = resolve_as_of_date(input_path=input_path, as_of_date=as_of_date)
    if effective_as_of_date is not None and not keep_incomplete_quarter:
        complete_quarter_start = effective_as_of_date.to_period("Q").start_time
        before_filter = len(titles)
        titles = titles.loc[titles["date"] < complete_quarter_start].copy()
        removed = before_filter - len(titles)
        print(
            "excluded "
            f"{removed:,} anime rows from incomplete/current-or-future quarters "
            f"using archive as-of date {effective_as_of_date.date()}"
        )

    if titles.empty:
        raise RuntimeError("No anime rows remain after applying the as-of-date quarter filter.")

    titles["quarter"] = titles["date"].dt.to_period("Q")
    titles["ds"] = titles["quarter"].dt.start_time

    global_mean_score = titles["score"].mean()
    vote_threshold = titles["votes"].quantile(vote_quantile)
    vote_threshold = max(float(vote_threshold), 1.0)

    titles["weighted_rating"] = (
        (titles["votes"] / (titles["votes"] + vote_threshold)) * titles["score"]
        + (vote_threshold / (titles["votes"] + vote_threshold)) * global_mean_score
    )

    expanded = titles.explode("themes").rename(columns={"themes": "theme"}).dropna(subset=["theme"]).copy()
    expanded["weighted_rating_times_votes"] = expanded["weighted_rating"] * expanded["votes"].clip(lower=1)

    quarterly = (
        expanded.groupby(["theme", "ds"], as_index=False)
        .agg(
            quarter=("quarter", "first"),
            title_count=("subject_id", "nunique"),
            total_votes=("votes", "sum"),
            total_favorites=("favorite_count", "sum"),
            avg_raw_score=("score", "mean"),
            avg_weighted_rating=("weighted_rating", "mean"),
            median_weighted_rating=("weighted_rating", "median"),
            weighted_rating_vote_numerator=("weighted_rating_times_votes", "sum"),
        )
        .sort_values(["theme", "ds"])
        .reset_index(drop=True)
    )

    quarterly["vote_weighted_rating"] = (
        quarterly["weighted_rating_vote_numerator"]
        / quarterly["total_votes"].replace(0, np.nan)
    )
    quarterly["avg_votes_per_title"] = quarterly["total_votes"] / quarterly["title_count"].replace(0, np.nan)
    quarterly["avg_favorites_per_title"] = quarterly["total_favorites"] / quarterly["title_count"].replace(0, np.nan)
    quarterly["quarter"] = quarterly["quarter"].astype(str)

    quarterly["rating_component"] = minmax_scale(quarterly["avg_weighted_rating"])
    quarterly["votes_component"] = minmax_scale(np.log1p(quarterly["total_votes"]))
    quarterly["favorites_component"] = minmax_scale(np.log1p(quarterly["total_favorites"]))
    quarterly["titles_component"] = minmax_scale(np.log1p(quarterly["title_count"]))
    quarterly["popularity_index"] = compute_popularity_index(
        quarterly,
        DEFAULT_POPULARITY_WEIGHTS,
    )
    quarterly["sample_reliability"] = 100.0 * (
        0.70 * quarterly["votes_component"] + 0.30 * quarterly["titles_component"]
    )
    quarterly["eligible_for_modeling"] = quarterly["title_count"] >= min_titles_per_quarter

    readiness = (
        quarterly.groupby("theme", as_index=False)
        .agg(
            first_quarter=("ds", "min"),
            last_quarter=("ds", "max"),
            observed_quarters=("ds", "size"),
            usable_quarters=("eligible_for_modeling", "sum"),
            total_titles=("title_count", "sum"),
            total_votes=("total_votes", "sum"),
            total_favorites=("total_favorites", "sum"),
            mean_titles_per_quarter=("title_count", "mean"),
            median_titles_per_quarter=("title_count", "median"),
        )
        .sort_values(["usable_quarters", "total_votes"], ascending=[False, False])
        .reset_index(drop=True)
    )

    readiness["span_quarters"] = readiness.apply(
        lambda row: quarter_span(row["first_quarter"], row["last_quarter"]),
        axis=1,
    )
    readiness["coverage_ratio"] = readiness["observed_quarters"] / readiness["span_quarters"].replace(0, np.nan)
    readiness["usable_coverage_ratio"] = readiness["usable_quarters"] / readiness["span_quarters"].replace(0, np.nan)
    readiness["ready_for_forecast"] = (
        (readiness["usable_quarters"] >= min_quarters)
        & (readiness["coverage_ratio"] >= min_coverage)
    )

    ready_themes = set(readiness.loc[readiness["ready_for_forecast"], "theme"])
    model_ready = quarterly[
        quarterly["theme"].isin(ready_themes) & quarterly["eligible_for_modeling"]
    ].copy()

    taxonomy_path = output_dir / "theme_taxonomy.csv"
    all_path = output_dir / "theme_quarterly_all.csv"
    ready_path = output_dir / "theme_quarterly_model_ready.csv"
    readiness_path = output_dir / "theme_readiness.csv"

    pd.DataFrame.from_records(taxonomy_records()).to_csv(
        taxonomy_path, index=False, encoding="utf-8-sig"
    )
    quarterly.to_csv(all_path, index=False, encoding="utf-8-sig")
    model_ready.to_csv(ready_path, index=False, encoding="utf-8-sig")
    readiness.to_csv(readiness_path, index=False, encoding="utf-8-sig")

    print(f"kept {len(titles):,} anime rows after filtering")
    print(f"matched {quarterly['theme'].nunique():,} standardized themes")
    print(f"themes ready for forecasting: {len(ready_themes):,}")
    print(f"saved taxonomy to {taxonomy_path}")
    print(f"saved quarterly dataset to {all_path}")
    print(f"saved model-ready dataset to {ready_path}")
    print(f"saved readiness summary to {readiness_path}")

    return {
        "taxonomy": taxonomy_path,
        "quarterly_all": all_path,
        "quarterly_ready": ready_path,
        "readiness": readiness_path,
    }


def main() -> None:
    args = parse_args()
    build_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        as_of_date=args.as_of_date,
        keep_incomplete_quarter=args.keep_incomplete_quarter,
        vote_quantile=args.vote_quantile,
        min_titles_per_quarter=args.min_titles_per_quarter,
        min_quarters=args.min_quarters,
        min_coverage=args.min_coverage,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
