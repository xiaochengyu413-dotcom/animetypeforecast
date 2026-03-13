from __future__ import annotations

from typing import Mapping

import pandas as pd


COMPONENT_COLUMNS: tuple[str, ...] = (
    "rating_component",
    "votes_component",
    "favorites_component",
    "titles_component",
)

DEFAULT_POPULARITY_WEIGHTS: dict[str, float] = {
    "rating_component": 0.45,
    "votes_component": 0.30,
    "favorites_component": 0.15,
    "titles_component": 0.10,
}


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for component in COMPONENT_COLUMNS:
        normalized[component] = float(weights.get(component, 0.0))

    total_weight = sum(normalized.values())
    if total_weight <= 0:
        raise ValueError("Popularity weights must sum to a positive value.")

    return {component: value / total_weight for component, value in normalized.items()}


def compute_popularity_index(
    frame: pd.DataFrame,
    weights: Mapping[str, float],
    scale: float = 100.0,
) -> pd.Series:
    normalized = normalize_weights(weights)
    score = pd.Series(0.0, index=frame.index, dtype=float)
    for component, weight in normalized.items():
        if component not in frame.columns:
            raise KeyError(f"Missing required component column: {component}")
        score = score + weight * pd.to_numeric(frame[component], errors="coerce").fillna(0.0)
    return scale * score


def weight_record(name: str, weights: Mapping[str, float]) -> dict[str, float | str]:
    normalized = normalize_weights(weights)
    return {"scenario": name, **normalized}
