from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_forecasts import (
    build_prophet_model,
    quarter_start_from_label,
    resolve_font,
    safe_name,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "generated" / "theme_quarterly_model_ready.csv"
DEFAULT_READINESS = SCRIPT_DIR / "generated" / "theme_readiness.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated" / "model_backtest_2024_2025"
DEFAULT_PROPHET_EVAL_DIR = SCRIPT_DIR / "generated" / "evaluation_2024_2025"
DEFAULT_TARGETS = ("popularity_index", "avg_weighted_rating")

MODEL_LINEAR = "linear_regression"
MODEL_PROPHET = "prophet"
MODEL_LSTM = "lstm"
MODEL_ORDER = (MODEL_LINEAR, MODEL_PROPHET, MODEL_LSTM)
MODEL_LABELS = {
    MODEL_LINEAR: "Linear Regression",
    MODEL_PROPHET: "Prophet",
    MODEL_LSTM: "LSTM",
}
MODEL_COLORS = {
    MODEL_LINEAR: "#4e79a7",
    MODEL_PROPHET: "#f28e2b",
    MODEL_LSTM: "#59a14f",
}

TORCH_SITE_PACKAGES = SCRIPT_DIR / ".venv-torch" / "Lib" / "site-packages"
if TORCH_SITE_PACKAGES.exists():
    import sys

    sys.path.insert(0, str(TORCH_SITE_PACKAGES))

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a linear regression baseline, Prophet, and an LSTM-style model "
            "on a historical backtest window."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prophet-evaluation-dir",
        type=Path,
        default=DEFAULT_PROPHET_EVAL_DIR,
        help="Directory containing existing Prophet backtest outputs to reuse when available.",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        default=None,
        help="Forecast target column. Can be passed multiple times.",
    )
    parser.add_argument("--test-start-quarter", type=str, default="2024Q1")
    parser.add_argument("--test-end-quarter", type=str, default="2025Q4")
    parser.add_argument("--min-train-points", type=int, default=16)
    parser.add_argument("--lookback", type=int, default=8)
    parser.add_argument("--lstm-hidden-size", type=int, default=12)
    parser.add_argument("--lstm-epochs", type=int, default=240)
    parser.add_argument("--lstm-learning-rate", type=float, default=0.02)
    parser.add_argument("--lstm-seed", type=int, default=42)
    parser.add_argument(
        "--lstm-backend",
        choices=("auto", "torch", "numpy"),
        default="auto",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.square(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100.0)


def quarter_ordinals(ds: pd.Series) -> np.ndarray:
    return np.asarray([period.ordinal for period in ds.dt.to_period("Q")], dtype=float)


def build_quarterly_sin_cos(ds: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    quarter_numbers = ds.dt.quarter.to_numpy(dtype=float)
    radians = 2.0 * np.pi * (quarter_numbers - 1.0) / 4.0
    return np.sin(radians), np.cos(radians)


def load_ready_theme_dataset(input_path: Path, readiness_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(input_path, encoding="utf-8-sig", parse_dates=["ds"])
    readiness = pd.read_csv(readiness_path, encoding="utf-8-sig")

    ready_themes = set(readiness.loc[readiness["ready_for_forecast"] == True, "theme"])  # noqa: E712
    frame = dataset.loc[dataset["theme"].isin(ready_themes)].copy()
    frame = frame.sort_values(["theme", "ds"]).reset_index(drop=True)
    return frame


def split_train_test(
    theme_frame: pd.DataFrame,
    test_start_quarter: str,
    test_end_quarter: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    ordered = theme_frame.sort_values("ds").reset_index(drop=True).copy()
    test_start = quarter_start_from_label(test_start_quarter)
    test_end = quarter_start_from_label(test_end_quarter)
    if test_start is None or test_end is None:
        raise ValueError("Both test_start_quarter and test_end_quarter must be provided.")

    test = ordered.loc[(ordered["ds"] >= test_start) & (ordered["ds"] <= test_end)].copy()
    if test.empty:
        return None

    train = ordered.loc[ordered["ds"] < test_start].copy()
    if train.empty:
        return None

    return train.reset_index(drop=True), test.reset_index(drop=True)


@dataclass
class LinearTrendModel:
    beta: np.ndarray
    reference_ordinal: float
    offset_mean: float
    offset_scale: float


def fit_linear_trend_model(train_frame: pd.DataFrame, target: str) -> LinearTrendModel:
    ordinals = quarter_ordinals(train_frame["ds"])
    reference = float(ordinals[0])
    offsets = ordinals - reference
    offset_mean = float(offsets.mean())
    offset_scale = float(offsets.std()) if float(offsets.std()) > 1e-6 else 1.0
    scaled_offsets = (offsets - offset_mean) / offset_scale

    quarters = train_frame["ds"].dt.quarter.to_numpy()
    design = np.column_stack(
        [
            np.ones(len(train_frame), dtype=float),
            scaled_offsets,
            (quarters == 2).astype(float),
            (quarters == 3).astype(float),
            (quarters == 4).astype(float),
        ]
    )
    response = train_frame[target].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(design, response, rcond=None)
    return LinearTrendModel(
        beta=beta,
        reference_ordinal=reference,
        offset_mean=offset_mean,
        offset_scale=offset_scale,
    )


def predict_linear_trend_model(
    model: LinearTrendModel,
    forecast_frame: pd.DataFrame,
) -> np.ndarray:
    ordinals = quarter_ordinals(forecast_frame["ds"])
    offsets = ordinals - model.reference_ordinal
    scaled_offsets = (offsets - model.offset_mean) / model.offset_scale
    quarters = forecast_frame["ds"].dt.quarter.to_numpy()
    design = np.column_stack(
        [
            np.ones(len(forecast_frame), dtype=float),
            scaled_offsets,
            (quarters == 2).astype(float),
            (quarters == 3).astype(float),
            (quarters == 4).astype(float),
        ]
    )
    return design @ model.beta


def predict_prophet_window(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target: str,
) -> np.ndarray:
    prophet_train = train_frame[["ds", target]].rename(columns={target: "y"})
    model = build_prophet_model()
    model.fit(prophet_train)
    forecast = model.predict(test_frame[["ds"]])[["ds", "yhat"]]
    merged = test_frame[["ds"]].merge(forecast, on="ds", how="left")
    return merged["yhat"].to_numpy(dtype=float)


def load_existing_prophet_results(
    evaluation_dir: Path,
    target: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    prediction_path = evaluation_dir / f"forecast_comparison_{target}.csv"
    metrics_path = evaluation_dir / f"metrics_{target}.csv"
    if not prediction_path.exists() or not metrics_path.exists():
        return {}, {}

    prediction_frame = pd.read_csv(prediction_path, encoding="utf-8-sig", parse_dates=["ds"])
    metrics_frame = pd.read_csv(metrics_path, encoding="utf-8-sig")

    prediction_lookup = {
        str(theme): group.sort_values("ds").reset_index(drop=True)
        for theme, group in prediction_frame.groupby("theme")
    }
    metrics_lookup = {
        str(row["theme"]): row
        for row in metrics_frame.to_dict(orient="records")
    }
    return prediction_lookup, metrics_lookup


@dataclass
class Standardizer:
    mean: float
    scale: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


def build_standardizer(values: np.ndarray, floor: float = 1e-6) -> Standardizer:
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if scale < floor:
        scale = 1.0
    return Standardizer(mean=mean, scale=scale)


@dataclass
class LSTMTrainingConfig:
    lookback: int
    hidden_size: int
    learning_rate: float
    epochs: int
    seed: int
    backend: str = "auto"
    gradient_clip: float = 5.0
    l2_penalty: float = 1e-4
    patience: int = 35


class NumpyLSTMForecaster:
    def __init__(self, config: LSTMTrainingConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.target_scaler: Standardizer | None = None
        self.gap_scaler: Standardizer | None = None
        self.params: dict[str, np.ndarray] | None = None

    def fit(self, train_frame: pd.DataFrame, target: str) -> None:
        values = train_frame[target].to_numpy(dtype=float)
        if len(values) <= self.config.lookback:
            raise ValueError(
                f"LSTM needs more than {self.config.lookback} training points, got {len(values)}."
            )

        ordinals = quarter_ordinals(train_frame["ds"])
        gaps = np.diff(ordinals, prepend=ordinals[0] - 1.0)

        self.target_scaler = build_standardizer(values)
        self.gap_scaler = build_standardizer(gaps)
        scaled_values = self.target_scaler.transform(values)
        scaled_gaps = self.gap_scaler.transform(gaps)
        sin_quarter, cos_quarter = build_quarterly_sin_cos(train_frame["ds"])

        samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        for index in range(self.config.lookback, len(train_frame)):
            sequence = scaled_values[index - self.config.lookback : index]
            aux = np.asarray(
                [sin_quarter[index], cos_quarter[index], scaled_gaps[index]],
                dtype=float,
            )
            samples.append((sequence, aux, float(scaled_values[index])))

        self._initialize_parameters()
        opt_state = self._new_optimizer_state()
        best_loss = math.inf
        best_params: dict[str, np.ndarray] | None = None
        stale_epochs = 0

        for _epoch in range(self.config.epochs):
            gradients = self._zero_like_params()
            total_loss = 0.0

            for sequence, aux, target_value in samples:
                prediction, cache = self._forward(sequence, aux)
                error = prediction - target_value
                total_loss += 0.5 * error * error
                sample_gradients = self._backward(error, cache)
                for name in gradients:
                    gradients[name] += sample_gradients[name]

            sample_count = float(len(samples))
            for name, grad in gradients.items():
                gradients[name] = grad / sample_count
                if name != "b_out":
                    gradients[name] += self.config.l2_penalty * self.params[name]

            self._clip_gradients(gradients)
            self._apply_adam_update(gradients, opt_state)

            mean_loss = total_loss / sample_count
            if mean_loss + 1e-7 < best_loss:
                best_loss = mean_loss
                best_params = {name: value.copy() for name, value in self.params.items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if best_params is not None:
            self.params = best_params

    def predict_recursive(
        self,
        train_frame: pd.DataFrame,
        forecast_frame: pd.DataFrame,
        target: str,
    ) -> np.ndarray:
        if self.target_scaler is None or self.gap_scaler is None or self.params is None:
            raise RuntimeError("LSTM model must be fitted before prediction.")

        history_values = train_frame[target].to_numpy(dtype=float)
        if len(history_values) < self.config.lookback:
            raise ValueError(
                f"Need at least {self.config.lookback} history values, got {len(history_values)}."
            )

        scaled_history = list(self.target_scaler.transform(history_values))
        previous_ordinal = float(quarter_ordinals(train_frame["ds"])[-1])
        predictions: list[float] = []

        for row in forecast_frame.itertuples(index=False):
            current_ds = pd.Timestamp(row.ds)
            quarter_num = int(current_ds.quarter)
            radians = 2.0 * np.pi * (quarter_num - 1.0) / 4.0
            current_ordinal = float(current_ds.to_period("Q").ordinal)
            scaled_gap = self.gap_scaler.transform(
                np.asarray([current_ordinal - previous_ordinal], dtype=float)
            )[0]
            aux = np.asarray([math.sin(radians), math.cos(radians), scaled_gap], dtype=float)
            sequence = np.asarray(scaled_history[-self.config.lookback :], dtype=float)
            predicted_scaled, _cache = self._forward(sequence, aux)
            predicted_value = float(self.target_scaler.inverse(np.asarray([predicted_scaled]))[0])
            predictions.append(predicted_value)
            scaled_history.append(predicted_scaled)
            previous_ordinal = current_ordinal

        return np.asarray(predictions, dtype=float)

    def _initialize_parameters(self) -> None:
        hidden = self.config.hidden_size
        input_size = 1
        aux_size = 3
        concat_size = hidden + input_size

        def randn(shape: tuple[int, ...], scale: float) -> np.ndarray:
            return self.rng.normal(loc=0.0, scale=scale, size=shape)

        gate_scale = 1.0 / math.sqrt(concat_size)
        output_scale = 1.0 / math.sqrt(hidden + aux_size)

        self.params = {
            "W_f": randn((hidden, concat_size), gate_scale),
            "W_i": randn((hidden, concat_size), gate_scale),
            "W_o": randn((hidden, concat_size), gate_scale),
            "W_g": randn((hidden, concat_size), gate_scale),
            "b_f": np.zeros((hidden, 1), dtype=float),
            "b_i": np.zeros((hidden, 1), dtype=float),
            "b_o": np.zeros((hidden, 1), dtype=float),
            "b_g": np.zeros((hidden, 1), dtype=float),
            "W_out": randn((1, hidden + aux_size), output_scale),
            "b_out": np.zeros((1, 1), dtype=float),
        }

    def _new_optimizer_state(self) -> dict[str, dict[str, np.ndarray] | int]:
        if self.params is None:
            raise RuntimeError("Parameters must be initialized before creating optimizer state.")
        return {
            "m": {name: np.zeros_like(value) for name, value in self.params.items()},
            "v": {name: np.zeros_like(value) for name, value in self.params.items()},
            "step": 0,
        }

    def _zero_like_params(self) -> dict[str, np.ndarray]:
        if self.params is None:
            raise RuntimeError("Parameters must be initialized before zeroing gradients.")
        return {name: np.zeros_like(value) for name, value in self.params.items()}

    def _clip_gradients(self, gradients: dict[str, np.ndarray]) -> None:
        squared_norm = 0.0
        for grad in gradients.values():
            squared_norm += float(np.sum(np.square(grad)))

        global_norm = math.sqrt(squared_norm)
        if global_norm <= self.config.gradient_clip or global_norm == 0.0:
            return

        scale = self.config.gradient_clip / global_norm
        for name in gradients:
            gradients[name] *= scale

    def _apply_adam_update(
        self,
        gradients: dict[str, np.ndarray],
        optimizer_state: dict[str, dict[str, np.ndarray] | int],
    ) -> None:
        if self.params is None:
            raise RuntimeError("Parameters must be initialized before optimizer updates.")

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        optimizer_state["step"] = int(optimizer_state["step"]) + 1
        step = int(optimizer_state["step"])

        m = optimizer_state["m"]
        v = optimizer_state["v"]
        assert isinstance(m, dict)
        assert isinstance(v, dict)

        for name, grad in gradients.items():
            m[name] = beta1 * m[name] + (1.0 - beta1) * grad
            v[name] = beta2 * v[name] + (1.0 - beta2) * np.square(grad)
            m_hat = m[name] / (1.0 - beta1**step)
            v_hat = v[name] / (1.0 - beta2**step)
            self.params[name] -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def _forward(self, sequence: np.ndarray, aux: np.ndarray) -> tuple[float, dict[str, object]]:
        if self.params is None:
            raise RuntimeError("Parameters must be initialized before forward passes.")

        hidden = self.config.hidden_size
        h_prev = np.zeros((hidden, 1), dtype=float)
        c_prev = np.zeros((hidden, 1), dtype=float)
        step_caches: list[dict[str, np.ndarray]] = []

        for value in sequence:
            x_t = np.asarray([[float(value)]], dtype=float)
            z_t = np.vstack([h_prev, x_t])
            f_t = self._sigmoid(self.params["W_f"] @ z_t + self.params["b_f"])
            i_t = self._sigmoid(self.params["W_i"] @ z_t + self.params["b_i"])
            o_t = self._sigmoid(self.params["W_o"] @ z_t + self.params["b_o"])
            g_t = np.tanh(self.params["W_g"] @ z_t + self.params["b_g"])
            c_t = f_t * c_prev + i_t * g_t
            h_t = o_t * np.tanh(c_t)
            step_caches.append(
                {
                    "z": z_t,
                    "f": f_t,
                    "i": i_t,
                    "o": o_t,
                    "g": g_t,
                    "c": c_t,
                    "c_prev": c_prev,
                }
            )
            h_prev = h_t
            c_prev = c_t

        aux_column = aux.reshape(-1, 1)
        output_input = np.vstack([h_prev, aux_column])
        y_hat = self.params["W_out"] @ output_input + self.params["b_out"]

        cache: dict[str, object] = {
            "steps": step_caches,
            "output_input": output_input,
            "hidden_final": h_prev,
        }
        return float(y_hat[0, 0]), cache

    def _backward(self, error: float, cache: dict[str, object]) -> dict[str, np.ndarray]:
        if self.params is None:
            raise RuntimeError("Parameters must be initialized before backward passes.")

        gradients = self._zero_like_params()
        dy = np.asarray([[float(error)]], dtype=float)

        output_input = cache["output_input"]
        hidden_final = cache["hidden_final"]
        step_caches = cache["steps"]
        assert isinstance(output_input, np.ndarray)
        assert isinstance(hidden_final, np.ndarray)
        assert isinstance(step_caches, list)

        gradients["W_out"] += dy @ output_input.T
        gradients["b_out"] += dy

        dh_next = self.params["W_out"][:, : self.config.hidden_size].T @ dy
        dc_next = np.zeros_like(hidden_final)

        for step_cache in reversed(step_caches):
            z_t = step_cache["z"]
            f_t = step_cache["f"]
            i_t = step_cache["i"]
            o_t = step_cache["o"]
            g_t = step_cache["g"]
            c_t = step_cache["c"]
            c_prev = step_cache["c_prev"]

            tanh_c = np.tanh(c_t)
            do_t = dh_next * tanh_c
            do_raw = do_t * o_t * (1.0 - o_t)

            dc_total = dh_next * o_t * (1.0 - np.square(tanh_c)) + dc_next
            df_t = dc_total * c_prev
            df_raw = df_t * f_t * (1.0 - f_t)

            di_t = dc_total * g_t
            di_raw = di_t * i_t * (1.0 - i_t)

            dg_t = dc_total * i_t
            dg_raw = dg_t * (1.0 - np.square(g_t))

            gradients["W_f"] += df_raw @ z_t.T
            gradients["W_i"] += di_raw @ z_t.T
            gradients["W_o"] += do_raw @ z_t.T
            gradients["W_g"] += dg_raw @ z_t.T
            gradients["b_f"] += df_raw
            gradients["b_i"] += di_raw
            gradients["b_o"] += do_raw
            gradients["b_g"] += dg_raw

            dz_t = (
                self.params["W_f"].T @ df_raw
                + self.params["W_i"].T @ di_raw
                + self.params["W_o"].T @ do_raw
                + self.params["W_g"].T @ dg_raw
            )
            dh_next = dz_t[: self.config.hidden_size, :]
            dc_next = dc_total * f_t

        return gradients

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))


class TorchSequenceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        sequence_output, _state = self.lstm(inputs)
        final_hidden = sequence_output[:, -1, :]
        return self.output(final_hidden).squeeze(-1)


class TorchLSTMForecaster:
    def __init__(self, config: LSTMTrainingConfig) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available for the standard LSTM implementation.")
        self.config = config
        self.target_scaler: Standardizer | None = None
        self.gap_scaler: Standardizer | None = None
        self.model: TorchSequenceLSTM | None = None

    def fit(self, train_frame: pd.DataFrame, target: str) -> None:
        values = train_frame[target].to_numpy(dtype=float)
        if len(values) <= self.config.lookback:
            raise ValueError(
                f"LSTM needs more than {self.config.lookback} training points, got {len(values)}."
            )

        ordinals = quarter_ordinals(train_frame["ds"])
        gaps = np.diff(ordinals, prepend=ordinals[0] - 1.0)
        sin_quarter, cos_quarter = build_quarterly_sin_cos(train_frame["ds"])

        self.target_scaler = build_standardizer(values)
        self.gap_scaler = build_standardizer(gaps)
        scaled_values = self.target_scaler.transform(values)
        scaled_gaps = self.gap_scaler.transform(gaps)

        features = np.column_stack([scaled_values, sin_quarter, cos_quarter, scaled_gaps]).astype(np.float32)

        xs: list[np.ndarray] = []
        ys: list[float] = []
        for index in range(self.config.lookback, len(train_frame)):
            xs.append(features[index - self.config.lookback : index])
            ys.append(float(scaled_values[index]))

        if not xs:
            raise ValueError("No training windows were generated for the LSTM.")

        x_all = torch.tensor(np.asarray(xs, dtype=np.float32), dtype=torch.float32)
        y_all = torch.tensor(np.asarray(ys, dtype=np.float32), dtype=torch.float32)

        validation_count = max(1, len(xs) // 5)
        if len(xs) - validation_count < 1:
            validation_count = 1
        train_count = max(1, len(xs) - validation_count)

        x_train = x_all[:train_count]
        y_train = y_all[:train_count]
        x_val = x_all[train_count:]
        y_val = y_all[train_count:]
        if len(x_val) == 0:
            x_val = x_train
            y_val = y_train

        torch.manual_seed(self.config.seed)
        self.model = TorchSequenceLSTM(input_size=4, hidden_size=self.config.hidden_size)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_penalty,
        )
        loss_fn = nn.MSELoss()

        best_state = None
        best_val_loss = float("inf")
        stale_epochs = 0

        for _epoch in range(self.config.epochs):
            self.model.train()
            optimizer.zero_grad()
            train_prediction = self.model(x_train)
            train_loss = loss_fn(train_prediction, y_train)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_prediction = self.model(x_val)
                val_loss = float(loss_fn(val_prediction, y_val).item())

            if val_loss + 1e-7 < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def predict_recursive(
        self,
        train_frame: pd.DataFrame,
        forecast_frame: pd.DataFrame,
        target: str,
    ) -> np.ndarray:
        if self.model is None or self.target_scaler is None or self.gap_scaler is None:
            raise RuntimeError("Torch LSTM model must be fitted before prediction.")

        values = train_frame[target].to_numpy(dtype=float)
        ordinals = quarter_ordinals(train_frame["ds"])
        gaps = np.diff(ordinals, prepend=ordinals[0] - 1.0)
        sin_quarter, cos_quarter = build_quarterly_sin_cos(train_frame["ds"])

        scaled_values = list(self.target_scaler.transform(values))
        scaled_gaps = list(self.gap_scaler.transform(gaps))
        sin_history = list(np.asarray(sin_quarter, dtype=float))
        cos_history = list(np.asarray(cos_quarter, dtype=float))
        previous_ordinal = float(ordinals[-1])
        predictions: list[float] = []

        for row in forecast_frame.itertuples(index=False):
            current_ds = pd.Timestamp(row.ds)
            quarter_num = int(current_ds.quarter)
            radians = 2.0 * np.pi * (quarter_num - 1.0) / 4.0
            current_ordinal = float(current_ds.to_period("Q").ordinal)
            current_gap = float(self.gap_scaler.transform(np.asarray([current_ordinal - previous_ordinal]))[0])
            current_sin = float(math.sin(radians))
            current_cos = float(math.cos(radians))

            sequence_features = np.column_stack(
                [
                    np.asarray(scaled_values[-self.config.lookback :], dtype=np.float32),
                    np.asarray(sin_history[-self.config.lookback :], dtype=np.float32),
                    np.asarray(cos_history[-self.config.lookback :], dtype=np.float32),
                    np.asarray(scaled_gaps[-self.config.lookback :], dtype=np.float32),
                ]
            )

            with torch.no_grad():
                inputs = torch.tensor(sequence_features[None, :, :], dtype=torch.float32)
                predicted_scaled = float(self.model(inputs).cpu().item())

            predicted_value = float(self.target_scaler.inverse(np.asarray([predicted_scaled]))[0])
            predictions.append(predicted_value)

            scaled_values.append(predicted_scaled)
            scaled_gaps.append(current_gap)
            sin_history.append(current_sin)
            cos_history.append(current_cos)
            previous_ordinal = current_ordinal

        return np.asarray(predictions, dtype=float)


def build_lstm_forecaster(config: LSTMTrainingConfig):
    backend = config.backend
    if backend == "torch":
        return TorchLSTMForecaster(config)
    if backend == "numpy":
        return NumpyLSTMForecaster(config)
    if backend == "auto" and TORCH_AVAILABLE:
        return TorchLSTMForecaster(config)
    return NumpyLSTMForecaster(config)


def evaluate_single_theme(
    theme_frame: pd.DataFrame,
    target: str,
    test_start_quarter: str,
    test_end_quarter: str,
    min_train_points: int,
    lstm_config: LSTMTrainingConfig,
    prophet_prediction_lookup: dict[str, pd.DataFrame] | None = None,
    prophet_metrics_lookup: dict[str, dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame] | None:
    split = split_train_test(theme_frame, test_start_quarter, test_end_quarter)
    if split is None:
        return None

    train_frame, test_frame = split
    if len(train_frame) < min_train_points:
        return None

    actual = test_frame[target].to_numpy(dtype=float)
    theme_name = str(theme_frame["theme"].iloc[0])

    linear_model = fit_linear_trend_model(train_frame=train_frame, target=target)
    linear_pred = predict_linear_trend_model(linear_model, test_frame)

    prophet_pred: np.ndarray
    prophet_override = None
    if prophet_prediction_lookup:
        prophet_override = prophet_prediction_lookup.get(theme_name)
    if prophet_override is not None:
        merged_prophet = test_frame[["ds"]].merge(
            prophet_override[["ds", "yhat"]],
            on="ds",
            how="left",
        )
        if merged_prophet["yhat"].isna().any():
            prophet_pred = predict_prophet_window(train_frame=train_frame, test_frame=test_frame, target=target)
        else:
            prophet_pred = merged_prophet["yhat"].to_numpy(dtype=float)
    else:
        prophet_pred = predict_prophet_window(train_frame=train_frame, test_frame=test_frame, target=target)

    lstm_model = build_lstm_forecaster(config=lstm_config)
    lstm_model.fit(train_frame=train_frame, target=target)
    lstm_pred = lstm_model.predict_recursive(train_frame=train_frame, forecast_frame=test_frame, target=target)

    prediction_map = {
        MODEL_LINEAR: linear_pred,
        MODEL_PROPHET: prophet_pred,
        MODEL_LSTM: lstm_pred,
    }

    metrics_rows: list[dict[str, object]] = []
    comparison_frame = test_frame[["theme", "ds", target]].copy()
    comparison_frame = comparison_frame.rename(columns={target: "actual"})

    for model_name in MODEL_ORDER:
        prediction = np.asarray(prediction_map[model_name], dtype=float)
        metric_row = {
            "theme": theme_name,
            "target": target,
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "train_points": int(len(train_frame)),
            "test_points": int(len(test_frame)),
            "mae": mae(actual, prediction),
            "mse": mse(actual, prediction),
            "rmse": rmse(actual, prediction),
            "mape": mape(actual, prediction),
            "train_end": train_frame["ds"].max().strftime("%Y-%m-%d"),
            "test_start": test_frame["ds"].min().strftime("%Y-%m-%d"),
            "test_end": test_frame["ds"].max().strftime("%Y-%m-%d"),
        }
        if model_name == MODEL_PROPHET and prophet_metrics_lookup:
            prophet_metric_row = prophet_metrics_lookup.get(theme_name)
            if prophet_metric_row is not None:
                metric_row["train_points"] = int(prophet_metric_row["train_points"])
                metric_row["test_points"] = int(prophet_metric_row["test_points"])
                metric_row["mae"] = float(prophet_metric_row["prophet_mae"])
                metric_row["rmse"] = float(prophet_metric_row["prophet_rmse"])
                metric_row["mape"] = float(prophet_metric_row["prophet_mape"])
                metric_row["mse"] = float(metric_row["rmse"]) ** 2
                metric_row["train_end"] = str(prophet_metric_row["train_end"])
                metric_row["test_start"] = str(prophet_metric_row["test_start"])
                metric_row["test_end"] = str(prophet_metric_row["test_end"])
        metrics_rows.append(metric_row)
        comparison_frame[model_name] = prediction

    return metrics_rows, comparison_frame


def build_theme_winner_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    winner_rows: list[dict[str, object]] = []
    for theme, group in metrics_df.groupby("theme"):
        winner_rows.append(
            {
                "theme": theme,
                "target": str(group["target"].iloc[0]),
                "mae_winner": str(group.loc[group["mae"].idxmin(), "model"]),
                "mse_winner": str(group.loc[group["mse"].idxmin(), "model"]),
                "rmse_winner": str(group.loc[group["rmse"].idxmin(), "model"]),
                "mape_winner": str(group.loc[group["mape"].idxmin(), "model"]),
            }
        )
    return pd.DataFrame(winner_rows).sort_values("theme").reset_index(drop=True)


def build_model_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    winner_table = build_theme_winner_table(metrics_df)
    summary_rows: list[dict[str, object]] = []
    for model_name, group in metrics_df.groupby("model"):
        model_name = str(model_name)
        summary_rows.append(
            {
                "target": str(group["target"].iloc[0]),
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "theme_count": int(group["theme"].nunique()),
                "mean_mae": float(group["mae"].mean()),
                "mean_mse": float(group["mse"].mean()),
                "mean_rmse": float(group["rmse"].mean()),
                "mean_mape": float(group["mape"].mean()),
                "median_mae": float(group["mae"].median()),
                "median_mse": float(group["mse"].median()),
                "median_rmse": float(group["rmse"].median()),
                "median_mape": float(group["mape"].median()),
                "mae_win_count": int((winner_table["mae_winner"] == model_name).sum()),
                "mse_win_count": int((winner_table["mse_winner"] == model_name).sum()),
                "rmse_win_count": int((winner_table["rmse_winner"] == model_name).sum()),
                "mape_win_count": int((winner_table["mape_winner"] == model_name).sum()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)


def write_markdown_summary(
    target: str,
    summary_df: pd.DataFrame,
    winner_df: pd.DataFrame,
    output_path: Path,
    test_start_quarter: str,
    test_end_quarter: str,
) -> None:
    lines = [
        f"# Model Backtest Summary: {target}",
        "",
        f"Backtest window: {test_start_quarter} to {test_end_quarter}",
        "",
        "## Mean Metrics",
        "",
        "| Model | Mean MAE | Mean MSE | Mean RMSE | Mean MAPE | MAE Wins | RMSE Wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            "| "
            f"{row.model_label} | {row.mean_mae:.4f} | {row.mean_mse:.4f} | "
            f"{row.mean_rmse:.4f} | {row.mean_mape:.4f} | "
            f"{row.mae_win_count} | {row.rmse_win_count} |"
        )

    lines.extend(
        [
            "",
            "## Theme Winners",
            "",
            "| Theme | MAE Winner | MSE Winner | RMSE Winner | MAPE Winner |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in winner_df.itertuples(index=False):
        lines.append(
            f"| {row.theme} | {row.mae_winner} | {row.mse_winner} | "
            f"{row.rmse_winner} | {row.mape_winner} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_theme_predictions(
    comparison_frame: pd.DataFrame,
    target: str,
    output_path: Path,
    font,
) -> None:
    ordered = comparison_frame.sort_values("ds").copy()
    if ordered.empty:
        return

    labels = ordered["ds"].dt.to_period("Q").astype(str).tolist()
    x_positions = np.arange(len(ordered), dtype=float)

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    ax.plot(x_positions, ordered["actual"], color="#1f1f1f", linewidth=2.4, marker="o", label="Actual")

    for model_name in MODEL_ORDER:
        ax.plot(
            x_positions,
            ordered[model_name],
            color=MODEL_COLORS[model_name],
            linewidth=1.8,
            marker="o",
            linestyle="--",
            label=MODEL_LABELS[model_name],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontproperties=font)
    ax.set_xlabel("Quarter")
    ax.set_ylabel(target)
    ax.grid(alpha=0.25)
    ax.legend(prop=font)
    ax.set_title(
        f"{ordered['theme'].iloc[0]}: {target} backtest comparison",
        fontproperties=font,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def compare_models_for_target(
    dataset: pd.DataFrame,
    output_dir: Path,
    prophet_evaluation_dir: Path,
    target: str,
    test_start_quarter: str,
    test_end_quarter: str,
    min_train_points: int,
    lstm_config: LSTMTrainingConfig,
    skip_plots: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots" / target
    if not skip_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    prophet_prediction_lookup, prophet_metrics_lookup = load_existing_prophet_results(
        evaluation_dir=prophet_evaluation_dir,
        target=target,
    )
    metrics_rows: list[dict[str, object]] = []
    comparison_frames: list[pd.DataFrame] = []
    font = resolve_font()

    for theme, theme_frame in dataset.groupby("theme"):
        result = evaluate_single_theme(
            theme_frame=theme_frame.copy(),
            target=target,
            test_start_quarter=test_start_quarter,
            test_end_quarter=test_end_quarter,
            min_train_points=min_train_points,
            lstm_config=lstm_config,
            prophet_prediction_lookup=prophet_prediction_lookup,
            prophet_metrics_lookup=prophet_metrics_lookup,
        )
        if result is None:
            continue

        theme_metrics, comparison_frame = result
        metrics_rows.extend(theme_metrics)
        comparison_frames.append(comparison_frame)

        if not skip_plots:
            plot_theme_predictions(
                comparison_frame=comparison_frame,
                target=target,
                output_path=plots_dir / f"{safe_name(str(theme))}_{target}_comparison.png",
                font=font,
            )

    if not metrics_rows:
        raise RuntimeError(f"No valid model-comparison results were produced for target {target}.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["theme", "model"]).reset_index(drop=True)
    comparison_df = pd.concat(comparison_frames, ignore_index=True).sort_values(["theme", "ds"]).reset_index(drop=True)
    winner_df = build_theme_winner_table(metrics_df)
    summary_df = build_model_summary(metrics_df)

    metrics_path = output_dir / f"metrics_{target}.csv"
    comparison_path = output_dir / f"window_predictions_{target}.csv"
    winners_path = output_dir / f"theme_winners_{target}.csv"
    summary_path = output_dir / f"summary_{target}.csv"
    markdown_path = output_dir / f"summary_{target}.md"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    winner_df.to_csv(winners_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_markdown_summary(
        target=target,
        summary_df=summary_df,
        winner_df=winner_df,
        output_path=markdown_path,
        test_start_quarter=test_start_quarter,
        test_end_quarter=test_end_quarter,
    )

    print(f"compared {len(comparison_frames):,} themes for target {target}")
    print(f"saved per-model metrics to {metrics_path}")
    print(f"saved backtest window predictions to {comparison_path}")
    print(f"saved per-theme winners to {winners_path}")
    print(f"saved model summary to {summary_path}")
    print(f"saved markdown summary to {markdown_path}")
    if not skip_plots:
        print(f"saved comparison plots to {plots_dir}")

    outputs = {
        "metrics": metrics_path,
        "predictions": comparison_path,
        "winners": winners_path,
        "summary": summary_path,
        "markdown": markdown_path,
    }
    if not skip_plots:
        outputs["plots_dir"] = plots_dir
    return outputs


def main() -> None:
    args = parse_args()
    targets = tuple(dict.fromkeys(args.targets or list(DEFAULT_TARGETS)))
    dataset = load_ready_theme_dataset(input_path=args.input, readiness_path=args.readiness)
    required_columns = {"theme", "ds", *targets}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    lstm_config = LSTMTrainingConfig(
        lookback=args.lookback,
        hidden_size=args.lstm_hidden_size,
        learning_rate=args.lstm_learning_rate,
        epochs=args.lstm_epochs,
        seed=args.lstm_seed,
        backend=args.lstm_backend,
    )

    for target in targets:
        compare_models_for_target(
            dataset=dataset,
            output_dir=args.output_dir,
            prophet_evaluation_dir=args.prophet_evaluation_dir,
            target=target,
            test_start_quarter=args.test_start_quarter,
            test_end_quarter=args.test_end_quarter,
            min_train_points=args.min_train_points,
            lstm_config=lstm_config,
            skip_plots=args.skip_plots,
        )


if __name__ == "__main__":
    main()
