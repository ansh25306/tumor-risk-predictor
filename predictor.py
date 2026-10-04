"""Model loading, input validation and inference.

Kept free of any UI code so it can be unit-tested and reused (CLI, API, notebook).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MAX_BATCH_ROWS = 5000
# Values beyond this multiple of the largest value seen in training are rejected
# as implausible (likely a unit or typing mistake) rather than merely warned about.
HARD_LIMIT_FACTOR = 3.0


class ModelLoadError(RuntimeError):
    """Raised when the model or metadata files are missing or unreadable."""


class ValidationError(ValueError):
    """Raised when user input is missing or invalid. `errors` lists every problem."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class Prediction:
    label: str
    probability_malignant: float
    probabilities: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return max(self.probabilities.values())


def load_artifacts(directory: Path | str = ARTIFACT_DIR):
    """Load (model, metadata). Raises ModelLoadError with a readable message."""
    directory = Path(directory)
    model_path = directory / "model.joblib"
    meta_path = directory / "metadata.json"
    for path in (model_path, meta_path):
        if not path.exists():
            raise ModelLoadError(
                f"Missing file: {path.name}. Run `python train.py` to create the model artifacts."
            )
    try:
        model = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text())
    except Exception as exc:  # noqa: BLE001 - surface any load problem to the UI
        raise ModelLoadError(f"Could not load model artifacts: {exc}") from exc
    return model, metadata


def feature_keys(metadata: Mapping[str, Any]) -> list[str]:
    return [f["key"] for f in metadata["features"]]


def validate_inputs(raw: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Validate one record.

    Returns (clean_values, warnings). Raises ValidationError listing *all* problems
    (missing, non-numeric, non-finite, negative, implausibly large). Values outside
    the training range but plausible produce a warning, not an error.
    """
    errors: list[str] = []
    warnings: list[str] = []
    clean: dict[str, float] = {}

    for feat in metadata["features"]:
        key, label = feat["key"], feat["label"]
        value = raw.get(key)

        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"{label} is required.")
            continue
        if isinstance(value, bool):
            errors.append(f"{label} must be a number.")
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number (got {value!r}).")
            continue
        if not math.isfinite(number):
            errors.append(f"{label} must be a finite number.")
            continue
        if number < 0:
            errors.append(f"{label} cannot be negative.")
            continue
        if number > feat["max"] * HARD_LIMIT_FACTOR:
            errors.append(
                f"{label} = {number:g} is implausibly large "
                f"(training data ranges {feat['min']:g}-{feat['max']:g})."
            )
            continue
        if number < feat["min"] or number > feat["max"]:
            warnings.append(
                f"{label} = {number:g} is outside the range seen in training "
                f"({feat['min']:g}-{feat['max']:g}); the prediction may be less reliable."
            )
        clean[key] = number

    if errors:
        raise ValidationError(errors)
    return clean, warnings


def _frame(values: Mapping[str, float], metadata: Mapping[str, Any]) -> pd.DataFrame:
    keys = feature_keys(metadata)
    return pd.DataFrame([[values[k] for k in keys]], columns=keys)


def predict(model, metadata: Mapping[str, Any], raw: Mapping[str, Any]) -> Prediction:
    """Validate `raw` and return a Prediction."""
    values, warnings = validate_inputs(raw, metadata)
    p_pos = float(model.predict_proba(_frame(values, metadata))[0, 1])
    names = metadata["class_names"]
    probs = {names[0]: 1.0 - p_pos, names[1]: p_pos}
    label = names[1] if p_pos >= 0.5 else names[0]
    return Prediction(label=label, probability_malignant=p_pos, probabilities=probs, warnings=warnings)


def contributions(model, metadata: Mapping[str, Any], raw: Mapping[str, Any]) -> pd.DataFrame | None:
    """Per-feature contribution to the log-odds of the positive class.

    Only defined for linear models (returns None otherwise). Positive values push
    toward "Malignant", negative toward "Benign".
    """
    clf = model.named_steps.get("clf")
    scaler = model.named_steps.get("scale")
    if clf is None or scaler is None or not hasattr(clf, "coef_"):
        return None
    values, _ = validate_inputs(raw, metadata)
    scaled = scaler.transform(_frame(values, metadata))[0]
    contrib = clf.coef_[0] * scaled
    labels = [f["label"] for f in metadata["features"]]
    df = pd.DataFrame({"Feature": labels, "Contribution": contrib})
    return df.reindex(df["Contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def predict_dataframe(model, metadata: Mapping[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Batch prediction. Every row is validated independently.

    Raises ValidationError for file-level problems (empty, missing columns, too many
    rows). Bad rows do not abort the batch: they get an `error` message and no prediction.
    """
    keys = feature_keys(metadata)
    if df.empty:
        raise ValidationError(["The uploaded file has no data rows."])
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValidationError([f"Missing required column(s): {', '.join(missing)}."])
    if len(df) > MAX_BATCH_ROWS:
        raise ValidationError([f"Too many rows ({len(df)}); the limit is {MAX_BATCH_ROWS}."])

    names = metadata["class_names"]
    rows = []
    for idx, record in df[keys].iterrows():
        try:
            values, warns = validate_inputs(record.to_dict(), metadata)
        except ValidationError as exc:
            rows.append({"row": idx + 2, "prediction": None, "probability_malignant": None,
                         "note": " ".join(exc.errors)})
            continue
        p_pos = float(model.predict_proba(_frame(values, metadata))[0, 1])
        rows.append({
            "row": idx + 2,  # +2: spreadsheet row number (1-based, header is row 1)
            "prediction": names[1] if p_pos >= 0.5 else names[0],
            "probability_malignant": round(p_pos, 4),
            "note": " ".join(warns),
        })
    return pd.DataFrame(rows)
