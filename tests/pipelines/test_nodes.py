"""Unit tests for pipeline nodes (no Kedro session needed — nodes are pure functions)."""
import pandas as pd
import pytest

from thesis_kedro_demo.pipelines.data_processing.nodes import (
    clean_data,
    engineer_features,
    validate_raw_data,
)
from thesis_kedro_demo.pipelines.model.nodes import (
    evaluate_model,
    split_data,
    train_model,
)

DP_PARAMS = {
    "required_columns": ["species", "island", "bill_length_mm", "bill_depth_mm",
                         "flipper_length_mm", "body_mass_g", "sex"],
    "min_rows": 5,
    "max_missing_fraction": 0.5,
    "target": "species",
}
MODEL_PARAMS = {
    "target": "species",
    "features": {
        "numeric": ["bill_length_mm", "bill_depth_mm", "flipper_length_mm",
                    "body_mass_g", "bill_ratio"],
        "categorical": ["island", "sex"],
    },
    "test_size": 0.25,
    "random_state": 0,
    "stratify": True,
    "hyperparameters": {"n_estimators": 5, "max_depth": 3, "min_samples_leaf": 1},
}


@pytest.fixture
def raw():
    rows = []
    for i in range(8):
        rows.append(["Adelie", "Torgersen", 39 + i * 0.1, 18.5, 185, 3700, "male"])
        rows.append(["Gentoo", " Biscoe ", 47 + i * 0.1, 15.0, 215, 5000, "Female"])
    rows.append(["Adelie", "Dream", None, 18.0, 190, 3600, "male"])  # incomplete
    return pd.DataFrame(rows, columns=DP_PARAMS["required_columns"])


def test_validate_passes_and_reports(raw):
    out, report = validate_raw_data(raw, DP_PARAMS)
    assert out.shape == raw.shape
    assert report["status"] == "passed"
    assert report["missing_values_per_column"]["bill_length_mm"] == 1


def test_validate_rejects_missing_column(raw):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_data(raw.drop(columns=["sex"]), DP_PARAMS)


def test_validate_rejects_too_few_rows(raw):
    with pytest.raises(ValueError, match="rows"):
        validate_raw_data(raw.head(3), DP_PARAMS)


def test_clean_drops_nans_and_normalises_text(raw):
    clean = clean_data(raw, DP_PARAMS)
    assert clean.isna().sum().sum() == 0
    assert set(clean["island"]) == {"torgersen", "biscoe"}
    assert set(clean["sex"]) == {"male", "female"}


def test_features_and_model_end_to_end(raw):
    table = engineer_features(clean_data(raw, DP_PARAMS), MODEL_PARAMS)
    assert "bill_ratio" in table.columns
    assert "island_biscoe" in table.columns
    X_train, X_test, y_train, y_test = split_data(table, MODEL_PARAMS)
    assert len(X_train) + len(X_test) == len(table)
    model = train_model(X_train, y_train, MODEL_PARAMS)
    metrics, preds = evaluate_model(model, X_test, y_test, MODEL_PARAMS)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(preds) == len(X_test)
