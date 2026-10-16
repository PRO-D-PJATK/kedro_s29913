"""Nodes for the ``model`` pipeline: split, train, evaluate.

All randomness is controlled by ``params:model.random_state`` so two runs with
the same configuration produce the same model and the same metrics.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_data(
    table: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the model input table into train and test sets.

    Args:
        table: Output of ``engineer_features``.
        params: ``params:model`` (uses ``target``, ``test_size``,
            ``random_state``, ``stratify``).

    Returns:
        ``X_train, X_test, y_train, y_test`` (targets as one-column DataFrames
        so they can be stored as CSV).
    """
    target = params["target"]
    X = table.drop(columns=[target])
    y = table[[target]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y[target] if params.get("stratify", False) else None,
    )
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict[str, Any]
) -> RandomForestClassifier:
    """Fit a random forest classifier with hyperparameters from config.

    Args:
        X_train: Training features.
        y_train: Training target (one column).
        params: ``params:model`` (uses ``hyperparameters``, ``random_state``).

    Returns:
        The fitted classifier (saved by the catalog as ``classifier``).
    """
    model = RandomForestClassifier(
        **params["hyperparameters"], random_state=params["random_state"]
    )
    model.fit(X_train, y_train.iloc[:, 0])
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score the model on the hold-out set.

    Returns:
        A metrics dictionary (saved to ``data/08_reporting/metrics.json``) and
        a table of test-set predictions.
    """
    y_true = y_test.iloc[:, 0]
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "n_test": int(len(y_true)),
        "per_class": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "feature_importances": {
            name: round(float(imp), 4)
            for name, imp in sorted(
                zip(X_test.columns, model.feature_importances_),
                key=lambda kv: kv[1],
                reverse=True,
            )
        },
        "config": {
            "model": type(model).__name__,
            "hyperparameters": params["hyperparameters"],
            "test_size": params["test_size"],
            "random_state": params["random_state"],
        },
    }
    logger.info(
        "Model accuracy on test set: %.3f (macro F1 %.3f)",
        metrics["accuracy"],
        metrics["f1_macro"],
    )

    predictions = pd.DataFrame(
        {"y_true": y_true.to_numpy(), "y_pred": y_pred}, index=y_test.index
    )
    predictions["correct"] = predictions["y_true"] == predictions["y_pred"]
    return metrics, predictions
