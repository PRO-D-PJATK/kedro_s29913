"""Nodes for the ``data_processing`` pipeline.

Every function is pure: it takes DataFrames / parameter dicts and returns new
objects. No file paths appear here — Kedro loads and saves data through the
catalog (``conf/base/catalog.yml``).
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def validate_raw_data(
    raw: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check that the raw table has the expected shape and columns.

    Args:
        raw: Raw penguins table as loaded from ``penguins_raw``.
        params: ``params:data_processing`` with ``required_columns``,
            ``min_rows``, ``max_missing_fraction`` and ``target``.

    Returns:
        The unchanged table (so downstream nodes only ever see validated data)
        and a data-quality report dictionary.

    Raises:
        ValueError: if a required column is missing, the table is too small,
            a column has too many missing values, or the target has no classes.
    """
    required = params["required_columns"]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Raw data is missing required columns: {missing_cols}")

    n_rows, n_cols = raw.shape
    if n_rows < params["min_rows"]:
        raise ValueError(
            f"Raw data has {n_rows} rows, expected at least {params['min_rows']}"
        )

    missing_fraction = raw[required].isna().mean()
    too_sparse = missing_fraction[missing_fraction > params["max_missing_fraction"]]
    if not too_sparse.empty:
        raise ValueError(
            "Columns exceed max_missing_fraction: " f"{too_sparse.round(3).to_dict()}"
        )

    target = params["target"]
    if raw[target].nunique() < 2:
        raise ValueError(f"Target '{target}' needs at least 2 classes")

    report = {
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "missing_values_per_column": {
            c: int(v) for c, v in raw[required].isna().sum().items()
        },
        "target_distribution": {
            str(k): int(v) for k, v in raw[target].value_counts().items()
        },
        "status": "passed",
    }
    return raw, report


def clean_data(validated: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Keep the required columns, drop incomplete rows and normalise text values.

    Args:
        validated: Output of :func:`validate_raw_data`.
        params: ``params:data_processing`` (uses ``required_columns``).

    Returns:
        A cleaned copy with no missing values and no duplicates.
    """
    df = validated[params["required_columns"]].copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip().str.lower()

    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df


def engineer_features(clean: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Add derived features and one-hot encode categorical columns.

    Args:
        clean: Output of :func:`clean_data`.
        params: ``params:model`` (uses ``features`` and ``target``).

    Returns:
        Model input table: numeric features, one-hot categorical features and
        the target column.
    """
    df = clean.copy()
    df["bill_ratio"] = df["bill_length_mm"] / df["bill_depth_mm"]

    numeric = params["features"]["numeric"]
    categorical = params["features"]["categorical"]
    target = params["target"]

    dummies = pd.get_dummies(df[categorical], prefix=categorical, dtype=int)
    table = pd.concat([df[numeric], dummies, df[[target]]], axis=1)
    return table
