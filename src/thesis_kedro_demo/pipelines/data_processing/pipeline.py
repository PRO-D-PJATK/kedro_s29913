"""``data_processing`` pipeline: raw CSV -> validated -> clean -> model input table."""
from kedro.pipeline import Node, Pipeline

from .nodes import clean_data, engineer_features, validate_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=validate_raw_data,
                inputs=["penguins_raw", "params:data_processing"],
                outputs=["penguins_validated", "data_quality_report"],
                name="validate_raw_data_node",
            ),
            Node(
                func=clean_data,
                inputs=["penguins_validated", "params:data_processing"],
                outputs="penguins_clean",
                name="clean_data_node",
            ),
            Node(
                func=engineer_features,
                inputs=["penguins_clean", "params:model"],
                outputs="model_input_table",
                name="engineer_features_node",
            ),
        ]
    )
