"""``model`` pipeline: model input table -> train/test split -> classifier -> metrics."""
from kedro.pipeline import Node, Pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=split_data,
                inputs=["model_input_table", "params:model"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            Node(
                func=train_model,
                inputs=["X_train", "y_train", "params:model"],
                outputs="classifier",
                name="train_model_node",
            ),
            Node(
                func=evaluate_model,
                inputs=["classifier", "X_test", "y_test", "params:model"],
                outputs=["metrics", "test_predictions"],
                name="evaluate_model_node",
            ),
        ]
    )
