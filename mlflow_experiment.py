import random

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, log_loss, roc_auc_score

from src.dataset import get_dataset
from src.models.dualemb import DualEmbPredictor


def determine_target(row):
    if row["team_score"] > row["opponent_score"]:
        return 0
    elif row["team_score"] == row["opponent_score"]:
        return 1
    else:
        return 2


dataset = get_dataset()

model_classes = [
    DualEmbPredictor,
]
folds_names = ["brazil", "international"]

folds_train = [dataset[dataset["fold"] != name] for name in folds_names]
folds_test = [dataset[dataset["fold"] == name] for name in folds_names]

results = pd.DataFrame({}, columns=["metric", "model", "fold", "iteration", "value"])

logged = {}

params_to_select = {
    "embedding_dim": list(range(5, 55, 5)),
    "num_epochs": list(range(5, 55, 5)),
    "hidden_dim": list(range(0, 55, 1)),
    "train_batch_size": list(range(16, 254 + 16, 16)),
    "train_learning_rate": [0.001, 0.01, 0.1],
    "update_learning_rate": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
}

while True:
    params = {
        "embedding_dim": random.choice(params_to_select["embedding_dim"]),
        "num_epochs": random.choice(params_to_select["num_epochs"]),
        "hidden_dim": random.choice(params_to_select["hidden_dim"]),
        "train_batch_size": random.choice(params_to_select["train_batch_size"]),
        "train_learning_rate": random.choice(params_to_select["train_learning_rate"]),
        "update_learning_rate": random.choice(params_to_select["update_learning_rate"]),
    }

    if str(params) in logged:
        continue

    # Log metrics to MLflow
    with mlflow.start_run():
        print(str(params))
        for iteration in range(1):
            for model_class in model_classes:
                for fold_train, fold_test, fold_test_name in zip(
                    folds_train, folds_test, folds_names
                ):
                    X_train = fold_train[
                        ["team_id", "opponent_id", "team_at_home", "opponent_at_home"]
                    ]
                    y_train = fold_train[["team_score", "opponent_score"]] / 10.0
                    X_test = fold_test[
                        ["team_id", "opponent_id", "team_at_home", "opponent_at_home"]
                    ]
                    y_test = fold_test[["team_score", "opponent_score"]] / 10.0
                    model = model_class(**params)
                    model.fit(X_train, y_train)
                    pred = model.predict_and_update(X_test, y_test)
                    max_pred = np.argmax(pred, axis=1)
                    target = fold_test.apply(determine_target, axis=1).to_numpy()
                    report = classification_report(
                        target,
                        max_pred,
                        target_names=["win", "draw", "loss"],
                        output_dict=True,
                    )
                    metrics = {
                        "accuracy": report["accuracy"],
                        "log_loss": log_loss(target, pred, labels=[0, 1, 2]),
                        "micro_auc_roc": roc_auc_score(
                            target, pred, average="micro", multi_class="ovr"
                        ),
                        "weighted_precision": report["weighted avg"]["precision"],
                        "weighted_recall": report["weighted avg"]["recall"],
                        "macro_precision": report["macro avg"]["precision"],
                        "macro_recall": report["macro avg"]["recall"],
                    }
                    for key, value in metrics.items():
                        results.loc[len(results)] = {
                            "metric": key,
                            "model": model_class.__name__,
                            "fold": fold_test_name,
                            "iteration": iteration + 1,
                            "value": value,
                        }

        # Log parameters
        mlflow.log_param("embedding_dim", 20)
        mlflow.log_param("num_epochs", 50)
        mlflow.log_param("hidden_dim", 10)
        mlflow.log_param("train_batch_size", 10)
        mlflow.log_param("train_learning_rate", 0.01)
        mlflow.log_param("update_learning_rate", 0.01)

        # Log metrics
        for metric, model, value in (
            results.groupby(["metric", "model"])["value"].mean().reset_index().values
        ):
            mlflow.log_metric(f"{metric}", value)

        print(
            str(results.groupby(["metric", "model"])["value"].mean().reset_index().values)
        )
        logged[str(params)] = 1
