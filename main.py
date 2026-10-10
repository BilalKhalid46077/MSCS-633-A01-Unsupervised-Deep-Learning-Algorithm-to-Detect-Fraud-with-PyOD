# Bilal Khalid

from pathlib import Path
import time

import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt


def main():
    # Use a fixed seed so the data split is reproducible.
    np.random.seed(42)

    data_path = Path("creditcard.csv")
    output_dir = Path("fraud_detection_outputs")
    output_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            "Place creditcard.csv in the same folder as main.py."
        )

    # Load data and validate the target column.
    data = pd.read_csv(data_path)
    if "Class" not in data.columns:
        raise ValueError("Expected a 'Class' target column.")

    if data.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    X = data.drop(columns=["Class"]).astype("float32")
    y = data["Class"].astype("int64")

    # Stratification preserves the rare fraud proportion in each split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Fit the AutoEncoder only on legitimate training transactions.
    X_train_normal = X_train.loc[y_train == 0]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
    X_test_scaled = np.asarray(X_test_scaled, dtype=np.float32)

    # Configure PyOD's neural-network anomaly detector.
    model = AutoEncoder(
        hidden_neuron_list=[30, 16, 8, 16, 30],
        epoch_num=20,
        batch_size=512,
        lr=0.001,
        contamination=0.01,
        preprocessing=False,
        random_state=42,
        verbose=1
    )

    print("Training AutoEncoder on legitimate transactions...")
    start = time.time()
    model.fit(X_train_scaled)
    print(f"Training time: {time.time() - start:.2f} seconds")

    # Higher scores indicate more unusual transactions.
    scores = model.decision_function(X_test_scaled)
    predictions = model.predict(X_test_scaled).astype(int)
    actual = y_test.to_numpy()

    # Evaluate predictions using the held-out test labels.
    matrix = confusion_matrix(actual, predictions, labels=[0, 1])
    metric_values = {
        "Accuracy": accuracy_score(actual, predictions),
        "Fraud precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "Fraud recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "Fraud F1-score": f1_score(
            actual, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(actual, scores),
        "Average precision": average_precision_score(actual, scores)
    }

    print("\nConfusion matrix (actual rows, predicted columns):")
    print(matrix)
    print("\nClassification report:")
    print(classification_report(
        actual, predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    print("Summary metrics:")
    for name, value in metric_values.items():
        print(f"{name}: {value:.4f}")

    # Save metrics and transaction-level predictions.
    pd.DataFrame([
        {"Metric": name, "Value": value}
        for name, value in metric_values.items()
    ]).to_csv(output_dir / "evaluation_metrics.csv", index=False)

    pd.DataFrame({
        "Actual_Class": actual,
        "Predicted_Class": predictions,
        "Anomaly_Score": scores
    }).to_csv(output_dir / "test_predictions.csv", index=False)

    # Save the confusion matrix plot.
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate", "Fraud"]
    ).plot(values_format="d")
    plt.title("AutoEncoder Fraud Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Save the precision-recall curve.
    PrecisionRecallDisplay.from_predictions(actual, scores)
    plt.title("AutoEncoder - Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close()

    print(f"\nAll output files saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
