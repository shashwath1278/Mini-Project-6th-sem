"""
Train and evaluate ML models for plastic-degrading enzyme prediction.

Models:
  1. Random Forest on k-mer features
  2. SVM on k-mer features
  3. Neural Network on ESM-2 embeddings

Evaluation: 5-fold stratified cross-validation with accuracy, precision,
recall, F1-score, and AUC-ROC.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
import joblib


def load_kmer_data(project_root: Path):
    """Load k-mer feature data."""
    data = np.load(project_root / "data" / "processed" / "features_kmer.npz", allow_pickle=True)
    return data["features"], data["labels"]


def load_esm_data(project_root: Path):
    """Load ESM-2 embedding data."""
    data = np.load(project_root / "data" / "embeddings" / "esm2_embeddings.npz", allow_pickle=True)
    return data["embeddings"], data["labels"]


def evaluate_model(model, X, y, model_name: str, n_splits: int = 5):
    """Run stratified k-fold cross-validation and return metrics."""
    # Adjust n_splits if dataset is too small
    min_class_count = min(np.sum(y == 0), np.sum(y == 1))
    if min_class_count < n_splits:
        n_splits = max(2, min_class_count)
        print(f"  (Adjusted to {n_splits}-fold CV due to small dataset)")
    if min_class_count < 2:
        print(f"  SKIPPING {model_name}: need at least 2 samples per class")
        return None, None, None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    metrics = {
        "accuracy": [], "precision": [], "recall": [],
        "f1": [], "auc_roc": [],
    }
    all_y_true, all_y_prob = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Clone model for each fold
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        y_pred = fold_model.predict(X_test)

        # Get probabilities for AUC
        if hasattr(fold_model, "predict_proba"):
            y_prob = fold_model.predict_proba(X_test)[:, 1]
        else:
            y_prob = fold_model.decision_function(X_test)

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        metrics["auc_roc"].append(roc_auc_score(y_test, y_prob))

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

    # Summary
    summary = {}
    for key in metrics:
        vals = metrics[key]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    print(f"\n  {model_name} (5-fold CV):")
    for key, val in summary.items():
        print(f"    {key:>12s}: {val['mean']:.4f} +/- {val['std']:.4f}")

    return summary, np.array(all_y_true), np.array(all_y_prob)


def train_final_model(model, X, y):
    """Train a final model on all data for deployment."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    return model, scaler


def plot_roc_curves(results: dict, output_path: Path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc = data["metrics"]["auc_roc"]["mean"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Plastic-Degrading Enzyme Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n  ROC curve saved to: {output_path}")


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "src" / "models" / "saved"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Model Training & Evaluation")
    print("=" * 60)

    # --- Load data ---
    try:
        X_kmer, y_kmer = load_kmer_data(project_root)
        print(f"\n  K-mer features: {X_kmer.shape}")
        has_kmer = True
    except FileNotFoundError:
        print("\n  WARNING: K-mer features not found. Run kmer_features.py first.")
        has_kmer = False

    try:
        X_esm, y_esm = load_esm_data(project_root)
        print(f"  ESM-2 embeddings: {X_esm.shape}")
        has_esm = True
    except FileNotFoundError:
        print("  WARNING: ESM-2 embeddings not found. Run esm_embeddings.py first.")
        has_esm = False

    if not has_kmer and not has_esm:
        print("\n  ERROR: No feature data found. Run feature extraction first.")
        return

    results = {}

    # --- Model 1: Random Forest on k-mers ---
    if has_kmer:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        )
        metrics, y_true, y_prob = evaluate_model(rf, X_kmer, y_kmer, "Random Forest (k-mer)")
        if metrics is not None:
            results["RF_kmer"] = {"metrics": metrics, "y_true": y_true, "y_prob": y_prob}

        # Train final model
        rf_final, rf_scaler = train_final_model(
            RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            X_kmer, y_kmer,
        )
        joblib.dump({"model": rf_final, "scaler": rf_scaler}, output_dir / "rf_kmer.joblib")

    # --- Model 2: SVM on k-mers ---
    if has_kmer:
        svm = SVC(kernel="rbf", probability=True, random_state=42)
        metrics, y_true, y_prob = evaluate_model(svm, X_kmer, y_kmer, "SVM (k-mer)")
        if metrics is not None:
            results["SVM_kmer"] = {"metrics": metrics, "y_true": y_true, "y_prob": y_prob}

    # --- Model 3: Neural Network on ESM-2 embeddings ---
    if has_esm:
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
        metrics, y_true, y_prob = evaluate_model(mlp, X_esm, y_esm, "MLP (ESM-2)")
        if metrics is not None:
            results["MLP_esm"] = {"metrics": metrics, "y_true": y_true, "y_prob": y_prob}

        # Train final model
        mlp_final, mlp_scaler = train_final_model(
            MLPClassifier(
                hidden_layer_sizes=(256, 128, 64), activation="relu",
                max_iter=500, random_state=42, early_stopping=True,
                validation_fraction=0.15,
            ),
            X_esm, y_esm,
        )
        joblib.dump({"model": mlp_final, "scaler": mlp_scaler}, output_dir / "mlp_esm.joblib")

    # --- Plot ROC curves ---
    if results:
        plot_roc_curves(results, output_dir / "roc_curves.png")

    # --- Save metrics ---
    metrics_out = {name: data["metrics"] for name, data in results.items()}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n  Models saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
