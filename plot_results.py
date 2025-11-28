#!/usr/bin/env python3
"""
PLOT RESULTS FOR BEARING FAULT DIAGNOSIS PROJECT

Generates:
- Bar chart of test accuracy for all models (traditional + deep learning)
- Training curves (accuracy & loss) for 1D CNN and LSTM
- Confusion matrices for each model (using saved predictions)
- Metrics table (Accuracy, F1, ROC-AUC) saved as CSV
- ROC curves (macro-average) for selected models (if probability files exist)

Saved under the results/ directory as PNG files.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------
# Paths and setup
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

sns.set(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------
# Helper: load label map
# ---------------------------------------------------------------------
def load_label_map():
    label_map_path = os.path.join(PROCESSED_DIR, "label_map.json")
    if not os.path.exists(label_map_path):
        print(f"-- label_map.json not found at {label_map_path}")
        return None, None

    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    # Keys are stringified ints if saved from json.dump(dict)
    # Ensure sorted by label index
    label_indices = sorted(int(k) for k in label_map.keys())
    class_names = [label_map[str(idx)] for idx in label_indices]

    return label_indices, class_names


# ---------------------------------------------------------------------
# Helper: load probability arrays for ROC/AUC
# ---------------------------------------------------------------------
def load_proba_for_model(model_name):
    """
    Load per-class probabilities for a given model, if available.

    Expects a CSV file in results/ named "<stem>_proba.csv", where
    stem is model_name with spaces replaced by underscores.

    The CSV should have columns:
      class_0, class_1, ..., true_label

    Returns:
        y_true (np.ndarray), y_proba (np.ndarray) or (None, None) if not found.
    """
    stem = model_name.replace(" ", "_")
    proba_path = os.path.join(RESULTS_DIR, f"{stem}_proba.csv")

    if not os.path.exists(proba_path):
        print(f"   -- No probability file found for {model_name} at {proba_path}")
        return None, None

    df = pd.read_csv(proba_path)
    if "true_label" not in df.columns:
        print(f"   -- 'true_label' column missing in {proba_path}")
        return None, None

    proba_cols = [c for c in df.columns if c.startswith("class_")]
    if not proba_cols:
        print(f"   -- No class_* probability columns in {proba_path}")
        return None, None

    y_true = df["true_label"].values
    y_proba = df[proba_cols].values

    print(f"   -- Loaded probabilities for {model_name} from {proba_path}")
    return y_true, y_proba


# ---------------------------------------------------------------------
# 0. Compute metrics table (Accuracy, F1, ROC-AUC)
# ---------------------------------------------------------------------
def compute_and_save_metrics():
    """
    Compute Accuracy, macro F1, weighted F1, and ROC-AUC (macro/micro when
    probabilities are available) per model using
    traditional_predictions.csv and deep_learning_predictions.csv.

    Saves a consolidated CSV: results/model_metrics.csv
    """
    trad_pred_path = os.path.join(RESULTS_DIR, "traditional_predictions.csv")
    dl_pred_path = os.path.join(RESULTS_DIR, "deep_learning_predictions.csv")

    if not os.path.exists(trad_pred_path) and not os.path.exists(dl_pred_path):
        print("-- No prediction CSVs found for metrics computation.")
        return
    
    print("-- Computing model metrics...")

    dfs = []
    if os.path.exists(trad_pred_path):
        dfs.append(pd.read_csv(trad_pred_path))
    if os.path.exists(dl_pred_path):
        dfs.append(pd.read_csv(dl_pred_path))

    preds_df = pd.concat(dfs, ignore_index=True)

    label_indices, _ = load_label_map()
    if label_indices is None:
        print("-- Could not load label_map.json; ROC-AUC will be skipped.")
        label_indices = None

    metrics_rows = []

    for model_name in preds_df["model"].unique():
        model_df = preds_df[preds_df["model"] == model_name]
        y_true = model_df["true_label"].values
        y_pred = model_df["predicted_label"].values

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        roc_auc_macro = np.nan
        roc_auc_micro = np.nan

        # If we have probabilities and label_map, compute ROC-AUC
        if label_indices is not None:
            y_true_proba, y_proba = load_proba_for_model(model_name)
            if y_true_proba is not None:
                if len(y_true_proba) != len(y_true):
                    print(
                        f"   -- Length mismatch for {model_name} "
                        f"(proba labels={len(y_true_proba)}, preds labels={len(y_true)}); skipping AUC."
                    )
                else:
                    classes_sorted = sorted(label_indices)
                    y_bin = label_binarize(y_true_proba, classes=classes_sorted)

                    try:
                        roc_auc_macro = roc_auc_score(
                            y_bin, y_proba, average="macro", multi_class="ovr"
                        )
                        roc_auc_micro = roc_auc_score(
                            y_bin, y_proba, average="micro", multi_class="ovr"
                        )
                    except Exception as e:
                        print(f"   -- Could not compute ROC-AUC for {model_name}: {e}")

        metrics_rows.append(
            {
                "Model": model_name,
                "Accuracy": acc,
                "F1_macro": f1_macro,
                "F1_weighted": f1_weighted,
                "ROC_AUC_macro": roc_auc_macro,
                "ROC_AUC_micro": roc_auc_micro,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values("Accuracy", ascending=False)

    out_path = os.path.join(RESULTS_DIR, "model_metrics.csv")
    metrics_df.to_csv(out_path, index=False)
    print(f"-- Saved metrics table (Accuracy/F1/ROC-AUC) to {out_path}")


# ---------------------------------------------------------------------
# 1. Bar chart of model accuracies
# ---------------------------------------------------------------------
def plot_model_accuracy_bar():
    """
    Creates a bar chart comparing test accuracy of all models
    (traditional + deep learning). Uses combined_results.csv if present,
    otherwise merges the two separate CSV files.
    """
    combined_path = os.path.join(RESULTS_DIR, "combined_results.csv")
    trad_path = os.path.join(RESULTS_DIR, "traditional_ml_results.csv")
    dl_path = os.path.join(RESULTS_DIR, "deep_learning_results.csv")

    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
    else:
        dfs = []
        if os.path.exists(trad_path):
            dfs.append(pd.read_csv(trad_path))
        if os.path.exists(dl_path):
            dfs.append(pd.read_csv(dl_path))
        if not dfs:
            print("-- No results CSVs found for accuracy bar plot.")
            return
        df = pd.concat(dfs, ignore_index=True)

    print("\n-- Plotting model accuracy comparison...")
    # Sort by accuracy
    df_sorted = df.sort_values("Accuracy", ascending=False)

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df_sorted,
        x="Model",
        y="Accuracy",
        hue="Type",
        dodge=False,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Test Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")

    # Annotate bars with percentage
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height * 100:.1f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = os.path.join(GRAPHS_DIR, "model_accuracy_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"--  Saved model accuracy bar chart to {out_path}")


# ---------------------------------------------------------------------
# 2. Training curves for deep learning models
# ---------------------------------------------------------------------
def plot_dl_training_curves():
    """
    Plots training & validation accuracy and loss for each deep model
    using the *_training_history.csv files.
    """
    history_files = [
        f for f in os.listdir(RESULTS_DIR) if f.endswith("_training_history.csv")
    ]

    if not history_files:
        print("-- No training history CSVs found for deep learning models.")
        return

    print("\n-- Plotting deep learning training curves...")
    for fname in history_files:
        model_key = fname.replace("_training_history.csv", "")
        history_path = os.path.join(RESULTS_DIR, fname)
        history = pd.read_csv(history_path)

        # Some Keras versions may use "accuracy"/"val_accuracy" or "sparse_categorical_accuracy"
        acc_cols = [c for c in history.columns if "acc" in c.lower()]
        loss_cols = [c for c in history.columns if "loss" in c.lower()]

        # Accuracy plot
        plt.figure(figsize=(7, 4))
        for col in acc_cols:
            plt.plot(history.index + 1, history[col], label=col)
        plt.title(f"{model_key} - Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        out_path_acc = os.path.join(GRAPHS_DIR, f"{model_key}_accuracy_history.png")
        plt.savefig(out_path_acc, dpi=300)
        plt.close()
        print(f"-- Saved {model_key} accuracy history to {out_path_acc}")

        # Loss plot
        plt.figure(figsize=(7, 4))
        for col in loss_cols:
            plt.plot(history.index + 1, history[col], label=col)
        plt.title(f"{model_key} - Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        out_path_loss = os.path.join(GRAPHS_DIR, f"{model_key}_loss_history.png")
        plt.savefig(out_path_loss, dpi=300)
        plt.close()
        print(f"-- Saved {model_key} loss history to {out_path_loss}")


# ---------------------------------------------------------------------
# 3. Confusion matrices for each model
# ---------------------------------------------------------------------
def plot_confusion_matrices():
    """
    Generates confusion matrix heatmaps for each model using the
    saved traditional_predictions.csv and deep_learning_predictions.csv.
    """
    trad_pred_path = os.path.join(RESULTS_DIR, "traditional_predictions.csv")
    dl_pred_path = os.path.join(RESULTS_DIR, "deep_learning_predictions.csv")

    if not os.path.exists(trad_pred_path) and not os.path.exists(dl_pred_path):
        print("-- No prediction CSVs found for confusion matrices.")
        return

    label_indices, class_names = load_label_map()
    if label_indices is None:
        print("-- Could not load label_map.json; skipping confusion matrices.")
        return
    
    print("\n-- Plotting confusion matrices for each model...")

    dfs = []
    if os.path.exists(trad_pred_path):
        dfs.append(pd.read_csv(trad_pred_path))
    if os.path.exists(dl_pred_path):
        dfs.append(pd.read_csv(dl_pred_path))

    preds_df = pd.concat(dfs, ignore_index=True)

    for model_name in preds_df["model"].unique():
        model_df = preds_df[preds_df["model"] == model_name]

        y_true = model_df["true_label"].values
        y_pred = model_df["predicted_label"].values

        cm = confusion_matrix(y_true, y_pred, labels=label_indices)
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"Confusion Matrix (normalized) - {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        out_path = os.path.join(GRAPHS_DIR, f"confusion_matrix_{model_name}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"-- Saved confusion matrix for {model_name} to {out_path}")


# ---------------------------------------------------------------------
# 4. ROC curves for selected models
# ---------------------------------------------------------------------
def plot_roc_curves(best_models=None):
    """
    Plots macro-average ROC curves for selected models (if probabilities
    are available). By default, tries ['RandomForest', '1D CNN'].
    """
    if best_models is None:
        best_models = ["RandomForest", "1D CNN"]

    label_indices, _ = load_label_map()
    if label_indices is None:
        print("-- Could not load label_map.json; skipping ROC curves.")
        return
    
    print("\n-- Plotting ROC curves for selected models...")

    classes_sorted = sorted(label_indices)

    for model_name in best_models:
        # Load probabilities
        y_true, y_proba = load_proba_for_model(model_name)
        if y_true is None or y_proba is None:
            print(f"-- Skipping ROC for {model_name} (no probabilities).")
            continue

        if y_proba.shape[1] != len(classes_sorted):
            print(
                f"-- Probability columns for {model_name} do not match number of classes; skipping ROC."
            )
            continue

        # Binarize labels
        y_bin = label_binarize(y_true, classes=classes_sorted)

        # Compute macro-average ROC
        fpr_dict = {}
        tpr_dict = {}
        for i in range(len(classes_sorted)):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])

        all_fpr = np.unique(
            np.concatenate([fpr_dict[i] for i in range(len(classes_sorted))])
        )
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes_sorted)):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= float(len(classes_sorted))

        roc_auc_macro = auc(all_fpr, mean_tpr)

        # Plot
        plt.figure(figsize=(6, 5))
        plt.plot(
            all_fpr,
            mean_tpr,
            label=f"{model_name} macro-average ROC (AUC = {roc_auc_macro:.3f})",
        )
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (macro-average) - {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()

        stem = model_name.replace(" ", "_")
        out_path = os.path.join(GRAPHS_DIR, f"roc_curve_{stem}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"-- Saved ROC curve for {model_name} to {out_path}")

# ---------------------------------------------------------------------
# 5. Plot RandomForest feature importance
# ---------------------------------------------------------------------
def plot_random_forest_feature_importance():
    """
    Loads the trained RandomForest model and plots feature importances
    for the six time-domain features used in the traditional pipeline.

    Saves plot to results/figures/randomforest_feature_importances.png
    """
    model_path = os.path.join(PROJECT_ROOT, "models", "randomforest.pkl")
    if not os.path.exists(model_path):
        print(
            f"-- RandomForest model file not found at {model_path}; skipping feature importance plot."
        )
        return

    print("\n-- Plotting RandomForest feature importances...")
    rf = joblib.load(model_path)

    if not hasattr(rf, "feature_importances_"):
        print("-- Loaded RandomForest has no feature_importances_ attribute; skipping.")
        return

    importances = rf.feature_importances_

    # Your traditional features are (from DataLoader.create_features):
    feature_names = ["mean", "std", "rms", "peak_to_peak", "skewness", "kurtosis"]

    # In case something weird happens, align lengths
    if len(importances) != len(feature_names):
        print(
            f"-- RF importances length ({len(importances)}) != number of features ({len(feature_names)}); "
            "will truncate to the smaller size."
        )
    n = min(len(importances), len(feature_names))
    importances = importances[:n]
    feature_names = feature_names[:n]

    # Sort features by importance (descending)
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]

    plt.figure(figsize=(7, 4))
    sns.barplot(x=sorted_importances, y=sorted_features)
    plt.title("RandomForest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    out_path = os.path.join(GRAPHS_DIR, "randomforest_feature_importances.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"-- Saved RandomForest feature importance plot to {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    print("--  Generating plots from results/ ...")
    compute_and_save_metrics()
    plot_model_accuracy_bar()
    plot_dl_training_curves()
    plot_confusion_matrices()
    plot_random_forest_feature_importance()
    plot_roc_curves(best_models=["RandomForest", "1D CNN"])
    print("--  Plot generation completed.")


if __name__ == "__main__":
    main()
