import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import confusion_matrix


def get_results_table(folder_path: str) -> pd.DataFrame:
    """
    Build a summary table of evaluation results for all models in a dataset folder.

    Macro and micro metrics are computed at the leaf (deepest hierarchy) level only.
    Hierarchical metrics are averaged across all hierarchy levels.

    Macro accuracy  : mean of per-class accuracies at the leaf level
    Micro accuracy  : overall (correct / total) accuracy at the leaf level
    Hier  accuracy  : mean of overall accuracy across every hierarchy level
    Macro F1        : mean of per-class F1 at the leaf level (already stored as overall.f1)
    Micro F1        : sample-weighted mean of per-class F1 at the leaf level
    Hier  F1        : mean of overall macro-F1 across every hierarchy level

    Args:
        folder_path: 'mnr' or 'whoi', or an explicit path to the training_result subfolder.

    Returns:
        pd.DataFrame with columns:
            backbone, expert,
            macro_accuracy, micro_accuracy, hier_accuracy,
            macro_f1,       micro_f1,       hier_f1
    """
    if folder_path in ("mnr", "whoi"):
        base = os.path.join(os.path.dirname(__file__), folder_path)
    else:
        base = folder_path

    dataset = os.path.basename(base)

    rows = []
    for run_dir in sorted(os.listdir(base)):
        eval_path = os.path.join(base, run_dir, "eval_results.json")
        if not os.path.isfile(eval_path):
            continue

        backbone, expert = _parse_run_name(run_dir, dataset)

        with open(eval_path) as f:
            data = json.load(f)

        level_results = data["level_results"]
        levels = sorted(level_results.keys(), key=int)
        deepest = levels[-1]

        # --- Macro / Micro at leaf (deepest) level ---
        leaf = level_results[deepest]
        per_class = leaf["per_class"]

        macro_accuracy = sum(c["accuracy"] for c in per_class.values()) / len(per_class)
        micro_accuracy = leaf["overall"]["accuracy"]
        macro_f1 = leaf["overall"]["f1"]  # stored value is already macro-averaged

        total_samples = sum(c["n_samples"] for c in per_class.values())
        micro_f1 = sum(c["f1"] * c["n_samples"] for c in per_class.values()) / total_samples

        # --- Hierarchical metrics (averaged across all levels) ---
        hier_accuracy = sum(level_results[l]["overall"]["accuracy"] for l in levels) / len(levels)
        hier_f1 = sum(level_results[l]["overall"]["f1"] for l in levels) / len(levels)

        rows.append({
            "backbone":        backbone,
            "expert":          expert,
            "macro_accuracy":  round(macro_accuracy, 4),
            "micro_accuracy":  round(micro_accuracy, 4),
            "hier_accuracy":   round(hier_accuracy, 4),
            "macro_f1":        round(macro_f1, 4),
            "micro_f1":        round(micro_f1, 4),
            "hier_f1":         round(hier_f1, 4),
        })

    return pd.DataFrame(rows, columns=[
        "backbone", "expert",
        "macro_accuracy", "micro_accuracy", "hier_accuracy",
        "macro_f1",       "micro_f1",       "hier_f1",
    ])


def plot_results_table(folder_path: str, save_path: str = None) -> plt.Figure:
    """
    Render the results table as a matplotlib figure with best values per column highlighted.

    Args:
        folder_path: 'mnr' or 'whoi', or an explicit path to the training_result subfolder.
        save_path:   Optional file path to save the figure (e.g. 'results.png').

    Returns:
        matplotlib Figure.
    """
    df = get_results_table(folder_path)

    metric_cols = ["macro_accuracy", "micro_accuracy", "hier_accuracy",
                   "macro_f1",       "micro_f1",       "hier_f1"]
    col_labels  = ["Backbone", "Expert",
                   "Macro Acc", "Micro Acc", "Hier Acc",
                   "Macro F1",  "Micro F1",  "Hier F1"]

    HIGHLIGHT   = "#2ecc71"   # green — best value in column
    HEADER_BG   = "#2c3e50"   # dark navy
    HEADER_FG   = "white"
    ROW_EVEN    = "#f0f4f8"
    ROW_ODD     = "white"
    TEXT_COLOR  = "#2c3e50"

    n_rows, n_cols = len(df), len(col_labels)
    fig_w = max(12, n_cols * 1.6)
    fig_h = max(2.5, (n_rows + 1) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Build cell text and colours
    cell_text   = []
    cell_colors = []

    # Find the best (max) row index for each metric column
    best = {col: df[col].idxmax() for col in metric_cols}

    for i, row in df.iterrows():
        base_bg  = ROW_EVEN if i % 2 == 0 else ROW_ODD
        text_row  = [row["backbone"], row["expert"]]
        color_row = [base_bg, base_bg]
        for col in metric_cols:
            text_row.append(f"{row[col]:.4f}")
            color_row.append(HIGHLIGHT if i == best[col] else base_bg)
        cell_text.append(text_row)
        cell_colors.append(color_row)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header row
    for col_idx in range(n_cols):
        cell = table[0, col_idx]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight="bold")

    # Style all body cells
    for row_idx in range(1, n_rows + 1):
        for col_idx in range(n_cols):
            cell = table[row_idx, col_idx]
            cell.set_text_props(color=TEXT_COLOR)
            cell.set_edgecolor("#d0d7de")

    dataset_label = os.path.basename(
        os.path.join(os.path.dirname(__file__), folder_path)
        if folder_path in ("mnr", "whoi") else folder_path
    ).upper()
    fig.suptitle(f"HierRoute Results — {dataset_label}", fontsize=13,
                 fontweight="bold", color=TEXT_COLOR, y=0.98)

    legend = mpatches.Patch(facecolor=HIGHLIGHT, edgecolor="grey",
                             label="Best in column")
    ax.legend(handles=[legend], loc="lower right",
              bbox_to_anchor=(1.0, -0.02), fontsize=9, framealpha=0.8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(run_path: str, save_path: str = None) -> plt.Figure:
    """
    Plot a confusion matrix over all leaf classes for a given model run.

    Leaf nodes are all nodes without children in the hierarchy, which may
    appear at different depth levels (e.g. Rotifer at level 2 and Cyclopoid
    at level 3 in MNR). All such nodes are included.

    Args:
        run_path:  Relative path from the training_result directory, e.g. 'mnr/swin_t_cnn',
                   or an explicit absolute path to the run folder.
        save_path: Optional file path to save the figure (e.g. 'cm.png').

    Returns:
        matplotlib Figure.
    """
    base = os.path.join(os.path.dirname(__file__), run_path) \
           if not os.path.isabs(run_path) else run_path

    npz  = np.load(os.path.join(base, "predictions.npz"))
    with open(os.path.join(base, "eval_results.json")) as f:
        eval_data = json.load(f)

    # predictions/targets are raw leaf node IDs — all leaves regardless of depth
    preds = npz["predictions"]
    trues = npz["targets"]

    # Build a combined node_id → class_name mapping across all levels
    level_results = eval_data["level_results"]
    id_to_name = {}
    for level_data in level_results.values():
        for name, info in level_data["per_class"].items():
            id_to_name[info["node_id"]] = name

    # Restrict to node IDs that actually appear in the true labels (i.e. true leaves)
    leaf_ids    = sorted(set(trues.tolist()))
    class_names = [id_to_name[i] for i in leaf_ids]

    cm = confusion_matrix(trues, preds, labels=leaf_ids)
    # Normalise rows to show recall per class
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm, row_sums, where=row_sums != 0).astype(float)

    n = len(class_names)
    fig_size = max(8, n * 0.75)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)

    run_label = os.path.relpath(base, os.path.dirname(__file__))
    ax.set_title(f"Leaf Confusion Matrix", fontsize=12, fontweight="bold", pad=12)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if cm_norm[i, j] > thresh else "#2c3e50")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _parse_run_name(run_dir: str, dataset: str):
    """Extract backbone and expert type from a run directory name."""
    parts = run_dir.split("_")
    expert = parts[-1]
    backbone = "_".join(p for p in parts[:-1] if p != dataset)
    return backbone, expert
