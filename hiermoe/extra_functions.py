import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

def set_seed(seed: int = 666):

    """
    Sets the random seed across Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int, optional): The seed value to use. Defaults to 666.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Visualize:
    """
    Reads saved artefacts from a model_dir (training_info.json, eval_results.json,
    predictions.npz) and produces publication-ready plots.

    Requires trainer.predict(save=True) to have been run so that per-level arrays
    are present in predictions.npz.
    """

    # Colour palette shared across methods
    _TRAIN_COL = "#4C72B0"   # blue  — train
    _VAL_COL   = "#DD8452"   # orange — validation
    _METRIC_COLS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]  # F1 / Acc / Prec / Rec

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)

        train_path = self.model_dir / "training_info.json"
        eval_path  = self.model_dir / "eval_results.json"
        pred_path  = self.model_dir / "predictions.npz"

        self.training_info = json.loads(train_path.read_text()) if train_path.exists() else None
        self.eval_results  = json.loads(eval_path.read_text())  if eval_path.exists()  else None
        self.pred_data     = dict(np.load(pred_path))           if pred_path.exists()  else None

        if self.training_info is None:
            print(f"  [Visualize] training_info.json not found in '{model_dir}' — plot_train unavailable")
        if self.eval_results is None or self.pred_data is None:
            print(f"  [Visualize] eval_results.json / predictions.npz not found — plot_pred unavailable")

    # ------------------------------------------------------------------ #
    #  Training curves                                                     #
    # ------------------------------------------------------------------ #

    def plot_train(self):
        """
        Three side-by-side line charts (Loss / Accuracy / F1),
        one line for train (blue) and one for validation (orange).
        Best validation epoch is marked with a vertical dashed line
        and a diamond marker on the validation curve.
        Saves train_metrics.png to model_dir.
        """
        if self.training_info is None:
            print("training_info.json not found — skipping plot_train.")
            return

        m          = self.training_info["training_metrics"]
        best_epoch = self.training_info["results"]["best_epoch"]
        n_epochs   = len(m["train_loss"])
        epochs     = np.arange(1, n_epochs + 1)

        metric_cfgs = [
            ("Loss",     "train_loss", "valid_loss"),
            ("Accuracy", "train_acc",  "valid_acc"),
            ("F1",       "train_f1",   "valid_f1"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Training Metrics per Epoch", fontsize=14, fontweight="bold")

        for ax, (title, train_key, val_key) in zip(axes, metric_cfgs):
            train_vals = np.array(m[train_key])
            val_vals   = np.array(m[val_key])

            ax.plot(epochs, train_vals, color=self._TRAIN_COL, linewidth=2,
                    marker="o", markersize=4, label="Train")
            ax.plot(epochs, val_vals,   color=self._VAL_COL,   linewidth=2,
                    marker="o", markersize=4, label="Validation")

            # Mark the best validation epoch
            if best_epoch is not None and 1 <= best_epoch <= n_epochs:
                bi = best_epoch - 1
                ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.5,
                           label=f"Best ep {best_epoch}", zorder=3)
                ax.scatter([best_epoch], [val_vals[bi]], color="black",
                           marker="D", s=60, zorder=5)

            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Epoch")
            ax.set_xticks(epochs)
            ax.set_xticklabels(epochs, fontsize=7,
                               rotation=45 if n_epochs > 20 else 0)
            ax.legend(fontsize=9)
            if title != "Loss":
                ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = self.model_dir / "train_metrics.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()

    # ------------------------------------------------------------------ #
    #  Prediction results                                                  #
    # ------------------------------------------------------------------ #

    def plot_pred(self):
        """
        For each hierarchy level:
          1. Confusion matrix — raw counts (left) and normalised / recall (right).
             An "Other" column captures predictions that landed at the wrong depth.
          2. Per-class bar chart — Acc / F1 / Precision / Recall, n annotated.

        Also calls plot_level_comparison() for an overall cross-level summary.
        Saves PNG files to model_dir.
        """
        if self.eval_results is None or self.pred_data is None:
            print("eval_results.json or predictions.npz not found — skipping plot_pred.")
            return

        level_results = self.eval_results["level_results"]

        for depth_str in sorted(level_results, key=int):
            depth     = int(depth_str)
            per_class = level_results[depth_str]["per_class"]

            classes     = sorted(per_class.items(), key=lambda x: x[1]["node_id"])
            class_names = [c[0] for c in classes]
            node_ids    = [c[1]["node_id"] for c in classes]
            nid_to_idx  = {nid: i for i, nid in enumerate(node_ids)}
            n           = len(class_names)

            preds_key = f"level_{depth}_preds"
            trues_key = f"level_{depth}_trues"

            if preds_key not in self.pred_data:
                print(f"  Level {depth} arrays missing — re-run trainer.predict(save=True).")
                continue

            preds = self.pred_data[preds_key].astype(int)
            trues = self.pred_data[trues_key].astype(int)

            # Build (n_true × n_pred+1) confusion matrix; last col = "Other"
            cm          = np.zeros((n, n + 1), dtype=int)
            col_labels  = class_names + ["Other"]
            for t, p in zip(trues, preds):
                ri = nid_to_idx.get(t, -1)
                ci = nid_to_idx.get(p, n)   # unknown pred → "Other"
                if ri >= 0:
                    cm[ri, ci] += 1

            # Drop "Other" column if empty
            if cm[:, n].sum() == 0:
                cm         = cm[:, :n]
                col_labels = class_names
            n_cols = cm.shape[1]

            # --- Confusion matrices ---
            fig, axes = plt.subplots(1, 2, figsize=(max(10, n_cols * 1.6 + 3), max(5, n * 1.2)))
            fig.suptitle(f"Level {depth} — Confusion Matrix", fontsize=13, fontweight="bold")

            for ax, data, title, fmt, vmax in [
                (axes[0], cm,                          "Counts",                    "d",    None),
                (axes[1], cm / cm.sum(1, keepdims=True).clip(1), "Normalised (recall / row)", ".2f", 1.0),
            ]:
                im = ax.imshow(data, cmap="Blues", aspect="auto",
                               vmin=0, vmax=vmax if vmax else data.max())
                ax.set_xticks(range(n_cols))
                ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
                ax.set_yticks(range(n))
                ax.set_yticklabels(class_names, fontsize=9)
                ax.invert_yaxis()
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title(title)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                thresh = data.max() / 2.
                for i in range(n):
                    for j in range(n_cols):
                        val = data[i, j]
                        ax.text(j, i, format(val, fmt), ha="center", va="center",
                                fontsize=8, color="white" if val > thresh else "black")

            plt.tight_layout()
            out = self.model_dir / f"confusion_matrix_level{depth}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
            plt.show()

            # --- Per-class metric bar chart ---
            self._plot_class_metrics(depth, per_class)

        # --- Cross-level overview ---
        self.plot_level_comparison()

    def _plot_class_metrics(self, depth, per_class):
        """Grouped bar chart: Acc / F1 / Precision / Recall per class at one level."""
        classes     = sorted(per_class.items(), key=lambda x: x[1]["node_id"])
        class_names = [c[0] for c in classes]
        n           = len(class_names)

        met_cfgs = [
            ("Accuracy",  "accuracy",  self._METRIC_COLS[0]),
            ("F1",        "f1",        self._METRIC_COLS[1]),
            ("Precision", "precision", self._METRIC_COLS[2]),
            ("Recall",    "recall",    self._METRIC_COLS[3]),
        ]
        x      = np.arange(n)
        w      = 0.18
        offset = [-1.5, -0.5, 0.5, 1.5]

        fig, ax = plt.subplots(figsize=(max(10, n * 1.6), 5))
        ax.set_title(f"Level {depth} — Per-class Metrics", fontsize=12, fontweight="bold")

        for (label, key, color), off in zip(met_cfgs, offset):
            vals = [per_class[cls][key] for cls in class_names]
            ax.bar(x + off * w, vals, w, label=label, color=color, alpha=0.85)

        # Annotate sample counts below x-axis
        n_samples = [per_class[cls]["n_samples"] for cls in class_names]
        for i, ns in enumerate(n_samples):
            ax.annotate(f"n={ns}", xy=(i, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -28), textcoords="offset points",
                        ha="center", fontsize=8, color="dimgray")

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=9, loc="lower right")

        plt.tight_layout()
        out = self.model_dir / f"class_metrics_level{depth}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()

    def plot_level_comparison(self):
        """
        Grouped bar chart comparing overall Acc / F1 / Precision / Recall
        across all hierarchy levels, with value labels on each bar.
        Saves level_comparison.png to model_dir.
        """
        if self.eval_results is None:
            print("eval_results.json not found — skipping plot_level_comparison.")
            return

        level_results = self.eval_results["level_results"]
        levels        = sorted(level_results, key=int)
        met_cfgs      = [
            ("Accuracy",  "accuracy",  self._METRIC_COLS[0]),
            ("F1",        "f1",        self._METRIC_COLS[1]),
            ("Precision", "precision", self._METRIC_COLS[2]),
            ("Recall",    "recall",    self._METRIC_COLS[3]),
        ]
        x      = np.arange(len(levels))
        w      = 0.18
        offset = [-1.5, -0.5, 0.5, 1.5]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Overall Metrics across Hierarchy Levels", fontsize=12, fontweight="bold")

        for (label, key, color), off in zip(met_cfgs, offset):
            vals = [level_results[l]["overall"][key] for l in levels]
            bars = ax.bar(x + off * w, vals, w, label=label, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Level {l}" for l in levels], fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)

        plt.tight_layout()
        out = self.model_dir / "level_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()

    def plot_class_size_vs_accuracy(self):
        """
        Scatter plot: class sample size vs accuracy for every class across all levels.
        Useful for diagnosing whether low accuracy is driven by data imbalance.
        Each level is a different colour.
        """
        if self.eval_results is None:
            print("eval_results.json not found.")
            return

        level_results = self.eval_results["level_results"]
        level_colors  = ["#4C72B0", "#55A868", "#C44E52"]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title("Class Size vs Accuracy across Hierarchy Levels",
                     fontsize=12, fontweight="bold")

        for (depth_str, color) in zip(sorted(level_results, key=int), level_colors):
            depth     = int(depth_str)
            per_class = level_results[depth_str]["per_class"]
            for cls_name, metrics in per_class.items():
                ax.scatter(metrics["n_samples"], metrics["accuracy"],
                           color=color, alpha=0.8, s=80, zorder=3)
                ax.annotate(cls_name, (metrics["n_samples"], metrics["accuracy"]),
                            textcoords="offset points", xytext=(5, 3), fontsize=7)

        # Legend patches
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=c, label=f"Level {d}")
                   for c, d in zip(level_colors, sorted(level_results, key=int))]
        ax.legend(handles=handles, fontsize=9)
        ax.set_xlabel("Number of test samples")
        ax.set_ylabel("Accuracy")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = self.model_dir / "size_vs_accuracy.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()
