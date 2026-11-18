"""Publication figures.

All figures share one style: thin marks, recessive grid, a fixed categorical
palette assigned by entity (summer is always orange, winter always blue; each
model keeps its colour across every figure), and direct labels where they fit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.tree import plot_tree

from battery_heating import config as C
from battery_heating.models import MODEL_LABELS

matplotlib.use("Agg")

# Palette (validated categorical slots and a single-hue sequential ramp).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREY = "#9a9891"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#ffffff"

SEASON_COLOR = {"summer": ORANGE, "winter": BLUE}
MODEL_COLOR = {
    "logistic": BLUE,
    "ridge": BLUE,
    "decision_tree": ORANGE,
    "random_forest": AQUA,
    "hist_gradient_boosting": YELLOW,
}
FEATURE_SET_MARKER = {"exogenous": "o", "exogenous-electrical": "^", "exogenous+thermal": "s"}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

DPI = 200


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.6,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def _season_legend(ax: plt.Axes, **kwargs) -> None:
    handles = [
        plt.Line2D([], [], marker="o", color=SEASON_COLOR[s], linestyle="", markersize=6, label=s.capitalize())
        for s in ("summer", "winter")
    ]
    ax.legend(handles=handles, **kwargs)


# --------------------------------------------------------------------------- #
# Descriptive figures
# --------------------------------------------------------------------------- #
def fig_heater_signal_flag(flag: pd.DataFrame, out_dir: Path) -> Path:
    """The original ``Heater Signal`` flag is constant within each of the 70 trips."""
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    rng = np.random.default_rng(C.RANDOM_STATE)
    for season, g in flag.groupby(C.SEASON):
        jitter = rng.uniform(-0.12, 0.12, len(g))
        ax.scatter(
            g["ambient_temp_c"],
            g["heater_signal"] + jitter,
            s=34,
            color=SEASON_COLOR[season],
            edgecolor=SURFACE,
            linewidth=0.8,
            alpha=0.9,
            label=season.capitalize(),
        )
    for _, r in flag[flag["heater_signal"] == 0].iterrows():
        ax.annotate(
            r[C.TRIP].replace("Trip", ""),
            (r["ambient_temp_c"], 0),
            xytext=(0, -11),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=INK_2,
        )
    ax.set_yticks([0, 1], ["0 (off)", "1 (on)"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Trip-mean ambient temperature (°C)")
    ax.set_ylabel("Heater Signal flag")
    ax.set_title("Raw Heater Signal flag by trip: constant within every trip, 0 in only 5 summer trips")
    ax.legend(loc="center right")
    return _save(fig, out_dir, "fig01_heater_signal_flag_by_trip")


def fig_trip_overview(summary: pd.DataFrame, out_dir: Path) -> Path:
    """Trip-level heater activity and heating power against ambient temperature."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for season, g in summary.groupby(C.SEASON):
        kw = dict(
            s=36, color=SEASON_COLOR[season], edgecolor=SURFACE, linewidth=0.8, alpha=0.9, label=season.capitalize()
        )
        axes[0].scatter(g["ambient_temp_c"], g["heater_on_share"] * 100, **kw)
        axes[1].scatter(g["ambient_temp_c"], g["heating_power_mean_w"] / 1000, **kw)
    axes[0].set_ylabel("Share of trip with heater on (%)")
    axes[1].set_ylabel("Trip-mean heating power (kW)")
    for ax, letter in zip(axes, "ab"):
        ax.set_xlabel("Trip-mean ambient temperature (°C)")
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    axes[0].legend(loc="center right")
    fig.suptitle("Heater use across the 50 heater-instrumented trips", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig02_trip_overview")


def fig_example_trips(table: pd.DataFrame, trips: list[str], out_dir: Path) -> Path:
    """Time series of heating power, cabin and ambient temperature for example trips."""
    set_style()
    n = len(trips)
    fig, axes = plt.subplots(n, 2, figsize=(9.2, 2.3 * n), sharex="row")
    axes = np.atleast_2d(axes)
    for i, trip in enumerate(trips):
        g = table[table[C.TRIP] == trip]
        t = g[C.TIME_SINCE_START] / 60
        season = g[C.SEASON].iloc[0]
        ax = axes[i, 0]
        ax.fill_between(t, 0, g[C.HEATING_POWER_W] / 1000, color=SEASON_COLOR[season], alpha=0.25, linewidth=0)
        ax.plot(t, g[C.HEATING_POWER_W] / 1000, color=SEASON_COLOR[season])
        ax.set_ylabel("Heating power (kW)")
        ax.set_ylim(0, 7.5)
        ax.set_title(f"{trip} ({season}): heater on {g[C.HEATER_ON].mean() * 100:.0f}% of the time", loc="left")
        ax2 = axes[i, 1]
        ax2.plot(t, g[C.CABIN_T], color=AQUA, label="Cabin")
        ax2.plot(t, g[C.AMBIENT_T], color=GREY, label="Ambient")
        ax2.axhline(g[C.TARGET_CABIN_T].iloc[0], color=INK_2, linestyle="--", linewidth=1, label="Target cabin")
        on = g[C.HEATER_ON].to_numpy().astype(bool)
        ax2.fill_between(t, 0, 1, where=on, color=YELLOW, alpha=0.15, linewidth=0, transform=ax2.get_xaxis_transform())
        ax2.set_ylim(
            min(g[C.AMBIENT_T].min(), g[C.CABIN_T].min()) - 2, max(g[C.CABIN_T].max(), g[C.TARGET_CABIN_T].iloc[0]) + 3
        )
        ax2.set_ylabel("Temperature (°C)")
        ax2.set_title("Temperatures (shaded: heater on)", loc="left")
        if i == 0:
            ax2.legend(loc="lower right", ncol=3)
        if i == n - 1:
            ax.set_xlabel("Time since trip start (min)")
            ax2.set_xlabel("Time since trip start (min)")
    fig.tight_layout()
    return _save(fig, out_dir, "fig03_example_trips")


def fig_target_distributions(table: pd.DataFrame, out_dir: Path) -> Path:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    bins = np.linspace(0, 7.5, 31)
    for season, g in table.groupby(C.SEASON):
        on = g[g[C.HEATER_ON] == 1]
        axes[0].hist(
            on[C.HEATING_POWER_W] / 1000,
            bins=bins,
            color=SEASON_COLOR[season],
            alpha=0.75,
            label=season.capitalize(),
            density=True,
        )
    axes[0].set_xlabel("Heating power while heater on (kW)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].set_title("Heating power is bimodal: ~1.7 kW or full 7 kW", loc="left")

    share = table.groupby([C.SEASON, C.TRIP], observed=True)[C.HEATER_ON].mean().reset_index()
    for j, season in enumerate(("summer", "winter")):
        s = share[share[C.SEASON] == season][C.HEATER_ON] * 100
        rng = np.random.default_rng(j)
        axes[1].scatter(
            rng.uniform(-0.18, 0.18, len(s)) + j,
            s,
            s=28,
            color=SEASON_COLOR[season],
            edgecolor=SURFACE,
            linewidth=0.7,
            alpha=0.9,
        )
        axes[1].hlines(s.median(), j - 0.28, j + 0.28, color=INK, linewidth=1.5)
    axes[1].set_xticks([0, 1], ["Summer\n(12 trips)", "Winter\n(38 trips)"])
    axes[1].set_ylabel("Share of trip with heater on (%)")
    axes[1].set_title("Per-trip heater-on share (bar: median)", loc="left")
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, "fig04_target_distributions")


def fig_feature_correlation(X: pd.DataFrame, out_dir: Path) -> Path:
    set_style()
    corr = X.rank().corr()  # Spearman
    labels = [C.FEATURE_LABELS.get(c, c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    data = np.ma.array(corr.to_numpy(), mask=mask)
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f3f2ef", BLUE])
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(i + 1):
            v = corr.iloc[i, j]
            if abs(v) >= 0.5 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=SURFACE if abs(v) > 0.75 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("Spearman correlation")
    ax.set_title("Rank correlation between exogenous features (|r| >= 0.5 labelled)", loc="left")
    return _save(fig, out_dir, "fig05_feature_correlation")


# --------------------------------------------------------------------------- #
# Model performance figures
# --------------------------------------------------------------------------- #
def fig_model_comparison(summary: pd.DataFrame, out_dir: Path) -> Path:
    """Dot-and-whisker comparison of cross-validated performance."""
    set_style()
    panels = [
        ("heater_on", "roc_auc", "ROC AUC (held-out trips)", (0.5, 1.0)),
        ("heater_on", "balanced_accuracy", "Balanced accuracy", (0.5, 1.0)),
        ("heating_power", "r2", "R² (held-out trips)", (None, 1.0)),
        ("heating_power", "rmse", "RMSE (W)", (0, None)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.8))
    feature_sets = list(C.FEATURE_SETS)
    models_cls = ["logistic", "decision_tree", "random_forest", "hist_gradient_boosting"]
    models_reg = ["ridge", "decision_tree", "random_forest", "hist_gradient_boosting"]
    for ax, (task, metric, title, lim) in zip(axes, panels):
        models = models_cls if task == "heater_on" else models_reg
        sub = summary[summary["task"] == task]
        ypos = []
        ylab = []
        y = 0
        for m in models:
            for fs in feature_sets:
                r = sub[(sub["model"] == m) & (sub["feature_set"] == fs)]
                if r.empty:
                    continue
                mean, sd = r[f"{metric}_mean"].iloc[0], r[f"{metric}_sd"].iloc[0]
                marker = FEATURE_SET_MARKER[fs]
                ax.errorbar(
                    mean,
                    y,
                    xerr=sd,
                    fmt=marker,
                    color=MODEL_COLOR[m],
                    markersize=6,
                    capsize=2,
                    elinewidth=1,
                    markeredgecolor=SURFACE,
                )
                ypos.append(y)
                ylab.append(f"{MODEL_LABELS[m]}\n{C.FEATURE_SET_LABELS[fs]}")
                y += 1
            y += 0.6
        ax.set_yticks(ypos, ylab, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlim(*lim)
        ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D(
            [],
            [],
            marker=FEATURE_SET_MARKER[fs],
            linestyle="",
            color=INK_2,
            label=f"Features: {C.FEATURE_SET_LABELS[fs]}",
        )
        for fs in feature_sets
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Cross-validated performance (mean ± SD over 5 leave-trips-out folds)", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig06_model_comparison")


def fig_roc_pr(y: np.ndarray, oof: dict[str, np.ndarray], out_dir: Path) -> Path:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    for name, p in oof.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, color=MODEL_COLOR[name], label=MODEL_LABELS[name])
        axes[1].plot(rec, prec, color=MODEL_COLOR[name], label=MODEL_LABELS[name])
    axes[0].plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1)
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("ROC (out-of-fold, exogenous features)", loc="left")
    axes[1].axhline(y.mean(), color=GREY, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall (dashed: prevalence)", loc="left")
    axes[0].legend(loc="lower right")
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.14, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, "fig07_roc_pr_curves")


def fig_pred_vs_actual(y: np.ndarray, pred: np.ndarray, season: np.ndarray, model_label: str, out_dir: Path) -> Path:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    hb = axes[0].hexbin(y / 1000, pred / 1000, gridsize=45, cmap=SEQ_BLUE, mincnt=1, bins="log", linewidths=0.2)
    axes[0].plot([0, 7.5], [0, 7.5], color=INK_2, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Observed heating power (kW)")
    axes[0].set_ylabel("Predicted heating power (kW)")
    axes[0].set_title(f"{model_label}: out-of-fold predictions", loc="left")
    axes[0].set_xlim(0, 7.5)
    axes[0].set_ylim(0, 7.5)
    cb = fig.colorbar(hb, ax=axes[0], shrink=0.8)
    cb.set_label("Seconds (log scale)")
    resid = (pred - y) / 1000
    bins = np.linspace(-6, 6, 49)
    for s in ("summer", "winter"):
        axes[1].hist(
            resid[season == s], bins=bins, color=SEASON_COLOR[s], alpha=0.75, label=s.capitalize(), density=True
        )
    axes[1].axvline(0, color=INK_2, linewidth=1)
    axes[1].set_xlabel("Residual, predicted minus observed (kW)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Residuals by season", loc="left")
    axes[1].legend()
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.14, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, "fig08_predicted_vs_actual")


def fig_per_trip(per_trip_cls: pd.DataFrame, per_trip_reg: pd.DataFrame, out_dir: Path) -> Path:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, df, col, ylabel, title in (
        (axes[0], per_trip_cls, "accuracy", "Accuracy on held-out trip", "Heater on/off: per-trip accuracy"),
        (axes[1], per_trip_reg, "rmse", "RMSE on held-out trip (W)", "Heating power: per-trip RMSE"),
    ):
        for j, season in enumerate(("summer", "winter")):
            s = df[df[C.SEASON] == season]
            rng = np.random.default_rng(j)
            ax.scatter(
                rng.uniform(-0.18, 0.18, len(s)) + j,
                s[col],
                s=28,
                color=SEASON_COLOR[season],
                edgecolor=SURFACE,
                linewidth=0.7,
                alpha=0.9,
            )
            ax.hlines(s[col].median(), j - 0.28, j + 0.28, color=INK, linewidth=1.5)
            worst = s.sort_values(col, ascending=(col == "accuracy")).head(2)
            for _, r in worst.iterrows():
                ax.annotate(
                    r[C.TRIP].replace("Trip", ""),
                    (j + 0.22, r[col]),
                    xytext=(4, 0),
                    textcoords="offset points",
                    fontsize=7,
                    color=INK_2,
                    va="center",
                )
        ax.set_xticks([0, 1], ["Summer", "Winter"])
        ax.set_ylabel(ylabel)
        ax.set_title(title + " (bar: median)", loc="left")
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.14, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir, "fig09_per_trip_performance")


# --------------------------------------------------------------------------- #
# Interpretation figures
# --------------------------------------------------------------------------- #
def fig_permutation_importance(pi_cls: pd.DataFrame, pi_reg: pd.DataFrame, out_dir: Path, top: int = 12) -> Path:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, df, title, color in (
        (axes[0], pi_cls, "Heater on/off (drop in ROC AUC)", BLUE),
        (axes[1], pi_reg, "Heating power (drop in R²)", ORANGE),
    ):
        d = df.head(top).iloc[::-1]
        ax.barh(
            d["label"],
            d["importance_mean"],
            xerr=d["importance_sd"],
            color=color,
            height=0.62,
            error_kw=dict(elinewidth=1, capsize=2, ecolor=INK_2),
        )
        ax.set_xlabel("Permutation importance on held-out trips")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.55, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.suptitle("Gradient boosting, exogenous features: mean ± SD over folds", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig10_permutation_importance")


def fig_shap_summary(
    values: np.ndarray, X_s: pd.DataFrame, task_label: str, name: str, out_dir: Path, top: int = 12
) -> Path:
    """Beeswarm of SHAP values with features ordered by mean |SHAP|."""
    import shap

    set_style()
    X_plot = X_s.rename(columns=C.FEATURE_LABELS)
    cmap = LinearSegmentedColormap.from_list("shap", ["#cde2fb", BLUE, "#0d366b"])
    shap.summary_plot(values, X_plot, max_display=top, show=False, cmap=cmap, plot_size=(8.2, 5.2), alpha=0.6)
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel(f"SHAP value ({task_label})")
    ax.set_title("SHAP values, gradient boosting, exogenous features", loc="left")
    for coll in ax.collections:
        coll.set_sizes([6])
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig_partial_dependence(pd_cls: pd.DataFrame, pd_reg: pd.DataFrame, out_dir: Path) -> Path:
    set_style()
    feats_c = list(dict.fromkeys(pd_cls["feature"]))
    feats_r = list(dict.fromkeys(pd_reg["feature"]))
    n = max(len(feats_c), len(feats_r))
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.0))
    for row, (df, feats, color, ylabel) in enumerate(
        ((pd_cls, feats_c, BLUE, "P(heater on)"), (pd_reg, feats_r, ORANGE, "Heating power (W)"))
    ):
        for j in range(n):
            ax = axes[row, j]
            if j >= len(feats):
                ax.axis("off")
                continue
            d = df[df["feature"] == feats[j]]
            ax.plot(d["x"], d["y"], color=color)
            ax.set_xlabel(d["label"].iloc[0], fontsize=8)
            if j == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, max(df["y"].max() * 1.1, 500))
    fig.suptitle(
        "Partial dependence, gradient boosting, exogenous features (top features by permutation importance)", y=1.01
    )
    fig.tight_layout()
    return _save(fig, out_dir, "fig12_partial_dependence")


def fig_linear_coefficients(coef_cls: pd.DataFrame, coef_reg: pd.DataFrame, out_dir: Path) -> Path:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ax, df, title, xlabel in (
        (axes[0], coef_cls, "Logistic regression: heater on/off", "Change in log-odds per 1 SD"),
        (axes[1], coef_reg, "Ridge regression: heating power", "Change in heating power (W) per 1 SD"),
    ):
        d = df.iloc[::-1]
        colors = [BLUE if v >= 0 else ORANGE for v in d["coef_mean"]]
        ax.barh(
            d["label"],
            d["coef_mean"],
            xerr=d["coef_sd"],
            color=colors,
            height=0.62,
            error_kw=dict(elinewidth=1, capsize=2, ecolor=INK_2),
        )
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.6, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.suptitle("Standardised coefficients (mean ± SD across folds); blue positive, orange negative", y=1.01)
    fig.tight_layout()
    return _save(fig, out_dir, "fig13_linear_coefficients")


def fig_decision_tree(tree, feature_names: list[str], class_names: list[str] | None, out_dir: Path, name: str) -> Path:
    set_style()
    fig, ax = plt.subplots(figsize=(12, 5.2))
    plot_tree(
        tree,
        feature_names=[C.FEATURE_LABELS.get(f, f) for f in feature_names],
        class_names=class_names,
        filled=False,
        rounded=True,
        impurity=False,
        proportion=True,
        precision=1,
        fontsize=7,
        ax=ax,
    )
    ax.grid(False)
    ax.set_title("Depth-3 decision tree fitted to all heater-instrumented trips (exogenous features)", loc="left")
    return _save(fig, out_dir, name)
