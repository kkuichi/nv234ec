"""Generovanie vizualizácií a obrázkov pre diplomovú prácu.

Modul obsahuje funkcie, ktoré zo CSV výstupov pipeline vytvárajú
obrázky do textu diplomovej práce aj príloh. Každá väčšia skupina grafov má
vlastnú funkciu a texty sú v slovenčine s diakritikou. Ak niektorý CSV
chýba, preskočí sa iba daný obrázok. Celý beh pokračuje.

Hlavný verejný vstup je generate_all_figures. Vygeneruje súhrnné obrázky
Obr. 1 až 17 použité v texte, rozhodovacie vizualizácie pre Experiment B,
SHAP/permutačné grafy a per-container vizualizácie.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "compute_decision_metrics",
    "generate_decision_figures",
    "generate_per_container_figures",
    "generate_shap_figure_expB",
    "generate_thesis_figures",
    "generate_all_figures",
]


# Farebná paleta je centralizovaná, aby všetky obrázky pôsobili
# jednotne naprieč celým projektom.
COLORS = {
    "hist": "#2E86AB",
    "lgbm": "#A23B72",
    "xgb": "#F18F01",
    "base": "#C73E1D",
    "light": "#95B8D1",
    "accent": "#3B1F2B",
    "actual": "#2E86AB",
    "pred": "#A23B72",
    "error": "#C73E1D",
}

MODEL_SHORT_NAMES = {
    "HistGradientBoosting": "HistGBR",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
}


def _setup_plot_style() -> None:
    """Nastaviť jednotný vizuálny štýl všetkých grafov.

    Používa sa font DejaVu Sans, ktorý má spoľahlivú podporu
    slovenskej diakritiky aj pri neinteraktívnom renderovaní
    cez backend Agg.
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _get_fig_dir(output_dir: str) -> Path:
    """Získať a podľa potreby vytvoriť priečinok pre obrázky."""
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _save_figure(fig: plt.Figure, output_dir: str, filename: str) -> None:
    """Uložiť obrázok do output_dir/figures a korektne ho zavrieť."""
    fig_dir = _get_fig_dir(output_dir)
    fig.savefig(fig_dir / filename, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    logger.info("  Obrázok: %s", filename)


def _read_csv_if_exists(path: str | Path) -> Optional[pd.DataFrame]:
    """Načítať CSV iba v prípade, že existuje."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _safe_percentage_labels(values: pd.Series, decimals: int = 0) -> list[str]:
    """Vytvoriť textové štítky s percentami pre osy alebo legendy."""
    fmt = f"{{:.{decimals}f}} %"
    return [fmt.format(v) for v in values]


def compute_decision_metrics(output_dir: str = "output") -> Dict:
    """Vypočítať rozhodovacie metriky z už uložených predikcií.

    Funkcia využíva už existujúce predictions.csv súbory a z nich
    dopočíta klasifikačné/rozhodovacie pohľady na Experiment B a C.
    Výsledné CSV/JSON súbory sú následne použiteľné aj pri generovaní
    obrázkov.
    """
    results: Dict = {}

    # --- Experiment B: urgentné kontajnery podľa prahu plnosti -----
    pred_b_path = Path(output_dir) / "exp_B" / "predictions.csv"
    if pred_b_path.exists():
        df_b = pd.read_csv(pred_b_path)
        y_true = df_b["actual"].values
        y_pred = df_b["predicted"].values

        rows_b = []
        detail_b = {}

        for threshold in [70, 80, 85, 90]:
            actual_urgent = (y_true >= threshold).astype(int)
            pred_urgent = (y_pred >= threshold).astype(int)

            tp = int(np.sum((pred_urgent == 1) & (actual_urgent == 1)))
            fp = int(np.sum((pred_urgent == 1) & (actual_urgent == 0)))
            fn = int(np.sum((pred_urgent == 0) & (actual_urgent == 1)))
            tn = int(np.sum((pred_urgent == 0) & (actual_urgent == 0)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            rows_b.append(
                {
                    "Prah (%)": threshold,
                    "Precision": round(precision, 3),
                    "Recall": round(recall, 3),
                    "F1": round(f1, 3),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "Skutočne urgentných": tp + fn,
                    "Podiel urgentných (%)": round(100 * (tp + fn) / len(y_true), 1),
                }
            )
            detail_b[f"threshold_{threshold}"] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

        # Úspešnosť výberu top-k kontajnerov pre hlavný prevádzkový prah 85 %.
        threshold_main = 85
        actual_urgent_mask = y_true >= threshold_main
        n_urgent = int(np.sum(actual_urgent_mask))
        sorted_idx = np.argsort(-y_pred)

        topk_rows = []
        for k_pct in [5, 10, 15, 20, 25, 30]:
            k = int(len(y_true) * k_pct / 100)
            hits = int(np.sum(actual_urgent_mask[sorted_idx[:k]]))
            hit_rate = hits / n_urgent if n_urgent > 0 else 0.0
            topk_rows.append(
                {
                    "Top-k (%)": k_pct,
                    "k (vzoriek)": k,
                    "Zachytených urgentných": hits,
                    "Z celkovo urgentných": n_urgent,
                    "Hit rate": round(hit_rate, 3),
                }
            )
        detail_b["top_k_hit_rate"] = topk_rows

        # Retrospektívna simulácia dispečera pri obmedzenej kapacite.
        dispatch_pct = 20
        k_dispatch = int(len(y_true) * dispatch_pct / 100)
        model_catches = int(np.sum(actual_urgent_mask[sorted_idx[:k_dispatch]]))
        random_expected = n_urgent * dispatch_pct / 100
        perfect_catches = int(
            np.sum((y_true >= threshold_main)[np.argsort(-y_true)[:k_dispatch]])
        )
        detail_b["dispatcher_simulation"] = {
            "capacity_pct": dispatch_pct,
            "capacity_n": k_dispatch,
            "model_catches": model_catches,
            "model_recall": model_catches / n_urgent if n_urgent > 0 else 0.0,
            "random_expected_catches": random_expected,
            "random_recall": dispatch_pct / 100,
            "perfect_catches": perfect_catches,
            "lift_vs_random": (
                model_catches / random_expected if random_expected > 0 else 0.0
            ),
            "n_total": len(y_true),
            "n_urgent": n_urgent,
        }

        pd.DataFrame(rows_b).to_csv(
            Path(output_dir) / "exp_B" / "decision_metrics.csv", index=False
        )
        pd.DataFrame(topk_rows).to_csv(
            Path(output_dir) / "exp_B" / "topk_hit_rate.csv", index=False
        )
        with open(Path(output_dir) / "exp_B" / "decision_metrics_detail.json", "w", encoding="utf-8") as f:
            json.dump(detail_b, f, indent=2, ensure_ascii=False)
        results["exp_B"] = detail_b

    # --- Experiment C: urgentnosť podľa počtu dní do prahu ----------
    pred_c_path = Path(output_dir) / "exp_C" / "predictions.csv"
    if pred_c_path.exists():
        df_c = pd.read_csv(pred_c_path)
        y_true_c = df_c["actual"].values
        y_pred_c = df_c["predicted"].values

        rows_c = []
        for days_threshold in [1, 2, 3, 5]:
            actual_urgent = (y_true_c <= days_threshold).astype(int)
            pred_urgent = (y_pred_c <= days_threshold).astype(int)

            tp = int(np.sum((pred_urgent == 1) & (actual_urgent == 1)))
            fp = int(np.sum((pred_urgent == 1) & (actual_urgent == 0)))
            fn = int(np.sum((pred_urgent == 0) & (actual_urgent == 1)))
            tn = int(np.sum((pred_urgent == 0) & (actual_urgent == 0)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            rows_c.append(
                {
                    "Prah (≤ dní)": days_threshold,
                    "Precision": round(precision, 3),
                    "Recall": round(recall, 3),
                    "F1": round(f1, 3),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                }
            )

        pd.DataFrame(rows_c).to_csv(
            Path(output_dir) / "exp_C" / "decision_metrics.csv", index=False
        )
        results["exp_C"] = rows_c

    return results


def generate_decision_figures(output_dir: str = "output") -> None:
    """Vytvoriť obrázky pre rozhodovacie metriky Experimentu B.

    Očakáva, že compute_decision_metrics už vytvorila
    súbor decision_metrics_detail.json.
    """
    _setup_plot_style()
    detail_path = Path(output_dir) / "exp_B" / "decision_metrics_detail.json"
    if not detail_path.exists():
        logger.info("Rozhodovacie metriky pre Experiment B nie sú dostupné.")
        return

    with open(detail_path, encoding="utf-8") as f:
        detail = json.load(f)

    thresholds = [70, 80, 85, 90]
    precisions = [detail[f"threshold_{t}"]["precision"] for t in thresholds]
    recalls = [detail[f"threshold_{t}"]["recall"] for t in thresholds]
    f1s = [detail[f"threshold_{t}"]["f1"] for t in thresholds]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x = np.arange(len(thresholds))
    width = 0.25
    ax.bar(x - width, precisions, width, label="Precision", color=COLORS["hist"], alpha=0.85)
    ax.bar(x, recalls, width, label="Recall", color=COLORS["lgbm"], alpha=0.85)
    ax.bar(x + width, f1s, width, label="F1", color=COLORS["xgb"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} %" for t in thresholds])
    ax.set_xlabel("Prah plnosti pre urgentný kontajner")
    ax.set_ylabel("Skóre")
    ax.set_title("(a) Precision / Recall / F1 pri rôznych prahoch")
    ax.legend(frameon=True, fancybox=False, edgecolor="#ccc")
    ax.set_ylim(0, 1.05)

    ax2 = axes[1]
    topk_data = detail.get("top_k_hit_rate", [])
    if topk_data:
        k_pcts = [item["Top-k (%)"] for item in topk_data]
        hit_rates = [float(item["Hit rate"]) for item in topk_data]
        ax2.plot(k_pcts, hit_rates, "o-", color=COLORS["hist"], lw=2, markersize=7, label="Model")
        ax2.plot(
            k_pcts,
            [k / 100 for k in k_pcts],
            "--",
            color=COLORS["base"],
            lw=1.5,
            label="Náhodný výber",
        )
        ax2.set_xlabel("Top-k % kontajnerov (podľa predikcie)")
        ax2.set_ylabel("Podiel zachytených urgentných")
        ax2.set_title("(b) Top-k hit rate (urgentné = ≥ 85 %)")
        ax2.legend(frameon=True, fancybox=False, edgecolor="#ccc")
        ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    _save_figure(fig, output_dir, "obr_decision_expB.png")

    disp = detail.get("dispatcher_simulation", {})
    if not disp:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    categories = ["Náhodný\nvýber", "Prioritizácia\nmodelom", "Dokonalý\noracle"]
    catches = [
        disp["random_expected_catches"],
        disp["model_catches"],
        disp["perfect_catches"],
    ]
    colors = [COLORS["base"], COLORS["hist"], COLORS["light"]]
    bars = ax.bar(categories, catches, color=colors, alpha=0.85, width=0.5)
    ax.axhline(
        disp["n_urgent"],
        color="gray",
        ls=":",
        lw=1,
        label=f"Celkovo urgentných: {disp['n_urgent']}",
    )
    for bar, val in zip(bars, catches):
        recall = val / disp["n_urgent"] if disp["n_urgent"] > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.0f}\n({recall:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Počet zachytených urgentných kontajnerov")
    ax.set_title("Simulácia dispečerského rozhodovania (kapacita 20 %)")
    ax.legend(frameon=True, fancybox=False, edgecolor="#ccc")
    fig.tight_layout()
    _save_figure(fig, output_dir, "obr_dispatcher_expB.png")


def generate_per_container_figures(
    output_dir: str = "output",
    n_best: int = 3,
    n_worst: int = 3,
) -> Optional[Dict]:
    """Vytvoriť per-container vizualizácie pre Experiment B.

    Funkcia identifikuje najlepšie a najhoršie predikované kontajnery
    podľa RMSE a následne vykreslí ich priebehy v čase.
    """
    _setup_plot_style()

    pred_path = Path(output_dir) / "exp_B" / "predictions.csv"
    if not pred_path.exists():
        logger.info("Vizualizácie po kontajneroch pre Experiment B nie sú dostupné.")
        return None

    df = pd.read_csv(pred_path)
    required_cols = {"container_id", "measured_at_utc", "actual", "predicted"}
    if not required_cols.issubset(df.columns):
        logger.info("predictions.csv neobsahuje všetky stĺpce pre vizualizácie po kontajneroch.")
        return None

    df["measured_at_utc"] = pd.to_datetime(df["measured_at_utc"])

    # Štatistiky po kontajneroch pomáhajú vybrať reprezentatívne série.
    stats_rows = []
    for container_id, group in df.groupby("container_id"):
        if len(group) < 5:
            continue
        ss_res = np.sum((group["actual"] - group["predicted"]) ** 2)
        ss_tot = np.sum((group["actual"] - group["actual"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        stats_rows.append(
            {
                "container_id": container_id,
                "rmse": float(np.sqrt(np.mean((group["actual"] - group["predicted"]) ** 2))),
                "r2": float(r2),
                "n_samples": int(len(group)),
                "trash_type": group["trash_type"].mode().iloc[0] if "trash_type" in group.columns else "N/A",
                "mean_actual": float(group["actual"].mean()),
                "std_actual": float(group["actual"].std(ddof=0)),
                "mean_error": float((group["predicted"] - group["actual"]).mean()),
            }
        )

    if not stats_rows:
        return None

    stats_df = pd.DataFrame(stats_rows).sort_values(["rmse", "n_samples"], ascending=[True, False])
    best_ids = stats_df.head(n_best)["container_id"].tolist()
    worst_ids = stats_df.tail(n_worst)["container_id"].tolist()

    analysis = {
        "best_containers": [],
        "worst_containers": [],
        "summary": {
            "n_containers": int(len(stats_df)),
            "median_rmse": float(stats_df["rmse"].median()),
            "mean_rmse": float(stats_df["rmse"].mean()),
        },
    }

    # Jednoduchá heuristika, aby mal JSON súhrn interpretačnú hodnotu.
    for _, row in stats_df.tail(n_worst).iterrows():
        reasons = []
        if row["std_actual"] < 5:
            reasons.append("nízka variabilita skutočných hodnôt")
        if abs(row["mean_error"]) > 5:
            reasons.append("systematická chyba bias")
        if row["n_samples"] < 20:
            reasons.append("málo vzoriek")
        if not reasons:
            reasons.append("vyššia náhodná chybovosť")
        analysis["worst_containers"].append(
            {
                "container_id": str(row["container_id"]),
                "rmse": float(row["rmse"]),
                "r2": float(row["r2"]),
                "n_samples": int(row["n_samples"]),
                "trash_type": str(row["trash_type"]),
                "mean_actual": float(row["mean_actual"]),
                "std_actual": float(row["std_actual"]),
                "mean_error": float(row["mean_error"]),
                "reasons": reasons,
            }
        )

    for _, row in stats_df.head(n_best).iterrows():
        analysis["best_containers"].append(
            {
                "container_id": str(row["container_id"]),
                "rmse": float(row["rmse"]),
                "r2": float(row["r2"]),
                "n_samples": int(row["n_samples"]),
                "trash_type": str(row["trash_type"]),
            }
        )

    with open(Path(output_dir) / "exp_B" / "container_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    def _plot_containers(container_ids: list, title_prefix: str, filename: str, is_worst: bool) -> None:
        n = len(container_ids)
        fig, axes = plt.subplots(n, 1, figsize=(12, 3.0 * n), sharex=False)
        if n == 1:
            axes = [axes]

        for index, container_id in enumerate(container_ids):
            ax = axes[index]
            group = df[df["container_id"] == container_id].sort_values("measured_at_utc")
            row = stats_df[stats_df["container_id"] == container_id].iloc[0]

            ax.plot(group["measured_at_utc"], group["actual"], "-", color=COLORS["actual"], lw=1.2, alpha=0.8, label="Skutočnosť")
            ax.plot(group["measured_at_utc"], group["predicted"], "-", color=COLORS["pred"], lw=1.2, alpha=0.8, label="Predikcia")
            ax.fill_between(group["measured_at_utc"], group["actual"], group["predicted"], alpha=0.15, color=COLORS["error"])
            ax.set_ylabel("Plnosť (%)")
            ax.set_ylim(-5, 105)

            title = (
                f"Kontajner {container_id} ({row['trash_type']}) - "
                f"RMSE = {row['rmse']:.1f} p. b., R² = {row['r2']:.3f}, "
                f"n = {row['n_samples']}"
            )
            if is_worst:
                info = [item for item in analysis["worst_containers"] if item["container_id"] == str(container_id)]
                if info:
                    title += f"\nDôvody: {', '.join(info[0]['reasons'])}"
            ax.set_title(title, fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            if index == 0:
                ax.legend(loc="upper right", fontsize=9, frameon=True, fancybox=False, edgecolor="#ccc")

        fig.suptitle(f"{title_prefix} (Experiment B)", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        _save_figure(fig, output_dir, filename)

    _plot_containers(best_ids, "Najlepšie predikované kontajnery", "obr_best_containers_expB.png", False)
    _plot_containers(worst_ids, "Najhoršie predikované kontajnery", "obr_worst_containers_expB.png", True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(stats_df["rmse"], bins=40, color=COLORS["actual"], alpha=0.8, edgecolor="white", linewidth=0.5)
    median_rmse = stats_df["rmse"].median()
    ax.axvline(median_rmse, color=COLORS["pred"], ls="--", lw=1.5, label=f"Medián RMSE: {median_rmse:.1f} p. b.")
    if len(stats_df) >= max(n_best, n_worst):
        best_max = stats_df.head(n_best)["rmse"].max()
        worst_min = stats_df.tail(n_worst)["rmse"].min()
        ax.axvspan(0, best_max, alpha=0.1, color="green", label=f"Top {n_best} (RMSE ≤ {best_max:.1f})")
        ax.axvspan(worst_min, stats_df["rmse"].max() * 1.05, alpha=0.1, color="red", label=f"Worst {n_worst} (RMSE ≥ {worst_min:.1f})")
    ax.set_xlabel("RMSE na kontajner (p. b.)")
    ax.set_ylabel("Počet kontajnerov")
    ax.set_title("Distribúcia per-container RMSE (Experiment B)")
    ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="#ccc")
    fig.tight_layout()
    _save_figure(fig, output_dir, "obr_rmse_distribution_expB.png")

    return analysis


def generate_shap_figure_expB(output_dir: str = "output") -> None:
    """Vytvoriť spoločný SHAP/permutačný obrázok pre Experiment B."""
    _setup_plot_style()

    shap_path = Path(output_dir) / "exp_B" / "shap_importance.csv"
    perm_path = Path(output_dir) / "exp_B" / "permutation_importance.csv"

    has_shap = shap_path.exists()
    has_perm = perm_path.exists()
    if not has_shap and not has_perm:
        logger.info("Žiadne SHAP alebo permutačné dáta pre Experiment B nie sú dostupné.")
        return

    n_panels = int(has_shap) + int(has_perm)
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    panel_index = 0
    if has_shap:
        shap_df = pd.read_csv(shap_path).head(20).sort_values("shap_importance", ascending=True)
        ax = axes[panel_index]
        bars = ax.barh(range(len(shap_df)), shap_df["shap_importance"].values, color=COLORS["hist"], alpha=0.85)
        ax.set_yticks(range(len(shap_df)))
        ax.set_yticklabels(shap_df["feature"].values, fontsize=9)
        ax.set_xlabel("Priemerná |SHAP hodnota|")
        ax.set_title("SHAP dôležitosť (Experiment B)")
        for bar, value in zip(bars, shap_df["shap_importance"].values):
            ax.text(value + 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8)
        panel_index += 1

    if has_perm:
        perm_df = pd.read_csv(perm_path).head(15).sort_values("importance_mean", ascending=True)
        ax = axes[panel_index]
        ax.barh(
            range(len(perm_df)),
            perm_df["importance_mean"].values,
            xerr=perm_df["importance_std_across_folds"].values,
            color=COLORS["lgbm"],
            alpha=0.8,
            capsize=3,
        )
        ax.set_yticks(range(len(perm_df)))
        ax.set_yticklabels(perm_df["feature"].values, fontsize=9)
        ax.set_xlabel("Permutačná dôležitosť (ΔRMSE)")
        ax.set_title("Permutačná dôležitosť (Experiment B)")

    fig.tight_layout()
    _save_figure(fig, output_dir, "obr_shap_expB.png")


def generate_thesis_figures(df_raw: pd.DataFrame, output_dir: str = "output") -> None:
    """Vygenerovať hlavné obrázky použité v texte diplomovej práce.

    Funkcia vychádza z pôvodného notebooku a z už existujúcich CSV
    výstupov vytvára ucelenú sadu obrázkov. Každý blok je obalený
    v try/except tak, aby lokálny problém neblokoval ostatné
    vizualizácie.
    """
    _setup_plot_style()
    logger.info("=" * 50)
    logger.info("Generovanie obrázkov pre diplomovú prácu")
    logger.info("=" * 50)

    # --- Obr. 1: distribúcia plnosti --------------------------------
    try:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        values = df_raw["percent_calculated"].dropna()
        ax.hist(values, bins=100, color=COLORS["hist"], edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.axvline(values.mean(), color=COLORS["base"], ls="--", lw=1.5, label=f"Priemer: {values.mean():.1f} %")
        ax.axvline(values.median(), color=COLORS["xgb"], ls="--", lw=1.5, label=f"Medián: {values.median():.0f} %")
        ax.set_xlabel("Plnosť kontajnera (%)")
        ax.set_ylabel("Počet meraní")
        ax.set_title("Distribúcia percentuálnej plnosti kontajnerov")
        ax.legend(frameon=True, fancybox=False, edgecolor="#ccc")
        ax.set_xlim(0, 100)
        _save_figure(fig, output_dir, "obr01_distribucia_plnosti.png")
    except Exception as exc:
        logger.warning("Obr. 1 zlyhal: %s", exc)

    # --- Obr. 2: sezónny trend --------------------------------------
    try:
        df_copy = df_raw.copy()
        df_copy["date"] = pd.to_datetime(df_copy["measured_at_utc"]).dt.date
        daily = df_copy.groupby("date")["percent_calculated"].mean()
        daily.index = pd.to_datetime(daily.index)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(daily.index, daily.values, alpha=0.3, color=COLORS["light"], lw=0.8, label="Denný priemer")
        ax.plot(daily.index, daily.rolling(30, center=True).mean(), color=COLORS["hist"], lw=2, label="30-dňový kĺzavý priemer")
        ax.set_xlabel("Dátum")
        ax.set_ylabel("Priemerná plnosť (%)")
        ax.set_title("Sezónny trend plnosti kontajnerov")
        ax.legend(frameon=True, fancybox=False, edgecolor="#ccc")
        _save_figure(fig, output_dir, "obr02_sezonny_trend.png")
    except Exception as exc:
        logger.warning("Obr. 2 zlyhal: %s", exc)

    # --- Obr. 3: heatmapa deň × hodina -------------------------------
    try:
        df_copy = df_raw.copy()
        df_copy["hour"] = pd.to_datetime(df_copy["measured_at_utc"]).dt.hour
        df_copy["dow"] = pd.to_datetime(df_copy["measured_at_utc"]).dt.dayofweek
        pivot = df_copy.pivot_table(values="percent_calculated", index="hour", columns="dow", aggfunc="mean")
        # Duplicitné "St" v pôvodnom notebooku je nahradené korektným
        # skrátením pre štvrtok.
        pivot.columns = ["Po", "Ut", "St", "Št", "Pi", "So", "Ne"]
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(7))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(0, 24, 2))
        ax.set_yticklabels(range(0, 24, 2))
        ax.set_xlabel("Deň v týždni")
        ax.set_ylabel("Hodina")
        ax.set_title("Priemerná plnosť podľa dňa v týždni a hodiny")
        fig.colorbar(image, ax=ax, shrink=0.8).set_label("Priemerná plnosť (%)")
        _save_figure(fig, output_dir, "obr03_heatmapa_den_hodina.png")
    except Exception as exc:
        logger.warning("Obr. 3 zlyhal: %s", exc)

    # --- Obr. 4: citlivostná analýza detekcie vývozov ----------------
    try:
        sensitivity_df = _read_csv_if_exists(Path(output_dir) / "sensitivity_collection_thresholds.csv")
        if sensitivity_df is not None:
            drop_values = sorted(sensitivity_df["drop_threshold"].unique())
            low_values = sorted(sensitivity_df["low_after_threshold"].unique())
            pivot = sensitivity_df.pivot_table(
                values="n_collections",
                index="drop_threshold",
                columns="low_after_threshold",
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            image = ax.imshow(pivot.values, aspect="auto", cmap="Blues", interpolation="nearest")
            ax.set_xticks(range(len(low_values)))
            ax.set_xticklabels([f"{int(v)} %" for v in low_values])
            ax.set_yticks(range(len(drop_values)))
            ax.set_yticklabels([f"{int(v)} p. b." for v in drop_values])
            ax.set_xlabel("Prah plnosti po zbere")
            ax.set_ylabel("Prah poklesu plnosti")
            ax.set_title("Počet detekovaných zberov podľa prahových hodnôt")
            for row in range(len(drop_values)):
                for col in range(len(low_values)):
                    value = int(pivot.values[row, col])
                    text_color = "white" if value > pivot.values.max() * 0.6 else "black"
                    ax.text(col, row, f"{value:,}", ha="center", va="center", fontsize=8, color=text_color)
            if -30.0 in drop_values and 50.0 in low_values:
                row_index = drop_values.index(-30.0)
                col_index = low_values.index(50.0)
                ax.add_patch(Rectangle((col_index - 0.5, row_index - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.5))
                ax.annotate(
                    "Zvolené\nnastavenie",
                    xy=(col_index, row_index),
                    xytext=(col_index + 1.4, row_index - 1.0),
                    fontsize=9,
                    color="red",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                )
            fig.colorbar(image, ax=ax, shrink=0.8).set_label("Počet detekovaných zberov")
            _save_figure(fig, output_dir, "obr04_citlivost_detekcie.png")
    except Exception as exc:
        logger.warning("Obr. 4 zlyhal: %s", exc)

    # --- Obr. 5: Optuna konvergencia --------------------------------
    try:
        trials_df = _read_csv_if_exists(Path(output_dir) / "optuna_trials" / "exp_A_HistGradientBoosting_trials.csv")
        if trials_df is not None:
            trials_df = trials_df[trials_df["state"] == "COMPLETE"].sort_values("number")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            ax1.scatter(trials_df["number"], trials_df["value"], s=12, alpha=0.4, color=COLORS["light"], zorder=2, label="Jednotlivé trialy")
            ax1.plot(trials_df["number"], trials_df["value"].cummin(), color=COLORS["hist"], lw=2, zorder=3, label="Najlepšia hodnota")
            ax1.set_xlabel("Číslo trialu")
            ax1.set_ylabel("CV RMSE (p. b.)")
            ax1.set_title("Konvergencia optimalizácie")
            ax1.legend(frameon=True, fancybox=False, edgecolor="#ccc")

            param_cols = [col for col in trials_df.columns if col.startswith("params_")]
            correlation_importance = {}
            for col in param_cols:
                try:
                    corr = abs(trials_df[col].astype(float).corr(trials_df["value"]))
                    if not np.isnan(corr):
                        correlation_importance[col.replace("params_", "")] = corr
                except Exception:
                    continue
            sorted_importance = sorted(correlation_importance.items(), key=lambda item: item[1], reverse=True)
            if sorted_importance:
                ax2.barh(range(len(sorted_importance)), [item[1] for item in sorted_importance], color=COLORS["hist"], alpha=0.8)
                ax2.set_yticks(range(len(sorted_importance)))
                ax2.set_yticklabels([item[0] for item in sorted_importance], fontsize=9)
                ax2.invert_yaxis()
                ax2.set_xlabel("|Korelácia| s CV RMSE")
                ax2.set_title("Citlivosť na hyperparametre")
            fig.tight_layout()
            _save_figure(fig, output_dir, "obr05_optuna_optimalizacia.png")
    except Exception as exc:
        logger.warning("Obr. 5 zlyhal: %s", exc)

    # --- Obr. 6-8: Experiment A -------------------------------------
    try:
        pred_a = _read_csv_if_exists(Path(output_dir) / "exp_A" / "predictions.csv")
        if pred_a is not None:
            actual = pred_a["actual"].values
            predicted = pred_a["predicted"].values
            residuals = actual - predicted

            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.scatter(actual, predicted, s=4, alpha=0.15, color=COLORS["hist"], rasterized=True)
            ax.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5, label="Ideálna predikcia")
            ax.set_xlabel("Skutočná plnosť (%)")
            ax.set_ylabel("Predikovaná plnosť (%)")
            ax.set_title("Experiment A - predikované vs. skutočné hodnoty")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect("equal")
            ax.legend(frameon=True, fancybox=False, edgecolor="#ccc")
            _save_figure(fig, output_dir, "obr06_expA_pred_vs_actual.png")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            ax1.scatter(predicted, residuals, s=4, alpha=0.15, color=COLORS["hist"], rasterized=True)
            ax1.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
            ax1.set_xlabel("Predikovaná plnosť (%)")
            ax1.set_ylabel("Reziduál (skut. - pred.)")
            ax1.set_title("Reziduál vs. predikcia")

            ax2.hist(residuals, bins=80, color=COLORS["hist"], edgecolor="white", linewidth=0.3, alpha=0.85)
            ax2.axvline(0, color="black", lw=1, ls="--", alpha=0.5)
            ax2.axvline(np.mean(residuals), color=COLORS["base"], lw=1.5, ls="--", label=f"Priemer: {np.mean(residuals):.2f}")
            ax2.set_xlabel("Reziduál (p. b.)")
            ax2.set_ylabel("Počet")
            ax2.set_title("Distribúcia reziduálov")
            ax2.legend(frameon=True, fancybox=False, edgecolor="#ccc")
            fig.tight_layout()
            _save_figure(fig, output_dir, "obr07_expA_residualy.png")

            if "trash_type" in pred_a.columns:
                rmse_by_type = pred_a.groupby("trash_type").apply(
                    lambda group: np.sqrt(np.mean((group["actual"] - group["predicted"]) ** 2))
                ).sort_values()
                fig, ax = plt.subplots(figsize=(8, 4.5))
                bars = ax.barh(range(len(rmse_by_type)), rmse_by_type.values, color=COLORS["hist"], alpha=0.8)
                ax.set_yticks(range(len(rmse_by_type)))
                ax.set_yticklabels(rmse_by_type.index, fontsize=10)
                ax.set_xlabel("RMSE (p. b.)")
                ax.set_title("RMSE podľa typu odpadu (Experiment A)")
                for bar, value in zip(bars, rmse_by_type.values):
                    ax.text(value + 0.3, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=10)
                _save_figure(fig, output_dir, "obr08_rmse_typ_odpadu.png")
    except Exception as exc:
        logger.warning("Obr. 6-8 zlyhali: %s", exc)

    # --- Obr. 9: Experiment B ---------------------------------------
    try:
        pred_b = _read_csv_if_exists(Path(output_dir) / "exp_B" / "predictions.csv")
        if pred_b is not None:
            fig, ax = plt.subplots(figsize=(6.5, 6))
            pred_b_plot = pred_b.sample(min(20000, len(pred_b)), random_state=42)
            ax.scatter(pred_b_plot["actual"], pred_b_plot["predicted"], s=1.5, alpha=0.05, color=COLORS["lgbm"], rasterized=True)
            ax.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5)
            ax.set_xlabel("Skutočná plnosť o 24 h (%)")
            ax.set_ylabel("Predikovaná plnosť o 24 h (%)")
            ax.set_title("Experiment B - 24-hodinová predikcia")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect("equal")
            _save_figure(fig, output_dir, "obr09_expB_pred_vs_actual.png")
    except Exception as exc:
        logger.warning("Obr. 9 zlyhal: %s", exc)

    # --- Obr. 10: Experiment C --------------------------------------
    try:
        pred_c = _read_csv_if_exists(Path(output_dir) / "exp_C" / "predictions.csv")
        if pred_c is not None:
            fig, ax = plt.subplots(figsize=(6.5, 6))
            pred_c_plot = pred_c.sample(min(20000, len(pred_c)), random_state=42)
            max_val = min(14, pred_c[["actual", "predicted"]].max().max() + 1)
            ax.scatter(pred_c_plot["actual"], pred_c_plot["predicted"], s=2, alpha=0.08, color=COLORS["xgb"], rasterized=True)
            ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.5)
            ax.set_xlabel("Skutočný čas do 85 % (dni)")
            ax.set_ylabel("Predikovaný čas do 85 % (dni)")
            ax.set_title("Experiment C - odhad času do prahu plnosti")
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_aspect("equal")
            _save_figure(fig, output_dir, "obr10_expC_pred_vs_actual.png")
    except Exception as exc:
        logger.warning("Obr. 10 zlyhal: %s", exc)

    # --- Obr. 11: modely oproti referenčnému modelu ------------------
    try:
        model_comp = _read_csv_if_exists(Path(output_dir) / "exp_A" / "model_comparison.csv")
        baseline_comp = _read_csv_if_exists(Path(output_dir) / "baseline_comparison.csv")
        if model_comp is not None:
            models = []
            for _, row in model_comp.iterrows():
                if "Baseline" not in str(row["model"]):
                    models.append((MODEL_SHORT_NAMES.get(row["model"], row["model"]), row["rmse"], "model"))
            if baseline_comp is not None:
                for _, row in baseline_comp.iterrows():
                    name = (
                        str(row["model"])
                        .replace("Same Day-of-Week", "Rovnaký deň")
                        .replace("Last Value (Naive)", "Posledná hodnota")
                        .replace("Seasonal Naive (7-day)", "Sezónny naivný (7d)")
                    )
                    models.append((name, row["rmse"], "baseline"))
            models.sort(key=lambda item: item[1])

            fig, ax = plt.subplots(figsize=(9, 4.5))
            bar_colors = [COLORS["hist"] if item[2] == "model" else COLORS["base"] for item in models]
            bars = ax.barh(range(len(models)), [item[1] for item in models], color=bar_colors, alpha=0.85)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([item[0] for item in models])
            ax.set_xlabel("RMSE (p. b.)")
            ax.set_title("Porovnanie gradient boosting modelov s referenčnými prístupmi")
            ax.invert_yaxis()
            for bar, item in zip(bars, models):
                ax.text(item[1] + 0.3, bar.get_y() + bar.get_height() / 2, f"{item[1]:.2f}", va="center", fontsize=10)
            ax.legend(
                handles=[
                    Patch(facecolor=COLORS["hist"], label="Gradient boosting"),
                    Patch(facecolor=COLORS["base"], label="Referenčné prístupy"),
                ],
                frameon=True,
                fancybox=False,
                edgecolor="#ccc",
            )
            _save_figure(fig, output_dir, "obr11_porovnanie_modelov_baseline.png")
    except Exception as exc:
        logger.warning("Obr. 11 zlyhal: %s", exc)

    # --- Obr. 12: SHAP dôležitosť -----------------------------------
    try:
        shap_df = _read_csv_if_exists(Path(output_dir) / "exp_A" / "shap_importance.csv")
        if shap_df is not None:
            shap_df = shap_df.head(20).sort_values("shap_importance", ascending=True)
            fig, ax = plt.subplots(figsize=(9, 6))
            bars = ax.barh(range(len(shap_df)), shap_df["shap_importance"].values, color=COLORS["hist"], alpha=0.85)
            ax.set_yticks(range(len(shap_df)))
            ax.set_yticklabels(shap_df["feature"].values, fontsize=9)
            ax.set_xlabel("Priemerná |SHAP hodnota|")
            ax.set_title("Top 20 príznakov podľa SHAP dôležitosti (Experiment A)")
            for bar, value in zip(bars, shap_df["shap_importance"].values):
                ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)
            _save_figure(fig, output_dir, "obr12_shap_dolezitost.png")
    except Exception as exc:
        logger.warning("Obr. 12 zlyhal: %s", exc)

    # --- Obr. 13: permutačná dôležitosť -----------------------------
    try:
        perm_df = _read_csv_if_exists(Path(output_dir) / "exp_A" / "permutation_importance.csv")
        if perm_df is not None:
            perm_df = perm_df.head(15).sort_values("importance_mean", ascending=True)
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.barh(
                range(len(perm_df)),
                perm_df["importance_mean"].values,
                xerr=perm_df["importance_std_across_folds"].values,
                color=COLORS["lgbm"],
                alpha=0.8,
                capsize=3,
            )
            ax.set_yticks(range(len(perm_df)))
            ax.set_yticklabels(perm_df["feature"].values, fontsize=9)
            ax.set_xlabel("Permutačná dôležitosť (zvýšenie RMSE)")
            ax.set_title("Top 15 príznakov podľa permutačnej dôležitosti (Experiment A)")
            _save_figure(fig, output_dir, "obr13_permutacna_dolezitost.png")
    except Exception as exc:
        logger.warning("Obr. 13 zlyhal: %s", exc)

    # --- Obr. 14: krížová validácia ---------------------------------
    try:
        cv_df = _read_csv_if_exists(Path(output_dir) / "cross_validation_results.csv")
        if cv_df is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
            for index, (model_name, color) in enumerate(
                zip(["HistGradientBoosting", "LightGBM", "XGBoost"], [COLORS["hist"], COLORS["lgbm"], COLORS["xgb"]])
            ):
                subset = cv_df[cv_df["model"] == model_name]
                if len(subset) == 0:
                    continue
                ax1.scatter([index] * len(subset), subset["rmse"], color=color, s=40, zorder=3, alpha=0.7)
                ax1.errorbar(index, subset["rmse"].mean(), yerr=subset["rmse"].std(), color=color, capsize=8, lw=2, fmt="_", markersize=20, zorder=4)
                ax2.scatter([index] * len(subset), subset["r2"], color=color, s=40, zorder=3, alpha=0.7)
                ax2.errorbar(index, subset["r2"].mean(), yerr=subset["r2"].std(), color=color, capsize=8, lw=2, fmt="_", markersize=20, zorder=4)
            ax1.set_xticks(range(3))
            ax1.set_xticklabels(["HistGBR", "LightGBM", "XGBoost"])
            ax1.set_ylabel("RMSE (p. b.)")
            ax1.set_title("RMSE naprieč foldmi")
            ax2.set_xticks(range(3))
            ax2.set_xticklabels(["HistGBR", "LightGBM", "XGBoost"])
            ax2.set_ylabel("R²")
            ax2.set_title("R² naprieč foldmi")
            fig.suptitle("5-fold krížová validácia (container holdout + temporálny cutoff)", fontsize=13, fontweight="bold", y=1.02)
            fig.tight_layout()
            _save_figure(fig, output_dir, "obr14_krizova_validacia.png")
    except Exception as exc:
        logger.warning("Obr. 14 zlyhal: %s", exc)

    # --- Obr. 15: stabilita pri viacerých seedoch --------------------
    try:
        stability_df = _read_csv_if_exists(Path(output_dir) / "stability" / "stability_results.csv")
        if stability_df is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
            for model_name, color in zip(["HistGradientBoosting", "LightGBM", "XGBoost"], [COLORS["hist"], COLORS["lgbm"], COLORS["xgb"]]):
                subset = stability_df[stability_df["model"] == model_name].sort_values("seed")
                if len(subset) == 0:
                    continue
                ax1.plot(subset["seed"], subset["rmse"], "o-", color=color, label=MODEL_SHORT_NAMES.get(model_name, model_name), markersize=6, lw=1.5)
                ax2.plot(subset["seed"], subset["r2"], "o-", color=color, label=MODEL_SHORT_NAMES.get(model_name, model_name), markersize=6, lw=1.5)
            ax1.set_xlabel("Seed")
            ax1.set_ylabel("RMSE (p. b.)")
            ax1.set_title("RMSE pri rôznych rozdeleniach")
            ax1.legend(frameon=True, fancybox=False, edgecolor="#ccc")
            ax2.set_xlabel("Seed")
            ax2.set_ylabel("R²")
            ax2.set_title("R² pri rôznych rozdeleniach")
            ax2.legend(frameon=True, fancybox=False, edgecolor="#ccc")
            fig.suptitle("Analýza stability (5 rôznych seedov)", fontsize=13, fontweight="bold", y=1.02)
            fig.tight_layout()
            _save_figure(fig, output_dir, "obr15_analyza_stability.png")
    except Exception as exc:
        logger.warning("Obr. 15 zlyhal: %s", exc)

    # --- Obr. 16: ablácia meteorologických príznakov ----------------
    try:
        weather_df = _read_csv_if_exists(Path(output_dir) / "weather_ablation" / "ablation_results.csv")
        if weather_df is not None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            labels = []
            for _, row in weather_df.iterrows():
                n_features = int(row["n_features"])
                if "WITHOUT" in row["variant"]:
                    labels.append(f"Bez počasia\n({n_features} prízn.)")
                else:
                    labels.append(f"S počasím\n({n_features} prízn.)")
            bars = ax.bar(range(len(weather_df)), weather_df["rmse"], color=[COLORS["xgb"], COLORS["hist"]][: len(weather_df)], alpha=0.85, width=0.5)
            ax.set_xticks(range(len(weather_df)))
            ax.set_xticklabels(labels)
            ax.set_ylabel("RMSE (p. b.)")
            ax.set_title("Vplyv meteorologických príznakov na presnosť predikcie")
            for idx, (_, row) in enumerate(weather_df.iterrows()):
                if pd.notna(row.get("rmse_ci_lower")) and pd.notna(row.get("rmse_ci_upper")):
                    ax.errorbar(
                        idx,
                        row["rmse"],
                        yerr=[[row["rmse"] - row["rmse_ci_lower"]], [row["rmse_ci_upper"] - row["rmse"]]],
                        color="black",
                        capsize=5,
                        lw=1.5,
                        fmt="none",
                    )
                ax.text(idx, row["rmse"] + 0.1, f"{row['rmse']:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax.set_ylim(weather_df["rmse"].min() - 1, weather_df["rmse"].max() + 1)
            _save_figure(fig, output_dir, "obr16_weather_ablacia.png")
    except Exception as exc:
        logger.warning("Obr. 16 zlyhal: %s", exc)

    # --- Obr. 17: súhrnný prehľad ------------------------------------
    try:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        configs = [
            ("exp_A", "model_comparison.csv", "Experiment A\n(predikcia pri vývoze)", "rmse", "RMSE (p. b.)"),
            ("exp_B", "metrics.csv", "Experiment B\n(24 h dopredu)", "rmse", "RMSE (p. b.)"),
            ("exp_C", "metrics.csv", "Experiment C\n(čas do 85 %)", None, "MAE (dni)"),
        ]
        for index, (exp_name, file_name, title, metric, ylabel) in enumerate(configs):
            ax = axes[index]
            metrics_df = _read_csv_if_exists(Path(output_dir) / exp_name / file_name)
            if metrics_df is None:
                continue
            metrics_df = metrics_df[~metrics_df["model"].astype(str).str.contains("Baseline", na=False)]
            names = [MODEL_SHORT_NAMES.get(model, model) for model in metrics_df["model"]]
            if metric is None:
                metric_column = "mae_days" if "mae_days" in metrics_df.columns else ("mae" if "mae" in metrics_df.columns else "rmse")
            else:
                metric_column = metric
            values = metrics_df[metric_column].values
            ax.bar(names, values, color=[COLORS["hist"], COLORS["lgbm"], COLORS["xgb"]][: len(names)], alpha=0.85)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            for j, value in enumerate(values):
                ax.text(j, value + value * 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
            ax.set_ylim(0, max(values) * 1.15 if len(values) else 1)
        fig.suptitle("Súhrnné porovnanie troch experimentov", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save_figure(fig, output_dir, "obr17_suhrnny_dashboard.png")
    except Exception as exc:
        logger.warning("Obr. 17 zlyhal: %s", exc)

    # --- Prílohy -----------------------------------------------------
    try:
        if "trash_type" in df_raw.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            for trash_type in df_raw["trash_type"].dropna().unique():
                subset = df_raw[df_raw["trash_type"] == trash_type]["percent_calculated"].dropna()
                ax.hist(subset, bins=50, alpha=0.5, label=trash_type)
            ax.set_xlabel("Plnosť (%)")
            ax.set_ylabel("Počet")
            ax.set_title("Distribúcia plnosti podľa typu odpadu")
            ax.legend(frameon=True, fancybox=False, edgecolor="#ccc")
            _save_figure(fig, output_dir, "obr_P1_typ_odpadu.png")
    except Exception as exc:
        logger.warning("Príloha P1 zlyhala: %s", exc)

    try:
        feature_groups = {
            "Časové": 8,
            "Lag a rolling": 35,
            "Fourierove": 10,
            "Sviatkové": 2,
            "Meteorologické": 79,
            "Priestorové": 48,
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            feature_groups.values(),
            labels=feature_groups.keys(),
            autopct="%1.0f%%",
            colors=[COLORS["hist"], COLORS["lgbm"], COLORS["xgb"], COLORS["base"], COLORS["light"], COLORS["accent"]],
            startangle=90,
            pctdistance=0.85,
        )
        ax.set_title("Rozdelenie 183 príznakov do tematických skupín")
        _save_figure(fig, output_dir, "obr_P3_feature_groups.png")
    except Exception as exc:
        logger.warning("Príloha P3 zlyhala: %s", exc)

    try:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
        plot_configs = [
            (Path(output_dir) / "exp_A" / "model_comparison.csv", ax1, "Experiment A", "rmse", "RMSE (p. b.)"),
            (Path(output_dir) / "exp_B" / "metrics.csv", ax2, "Experiment B", "rmse", "RMSE (p. b.)"),
            (Path(output_dir) / "exp_C" / "metrics.csv", ax3, "Experiment C", "rmse", "RMSE (dni)"),
        ]
        for metrics_path, ax, title, metric, ylabel in plot_configs:
            metrics_df = _read_csv_if_exists(metrics_path)
            if metrics_df is None:
                continue
            metrics_df = metrics_df[~metrics_df["model"].astype(str).str.contains("Baseline", na=False)]
            names = [MODEL_SHORT_NAMES.get(model, model) for model in metrics_df["model"]]
            metric_column = "rmse_days" if "rmse_days" in metrics_df.columns else metric
            values = metrics_df[metric_column].values if metric_column in metrics_df.columns else metrics_df["rmse"].values
            ax.bar(names, values, color=[COLORS["hist"], COLORS["lgbm"], COLORS["xgb"]][: len(names)], alpha=0.85)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            for index, value in enumerate(values):
                ax.text(index, value + value * 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
        fig.suptitle("Porovnanie troch gradient boosting modelov", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save_figure(fig, output_dir, "obr_P5_porovnanie_modelov.png")
    except Exception as exc:
        logger.warning("Príloha P5 zlyhala: %s", exc)

    logger.info("Všetky dostupné obrázky boli vygenerované do %s/figures/.", output_dir)


def generate_all_figures(df_raw: pd.DataFrame, output_dir: str = "output") -> None:
    """Orchestrátor pre všetky typy vizualizácií.

    Funkcia je určená na volanie z main.py po ukončení hlavných
    experimentov a exporte CSV výsledkov.
    """
    compute_decision_metrics(output_dir)
    generate_thesis_figures(df_raw=df_raw, output_dir=output_dir)
    generate_decision_figures(output_dir)
    generate_shap_figure_expB(output_dir)
    generate_per_container_figures(output_dir)
