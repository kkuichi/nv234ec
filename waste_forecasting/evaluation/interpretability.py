"""Interpretácia modelov: SHAP hodnoty.

SHAP analýzu púšťam na víťazný model z každého experimentu (víťaz je
model s najnižším RMSE). Pri LightGBM a XGBoost treba najprv sanitizovať
názvy stĺpcov, inak nastáva známy nesúlad medzi názvami v DataFrame
a internými názvami príznakov modelu.

Používam shap.TreeExplainer, ktorý pre stromové modely počíta presné
Shapleyove hodnoty a je výrazne rýchlejší ako všeobecné KernelExplainer.
Shapleyova hodnota sama o sebe je priemerný marginálny príspevok
príznaku k predikcii naprieč všetkými možnými kombináciami príznakov.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config import CONFIG
from waste_forecasting.features.encoding import sanitize_feature_names

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = logging.getLogger(__name__)

__all__ = ['compute_shap_importance']


def compute_shap_importance(
    model: object,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str,
    max_samples: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Vypočítať SHAP hodnoty na vzorke testovacej množiny.

    Pre LightGBM a XGBoost modely sa automaticky aplikuje
    sanitizácia názvov stĺpcov. Výsledkom je zoznam príznakov
    zoradených zostupne podľa priemernej absolútnej SHAP hodnoty.
    """
    if not HAS_SHAP:
        logger.warning("SHAP not installed, skipping SHAP analysis")
        return None

    max_samples = max_samples or CONFIG.SHAP_MAX_SAMPLES

    logger.info("Computing SHAP values (max %d samples)...", max_samples)

    try:
        # Pre veľké testovacie množiny náhodne vzorkujeme, aby
        # výpočet zostal v rozumných časových medziach.
        if len(X_val) > max_samples:
            sample_idx = np.random.choice(len(X_val), max_samples, replace=False)
            X_sample = X_val.iloc[sample_idx].copy()
        else:
            X_sample = X_val.copy()

        # Zistíme, či model vyžaduje sanitizáciu názvov stĺpcov.
        model_class_name = type(model).__name__
        needs_sanitize = (
            "XGB" in model_class_name
            or "LGBM" in model_class_name
            or "LightGBM" in model_class_name
        )
        if needs_sanitize:
            X_sample, _ = sanitize_feature_names(X_sample)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Priemerná absolútna SHAP hodnota reprezentuje priemerný
        # vplyv príznaku na predikciu naprieč všetkými vzorkami.
        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "shap_importance": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("shap_importance", ascending=False)

        importance_df.to_csv(f"{output_dir}/shap_importance.csv", index=False)

        logger.info("SHAP analysis complete. Top features:")
        for _, row in importance_df.head(5).iterrows():
            logger.info("  %s: %.4f", row["feature"], row["shap_importance"])

        return importance_df

    except Exception as exc:
        logger.warning("SHAP analysis failed: %s", exc)
        return None
