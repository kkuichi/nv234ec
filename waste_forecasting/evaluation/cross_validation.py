"""5-fold krížová validácia s container holdout a temporálnym cutoffom.

Modul poskytuje funkciu run_cross_validation, ktorá vykoná
kompletnú krížovú validáciu nad tréningovou množinou pre všetky
modely z CONFIG.MODELS_TO_COMPARE. Používa rovnakú
stratégiu delenia ako hlavné experimenty (container holdout
s temporálnym cutoffom), takže výsledky sú konzistentné a priamo
interpretovateľné.

CV sa vykonáva v event-based režime (nad detegovanými vývozmi),
teda výsledky sú najbližšie k Experimentu A. Slúži ako dodatočné
overenie stability metrík naprieč foldmi.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from config import CONFIG
from waste_forecasting.data.preprocessing import detect_collections
from waste_forecasting.data.splitting import create_temporal_cv_folds
from waste_forecasting.features.encoding import SmartCategoricalEncoder
from waste_forecasting.features.fourier import add_fourier_features
from waste_forecasting.features.holiday import create_holiday_features
from waste_forecasting.features.spatial import add_geolocation_features
from waste_forecasting.features.temporal import (
    create_event_dataset,
    add_event_features,
    get_feature_columns,
)
from waste_forecasting.models.metrics import calculate_all_metrics
from waste_forecasting.models.training import (
    is_model_available,
    get_default_params,
    predict_model,
    train_model,
)

logger = logging.getLogger(__name__)

__all__ = ['run_cross_validation']


def run_cross_validation(
    df: pd.DataFrame,
    test_containers: set,
    output_dir: str,
) -> pd.DataFrame:
    """Vykonať k-fold krížovú validáciu pre všetky porovnávané modely.

    Funkcia najprv pripraví event-based dataset (analogicky k
    Experimentu A), vylúči testovacie kontajnery a z trénovacej
    množiny vytvorí N_FOLDS foldov. Pre každý fold a každý
    dostupný model natrénuje a vyhodnotí metriky.
    """
    logger.info("=" * 70)
    logger.info(f"{CONFIG.N_FOLDS}-FOLD CROSS-VALIDATION (MULTI-MODEL)")
    logger.info("=" * 70)

    # --- Príprava event-based datasetu ------------------------------
    df = detect_collections(df)
    df_events = create_event_dataset(df)
    df_events = add_event_features(df_events)
    df_events = add_fourier_features(df_events)
    df_events = create_holiday_features(df_events)
    df_events = add_geolocation_features(df_events)
    df_events = df_events.dropna(subset=['target_lag_1', 'target_lag_2', 'target_lag_3'])

    # Vylúčenie testovacích kontajnerov - CV musí prebehnúť iba
    # na tréningovej populácii.
    df_cv = df_events[~df_events['container_id'].isin(test_containers)]
    folds = create_temporal_cv_folds(df_cv, CONFIG.N_FOLDS, CONFIG.SEED, CONFIG.CV_TYPE)

    all_fold_results = []

    for fold_idx, (train_df, val_df) in enumerate(folds):
        logger.info(f"Fold {fold_idx + 1}...")

        # Kódovanie kategoriálnych premenných je fitnuté na trénovacom
        # folde a aplikované na validačnú časť (bez leakage).
        encoder = SmartCategoricalEncoder()
        train_df = encoder.fit_transform(train_df)
        val_df = encoder.transform(val_df)

        feature_cols = get_feature_columns(train_df)
        ohe_cols = encoder.get_all_feature_names()
        feature_cols = feature_cols + [c for c in ohe_cols if c not in feature_cols]
        feature_cols = [c for c in feature_cols if c in train_df.columns]

        X_train = train_df[feature_cols]; y_train = train_df['target'].values
        X_val = val_df[feature_cols]; y_val = val_df['target'].values

        # Pre každý model natrénujeme s predvolenými parametrami.
        # CV tu slúži ako odhad variability, nie ako optimalizácia -
        # preto neladíme hyperparametre.
        for model_name in CONFIG.MODELS_TO_COMPARE:
            if not is_model_available(model_name):
                continue

            params = get_default_params(model_name)
            model, y_pred, _ = train_model(
                model_name, X_train, y_train, X_val, y_val,
                params=params, clip_range=(0, 100),
            )
            if y_pred is None:
                y_pred = predict_model(model, X_val, model_name, (0, 100))

            metrics = calculate_all_metrics(y_val, y_pred, with_ci=False)
            metrics['fold'] = fold_idx + 1
            metrics['model'] = model_name
            all_fold_results.append(metrics)
            logger.info(f"  {model_name}: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")

    results_df = pd.DataFrame(all_fold_results)

    # Agregované výsledky naprieč foldmi - priemer ± smerodajná
    # odchýlka pre RMSE a R² dávajú prehľadný obraz variability.
    logger.info(f"\nCV Results:")
    for model_name in results_df['model'].unique():
        subset = results_df[results_df['model'] == model_name]
        logger.info(f"  {model_name}: RMSE = {subset['rmse'].mean():.2f} ± {subset['rmse'].std():.2f}, "
                    f"R2 = {subset['r2'].mean():.3f} ± {subset['r2'].std():.3f}")

    results_df.to_csv(f'{output_dir}/cross_validation_results.csv', index=False)

    return results_df
