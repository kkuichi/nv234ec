"""Multi-seed analýza stability výsledkov modelov.

Funkcia run_stability_analysis opakovane púšťa pipeline s rôznymi seedmi
zo zoznamu STABILITY_SEEDS a zbiera metriky z jednotlivých behov. Ak je
smerodajná odchýlka metrík naprieč seedmi nízka, znamená to, že výsledky
nezávisia od konkrétnej voľby náhodného rozdelenia kontajnerov.

Stabilita sa hodnotí na event-based úlohe rovnako ako v Experimente A.
Pre každý seed sa vyberie nová testovacia množina a modely sa trénujú
s predvolenými hyperparametrami (Optuna tuning je tu vypnutý, aby beh
nebol príliš pomalý).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np 
import pandas as pd

from config import CONFIG
from waste_forecasting.data.preprocessing import (
    detect_collections,
    filter_containers_by_min_samples,
)
from waste_forecasting.data.splitting import get_unified_test_containers
from waste_forecasting.features.encoding import SmartCategoricalEncoder
from waste_forecasting.features.fourier import add_fourier_features
from waste_forecasting.features.holiday import create_holiday_features
from waste_forecasting.features.spatial import add_geolocation_features
from waste_forecasting.features.temporal import (
    add_event_features,
    create_event_dataset,
    get_feature_columns,
)
from waste_forecasting.features.weather import add_all_weather_features
from waste_forecasting.models.metrics import calculate_all_metrics
from waste_forecasting.models.training import (
    is_model_available,
    get_default_params,
    predict_model,
    train_model,
)

logger = logging.getLogger(__name__)

__all__ = ['run_stability_analysis']


def run_stability_analysis(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Vykonať multi-seed analýzu stability naprieč všetkými modelmi.

    Pre každý seed zo zoznamu CONFIG.STABILITY_SEEDS
    vytvorí nezávislú testovaciu množinu a natrénuje všetky dostupné
    modely. Agregované výsledky (priemer ± smerodajná odchýlka RMSE
    naprieč seedmi) slúžia ako empirický dôkaz robustnosti pipeline.
    """
    logger.info("=" * 70)
    logger.info("STABILITY ANALYSIS (MULTI-MODEL)")
    logger.info("=" * 70)

    stab_dir = f'{output_dir}/stability'
    os.makedirs(stab_dir, exist_ok=True)

    # --- Jednorazová príprava event-based datasetu ------------------
    # Dataset sa pripravuje iba raz; mení sa len rozdelenie
    # na train/test podľa aktuálneho seedu.
    df_det = detect_collections(df)
    df_events = create_event_dataset(df_det)
    df_events = add_event_features(df_events)
    df_events = add_fourier_features(df_events)
    df_events = create_holiday_features(df_events)
    df_events = add_geolocation_features(df_events)
    df_events = df_events.dropna(subset=['target_lag_1', 'target_lag_2'])

    all_results = []

    for seed in CONFIG.STABILITY_SEEDS:
        # Nezávislá testovacia množina pre každý seed.
        test_containers = get_unified_test_containers(df, seed=seed)
        train_ids = [c for c in df_events['container_id'].unique() if c not in test_containers]
        df_train = df_events[df_events['container_id'].isin(train_ids)].copy()
        df_test = df_events[df_events['container_id'].isin(test_containers)].copy()

        if len(df_test) == 0:
            continue

        encoder = SmartCategoricalEncoder()
        df_train = encoder.fit_transform(df_train)
        df_test = encoder.transform(df_test)

        feature_cols = get_feature_columns(df_train)
        ohe_cols = encoder.get_all_feature_names()
        feature_cols = feature_cols + [c for c in ohe_cols if c not in feature_cols]
        feature_cols = [c for c in feature_cols if c in df_train.columns]

        X_train = df_train[feature_cols]; y_train = df_train['target'].values
        X_test = df_test[feature_cols]; y_test = df_test['target'].values

        # Trénujeme všetky modely s predvolenými parametrami -
        # cieľom je izolovať efekt seedu, nie efekt hyperparametrov.
        for model_name in CONFIG.MODELS_TO_COMPARE:
            if not is_model_available(model_name):
                continue

            params = get_default_params(model_name)
            model, _, _ = train_model(model_name, X_train, y_train, params=params)
            y_pred = predict_model(model, X_test, model_name, (0, 100))
            metrics = calculate_all_metrics(y_test, y_pred, with_ci=False)
            metrics['seed'] = seed
            metrics['model'] = model_name
            all_results.append(metrics)
            logger.info(f"  Seed {seed}, {model_name}: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        logger.info(f"\nStability Results:")
        for model_name in results_df['model'].unique():
            subset = results_df[results_df['model'] == model_name]
            logger.info(f"  {model_name}: RMSE = {subset['rmse'].mean():.2f} ± {subset['rmse'].std():.2f}")

        results_df.to_csv(f'{stab_dir}/stability_results.csv', index=False)

    return results_df
