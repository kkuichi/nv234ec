"""Experiment A - event-based predikcia plnosti v momente vývozu.

Prvý z troch hlavných experimentov. Úlohou je predikovať plnosť
kontajnera tesne pred jeho vývozom. Vzorky sú iba detegované vývozy
(is_collection == 1), cieľovou premennou je hodnota plnosti pred
vývozom (pct_prev).

Postupnosť krokov v pipeline experimentu:

1. Detekcia vývozov cez detect_collections.
2. Zostavenie event datasetu a pridanie lag/rolling príznakov.
3. Doplnenie sezónnych, sviatkových, geolokačných a meteorologických
   príznakov.
4. Filtrovanie kontajnerov s príliš málo vývozmi.
5. Rozdelenie na trénovaciu a testovaciu množinu podľa test_containers.
6. Kódovanie kategoriálnych premenných.
7. Anti-leakage a korelačná kontrola.
8. Tréning všetkých troch modelov s voliteľným Optuna tuningom.
9. Výber víťaza podľa najnižšieho RMSE a výpočet SHAP a permutačnej
   dôležitosti na víťaznom modeli.
10. Export predikcií, porovnania modelov a metrík.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
import pandas as pd

from config import CONFIG
from waste_forecasting.data.preprocessing import (
    check_feature_correlations,
    detect_collections,
    filter_containers_by_min_samples,
    verify_no_data_leakage,
)
from waste_forecasting.data.loading import cleanup_dataframes
from waste_forecasting.evaluation.interpretability import (
    compute_shap_importance,
)
from waste_forecasting.evaluation.weather_ablation import analyze_weather_impact_wrapper as analyze_weather_impact
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
from waste_forecasting.models.tuning import MultiModelTuner

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = logging.getLogger(__name__)

__all__ = ['run_experiment_A']


def run_experiment_A(
    df: pd.DataFrame,
    test_containers: set,
    output_dir: str,
    enable_tuning: bool = None,
    enable_shap: bool = None,
) -> Dict:
    """Spustiť Experiment A: event-based predikciu plnosti kontajnera.

    Funkcia predstavuje kompletný beh experimentu od detekcie vývozov
    po export všetkých výsledkov. Pracuje s kópiou vstupného DataFrame,
    aby nezmenila pôvodné dáta volajúceho.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT A: EVENT-BASED PREDICTION (MULTI-MODEL)")
    logger.info("=" * 70)

    enable_tuning = enable_tuning if enable_tuning is not None else CONFIG.ENABLE_HYPERPARAMETER_TUNING
    enable_shap = enable_shap if enable_shap is not None else CONFIG.ENABLE_SHAP

    exp_dir = f'{output_dir}/exp_A'
    os.makedirs(exp_dir, exist_ok=True)

    # --- Príprava dát a generovanie príznakov -----------------------
    df_det = detect_collections(df)
    df_events = create_event_dataset(df_det)
    df_events = add_event_features(df_events)
    df_events = add_fourier_features(df_events)
    df_events = create_holiday_features(df_events)
    df_events = add_geolocation_features(df_events)
    df_events = add_all_weather_features(df_events, mode="event")

    # Odfiltrujeme kontajnery s málo vývozmi, aby sme neučili model
    # na šumových kontajneroch.
    df_events, filter_info = filter_containers_by_min_samples(
        df_events,
        min_samples=CONFIG.MIN_COLLECTIONS_PER_CONTAINER,
        sample_col='container_id',
        output_dir=exp_dir,
        tag='events',
    )

    cleanup_dataframes(df_det)

    # --- Rozdelenie trénovacej a testovacej množiny ------------------
    train_containers = [c for c in df_events['container_id'].unique() if c not in test_containers]
    df_train_all = df_events[df_events['container_id'].isin(train_containers)].copy()
    df_test = df_events[df_events['container_id'].isin(test_containers)].copy()

    cleanup_dataframes(df_events)

    logger.info(f"Train: {len(train_containers)} containers, {len(df_train_all):,} samples")
    logger.info(f"Test: {len(test_containers)} containers, {len(df_test):,} samples")

    # Uchovanie stĺpca trash_type pred kódovaním pre neskoršiu
    # per-trash-type analýzu predikcií.
    test_trash_types = df_test['trash_type'].values.copy() if 'trash_type' in df_test.columns else None

    # --- Kódovanie kategoriálnych premenných -------------------------
    encoder = SmartCategoricalEncoder()
    df_train_all = encoder.fit_transform(df_train_all)
    df_test = encoder.transform(df_test)

    logger.info(encoder.get_cardinality_report())

    # --- Zostavenie zoznamu príznakov a validácia --------------------
    feature_cols = get_feature_columns(df_train_all)
    ohe_cols = encoder.get_all_feature_names()
    feature_cols = feature_cols + [c for c in ohe_cols if c not in feature_cols]
    feature_cols = [c for c in feature_cols if c in df_train_all.columns]

    logger.info(f"Features: {len(feature_cols)}")
    verify_no_data_leakage(df_train_all, 'target', feature_cols)
    check_feature_correlations(df_train_all, feature_cols)

    X_train = df_train_all[feature_cols]
    y_train = df_train_all['target'].values
    X_test = df_test[feature_cols]
    y_test = df_test['target'].values

    # --- Last-value baseline na identickom filtri vzoriek -----------
    # Baseline predikuje plnosť z predchádzajúceho vývozu (target_lag_1).
    # Vzorky, pre ktoré baseline nemá definovanú hodnotu (prvý vývoz
    # kontajnera v teste), sú odfiltrované - rovnaký filter sa aplikuje
    # aj na hlavný model, aby bolo porovnanie férové.
    baseline_values = df_test['target_lag_1'].values
    valid_mask = ~np.isnan(baseline_values)
    n_total, n_valid = len(y_test), valid_mask.sum()
    logger.info(f"Baseline comparison: {n_valid}/{n_total} samples ({100*n_valid/n_total:.1f}%)")

    y_test_filtered = y_test[valid_mask]
    baseline_filtered = np.clip(baseline_values[valid_mask], 0, 100)
    baseline_metrics = calculate_all_metrics(y_test_filtered, baseline_filtered, with_ci=True)

    # --- Multi-model tréning a evaluácia ----------------------------
    model_results = {}
    primary_model = None
    primary_model_name = None

    for model_name in CONFIG.MODELS_TO_COMPARE:
        if not is_model_available(model_name):
            logger.warning(f"{model_name} not available, skipping")
            continue

        logger.info(f"\n{'─' * 50}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'─' * 50}")

        model_dir = f'{exp_dir}/{model_name.lower()}'
        os.makedirs(model_dir, exist_ok=True)

        # Ladenie hyperparametrov (ak je povolené a Optuna je dostupná).
        optuna_study = None
        if enable_tuning and HAS_OPTUNA:
            tuner = MultiModelTuner(model_name=model_name)
            best_params = tuner.tune(df_train_all, feature_cols)
            optuna_study = tuner.study
        else:
            best_params = get_default_params(model_name)

        # Finálny tréning s nájdenými (alebo predvolenými) parametrami.
        model, _, training_info = train_model(model_name, X_train, y_train, params=best_params)
        training_info['optuna_study'] = optuna_study

        # Predikcia na testovacej množine.
        y_pred_test = predict_model(model, X_test, model_name, (0, 100))
        y_pred_filtered = y_pred_test[valid_mask]

        # Dve sady metrík: na rovnakom filtri ako baseline (pre férové
        # porovnanie) a na celej testovacej množine (pre hlavný reporting).
        test_metrics = calculate_all_metrics(y_test_filtered, y_pred_filtered, with_ci=True)
        test_metrics_all = calculate_all_metrics(y_test, y_pred_test, with_ci=True)

        logger.info(f"  {model_name} (identical filter, n={test_metrics['n']}):")
        logger.info(f"    RMSE:    {test_metrics['rmse']:.2f} [{test_metrics.get('rmse_ci_lower', 0):.2f}, {test_metrics.get('rmse_ci_upper', 0):.2f}]")
        logger.info(f"    R2:      {test_metrics['r2']:.3f}")

        model_results[model_name] = {
            'model': model,
            'y_pred': y_pred_test,
            'metrics': test_metrics,
            'metrics_all': test_metrics_all,
            'best_params': best_params,
            'training_info': training_info,
        }

    # --- Výber víťazného modelu pre SHAP ----------------------------
    primary_model_name = min(model_results, key=lambda k: model_results[k]['metrics']['rmse'])
    primary_model = model_results[primary_model_name]['model']
    logger.info(f"Primary model for SHAP: {primary_model_name} "
                f"(RMSE={model_results[primary_model_name]['metrics']['rmse']:.2f})")

    # --- SHAP analýza víťazného modelu -------------------------------
    shap_importance = None
    if enable_shap and HAS_SHAP and primary_model is not None:
        shap_importance = compute_shap_importance(primary_model, X_train, X_test, feature_cols, exp_dir)

    # --- Porovnanie modelov voči baseline ----------------------------
    comparison_rows = []
    for m_name, m_res in model_results.items():
        row = {'model': m_name}
        row.update(m_res['metrics'])
        comparison_rows.append(row)
    comparison_rows.append({'model': 'Baseline (Last Value)', **baseline_metrics})

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(f'{exp_dir}/model_comparison.csv', index=False)
    logger.info(f"\nModel Comparison (Experiment A):")
    logger.info(comparison_df[['model', 'rmse', 'mae', 'r2']].to_string(index=False))

    # --- Analýza vplyvu počasia --------------------------------------
    weather_analysis = analyze_weather_impact(df_test, 'target', exp_dir)
    weather_feature_count = sum(1 for f in feature_cols if any(w in f.lower() for w in
                               ['temp', 'precip', 'rain', 'humid', 'wind', 'press', 'weather', 'comfort', 'heat', 'chill', 'season']))
    logger.info(f"Weather features used: {weather_feature_count}")

    # --- Uloženie predikcií pre následnú vizualizáciu ----------------
    best_mn = min(model_results, key=lambda k: model_results[k]['metrics']['rmse'])
    pred_df = pd.DataFrame({'actual': y_test, 'predicted': model_results[best_mn]['y_pred']})
    if test_trash_types is not None:
        pred_df['trash_type'] = test_trash_types
    pred_df['container_id'] = df_test['container_id'].values
    pred_df.to_csv(f'{exp_dir}/predictions.csv', index=False)
    logger.info(f"Saved {len(pred_df)} predictions to {exp_dir}/predictions.csv")

    cleanup_dataframes(df_train_all, df_test)

    # Vrátenie metrík víťazného modelu (konzistentné so SHAP a predikciami).
    primary_res = model_results[primary_model_name]
    return {
        'test_metrics': primary_res.get('metrics', {}),
        'test_metrics_all': primary_res.get('metrics_all', {}),
        'baseline_metrics': baseline_metrics,
        'best_params': primary_res.get('best_params', {}),
        'primary_model_name': primary_model_name,
        'n_identical': n_valid,
        'n_total': n_total,
        'model': primary_model,
        'shap_importance': shap_importance,
        'weather_analysis': weather_analysis,
        'n_weather_features': weather_feature_count,
        'model_comparison': {name: res['metrics'] for name, res in model_results.items()},
        'all_model_results': model_results,
    }
