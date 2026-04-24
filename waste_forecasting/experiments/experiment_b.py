"""Experiment B - 24-hodinová predikcia plnosti kontajnera.

Druhý z troch hlavných experimentov a zároveň hlavný prevádzkový výstup
celého riešenia. Model predikuje plnosť kontajnera s horizontom 24 hodín
dopredu na pravidelne prevzorkovanej časovej sérii (krok je RESAMPLE_HOURS
z konfigurácie).

Postupnosť krokov v pipeline experimentu:

1. Prevzorkovanie časovej série a zostavenie time datasetu s
   cieľovou premennou posunutou o 24 hodín dopredu.
2. Filtrovanie kontajnerov s nedostatkom meraní.
3. Generovanie temporálnych, sezónnych, sviatkových, geolokačných
   a meteorologických príznakov.
4. Rozdelenie na trénovaciu a testovaciu množinu podľa test_containers
   a následne train/val temporálne delenie v rámci trénovacích kontajnerov.
5. Kódovanie kategoriálnych premenných (fit na trénovacej časti,
   transform na val + test).
6. Tréning všetkých troch modelov s voliteľným Optuna tuningom.
7. Výber víťaza, SHAP a permutačná dôležitosť na víťazovi.
8. Export metrík, porovnania a predikcií s metadátami.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
import pandas as pd

from config import CONFIG
from waste_forecasting.data.preprocessing import filter_containers_by_min_samples
from waste_forecasting.data.splitting import temporal_train_val_split_per_container
from waste_forecasting.evaluation.interpretability import (
    compute_shap_importance,
)
from waste_forecasting.features.encoding import SmartCategoricalEncoder
from waste_forecasting.features.fourier import add_fourier_features
from waste_forecasting.features.holiday import create_holiday_features
from waste_forecasting.features.spatial import add_geolocation_features
from waste_forecasting.features.temporal import (
    add_time_features,
    create_time_dataset,
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

__all__ = ['run_experiment_B']


def run_experiment_B(
    df: pd.DataFrame,
    test_containers: set,
    output_dir: str,
    enable_tuning: bool = None,
    enable_shap: bool = None,
) -> Dict:
    """Spustiť Experiment B: 24-hodinovú predikciu plnosti.

    Na rozdiel od Experimentu A pracuje s pravidelne prevzorkovanou
    časovou sériou a predikuje plnosť o 24 hodín dopredu. Ide o
    hlavný prevádzkový výstup pipeline, ktorý je priamo použiteľný
    pre plánovanie dennej trasy zberu.
    """
    logger.info("=" * 70)
    logger.info("Experiment B: 24-hodinová predikcia plnosti naprieč modelmi")
    logger.info("=" * 70)

    enable_tuning = enable_tuning if enable_tuning is not None else CONFIG.ENABLE_HYPERPARAMETER_TUNING
    enable_shap = enable_shap if enable_shap is not None else CONFIG.ENABLE_SHAP

    exp_dir = f'{output_dir}/exp_B'
    os.makedirs(exp_dir, exist_ok=True)

    # --- Príprava dát a tvorba príznakov -----------------------------
    df_time = create_time_dataset(df, horizon_hours=24)
    df_time, filter_info = filter_containers_by_min_samples(
        df_time, min_samples=CONFIG.MIN_MEASUREMENTS_PER_CONTAINER,
        sample_col='container_id', output_dir=exp_dir, tag='time',
    )
    df_time = add_time_features(df_time)
    df_time = add_fourier_features(df_time)
    df_time = create_holiday_features(df_time)
    df_time = add_geolocation_features(df_time)
    df_time = add_all_weather_features(df_time, mode="time", step_hours=CONFIG.RESAMPLE_HOURS)
    df_time = df_time.dropna(subset=['target', 'current_fill'])
    logger.info(f"Časový dataset: {len(df_time):,} vzoriek")

    # --- Rozdelenie dát ----------------------------------------------
    train_containers = [c for c in df_time['container_id'].unique() if c not in test_containers]
    df_train_full = df_time[df_time['container_id'].isin(train_containers)]
    df_test = df_time[df_time['container_id'].isin(test_containers)]
    df_train, df_val = temporal_train_val_split_per_container(df_train_full)

    # Kópia pre permutačnú dôležitosť - počítame ju na plnej
    # trénovacej množine, nie na delení tréning/validácia.
    df_train_all = df_train_full.copy()

    # Zachovanie metadát (kontajner, čas, typ odpadu) pred kódovaním -
    # tieto stĺpce sú potrebné pre neskoršiu vizualizáciu po kontajneroch.
    test_meta = df_test[['container_id', 'measured_at_utc', 'trash_type']].copy()

    # --- Kódovanie kategoriálnych premenných ------------------------
    encoder = SmartCategoricalEncoder()
    df_train = encoder.fit_transform(df_train)
    df_val = encoder.transform(df_val)
    df_test = encoder.transform(df_test)
    df_train_all = encoder.transform(df_train_all)

    feature_cols = get_feature_columns(df_train)
    ohe_cols = encoder.get_all_feature_names()
    feature_cols = feature_cols + [c for c in ohe_cols if c not in feature_cols]
    feature_cols = [c for c in feature_cols if c in df_train.columns]

    X_train = df_train[feature_cols]; y_train = df_train['target'].values
    X_val = df_val[feature_cols]; y_val = df_val['target'].values
    X_test = df_test[feature_cols]; y_test = df_test['target'].values

    # --- Sezónny perzistentný referenčný model -----------------------
    # Referenčný model predikuje plnosť spred 24 hodín, alebo aktuálnu,
    # ak 24-hodinový lag nie je k dispozícii.
    baseline = np.clip(df_test['fill_lag_24h'].fillna(df_test['current_fill']).values, 0, 100)
    baseline_metrics = calculate_all_metrics(y_test, baseline, with_ci=False)

    # --- Tréning viacerých modelov ----------------------------------
    model_results = {}
    primary_model = None
    primary_model_name = None

    for model_name in CONFIG.MODELS_TO_COMPARE:
        if not is_model_available(model_name):
            continue
        logger.info(f"\nTrénujem {model_name}")
        model_dir = f'{exp_dir}/{model_name.lower()}'
        os.makedirs(model_dir, exist_ok=True)

        optuna_study = None
        if enable_tuning and HAS_OPTUNA:
            tuner = MultiModelTuner(model_name=model_name)
            params = tuner.tune(df_train, feature_cols)
            optuna_study = tuner.study
        else:
            params = get_default_params(model_name)

        model, _, training_info = train_model(
            model_name, X_train, y_train, X_val, y_val,
            params=params, clip_range=(0, 100),
        )
        training_info['optuna_study'] = optuna_study
        y_pred_test = predict_model(model, X_test, model_name, (0, 100))

        test_metrics = calculate_all_metrics(y_test, y_pred_test, with_ci=True)

        logger.info(f"  {model_name}: RMSE={test_metrics['rmse']:.2f}, R2={test_metrics['r2']:.3f}")

        model_results[model_name] = {
            'model': model, 'y_pred': y_pred_test, 'metrics': test_metrics,
            'best_params': params, 'training_info': training_info,
        }

    # --- Výber víťazného modelu pre SHAP ----------------------------
    primary_model_name = min(model_results, key=lambda k: model_results[k]['metrics']['rmse'])
    primary_model = model_results[primary_model_name]['model']
    logger.info(f"Model použitý pre SHAP: {primary_model_name} "
                f"(RMSE={model_results[primary_model_name]['metrics']['rmse']:.2f})")
    # --- SHAP analýza víťazného modelu ------------------------------
    shap_importance = None
    if enable_shap and HAS_SHAP and primary_model is not None:
        shap_importance = compute_shap_importance(
            primary_model, X_train, X_test, feature_cols, exp_dir,
        )

    # --- Porovnanie modelov a export metrík -------------------------
    comparison_rows = [{'model': name, **res['metrics']} for name, res in model_results.items()]
    comparison_rows.append({'model': 'Baseline_Seasonal', **baseline_metrics})
    metrics_df = pd.DataFrame(comparison_rows)
    metrics_df.to_csv(f'{exp_dir}/metrics.csv', index=False)

    logger.info("\nPorovnanie modelov (Experiment B):")
    logger.info(metrics_df[['model', 'rmse', 'mae', 'r2']].to_string(index=False))

    # --- Export predikcií s metadátami ------------------------------
    # Predikcie zahŕňajú container_id, čas a typ odpadu, aby bolo
    # možné robiť analýzy po kontajneroch a podľa typu odpadu.
    best_mn = min(model_results, key=lambda k: model_results[k]['metrics']['rmse'])
    pred_df = test_meta.reset_index(drop=True).copy()
    pred_df['actual'] = y_test
    pred_df['predicted'] = model_results[best_mn]['y_pred']
    pred_df['error'] = pred_df['predicted'] - pred_df['actual']
    pred_df['abs_error'] = pred_df['error'].abs()
    pred_df.to_csv(f'{exp_dir}/predictions.csv', index=False)
    logger.info(f"Uložených predikcií s metadátami: {len(pred_df)} do {exp_dir}/predictions.csv")

    # Vrátenie výsledkov víťazného modelu.
    winner = model_results[primary_model_name]
    return {
        'data_filter': filter_info,
        'test_metrics': winner.get('metrics', {}),
        'primary_model_name': primary_model_name,
        'baseline_metrics': baseline_metrics,
        'model_comparison': {name: res['metrics'] for name, res in model_results.items()},
        'all_model_results': model_results,
        'shap_importance': shap_importance,
    }
