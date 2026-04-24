"""Experiment C - odhad počtu dní do dosiahnutia 85 % plnosti.

Tretí experiment odhaduje počet dní, za ktoré kontajner dosiahne prah
FILL_THRESHOLD (predvolene 85 %). Výstup pomáha pri dlhodobom plánovaní
zvozu. Ak model odhadne, že kontajner bude plný o 3 dni, dispečer má čas
zaradiť ho do plánu.

Pozor na selekčné skreslenie: do datasetu tohto experimentu vstupujú iba
kontajnery, pri ktorých bol aspoň raz pozorovaný prechod cez prah.
Informácie o pokrytí sa ukladajú do selection_bias_info. Pre štatisticky
korektnú alternatívu s cenzorovanými pozorovaniami pozri survival_baseline.py.

Víťaz sa vyberá podľa MAE v dňoch (nie RMSE), pretože MAE je robustnejšia
pre exponenciálne rozdelené cieľové hodnoty, ktoré bývajú pri odhade času
do udalosti bežné.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import CONFIG
from waste_forecasting.data.preprocessing import (
    filter_containers_by_min_samples,
    resample_time_series,
)
from waste_forecasting.data.splitting import temporal_train_val_split_per_container
from waste_forecasting.features.encoding import SmartCategoricalEncoder
from waste_forecasting.features.fourier import add_fourier_features
from waste_forecasting.features.spatial import add_geolocation_features
from waste_forecasting.features.temporal import (
    add_ttt_features,
    create_time_to_threshold_dataset,
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

logger = logging.getLogger(__name__)

__all__ = ['run_experiment_C']


def run_experiment_C(
    df: pd.DataFrame,
    test_containers: set,
    output_dir: str,
    enable_tuning: bool = None,
) -> Optional[Dict]:
    """Spustiť Experiment C: odhad počtu dní do prahu plnosti.

    Na rozdiel od experimentov A a B je Experiment C vyhlásený
    ako exploratórny. Cieľová premenná je spojité číslo dní
    (nie plnosť v percentách), orezané do intervalu
    [0, CONFIG.MAX_TTT_DAYS].
    """
    logger.info("=" * 70)
    logger.info(f"Experiment C: čas do dosiahnutia {CONFIG.FILL_THRESHOLD}% plnosti naprieč modelmi")
    logger.info("=" * 70)

    enable_tuning = enable_tuning if enable_tuning is not None else CONFIG.ENABLE_HYPERPARAMETER_TUNING

    exp_dir = f'{output_dir}/exp_C'
    os.makedirs(exp_dir, exist_ok=True)

    # --- Príprava dát -----------------------------------------------
    df_resampled = resample_time_series(df)
    df_ttt, selection_bias_info = create_time_to_threshold_dataset(df_resampled)
    df_ttt, filter_info = filter_containers_by_min_samples(
        df_ttt, min_samples=CONFIG.MIN_MEASUREMENTS_PER_CONTAINER,
        sample_col='container_id', output_dir=exp_dir, tag='ttt',
    )

    # Selekčné skreslenie je explicitne uložené pre spätnú
    # interpretovateľnosť výsledkov.
    with open(f'{exp_dir}/selection_bias_info.json', 'w') as f:
        json.dump(selection_bias_info, f, indent=2)

    if len(df_ttt) < 100:
        logger.warning("Nedostatok dát pre Experiment C")
        return None

    df_ttt = add_ttt_features(df_ttt)
    df_ttt = add_fourier_features(df_ttt)
    df_ttt = add_geolocation_features(df_ttt)
    df_ttt = add_all_weather_features(df_ttt)
    df_ttt = df_ttt.dropna(subset=['target', 'current_fill'])

    # --- Rozdelenie dát ---------------------------------------------
    train_containers = [c for c in df_ttt['container_id'].unique() if c not in test_containers]
    df_train_full = df_ttt[df_ttt['container_id'].isin(train_containers)]
    df_test = df_ttt[df_ttt['container_id'].isin(test_containers)]

    if len(df_train_full) < 50 or len(df_test) < 20:
        logger.warning("Nedostatok dát po rozdelení")
        return None

    df_train, df_val = temporal_train_val_split_per_container(df_train_full)

    # --- Kódovanie kategoriálnych premenných ------------------------
    encoder = SmartCategoricalEncoder()
    df_train = encoder.fit_transform(df_train)
    df_val = encoder.transform(df_val)
    df_test = encoder.transform(df_test)

    feature_cols = get_feature_columns(df_train)
    ohe_cols = encoder.get_all_feature_names()
    feature_cols = feature_cols + [c for c in ohe_cols if c not in feature_cols]
    feature_cols = [c for c in feature_cols if c in df_train.columns]

    X_train = df_train[feature_cols]; y_train = df_train['target'].values
    X_test = df_test[feature_cols]; y_test = df_test['target'].values

    # --- Lineárny referenčný model ----------------------------------
    # Referenčný model používa lineárnu extrapoláciu tempa plnenia
    # (estimated_time_linear z TTT príznaku). Ide o prirodzenú
    # a jednoduchú referenciu, ktorú by pokročilý model mal prekonať.
    baseline = np.clip(
        df_test['estimated_time_linear'].fillna(df_test['target'].mean()).values,
        0, CONFIG.MAX_TTT_DAYS,
    )
    baseline_metrics = calculate_all_metrics(y_test, baseline, with_ci=False)

    # --- Tréning viacerých modelov ----------------------------------
    model_results = {}
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

        # Predikcie sú orezané do intervalu [0, MAX_TTT_DAYS].
        model, _, training_info = train_model(
            model_name, X_train, y_train, params=params,
            clip_range=(0, CONFIG.MAX_TTT_DAYS),
        )
        training_info['optuna_study'] = optuna_study
        y_pred_test = predict_model(model, X_test, model_name, (0, CONFIG.MAX_TTT_DAYS))

        test_metrics = calculate_all_metrics(y_test, y_pred_test, with_ci=True)

        logger.info(f"  {model_name} (dni): RMSE={test_metrics['rmse']:.2f}, MAE={test_metrics['mae']:.2f}")

        model_results[model_name] = {
            'model': model, 'y_pred': y_pred_test, 'metrics': test_metrics,
            'best_params': params, 'training_info': training_info,
        }

    # --- Porovnanie modelov -----------------------------------------
    comparison_rows = [{'model': name, 'rmse_days': res['metrics']['rmse'],
                        'mae_days': res['metrics']['mae'], 'r2': res['metrics']['r2']}
                       for name, res in model_results.items()]
    comparison_rows.append({'model': 'Baseline_Linear', 'rmse_days': baseline_metrics['rmse'],
                           'mae_days': baseline_metrics['mae'], 'r2': baseline_metrics['r2']})
    metrics_df = pd.DataFrame(comparison_rows)
    metrics_df.to_csv(f'{exp_dir}/metrics.csv', index=False)

    logger.info("\nPorovnanie modelov (Experiment C):")
    logger.info(metrics_df.to_string(index=False))

    # --- Export predikcií -------------------------------------------
    # Víťaz sa v Experimente C určuje podľa MAE, ktorá je robustnejšia
    # voči dlhým chvostom rozdelenia TTT hodnôt.
    best_mn = min(model_results, key=lambda k: model_results[k]['metrics'].get('mae', model_results[k]['metrics']['rmse']))
    pred_df = pd.DataFrame({'actual': y_test, 'predicted': model_results[best_mn]['y_pred']})
    pred_df.to_csv(f'{exp_dir}/predictions.csv', index=False)
    logger.info(f"Uložených predikcií: {len(pred_df)} do {exp_dir}/predictions.csv")

    winner = model_results[best_mn]
    return {
        'data_filter': filter_info,
        'test_metrics': winner.get('metrics', {}),
        'primary_model_name': best_mn,
        'baseline_metrics': baseline_metrics,
        'selection_bias_info': selection_bias_info,
        'model_comparison': {name: res['metrics'] for name, res in model_results.items()},
        'all_model_results': model_results,
    }
