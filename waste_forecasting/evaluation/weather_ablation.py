"""Ablačné štúdie meteorologických príznakov.

Cieľom je odhadnúť, ako sa zmení testovacie RMSE po pridaní
meteorologických príznakov. Postup porovnáva dva varianty pipeline:
jeden so všetkými meteorologickými príznakmi a druhý bez nich.
Rozdiel v testovacom RMSE odhaduje prínos počasia.

Sú k dispozícii dve samostatné funkcie: run_weather_ablation_study pre
predikciu pri vývoze (Experiment A) a run_weather_ablation_expB pre
predikciu na pravidelnej časovej osi (Experiment B). Obe ukladajú
výsledky ako CSV tabuľky v príslušnom podpriečinku výstupu.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import CONFIG
from waste_forecasting.data.loading import cleanup_dataframes
from waste_forecasting.data.preprocessing import (
    detect_collections,
    filter_containers_by_min_samples,
)
from waste_forecasting.data.splitting import temporal_train_val_split_per_container
from waste_forecasting.features.encoding import SmartCategoricalEncoder
from waste_forecasting.features.fourier import add_fourier_features
from waste_forecasting.features.holiday import create_holiday_features
from waste_forecasting.features.spatial import add_geolocation_features
from waste_forecasting.features.temporal import (
    add_event_features,
    add_time_features,
    create_event_dataset,
    create_time_dataset,
    get_feature_columns,
)
from waste_forecasting.features.weather import add_all_weather_features, analyze_weather_impact
from waste_forecasting.models.metrics import calculate_all_metrics
from waste_forecasting.models.training import train_histgb_model

logger = logging.getLogger(__name__)

analyze_weather_impact_wrapper = analyze_weather_impact

__all__ = ['run_weather_ablation_study', 'run_weather_ablation_expB']


def run_weather_ablation_study(
    df: pd.DataFrame,
    test_containers: set,
    output_dir: str,
) -> Dict:
    """Porovnať výkon modelu s meteorologickými príznakmi a bez nich.

    Porovná dva varianty Experimentu A: s meteorologickými príznakmi
    a bez nich. Rozdiel v testovacom RMSE ukazuje, či meteorologické dáta
    modelu pomohli.
    """
    logger.info("=" * 70)
    logger.info("Ablácia meteorologických príznakov: s počasím a bez počasia")
    logger.info("=" * 70)
    
    abl_dir = f'{output_dir}/weather_ablation'
    os.makedirs(abl_dir, exist_ok=True)
    
    # Príprava základných dát (ešte bez meteorologických príznakov)
    df_det = detect_collections(df.copy())
    df_events = create_event_dataset(df_det)
    # Filtrovanie: vyžadujeme dostatok vzoriek na kontajner pre tréning aj test.
    df_events, filter_info = filter_containers_by_min_samples(
        df_events,
        min_samples=CONFIG.MIN_COLLECTIONS_PER_CONTAINER,
        sample_col='container_id',
        output_dir=abl_dir,
        tag='events'
    )

    df_events = add_event_features(df_events)
    df_events = add_fourier_features(df_events)
    df_events = create_holiday_features(df_events)
    df_events = add_geolocation_features(df_events)
    
    # Rozdelenie na tréningovú a testovaciu množinu podľa kontajnerov
    train_containers = [c for c in df_events['container_id'].unique() if c not in test_containers]
    
    results = {}
    
    for variant_name, add_weather in [('WITHOUT_WEATHER', False), ('WITH_WEATHER', True)]:
        logger.info(f"\nTrénujem variant {variant_name}...")
        
        df_variant = df_events.copy()
        
        if add_weather:
            df_variant = add_all_weather_features(df_variant)
        
        df_train = df_variant[df_variant['container_id'].isin(train_containers)].copy()
        df_test = df_variant[df_variant['container_id'].isin(test_containers)].copy()
        
        encoder = SmartCategoricalEncoder()
        df_train = encoder.fit_transform(df_train)
        df_test = encoder.transform(df_test)
        
        feature_cols = get_feature_columns(df_train)
        ohe_cols = encoder.get_all_feature_names()
        feature_cols = feature_cols + [c for c in ohe_cols if c not in feature_cols]
        feature_cols = [c for c in feature_cols if c in df_train.columns]
        
        X_train = df_train[feature_cols]
        y_train = df_train['target'].values
        X_test = df_test[feature_cols]
        y_test = df_test['target'].values
        
        model, _, _ = train_histgb_model(X_train, y_train, clip_range=(0, 100))
        y_pred = np.clip(model.predict(X_test), 0, 100)
        
        metrics = calculate_all_metrics(y_test, y_pred, with_ci=True)
        metrics['n_features'] = len(feature_cols)
        
        results[variant_name] = metrics
        logger.info(f"  {variant_name}: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}, príznaky={len(feature_cols)}")
    
    # Kvantifikácia prínosu meteorologických príznakov
    improvement = results['WITHOUT_WEATHER']['rmse'] - results['WITH_WEATHER']['rmse']
    improvement_pct = 100 * improvement / results['WITHOUT_WEATHER']['rmse']
    
    results['improvement'] = {
        'rmse_reduction': improvement,
        'rmse_reduction_pct': improvement_pct,
        'feature_increase': results['WITH_WEATHER']['n_features'] - results['WITHOUT_WEATHER']['n_features']
    }
    
    logger.info("\nVplyv meteorologických príznakov:")
    logger.info(f"  Zníženie RMSE: {improvement:.2f} ({improvement_pct:+.1f} %)")
    logger.info(f"  Pridané príznaky: {results['improvement']['feature_increase']}")
    
    # Uloženie výsledkov do CSV
    results_df = pd.DataFrame([
        {'variant': 'WITHOUT_WEATHER', **results['WITHOUT_WEATHER']},
        {'variant': 'WITH_WEATHER', **results['WITH_WEATHER']}
    ])
    results_df.to_csv(f'{abl_dir}/ablation_results.csv', index=False)
    
    with open(f'{abl_dir}/ablation_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    cleanup_dataframes(df_det, df_events)
    
    return results


def run_weather_ablation_expB(df: pd.DataFrame, test_containers: set, output_dir: str) -> Dict:
    """Variant k run_weather_ablation_study pre Experiment B. Namiesto datasetu
    vývozov používa dataset na 24-hodinovú predikciu. Natrénujú sa dva modely:
    s meteorologickými príznakmi a bez nich. Variant bez počasia odstráni aj
    odvodené príznaky s meteorologickými názvami (temp, precip, humid a ďalšie),
    aby sa počasie nedostalo do modelu cez interakčné členy.

    Na rozdiel od Experimentu A sa tu aplikuje aj vnútorné časové validačné
    delenie (temporal_train_val_split_per_container), takže tréningová množina
    nevidí budúce záznamy rovnakého kontajnera.
    """
    logger.info("=" * 70)
    logger.info("Ablácia meteorologických príznakov: Experiment B")
    logger.info("=" * 70)
    
    abl_dir = f'{output_dir}/weather_ablation'
    os.makedirs(abl_dir, exist_ok=True)
    
    # Príprava dát pre časovú predikciu (24h horizont)
    df_time = create_time_dataset(df.copy(), horizon_hours=24)
    df_time = add_time_features(df_time)
    df_time = add_fourier_features(df_time)
    df_time = create_holiday_features(df_time)
    df_time = add_geolocation_features(df_time)
    
    train_containers = [c for c in df_time['container_id'].unique() if c not in test_containers]
    
    results = {}
    for variant, use_weather in [('with_weather', True), ('without_weather', False)]:
        df_var = df_time.copy()
        if use_weather:
            df_var = add_all_weather_features(df_var, mode="time", step_hours=CONFIG.RESAMPLE_HOURS)
        
        encoder = SmartCategoricalEncoder()
        
        df_train = df_var[df_var['container_id'].isin(train_containers)].copy()
        df_test_var = df_var[df_var['container_id'].isin(test_containers)].copy()
        
        df_train, df_val = temporal_train_val_split_per_container(df_train)
        df_train = encoder.fit_transform(df_train)
        df_test_var = encoder.transform(df_test_var)
        
        feature_cols = get_feature_columns(df_train)
        ohe_cols = encoder.get_all_feature_names()
        feature_cols = feature_cols + [c for c in ohe_cols if c not in feature_cols]
        feature_cols = [c for c in feature_cols if c in df_train.columns]
        
        if not use_weather:
            weather_terms = ['temp', 'precip', 'rain', 'humid', 'wind', 'press', 
                           'weather', 'comfort', 'heat', 'chill', 'season']
            feature_cols = [f for f in feature_cols if not any(w in f.lower() for w in weather_terms)]
        
        X_train = df_train[feature_cols]
        y_train = df_train['target'].values
        X_test = df_test_var[feature_cols]
        y_test = df_test_var['target'].values
        
        model, _, _ = train_histgb_model(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, 100)
        metrics = calculate_all_metrics(y_test, y_pred, with_ci=True)
        
        results[variant] = {
            'n_features': len(feature_cols),
            'rmse': metrics['rmse'],
            'r2': metrics['r2'],
            'rmse_ci_lower': metrics.get('rmse_ci_lower'),
            'rmse_ci_upper': metrics.get('rmse_ci_upper'),
        }
        logger.info(f"  {variant}: {len(feature_cols)} príznakov, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")
    
    with open(f'{abl_dir}/ablation_expB.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"  Vplyv počasia v Experimente B: RMSE {results['without_weather']['rmse']:.2f} -> {results['with_weather']['rmse']:.2f}")
    
    cleanup_dataframes(df_time)
    return results

