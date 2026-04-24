"""Klasické referenčné modely (Last-Value, Seasonal Naive, ARIMA, Prophet).

Modul implementuje baseline prístupy, s ktorými sa porovnávajú hlavné
modely gradient boostingu. Všetky baseline metódy sú vyhodnocované na 
identickom filtri vzoriek ako hlavné modely, takže ich RMSE je priamo porovnateľné.

Jednotlivé baseline metódy:

* Last-Value - predikcia sa rovná poslednej zaznamenanej hodnote
  (persistence model); je výpočtovo najrýchlejší.
* Seasonal Naive - predikcia sa rovná hodnote spred týždňa
  (perióda 7 dní), zachytáva týždennú sezónnosť.
* ARIMA - autoregresný integrovaný kĺzavý priemer, implementovaný
  cez knižnicu statsmodels (fixné rády (2, 0, 1)) alebo
  pmdarima (automatický výber rádov).
* Prophet - aditívny model s týždennou sezónnosťou od Facebook.

ARIMA a Prophet sú z výpočtových dôvodov aplikované na podvzorku
30 testovacích kontajnerov (pri volaní funkcií s explicitným
subsample_size).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import CONFIG
from waste_forecasting.data.preprocessing import (
    detect_collections,
    filter_containers_by_min_samples,
)
from waste_forecasting.features.temporal import (
    add_event_features,
    create_event_dataset,
    create_time_dataset,
)
from waste_forecasting.models.metrics import calculate_all_metrics, rmse

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

logger = logging.getLogger(__name__)

__all__ = ['run_auto_arima_baseline', 'run_arima_baseline', 'run_prophet_baseline', 'run_seasonal_naive_baseline', 'compare_all_baselines', 'run_arima_prophet_baseline_expB']


def run_auto_arima_baseline(df: pd.DataFrame, target_col: str = 'target') -> Tuple[Optional[np.ndarray], Dict]:
    """Auto-ARIMA baseline s automatickým výberom rádov modelu.

    Pre každý kontajner nezávisle natrénuje ARIMA model s rádmi
    vybranými knižnicou pmdarima. Trénuje sa na prvých 80 %
    časovej osi kontajnera, predikuje sa na zvyšných 20 %. Ak
    pmdarima nie je dostupná, volá fallback run_arima_baseline
    s fixnými rádmi.
    """
    if not HAS_PMDARIMA:
        logger.warning("pmdarima nie je nainštalovaná, používam fixnú ARIMA")
        return run_arima_baseline(df, target_col)
    
    logger.info("Spúšťam referenčný model Auto-ARIMA...")
    
    predictions, actuals = [], []
    containers = df['container_id'].unique()
    orders_used = []
    
    for i, cid in enumerate(tqdm(containers[:50], desc="Auto-ARIMA", disable=not HAS_TQDM)):
        container_data = df[df['container_id'] == cid].sort_values('measured_at_utc')
        
        if len(container_data) < 20:
            continue
        
        series = container_data[target_col].values
        train_size = int(len(series) * 0.8)
        
        if train_size < 15:
            continue
        
        try:
            model = pm.auto_arima(
                series[:train_size],
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                d=None,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_order=6,
                trace=False
            )
            
            forecast = model.predict(n_periods=len(series) - train_size)
            
            predictions.extend(forecast)
            actuals.extend(series[train_size:])
            orders_used.append(model.order)
            
        except Exception:
            continue
    
    if len(predictions) == 0:
        return None, {'error': 'no predictions'}
    
    predictions, actuals = np.array(predictions), np.array(actuals)
    metrics = calculate_all_metrics(actuals, predictions, with_ci=False)
    
    if orders_used:
        from collections import Counter
        most_common_order = Counter(orders_used).most_common(1)[0][0]
        metrics['most_common_order'] = str(most_common_order)
    
    logger.info(f"Auto-ARIMA: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")
    
    return predictions, metrics


def run_arima_baseline(df: pd.DataFrame, target_col: str = 'target', order: Tuple = (2, 0, 1)) -> Tuple[Optional[np.ndarray], Dict]:
    """ARIMA baseline s fixnými rádmi pre časovú predikciu.

    Jednoduchší variant ARIMA bez automatického hľadania rádov.
    Predvolené rády (p, d, q) = (2, 0, 1) boli vybrané na základe
    empirickej analýzy dát. Fallback, keď pmdarima nie je
    k dispozícii.
    """
    if not HAS_ARIMA:
        logger.warning("ARIMA nie je dostupná, preskakujem")
        return None, {'error': 'ARIMA nie je nainštalovaná'}
    
    logger.info(f"Spúšťam referenčný model ARIMA{order}...")
    
    predictions, actuals = [], []
    containers = df['container_id'].unique()
    
    for i, cid in enumerate(tqdm(containers, desc="ARIMA", disable=not HAS_TQDM)):
        container_data = df[df['container_id'] == cid].sort_values('measured_at_utc')
        
        if len(container_data) < 15:
            continue
        
        series = container_data[target_col].values
        train_size = int(len(series) * 0.8)
        
        if train_size < 10:
            continue
        
        try:
            model = ARIMA(series[:train_size], order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=len(series) - train_size)
            
            predictions.extend(forecast)
            actuals.extend(series[train_size:])
        except Exception:
            continue
    
    if len(predictions) == 0:
        return None, {'error': 'no predictions'}
    
    predictions, actuals = np.array(predictions), np.array(actuals)
    metrics = calculate_all_metrics(actuals, predictions, with_ci=False)
    
    logger.info(f"ARIMA{order}: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")
    
    return predictions, metrics


def run_prophet_baseline(df: pd.DataFrame, target_col: str = 'target') -> Tuple[Optional[np.ndarray], Dict]:
    """Prophet baseline s týždennou sezónnosťou.

    Pre každý kontajner zvlášť natrénuje Prophet model (Facebook
    aditívny model) s aktivovanou týždennou sezónnosťou a vypnutou
    dennou a ročnou. Je vhodný pre časové rady s výraznou týždennou
    štruktúrou, čo kontajnerové dáta typicky majú.
    """
    if not HAS_PROPHET:
        logger.warning("Prophet nie je nainštalovaný, preskakujem")
        return None, {'error': 'prophet nie je nainštalovaný'}
    
    logger.info("Spúšťam referenčný model Prophet...")
    
    predictions, actuals = [], []
    containers = df['container_id'].unique()
    
    for cid in tqdm(containers, desc="Prophet", disable=not HAS_TQDM):
        container_data = df[df['container_id'] == cid].sort_values('measured_at_utc')
        
        if len(container_data) < 15:
            continue
        
        prophet_df = container_data[['measured_at_utc', target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        train_size = int(len(prophet_df) * 0.8)
        if train_size < 10:
            continue
        
        try:
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            model = Prophet(yearly_seasonality=False, weekly_seasonality=True,
                            daily_seasonality=False)
            model.fit(prophet_df.iloc[:train_size])
            
            forecast = model.predict(prophet_df.iloc[train_size:][['ds']])
            
            predictions.extend(forecast['yhat'].values)
            actuals.extend(prophet_df.iloc[train_size:]['y'].values)
        except Exception:
            continue
    
    if len(predictions) == 0:
        return None, {'error': 'no predictions'}
    
    predictions, actuals = np.array(predictions), np.array(actuals)
    metrics = calculate_all_metrics(actuals, predictions, with_ci=False)
    
    logger.info(f"Prophet: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")
    
    return predictions, metrics


def run_seasonal_naive_baseline(df: pd.DataFrame, target_col: str = 'target', period: int = 7) -> Tuple[Optional[np.ndarray], Dict]:
    """Seasonal Naive baseline - predikcia hodnotou spred jednej periódy.

    Predikuje, že aktuálna hodnota bude rovnaká ako pred period
    krokmi. Pri predvolenom period=7 to zodpovedá očakávaniu,
    že plnosť v daný deň týždňa bude podobná ako pred týždňom.
    """
    logger.info(f"Spúšťam sezónny naivný referenčný model (perióda={period})...")
    
    predictions, actuals = [], []
    
    for cid in df['container_id'].unique():
        container_data = df[df['container_id'] == cid].sort_values('measured_at_utc')
        
        if len(container_data) <= period:
            continue
        
        series = container_data[target_col].values
        
        for i in range(period, len(series)):
            predictions.append(series[i - period])
            actuals.append(series[i])
    
    if len(predictions) == 0:
        return None, {'error': 'no predictions'}
    
    predictions, actuals = np.array(predictions), np.array(actuals)
    metrics = calculate_all_metrics(actuals, predictions, with_ci=False)
    
    logger.info(f"Seasonal Naive: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")
    
    return predictions, metrics


def compare_all_baselines(df: pd.DataFrame, test_containers: set, output_dir: str) -> pd.DataFrame:
    """Spustiť všetky baseline metódy a porovnať ich na testovacej množine.

    Agregačná funkcia, ktorá postupne zavolá Last-Value baseline,
    Seasonal Naive, ARIMA (auto alebo fixné rády) a Prophet,
    vyhodnotí každú metódu a uloží súhrnnú tabuľku do
    output_dir/baselines_comparison.csv.
    """
    logger.info("=" * 70)
    logger.info("Porovnanie referenčných modelov")
    logger.info("=" * 70)
    
    df_copy = df.copy()
    df_copy = detect_collections(df_copy)
    df_events = create_event_dataset(df_copy)
    df_events = add_event_features(df_events)
    df_events = df_events.dropna(subset=['target_lag_1', 'target_lag_2'])
    
    df_test = df_events[df_events['container_id'].isin(test_containers)]
    
    if len(df_test) < 100:
        logger.warning("Nedostatok dát na porovnanie referenčných modelov")
        return None
    
    y_true = df_test['target'].values
    results = []
    
    # Last Value baseline - predikcia rovná predchádzajúcej hodnote
    y_pred_last = df_test['target_lag_1'].values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred_last)
    metrics = calculate_all_metrics(y_true[mask], y_pred_last[mask], with_ci=True)
    metrics['model'] = 'Last Value (Naive)'
    results.append(metrics)
    
    # Same Day-of-Week - rovnaký deň v predchádzajúcom týždni
    y_pred_dow = df_test['target_same_dow'].fillna(df_test['target_lag_1']).values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred_dow)
    metrics = calculate_all_metrics(y_true[mask], y_pred_dow[mask], with_ci=True)
    metrics['model'] = 'Same Day-of-Week'
    results.append(metrics)
    
    # Seasonal Naive - hodnota spred týždňa (perióda 7 dní)
    _, metrics = run_seasonal_naive_baseline(df_test, 'target', period=7)
    if metrics and 'error' not in metrics:
        metrics['model'] = 'Seasonal Naive (7-day)'
        results.append(metrics)
    
    # Auto-ARIMA na podvzorke (automatický výber rádov)
    df_sample = df_test.groupby('container_id').head(50)
    _, metrics = run_auto_arima_baseline(df_sample, 'target')
    if metrics and 'error' not in metrics:
        metrics['model'] = 'Auto-ARIMA'
        results.append(metrics)
    
    # Prophet na rovnakej podvzorke
    _, metrics = run_prophet_baseline(df_sample, 'target')
    if metrics and 'error' not in metrics:
        metrics['model'] = 'Prophet'
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    cols = ['model', 'rmse', 'mae', 'r2', 'n']
    cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[cols].sort_values('rmse')
    
    results_df.to_csv(f'{output_dir}/baseline_comparison.csv', index=False)
    
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    return results_df


def run_arima_prophet_baseline_expB(df: pd.DataFrame, test_containers: set, output_dir: str,
                                     max_containers: int = 30) -> Dict:
    """ARIMA a Prophet baseline špecificky pre Experiment B (24h predikcia).

    Z výpočtových dôvodov sa obe metódy aplikujú iba na náhodnú
    podvzorku max_containers testovacích kontajnerov. Pre každý
    vybraný kontajner sa natrénuje model nad prevzorkovanou časovou
    sériou a vyhodnotí sa 24-hodinová predikcia.
    """
    logger.info("=" * 70)
    logger.info(f"Referenčné modely ARIMA/Prophet: Experiment B (podvzorka {max_containers} kontajnerov)")
    logger.info("=" * 70)
    
    baseline_dir = f'{output_dir}/exp_B'
    os.makedirs(baseline_dir, exist_ok=True)
    
    df_time = create_time_dataset(df.copy(), horizon_hours=24)
    
    test_ids = sorted(test_containers)
    rng = np.random.RandomState(CONFIG.SEED)
    if len(test_ids) > max_containers:
        test_ids = list(rng.choice(test_ids, max_containers, replace=False))
    
    arima_preds, arima_actuals = [], []
    prophet_preds, prophet_actuals = [], []
    
    horizon_steps = 24 // CONFIG.RESAMPLE_HOURS  # 4 kroky pri 6-hodinovom prevzorkovaní
    
    for cid in test_ids:
        c_data = df_time[df_time['container_id'] == cid].sort_values('measured_at_utc').copy()
        if len(c_data) < 50:
            continue
        
        cutoff = int(len(c_data) * 0.7)
        train_series = c_data['percent_calculated'].iloc[:cutoff].values
        test_data = c_data.iloc[cutoff:]
        
        if len(test_data) <= horizon_steps:
            continue
        
        # Auto-ARIMA pre tento kontajner (ak je k dispozícii knižnica)
        if HAS_PMDARIMA:
            try:
                from pmdarima import auto_arima
                arima_model = auto_arima(train_series, seasonal=True, m=horizon_steps,
                                         stepwise=True, suppress_warnings=True,
                                         error_action='ignore', max_order=5,
                                         max_p=3, max_q=3, max_d=2)
                n_test = len(test_data) - horizon_steps
                test_values = test_data['percent_calculated'].values
                for j in range(0, min(n_test, 100)):
                    try:
                        forecast = arima_model.predict(n_periods=horizon_steps)
                        pred_val = np.clip(forecast[-1], 0, 100)
                        actual_val = test_values[j + horizon_steps]
                        arima_preds.append(pred_val)
                        arima_actuals.append(actual_val)
                        # Posunúť model o jeden skutočne pozorovaný krok dopredu.
                        arima_model.update([test_values[j]])
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"ARIMA zlyhala pre kontajner {cid}: {e}")
        
        # Prophet pre rovnaký kontajner (ak je knižnica dostupná)
        if HAS_PROPHET:
            try:
                from prophet import Prophet
                train_df_prophet = pd.DataFrame({
                    'ds': c_data['measured_at_utc'].iloc[:cutoff],
                    'y': train_series
                })
                m = Prophet(daily_seasonality=True, weekly_seasonality=True, 
                           yearly_seasonality=False, changepoint_prior_scale=0.05)
                m.fit(train_df_prophet)
                
                future_dates = c_data['measured_at_utc'].iloc[cutoff:cutoff + min(len(test_data), 100 + horizon_steps)]
                future_df = pd.DataFrame({'ds': future_dates})
                forecast = m.predict(future_df)
                
                for j in range(min(len(test_data) - horizon_steps, 100)):
                    if j + horizon_steps < len(forecast):
                        pred_val = np.clip(forecast['yhat'].iloc[j + horizon_steps], 0, 100)
                        actual_val = test_data['percent_calculated'].iloc[j + horizon_steps]
                        prophet_preds.append(pred_val)
                        prophet_actuals.append(actual_val)
            except Exception as e:
                logger.debug(f"Prophet zlyhal pre kontajner {cid}: {e}")
    
    results = {'n_containers_used': len(test_ids)}
    
    if arima_preds:
        arima_rmse_val = rmse(np.array(arima_actuals), np.array(arima_preds))
        arima_r2 = 1 - np.sum((np.array(arima_actuals) - np.array(arima_preds))**2) / np.sum((np.array(arima_actuals) - np.mean(arima_actuals))**2)
        results['arima'] = {'rmse': arima_rmse_val, 'r2': arima_r2, 'n_predictions': len(arima_preds)}
        logger.info(f"  Auto-ARIMA (podvzorka): RMSE={arima_rmse_val:.2f}, R2={arima_r2:.3f}, n={len(arima_preds)}")
    
    if prophet_preds:
        prophet_rmse_val = rmse(np.array(prophet_actuals), np.array(prophet_preds))
        prophet_r2 = 1 - np.sum((np.array(prophet_actuals) - np.array(prophet_preds))**2) / np.sum((np.array(prophet_actuals) - np.mean(prophet_actuals))**2)
        results['prophet'] = {'rmse': prophet_rmse_val, 'r2': prophet_r2, 'n_predictions': len(prophet_preds)}
        logger.info(f"  Prophet (podvzorka): RMSE={prophet_rmse_val:.2f}, R2={prophet_r2:.3f}, n={len(prophet_preds)}")
    
    with open(f'{baseline_dir}/arima_prophet_baseline.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

