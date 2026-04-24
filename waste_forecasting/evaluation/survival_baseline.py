"""Survival-analýza ako baseline pre Experiment C.

Experiment C v hlavnej pipeline rieši time-to-threshold regresne,
teda predikuje spojitý počet dní, kým kontajner dosiahne
CONFIG.FILL_THRESHOLD. Tento prístup však nevie priamo pracovať
s pravostranným cenzorovaním. Kontajnery, ktoré počas sledovaného
horizontu prah nedosiahli, musia byť buď vylúčené, čo vytvára
selekčné skreslenie, alebo im je priradená hodnota MAX_TTT_DAYS,
čo môže skresľovať tréning.

Modul preto dopĺňa dva klasické nástroje z analýzy prežívania,
ktoré s cenzorovaním pracujú natívne: Kaplan-Meier a Cox
Proportional Hazards. Kaplan-Meier odhaduje funkciu prežitia
neparametricky, Cox PH je semi-parametrický regresný model
s aktuálnou plnosťou ako prediktorom.

Modely sa spúšťajú cez knižnicu lifelines. Ak knižnica nie je
nainštalovaná, funkcia vráti slovník s kľúčom error a hlavná
pipeline pokračuje bez survival baseline.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd

from config import CONFIG
from waste_forecasting.data.loading import cleanup_dataframes
from waste_forecasting.data.preprocessing import detect_collections
from waste_forecasting.models.metrics import mae, rmse

logger = logging.getLogger(__name__)

__all__ = ['run_survival_baseline_expC']


def run_survival_baseline_expC(df: pd.DataFrame, test_containers: set, output_dir: str) -> Dict:
    """Spustí Kaplan-Meier a Cox PH baseline pre Experiment C.

    Pre každé meranie pod prahom CONFIG.FILL_THRESHOLD, typicky 85 %,
    sa skonštruuje dvojica (duration, event): čas do najbližšieho
    dosiahnutia prahu, alebo cenzorovaná hodnota MAX_TTT_DAYS s
    event=False, ak prah nebol počas nasledujúcich CONFIG.MAX_TTT_DAYS
    dní dosiahnutý.

    Kaplan-Meier sa trénuje iba na dĺžkach a príznaku udalosti, bez
    kovariátov, a ako predikciu poskytuje mediánový čas prežitia.
    Cox PH pridáva jednu kovariátu, aktuálnu plnosť, a predikuje
    mediánový čas do prahu podmienený touto hodnotou.

    Vyhodnotenie prebieha len na testovacích kontajneroch a iba na
    vzorkách s pozorovanou udalosťou, aby bolo porovnanie s regresným
    Experimentom C konzistentné. Pri Cox PH sa z predikcií odstránia
    nekonečné hodnoty a zvyšné hodnoty sa orežú na interval
    [0, MAX_TTT_DAYS].
    """
    logger.info("=" * 70)
    logger.info("SURVIVAL REFERENČNÝ MODEL: EXPERIMENT C")
    logger.info("=" * 70)
    
    surv_dir = f'{output_dir}/exp_C'
    os.makedirs(surv_dir, exist_ok=True)
    
    try:
        from lifelines import KaplanMeierFitter, CoxPHFitter
        HAS_LIFELINES = True
    except ImportError:
        logger.warning("lifelines nie je nainštalovaný. Inštalácia: pip install lifelines")
        HAS_LIFELINES = False
    
    if not HAS_LIFELINES:
        return {'error': 'lifelines nie je nainštalovaný'}
    
    # Vybudovanie TTT datasetu s informáciou o cenzorovaní
    df_det = detect_collections(df.copy())
    
    # Pre každé meranie vypočítaj čas do dosiahnutia prahu 85 %
    results_list = []
    for cid in df_det['container_id'].unique():
        c_data = df_det[df_det['container_id'] == cid].sort_values('measured_at_utc')
        pcts = c_data['percent_calculated'].values
        times = c_data['measured_at_utc'].values
        
        for i in range(len(c_data)):
            current_pct = pcts[i]
            if current_pct >= CONFIG.FILL_THRESHOLD:
                continue
            
            # Hľadáme čas, kedy sa prah dosiahol
            event_observed = False
            duration = CONFIG.MAX_TTT_DAYS  # predvolene cenzorované na maximum
            
            for j in range(i + 1, len(c_data)):
                if pcts[j] >= CONFIG.FILL_THRESHOLD:
                    delta = (times[j] - times[i]) / np.timedelta64(1, 'D')
                    if delta <= CONFIG.MAX_TTT_DAYS:
                        duration = delta
                        event_observed = True
                    break
            
            results_list.append({
                'container_id': cid,
                'duration': max(duration, 0.01),
                'event': event_observed,
                'percent_calculated': current_pct,
                'is_test': cid in test_containers,
            })
    
    surv_df = pd.DataFrame(results_list)
    cleanup_dataframes(df_det)
    
    if len(surv_df) == 0:
        logger.warning("Nevznikli žiadne dáta pre survival model")
        return {'error': 'no data'}
    
    train_surv = surv_df[~surv_df['is_test']].copy()
    test_surv = surv_df[surv_df['is_test']].copy()
    
    logger.info(f"  Survival data: tréning={len(train_surv)}, test={len(test_surv)}")
    logger.info(f"  Pozorované udalosti: tréning={train_surv['event'].sum()}/{len(train_surv)}, test={test_surv['event'].sum()}/{len(test_surv)}")
    
    results = {}
    
    # Kaplan-Meier: mediánový čas prežitia ako konštantná predikcia
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(train_surv['duration'], event_observed=train_surv['event'])
        km_median = kmf.median_survival_time_
        
        # Použijeme medián ako jednu konštantnú hodnotu pre všetky testovacie prípady
        test_events = test_surv[test_surv['event']].copy()
        if len(test_events) > 0:
            km_pred = np.full(len(test_events), km_median)
            km_actual = test_events['duration'].values
            km_rmse = rmse(km_actual, km_pred)
            km_mae = mae(km_actual, km_pred)
            km_r2 = 1 - np.sum((km_actual - km_pred)**2) / np.sum((km_actual - np.mean(km_actual))**2)
            
            results['kaplan_meier'] = {
                'median_survival': float(km_median),
                'rmse': float(km_rmse),
                'mae': float(km_mae),
                'r2': float(km_r2),
                'n': len(test_events),
            }
            logger.info(f"  Kaplan-Meier: medián={km_median:.2f} dňa, RMSE={km_rmse:.2f} dňa, MAE={km_mae:.2f} dňa, R2={km_r2:.3f}")
    except Exception as e:
        logger.warning(f"  Kaplan-Meier zlyhal: {e}")
    
    # Cox Proportional Hazards: aktuálne percento plnosti ako kovariát
    try:
        cox_train = train_surv[['duration', 'event', 'percent_calculated']].copy()
        cox_train = cox_train.dropna()
        
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_train, duration_col='duration', event_col='event')
        
        test_events = test_surv[test_surv['event']].copy()
        test_events = test_events.dropna(subset=['percent_calculated'])
        
        if len(test_events) > 0:
            cox_median_preds = cph.predict_median(test_events[['percent_calculated']])
            cox_pred = cox_median_preds.values.flatten()
            
            # Ošetrenie nekonečných predikcií (tie odpovedajú cenzorovaným prípadom)
            valid = np.isfinite(cox_pred)
            if valid.sum() > 0:
                cox_actual = test_events['duration'].values[valid]
                cox_pred_valid = np.clip(cox_pred[valid], 0, CONFIG.MAX_TTT_DAYS)
                
                cox_rmse = rmse(cox_actual, cox_pred_valid)
                cox_mae_val = mae(cox_actual, cox_pred_valid)
                cox_r2 = 1 - np.sum((cox_actual - cox_pred_valid)**2) / np.sum((cox_actual - np.mean(cox_actual))**2)
                
                results['cox_ph'] = {
                    'rmse': float(cox_rmse),
                    'mae': float(cox_mae_val),
                    'r2': float(cox_r2),
                    'n': int(valid.sum()),
                    'concordance': float(cph.concordance_index_),
                }
                logger.info(f"  Cox PH: RMSE={cox_rmse:.2f} dňa, MAE={cox_mae_val:.2f} dňa, R2={cox_r2:.3f}, C-index={cph.concordance_index_:.3f}")
    except Exception as e:
        logger.warning(f"  Cox PH zlyhal: {e}")
    
    with open(f'{surv_dir}/survival_baseline.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    cleanup_dataframes(surv_df, train_surv, test_surv)
    return results

