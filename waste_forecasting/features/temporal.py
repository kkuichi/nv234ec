"""Tvorba časových príznakov: lagy, kĺzavé štatistiky a experimentálne datasety.

Modul implementuje konštrukciu sád príznakov pre všetky tri experimenty:

* Experiment A: lagy a kĺzavé príznaky počítané nad detegovanými vývozmi.
  Cieľová premenná je plnosť pred vývozom.
* Experiment B: lagy a kĺzavé príznaky nad pravidelne prevzorkovanou
  časovou sériou. Cieľová premenná je hodnota plnosti o 24 hodín neskôr.
* Experiment C: príznaky pre odhad počtu dní do dosiahnutia 85 % plnosti,
  vrátane lineárnej extrapolácie.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import CONFIG
from waste_forecasting.data.preprocessing import resample_time_series

logger = logging.getLogger(__name__)

__all__ = [
    'create_event_dataset',
    'add_event_features',
    'create_time_dataset',
    'add_time_features',
    'create_time_to_threshold_dataset',
    'add_ttt_features',
    'add_missing_indicators',
    'get_feature_columns',
]


def create_event_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Zostaviť dataset vývozov pre Experiment A.

    Z tabuľky meraní vyberie iba riadky s detegovaným vývozom
    (is_collection == 1) a ako cieľovú premennú priradí
    hodnotu plnosti pred vývozom (pct_prev). Zároveň odvodí
    bežné kalendárne príznaky z časovej pečiatky.
    """
    collections = df[df['is_collection'] == 1].copy()
    collections['target'] = collections['pct_prev']

    # Kalendárne príznaky extrahované zo časovej pečiatky.
    collections['date'] = collections['measured_at_utc'].dt.normalize()
    collections['hour'] = collections['measured_at_utc'].dt.hour
    collections['day_of_week'] = collections['measured_at_utc'].dt.dayofweek
    collections['day_of_month'] = collections['measured_at_utc'].dt.day
    collections['month'] = collections['measured_at_utc'].dt.month
    collections['week_of_year'] = collections['measured_at_utc'].dt.isocalendar().week.astype(int)
    collections['is_weekend'] = (collections['day_of_week'] >= 5).astype(int)

    collections = collections.dropna(subset=['target'])

    logger.info(f"Dataset vývozov: {len(collections):,} vývozov, {collections['container_id'].nunique()} kontajnerov")

    return collections


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pridať lagy a kĺzavé príznaky pre predikciu pri vývoze.

    Pre každý kontajner vygeneruje:

    * target_lag_{1..10} - hodnoty cieľovej premennej v predchádzajúcich
      vývozoch;
    * interval_lag_{1..3} - počet dní medzi predchádzajúcimi vývozmi;
    * target_roll_{mean,std,max}_{3,5,7} - kĺzavé štatistiky cieľovej
      premennej cez okná 3, 5, 7 vývozov;
    * target_same_dow - lag rovnakého dňa v týždni;
    * target_trend - rozdiel posledných dvoch cieľov;
    * fill_rate - tempo plnenia (cieľ / interval);
    * container_hist_{mean,std} - expandujúci priemer a smerodajná
      odchýlka cieľovej premennej.
    """
    df = df.sort_values(['container_id', 'measured_at_utc']).copy()

    # Lagy cieľovej premennej.
    for lag in CONFIG.EVENT_LAGS:
        df[f'target_lag_{lag}'] = df.groupby('container_id')['target'].shift(lag)

    # Intervaly medzi vývozmi.
    df['days_since_prev_collection'] = df.groupby('container_id')['date'].diff().dt.days
    for lag in [1, 2, 3]:
        df[f'interval_lag_{lag}'] = df.groupby('container_id')['days_since_prev_collection'].shift(lag)

    # Kĺzavé štatistiky (shift(1) zabraňuje použitiu aktuálnej hodnoty).
    for window in CONFIG.EVENT_ROLL_WINDOWS:
        df[f'target_roll_mean_{window}'] = df.groupby('container_id')['target'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'target_roll_std_{window}'] = df.groupby('container_id')['target'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
        df[f'target_roll_max_{window}'] = df.groupby('container_id')['target'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )
        df[f'interval_roll_mean_{window}'] = df.groupby('container_id')['days_since_prev_collection'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Sezónne lagy a trend.
    df['target_same_dow'] = df.groupby(['container_id', 'day_of_week'])['target'].shift(1)
    df['target_trend'] = df['target_lag_1'] - df['target_lag_2']
    df['fill_rate'] = df['target_lag_1'] / df['interval_lag_1'].replace(0, np.nan)

    # Historický priemer/odchýlka (expanding window s shift(1)).
    df['container_hist_mean'] = df.groupby('container_id')['target'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['container_hist_std'] = df.groupby('container_id')['target'].transform(
        lambda x: x.shift(1).expanding().std()
    )

    df = add_missing_indicators(df, ['target_lag_1', 'target_lag_2', 'target_lag_3', 'target_same_dow'])

    return df


def create_time_dataset(df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """Zostaviť časový dataset pre Experiment B.

    Prevzorkuje vstupné merania na pravidelný krok
    CONFIG.RESAMPLE_HOURS a ako cieľovú premennú
    priradí hodnotu plnosti posunutú o horizon_hours dopredu.
    """
    df_resampled = resample_time_series(df)

    # Počet krokov prevzorkovania, ktoré zodpovedajú horizontu predikcie.
    n_steps = horizon_hours // CONFIG.RESAMPLE_HOURS
    df_resampled['target'] = df_resampled.groupby('container_id')['percent_calculated'].shift(-n_steps)

    df_resampled['date'] = df_resampled['measured_at_utc'].dt.normalize()
    df_resampled['hour'] = df_resampled['measured_at_utc'].dt.hour
    df_resampled['day_of_week'] = df_resampled['measured_at_utc'].dt.dayofweek
    df_resampled['month'] = df_resampled['measured_at_utc'].dt.month
    df_resampled['is_weekend'] = (df_resampled['day_of_week'] >= 5).astype(int)

    return df_resampled.dropna(subset=['target'])


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pridať lagy a kĺzavé príznaky pre časovú predikciu.

    Na rozdiel od verzie pre vývozy pracuje s pravidelnou
    časovou sériou. Vyrába:

    * fill_lag_{1..48} - lag hodnoty plnosti v krokoch prevzorkovania;
    * fill_diff_1 a fill_diff_24h - rozdiel oproti 1. a 24-hodinovému lagu;
    * fill_roll_{mean,std}_{6,12,24} - kĺzavé štatistiky cez okná;
    * had_drop a had_drop_24h - indikátor výrazného poklesu
      a rolling flag za posledných 24 hodín;
    * container_fill_mean - expandujúci priemer plnosti.
    """
    df = df.sort_values(['container_id', 'measured_at_utc']).copy()

    df['current_fill'] = df['percent_calculated']

    # Jednoduché lagy.
    for lag in CONFIG.TIME_LAGS:
        df[f'fill_lag_{lag}'] = df.groupby('container_id')['percent_calculated'].shift(lag)

    df['fill_diff_1'] = df['percent_calculated'] - df.groupby('container_id')['percent_calculated'].shift(1)

    # Lag presne o 24 hodín dozadu (závisí od kroku prevzorkovania).
    steps_24h = 24 // CONFIG.RESAMPLE_HOURS
    df['fill_lag_24h'] = df.groupby('container_id')['percent_calculated'].shift(steps_24h)
    df['fill_diff_24h'] = df['percent_calculated'] - df['fill_lag_24h']

    # Kĺzavé okná (shift(1) zabraňuje použitiu aktuálnej hodnoty).
    for window in CONFIG.TIME_ROLL_WINDOWS:
        df[f'fill_roll_mean_{window}'] = df.groupby('container_id')['percent_calculated'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'fill_roll_std_{window}'] = df.groupby('container_id')['percent_calculated'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    # Detekcia prudkého poklesu ako proxy pre vývoz na prevzorkovanej sérii.
    df['had_drop'] = (df['fill_diff_1'] < -25).astype(int)
    df['had_drop_24h'] = df.groupby('container_id')['had_drop'].transform(
        lambda x: x.shift(1).rolling(steps_24h, min_periods=1).max()
    ).fillna(0).astype(int)

    df['container_fill_mean'] = df.groupby('container_id')['percent_calculated'].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    return df


def create_time_to_threshold_dataset(df_resampled: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Zostaviť dataset pre Experiment C, teda čas do dosiahnutia prahu.

    Pre každé meranie sa hľadá počet krokov do najbližšieho prekročenia
    prahu CONFIG.FILL_THRESHOLD. Ak medzi aktuálnym časom
    a dosiahnutím prahu dôjde k výraznému poklesu plnosti (predpokladaný
    vývoz, pokles > 25 p. b.), hľadanie sa preruší a kontajner sa pre
    tento časový bod nezaradí.

    Funkcia loguje WARNING o selekčnom skreslení (SELECTION BIAS),
    pretože do výsledného datasetu vstupujú iba kontajnery, pri ktorých 
    bol aspoň raz pozorovaný prechod cez prah.
    """
    logger.info(f"Vytváram TTT dataset (prah: {CONFIG.FILL_THRESHOLD} %)")

    df = df_resampled.copy()
    n_containers_total = df['container_id'].nunique()
    containers_with_threshold = set()

    def calc_ttt(group):
        """Pre každý záznam spočítať počet hodín do prahu 85 %."""
        pct = group['percent_calculated'].values
        n = len(pct)
        result = np.full(n, np.nan)
        cid = group['container_id'].iloc[0]

        for i in range(n):
            if pct[i] >= CONFIG.FILL_THRESHOLD:
                # Hodnota je už nad prahom, preto je čas do dosiahnutia prahu rovný 0.
                result[i] = 0
                containers_with_threshold.add(cid)
                continue

            # Hľadáme najbližší nasledujúci prechod cez prah,
            # maximálne 100 krokov dopredu.
            for j in range(i + 1, min(i + 100, n)):
                # Ak medzi aktuálnym a budúcim časom došlo k výraznému
                # poklesu (>25 p. b.), pravdepodobne ide o vývoz -
                # hľadanie prerušíme.
                if j > i and pct[j] < pct[j-1] - 25:
                    break
                if pct[j] >= CONFIG.FILL_THRESHOLD:
                    result[i] = (j - i) * CONFIG.RESAMPLE_HOURS
                    containers_with_threshold.add(cid)
                    break

        return pd.Series(result, index=group.index)

    df['time_to_threshold_hours'] = df.groupby('container_id', group_keys=False).apply(calc_ttt)

    # Odstránime záznamy bez platného TTT a obmedzíme na 14-dňový horizont.
    df = df.dropna(subset=['time_to_threshold_hours'])
    df = df[(df['time_to_threshold_hours'] > 0) &
            (df['time_to_threshold_hours'] <= CONFIG.MAX_TTT_DAYS * 24)]

    # Cieľovú premennú konvertujeme z hodín na dni.
    df['target'] = df['time_to_threshold_hours'] / 24

    n_containers_with_ttt = len(containers_with_threshold)
    selection_bias_info = {
        'n_containers_total': n_containers_total,
        'n_containers_with_ttt': n_containers_with_ttt,
        'coverage_pct': 100 * n_containers_with_ttt / n_containers_total if n_containers_total > 0 else 0,
        'threshold': CONFIG.FILL_THRESHOLD,
    }

    logger.info(f"TTT dataset: {len(df):,} vzoriek, priemerný TTT: {df['target'].mean():.1f} dňa")
    logger.warning(
        f"Selekčné skreslenie: TTT dataset pokrýva {n_containers_with_ttt}/{n_containers_total} "
        f"kontajnerov ({selection_bias_info['coverage_pct']:.1f} %)."
    )

    return df, selection_bias_info


def add_ttt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pridať príznaky pre Experiment C, teda čas do dosiahnutia prahu.

    Okrem štandardných lagov generuje aj derivované príznaky
    fill_gap (vzdialenosť od prahu), fill_rate_per_hour
    (odhadnuté tempo plnenia) a estimated_time_linear
    (lineárna extrapolácia dní do prahu).
    """
    df = df.copy()

    df['current_fill'] = df['percent_calculated']
    df['fill_gap'] = CONFIG.FILL_THRESHOLD - df['percent_calculated']

    steps_24h = 24 // CONFIG.RESAMPLE_HOURS
    for lag in [1, 2, steps_24h, steps_24h*2]:
        col_name = f'fill_lag_{lag}'
        if col_name not in df.columns:
            df[col_name] = df.groupby('container_id')['percent_calculated'].shift(lag)

    df['fill_change_24h'] = df['percent_calculated'] - df.get(f'fill_lag_{steps_24h}', df['percent_calculated'])
    df['fill_rate_per_hour'] = df['fill_change_24h'] / 24

    # Lineárna extrapolácia: za predpokladu konštantného tempa plnenia
    # by sa prah dosiahol za fill_gap / fill_rate_per_hour hodín.
    df['estimated_time_linear'] = df['fill_gap'] / (df['fill_rate_per_hour'].replace(0, np.nan) * 24)
    df['estimated_time_linear'] = df['estimated_time_linear'].clip(0, CONFIG.MAX_TTT_DAYS)

    if 'hour' not in df.columns:
        df['hour'] = df['measured_at_utc'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['measured_at_utc'].dt.dayofweek
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    return df


def add_missing_indicators(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Pridať binárne indikátory chýbajúcich hodnôt pre zadané stĺpce.

    Pre každý zadaný stĺpec sa vytvorí nový stĺpec s príponou
    _missing obsahujúci 1, ak je hodnota NaN, inak 0.
    Stromové modely môžu tieto indikátory využiť na rozlíšenie
    medzi skutočnou hodnotou a chýbajúcou informáciou.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isna().astype(np.int8)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Vrátiť zoznam numerických stĺpcov vhodných ako príznaky.

    Z všetkých numerických stĺpcov odfiltruje identifikátory,
    časové pečiatky, cieľovú premennú a pomocné stĺpce, ktoré
    sa nesmú dostať do modelu (kvôli riziku data leakage alebo
    z dôvodu irelevantnosti).
    """
    exclude = {
        'id', 'container_id', 'code', 'sensor_code', 'station_id', 'station_name',
        'station_number', 'measured_at_utc', 'prediction_utc', 'date', '_split',
        'target', 'pct_prev', 'pct_change', 'percent_calculated',
        'time_to_threshold_hours', 'time_prev', 'is_collection', 'had_drop',
        'trash_type', 'container_type', 'district', 'accessibility',
        'sensor_supplier', 'cleaning_duration', 'capacity_class',
        'geo_bin_lat', 'geo_bin_lon', 'pct_next'
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]
