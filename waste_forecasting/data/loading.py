"""Načítanie a prvotné predspracovanie surových meraní plnosti.

Modul implementuje načítanie vstupného CSV, overenie povinných stĺpcov,
odstránenie neplatných riadkov (neznámy typ odpadu, percento mimo
intervalu 0 - 100) a odvodenie pomocných stĺpcov (capacity_class,
district_num). Súčasťou modulu je aj pamäťová optimalizácia
konverziou numerických typov na užšie reprezentácie.
"""

from __future__ import annotations

import gc
import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)

__all__ = [
    'reduce_memory_usage',
    'cleanup_dataframes',
    'parse_capacity_from_type',
    'classify_capacity',
    'add_capacity_class_per_container',
    'load_and_preprocess_data',
]


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Znížiť pamäťovú stopu DataFrame downcastom číselných stĺpcov.

    Pre každý numerický stĺpec (okrem object a category) sa
    určí najmenší numpy typ, ktorý pokryje rozsah hodnôt v stĺpci.
    Optimalizácia je bezstratová, teda zachováva všetky hodnoty.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()

            # Celočíselný downcast: hľadáme najužší signed integer,
            # ktorý pokryje aktuálny rozsah hodnôt.
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            # Pre float stĺpce konvertujeme na float32 ak to rozsah dovolí.
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        logger.info(f"Memory: {start_mem:.1f}MB → {end_mem:.1f}MB ({reduction:.1f}% reduction)")

    return df


def cleanup_dataframes(*dfs) -> None:
    """Explicitne uvoľniť referencie na DataFrame objekty a vyvolať GC.

    Používané pri výpočtoch nad veľkými tabuľkami po ukončení
    medzifázy, aby sa skrátila špičková pamäťová stopa pipeline.
    """
    for df in dfs:
        if df is not None:
            del df
    gc.collect()


def parse_capacity_from_type(container_type: Optional[str]) -> float:
    """Extrahovať nominálnu kapacitu kontajnera z reťazca s typom.

    Parsuje vedúce číslo z reťazcov ako "3000 Podzemní SV" alebo
    "1100 Nadzemní". Pri chýbajúcej alebo neparsovateľnej hodnote
    vracia NaN.
    """
    if pd.isna(container_type) or not isinstance(container_type, str):
        return np.nan
    match = re.match(r"\s*(\d+)", container_type)
    return float(match.group(1)) if match else np.nan


def classify_capacity(capacity: float) -> str:
    """Klasifikovať kontajner do kapacitnej triedy na základe prahov.

    Hranice oddeľujúce kategórie sú v CONFIG.CAPACITY_THRESHOLD_LOW
    (≤ → 'low') a CONFIG.CAPACITY_THRESHOLD_HIGH
    (≥ → 'high'); hodnoty medzi nimi spadajú do triedy 'medium'.
    """
    if pd.isna(capacity):
        return 'unknown'
    if capacity <= CONFIG.CAPACITY_THRESHOLD_LOW:
        return 'low'
    if capacity >= CONFIG.CAPACITY_THRESHOLD_HIGH:
        return 'high'
    return 'medium'


def add_capacity_class_per_container(df: pd.DataFrame) -> pd.DataFrame:
    """Pre každý kontajner odvodiť stĺpec capacity_class.

    Z dôvodu občasných nekonzistencií v surových dátach (jeden kontajner
    niekedy obsahuje viac hodnôt container_type) sa ako
    reprezentatívna hodnota berie modus na úrovni kontajnera. Táto
    hodnota sa následne parsuje a klasifikuje.
    """
    df = df.copy()

    # Pre každý kontajner vyberieme modus hodnoty container_type
    # (ignoruje NaN); ak žiadna validná hodnota nie je, vraciame NaN.
    ct_mode = df.groupby('container_id')['container_type'].agg(
        lambda s: s.dropna().mode().iloc[0] if len(s.dropna().mode()) > 0 else np.nan
    )

    capacity = ct_mode.apply(parse_capacity_from_type)
    capacity_class = capacity.apply(classify_capacity).rename('capacity_class')

    # Ak stĺpec už v DataFrame existuje, zahodíme ho pred merge-om.
    if 'capacity_class' in df.columns:
        df = df.drop(columns=['capacity_class'])

    return df.merge(capacity_class.reset_index(), on='container_id', how='left')


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Načítať CSV s meraniami plnosti a vykonať prvotné predspracovanie.

    Postup spracovania:

    1. Overí existenciu súboru (pri chybe vyvolá FileNotFoundError).
    2. Načíta CSV cez pd.read_csv.
    3. Overí prítomnosť povinných stĺpcov container_id,
       measured_at_utc, percent_calculated, trash_type.
    4. Skonvertuje časové pečiatky na pd.Timestamp a usporiada
       záznamy podľa (container_id, measured_at_utc).
    5. Odstráni riadky so stringom 'neznamy' alebo NaN v stĺpci
       trash_type a riadky s percent_calculated mimo [0, 100].
    6. Zneplatní outliery teploty mimo intervalu [-30, 50] °C.
    7. Binarizuje indikátor firealarm podľa podmienky > 0.
    8. Ak je prítomný stĺpec container_type, odvodí capacity_class.
    9. Z district extrahuje číselný obvod do district_num.
    10. Zredukuje pamäťovú stopu pomocou reduce_memory_usage.
    """
    logger.info("=" * 70)
    logger.info("Načítanie dát")
    logger.info("=" * 70)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Súbor s dátami sa nenašiel: {filepath}")

    required_columns = ['container_id', 'measured_at_utc', 'percent_calculated', 'trash_type']

    df = pd.read_csv(filepath)

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    logger.info(f"Načítaných riadkov: {len(df):,}")

    df['measured_at_utc'] = pd.to_datetime(df['measured_at_utc'])
    df = df.sort_values(['container_id', 'measured_at_utc']).reset_index(drop=True)

    # Odstránenie neznámych typov odpadu.
    initial = len(df)
    df = df[df['trash_type'].notna() & (df['trash_type'] != 'neznamy')]
    logger.info(f"Odstránených riadkov s neznámym alebo chýbajúcim typom odpadu: {initial - len(df):,}")

    # Odstránenie neplatných hodnôt plnosti.
    initial = len(df)
    df = df[(df['percent_calculated'] >= 0) & (df['percent_calculated'] <= 100)]
    logger.info(f"Odstránených riadkov s neplatnou plnosťou: {initial - len(df):,}")

    # Teplota môže chýbať v surovom CSV - ošetríme oba prípady.
    if 'temperature' in df.columns:
        temp_outliers = ((df['temperature'] < -30) | (df['temperature'] > 50)).sum()
        df.loc[(df['temperature'] < -30) | (df['temperature'] > 50), 'temperature'] = np.nan
        if temp_outliers > 0:
            logger.info(f"Počet odľahlých hodnôt teploty nastavených na NaN: {temp_outliers:,}")
    else:
        df['temperature'] = np.nan  # Zachovanie stĺpca pre ďalšie kroky.

    # Indikátor požiarneho alarmu binarizujeme alebo dopĺňame nulou.
    if 'firealarm' in df.columns:
        df['firealarm'] = (df['firealarm'] > 0).astype(int)
    else:
        df['firealarm'] = 0

    # Odvodenie kapacitnej triedy iba ak je k dispozícii container_type.
    if 'container_type' in df.columns:
        df = add_capacity_class_per_container(df)
    else:
        df['capacity_class'] = 'unknown'

    # Z označenia pražského obvodu vyparsujeme číslo (napr. "praha-7" → 7).
    if 'district' in df.columns:
        df['district_num'] = df['district'].astype(str).str.extract(r'praha-(\d+)').astype(float)
    else:
        df['district_num'] = np.nan

    df = reduce_memory_usage(df)

    logger.info(f"Finálny dataset: {len(df):,} riadkov, {df['container_id'].nunique()} kontajnerov")

    cap_dist = df.groupby('capacity_class')['container_id'].nunique()
    logger.info(f"Rozdelenie podľa kapacity:\n{cap_dist.to_string()}")

    return df
