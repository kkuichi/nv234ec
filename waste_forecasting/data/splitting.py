"""Stratégie rozdeľovania dát pre tréning a validáciu modelov.

Základom je container holdout - deterministicky vyberiem podmnožinu
kontajnerov ako testovaciu množinu, aby model počas tréningu nikdy nevidel
merania testovacieho kontajnera. Druhou technikou je temporálne rozdelenie
v rámci kontajnera, kde z časovej osi vyčlením novšiu časť ako validačnú.

Kombináciou týchto dvoch dostávam container holdout CV s temporálnym
cutoffom - žiadny tréningový záznam nie je novší ako najstarší validačný
záznam. Ako alternatíva je k dispozícii aj čisto temporálne expanding
window CV. Pre kapacitne segmentované modely modul ponúka ešte
kapacitne stratifikovaný výber testovacej množiny v rámci jedného segmentu.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)

__all__ = [
    'get_unified_test_containers',
    'temporal_train_val_split_per_container',
    'create_temporal_cv_folds',
    'create_container_holdout_cv_folds',
    'create_expanding_window_cv_folds',
    'get_test_containers_for_capacity',
]


def get_unified_test_containers(df: pd.DataFrame, seed: Optional[int] = None) -> set:
    """Deterministicky vybrať podmnožinu kontajnerov ako testovaciu množinu.

    Veľkosť testovacej množiny je daná parametrom
    CONFIG.TEST_CONTAINER_FRACTION. Pri rovnakom seede
    vždy vráti rovnakú množinu, čo je kľúčové pre reprodukovateľnosť
    naprieč krokmi pipeline.
    """
    seed = seed if seed is not None else CONFIG.SEED
    rng = np.random.RandomState(seed)

    all_containers = df['container_id'].unique()
    n_test = int(len(all_containers) * CONFIG.TEST_CONTAINER_FRACTION)

    test_containers = set(rng.choice(all_containers, size=n_test, replace=False))
    logger.info(f"Selected {len(test_containers)} test containers (seed={seed})")

    return test_containers


def temporal_train_val_split_per_container(
    df: pd.DataFrame,
    val_fraction: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Temporálne rozdeliť dáta na train/val v rámci každého kontajnera.

    Pre každý kontajner sa jeho časová os usporiada vzostupne a posledných
    val_fraction záznamov sa označí ako validačné. Tento prístup
    simuluje budúce predikcie - trénujeme na minulosti, validujeme
    na novšom období.
    """
    val_fraction = val_fraction or CONFIG.VAL_FRACTION_PER_CONTAINER

    def split_container(group):
        n = len(group)
        n_val = max(1, int(n * val_fraction))
        group = group.sort_values('measured_at_utc')
        group['_split'] = 'train'
        group.iloc[-n_val:, group.columns.get_loc('_split')] = 'val'
        return group

    df = df.groupby('container_id', group_keys=False).apply(split_container)

    train_df = df[df['_split'] == 'train'].drop(columns=['_split'])
    val_df = df[df['_split'] == 'val'].drop(columns=['_split'])

    return train_df, val_df


def create_temporal_cv_folds(
    df: pd.DataFrame,
    n_folds: Optional[int] = None,
    seed: Optional[int] = None,
    cv_type: Optional[str] = None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Vytvoriť CV foldy pomocou zvolenej stratégie.

    Dispatcher, ktorý podľa cv_type zavolá buď
    create_container_holdout_cv_folds, alebo
    create_expanding_window_cv_folds.
    """
    n_folds = n_folds or CONFIG.N_FOLDS
    seed = seed if seed is not None else CONFIG.SEED
    cv_type = cv_type or CONFIG.CV_TYPE

    if cv_type == 'expanding_window':
        return create_expanding_window_cv_folds(df, n_folds)
    else:
        return create_container_holdout_cv_folds(df, n_folds, seed)


def create_container_holdout_cv_folds(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Container holdout CV s dodatočným temporálnym cutoffom.

    Kontajnery sú náhodne rozdelené do n_folds skupín. V každom
    folde je jedna skupina validačná a ostatné sú trénovacie. V rámci
    validačných kontajnerov sa vyčlení novšia polovica časovej osi
    ako validačné dáta.

    Dodatočne sa aplikuje temporálny cutoff: trénovacie záznamy,
    ktoré sú novšie ako najstarší validačný záznam, sú odfiltrované.
    Toto zabraňuje tomu, aby tréningová množina obsahovala informácie
    z časovej oblasti, v ktorej sa vykonáva validácia.
    """
    containers = df['container_id'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(containers)

    container_folds = np.array_split(containers, n_folds)

    folds = []
    for fold_idx in range(n_folds):
        val_containers = set(container_folds[fold_idx])
        train_containers = set(containers) - val_containers

        train_df = df[df['container_id'].isin(train_containers)].copy()

        val_container_df = df[df['container_id'].isin(val_containers)].copy()
        _, val_df = temporal_train_val_split_per_container(val_container_df, val_fraction=0.5)

        # Temporálny cutoff: trénovacie dáta nesmú obsahovať záznamy
        # novšie ako najstarší validačný záznam.
        if "measured_at_utc" in train_df.columns and "measured_at_utc" in val_df.columns:
            train_df["measured_at_utc"] = pd.to_datetime(train_df["measured_at_utc"], errors="coerce")
            val_df["measured_at_utc"] = pd.to_datetime(val_df["measured_at_utc"], errors="coerce")

            cutoff = val_df["measured_at_utc"].min()
            train_df = train_df[train_df["measured_at_utc"] < cutoff].copy()
            val_df = val_df[val_df["measured_at_utc"] >= cutoff].copy()

            logger.info(f"CV time cutoff applied. cutoff={cutoff}. train_rows={len(train_df)}, val_rows={len(val_df)}")

        folds.append((train_df, val_df))
        logger.info(f"  Fold {fold_idx+1}: Train {len(train_df):,}, Val {len(val_df):,}")

    return folds


def create_expanding_window_cv_folds(
    df: pd.DataFrame,
    n_folds: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding window CV so striktne temporálnym rozdelením.

    Časová os datasetu sa rozdelí na proporcie: prvý fold trénuje na
    prvých 60 % časového rozsahu a validuje na nasledujúcich 10 %,
    každý ďalší fold posúva hranicu o 10 %. Výsledok je
    stále rastúca trénovacia množina a posúvajúca sa validácia,
    čo zodpovedá časovému horizontu skutočného prevádzkového nasadenia.
    """
    df = df.sort_values('measured_at_utc')
    min_time = df['measured_at_utc'].min()
    max_time = df['measured_at_utc'].max()
    time_range = (max_time - min_time).total_seconds()

    val_window_pct = 0.10
    initial_train_pct = 0.60

    folds = []
    for fold_idx in range(n_folds):
        train_end_pct = initial_train_pct + fold_idx * val_window_pct
        val_end_pct = train_end_pct + val_window_pct

        if val_end_pct > 1.0:
            break

        train_end_time = min_time + timedelta(seconds=time_range * train_end_pct)
        val_end_time = min_time + timedelta(seconds=time_range * val_end_pct)

        train_df = df[df['measured_at_utc'] < train_end_time].copy()
        val_df = df[(df['measured_at_utc'] >= train_end_time) &
                    (df['measured_at_utc'] < val_end_time)].copy()

        if len(train_df) > 0 and len(val_df) > 0:
            folds.append((train_df, val_df))
            logger.info(f"  Fold {fold_idx+1} (Expanding): Train {len(train_df):,}, Val {len(val_df):,}")

    return folds


def get_test_containers_for_capacity(
    df: pd.DataFrame,
    capacity_class: str,
    seed: Optional[int] = None,
) -> set:
    """Stabilne vybrať testovaciu množinu v rámci jedného kapacitného segmentu.

    Pre každý segment používame iný seed offset, aby testovacie
    množiny segmentov boli navzájom nezávislé a súčasne deterministické
    pre daný hlavný seed.
    """
    seed = seed if seed is not None else CONFIG.SEED
    seed_offsets = {'low': 0, 'medium': 100, 'high': 200, 'unknown': 300}
    rng = np.random.RandomState(int(seed) + seed_offsets.get(capacity_class, 500))
    df_cap = df[df['capacity_class'] == capacity_class]
    all_containers = df_cap['container_id'].unique()
    n_test = int(len(all_containers) * CONFIG.TEST_CONTAINER_FRACTION)
    n_test = max(1, n_test) if len(all_containers) > 0 else 0
    return set(rng.choice(all_containers, size=n_test, replace=False)) if n_test > 0 else set()
