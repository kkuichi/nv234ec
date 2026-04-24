"""Predspracovanie dát: detekcia vývozov, prevzorkovanie, anti-leakage kontroly.

Modul implementuje heuristickú detekciu vývozov odpadu (slúži ako proxy
cieľovej premennej pre Experiment A), prevzorkovanie časovej série na
pravidelný krok pre Experiment B, filtrovanie kontajnerov s nedostatkom
meraní a dve validačné kontroly - overenie absencie data leakage
a detekciu vysokých korelácií medzi príznakmi.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)

__all__ = [
    'detect_collections',
    'sensitivity_analysis_collections',
    'resample_time_series',
    'filter_containers_by_min_samples',
    'verify_no_data_leakage',
    'check_feature_correlations',
]


def detect_collections(
    df: pd.DataFrame,
    drop_threshold: Optional[float] = None,
    low_after: Optional[float] = None,
    high_before: Optional[float] = None,
    use_future_check: Optional[bool] = None,
) -> pd.DataFrame:
    """Heuristicky detegovať vývozy odpadu z časovej série plnosti.

    Vývoz je indikovaný prudkým poklesom plnosti pri splnení niekoľkých
    podmienok súčasne:

    * pokles oproti predchádzajúcemu meraniu je nižší ako adaptívny
      prah (max(10. percentil poklesov kontajnera, drop_threshold));
    * časový odstup medzi meraniami nie je väčší ako
      CONFIG.MAX_HOURS_BETWEEN_MEASUREMENTS;
    * aktuálna plnosť (po vývoze) je pod low_after;
    * predchádzajúca plnosť (pred vývozom) je nad high_before;
    * voliteľne aj kontrola budúceho merania (vypnutá predvolene,
      pretože zavádza mierny look-ahead bias).
    """
    drop_threshold = drop_threshold if drop_threshold is not None else CONFIG.COLLECTION_DROP_THRESHOLD
    low_after = low_after if low_after is not None else CONFIG.LOW_AFTER_THRESHOLD
    high_before = high_before if high_before is not None else CONFIG.HIGH_BEFORE_THRESHOLD
    use_future_check = use_future_check if use_future_check is not None else CONFIG.USE_PCT_NEXT_IN_COLLECTION_DETECTION

    df = df.copy()

    # Predchádzajúca hodnota plnosti a časová pečiatka v rámci kontajnera.
    df['pct_prev'] = df.groupby('container_id')['percent_calculated'].shift(1)
    df['pct_change'] = df['percent_calculated'] - df['pct_prev']
    df['time_prev'] = df.groupby('container_id')['measured_at_utc'].shift(1)
    df['hours_since_prev'] = (df['measured_at_utc'] - df['time_prev']).dt.total_seconds() / 3600

    # Adaptívny prah: berieme maximum z globálneho prahu a 10. percentilu
    # poklesov daného kontajnera. Zohľadňuje individuálnu variabilitu
    # senzora a typickú frekvenciu vývozov.
    container_thresholds = df.groupby('container_id')['pct_change'].transform(lambda x: x.quantile(0.10))
    adaptive_threshold = np.maximum(container_thresholds, drop_threshold)

    if use_future_check:
        df['pct_next'] = df.groupby('container_id')['percent_calculated'].shift(-1)
        # Budúce meranie nemá „zasa stúpnuť" - ak po vývoze plnosť
        # pokračuje v nízkych hodnotách, je to ďalšia indikácia vývozu.
        future_ok = (
            (df['pct_next'].isna()) |
            ((df['pct_next'] < 70) & (df['pct_next'] < df['pct_prev']))
        )
    else:
        future_ok = True

    df['is_collection'] = (
        (df['pct_change'] <= adaptive_threshold) &
        (df['hours_since_prev'] <= CONFIG.MAX_HOURS_BETWEEN_MEASUREMENTS) &
        (df['percent_calculated'] < low_after) &
        (df['pct_prev'] > high_before) &
        future_ok
    ).astype(int)

    df = df.drop(columns=['pct_next'], errors='ignore')

    n_collections = int(df['is_collection'].sum())
    n_containers = int(df['container_id'].nunique())
    avg_per_container = n_collections / n_containers if n_containers else 0
    logger.info(f"Detected {n_collections:,} collections ({avg_per_container:.1f} per container avg)")

    return df


def sensitivity_analysis_collections(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Vykonať analýzu citlivosti detektora vývozov na prahové hodnoty.

    Iteruje mriežku hodnôt drop_threshold × low_after a pre každú
    kombináciu zaznamená počet detegovaných vývozov. Výsledok slúži
    na posúdenie robustnosti detekčnej heuristiky voči voľbe prahov.
    """
    logger.info("=" * 70)
    logger.info("SENSITIVITY ANALYSIS: Collection Detection Thresholds")
    logger.info("=" * 70)

    results = []

    for drop_thresh in CONFIG.SENSITIVITY_DROP_THRESHOLDS:
        for low_after in CONFIG.SENSITIVITY_LOW_AFTER:
            df_temp = detect_collections(
                df.copy(),
                drop_threshold=drop_thresh,
                low_after=low_after,
                high_before=CONFIG.HIGH_BEFORE_THRESHOLD
            )

            n_collections = df_temp['is_collection'].sum()
            n_containers = df_temp['container_id'].nunique()

            results.append({
                'drop_threshold': drop_thresh,
                'low_after_threshold': low_after,
                'n_collections': n_collections,
                'collections_per_container': n_collections / n_containers
            })

            del df_temp
            gc.collect()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/sensitivity_collection_thresholds.csv', index=False)

    return results_df


def resample_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Prevzorkovať časovú sériu meraní na pravidelný krok.

    Pre každý kontajner použije resample(last) s krokom
    CONFIG.RESAMPLE_HOURS a medzery vyplní dopredným
    doplnením s limitom dve vzorky (ffill(limit=2)). Dôležité
    metadáta (container_type, trash_type, district,
    capacity_class, súradnice) sú prenesené z prvého neprázdneho
    záznamu a replikované na všetky prevzorkované riadky.
    """
    df = df.sort_values(['container_id', 'measured_at_utc']).copy()
    resample_str = f'{CONFIG.RESAMPLE_HOURS}h'

    meta_cols = ['container_type', 'trash_type', 'district', 'district_num',
                 'capacity_class', 'latitude', 'longitude']

    resampled_chunks = []

    for cid, group in df.groupby('container_id'):
        group = group.set_index('measured_at_utc')
        series = group['percent_calculated'].resample(resample_str).last()
        series = series.ffill(limit=2)

        chunk = series.reset_index()
        chunk.columns = ['measured_at_utc', 'percent_calculated']
        chunk['container_id'] = cid

        # Doplnenie statických metadát z originálnej skupiny.
        for col in meta_cols:
            if col in group.columns:
                vals = group[col].dropna()
                chunk[col] = vals.iloc[0] if len(vals) > 0 else np.nan

        resampled_chunks.append(chunk)

    result = pd.concat(resampled_chunks, ignore_index=True)
    return result.dropna(subset=['percent_calculated'])


def filter_containers_by_min_samples(
    df: pd.DataFrame,
    min_samples: int,
    sample_col: str,
    output_dir: Optional[str] = None,
    tag: str = 'dataset',
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Vyfiltrovať kontajnery s nedostatočným počtom vzoriek.

    Kontajnery s počtom záznamov pod min_samples sú odstránené
    z tréningu, aby sa predišlo modelovaniu s príliš krátkou históriou.
    Pri zadanom output_dir sa navyše uloží diagnostická tabuľka.
    """
    if df.empty:
        return df, {'kept_containers': 0, 'dropped_containers': 0, 'min_samples': min_samples}

    counts = df.groupby(sample_col).size().rename('n_samples').reset_index()
    keep_ids = set(counts[counts['n_samples'] >= min_samples][sample_col].values)
    dropped = counts[~counts[sample_col].isin(keep_ids)].copy()
    df_f = df[df[sample_col].isin(keep_ids)].copy()

    info = {
        'min_samples': int(min_samples),
        'kept_containers': int(len(keep_ids)),
        'dropped_containers': int(len(dropped)),
        'kept_rows': int(len(df_f)),
        'original_rows': int(len(df)),
    }

    logger.info(
        f"Filter {tag}: kept {info['kept_containers']:,} containers and {info['kept_rows']:,} rows "
        f"(dropped {info['dropped_containers']:,} containers)"
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        counts.to_csv(os.path.join(output_dir, f'{tag}_container_sample_counts.csv'), index=False)
        if len(dropped) > 0:
            dropped.to_csv(os.path.join(output_dir, f'{tag}_dropped_containers.csv'), index=False)

    return df_f, info


def verify_no_data_leakage(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> bool:
    """Overiť, že lag príznaky neobsahujú informácie z budúcnosti.

    Kontrola prebieha na vzorke prvých 10 kontajnerov: pre každý z nich
    overí, že prvý riadok obsahuje NaN v stĺpcoch obsahujúcich
    v názve podreťazec 'lag'. Prvý riadok kontajnera nemá
    predchádzajúcu hodnotu, takže lag stĺpec musí byť NaN.
    """
    logger.info("Verifying no data leakage...")

    sample_containers = df['container_id'].unique()[:10]
    issues_found = 0

    for cid in sample_containers:
        container_data = df[df['container_id'] == cid].sort_values('measured_at_utc')

        if len(container_data) < 3:
            continue

        first_row = container_data.iloc[0]
        lag_cols = [c for c in feature_cols if 'lag' in c.lower() and c in container_data.columns]

        for col in lag_cols[:5]:
            if not pd.isna(first_row[col]):
                logger.warning(f"Potential leakage: {col} not NaN for first row of {cid}")
                issues_found += 1

    if issues_found == 0:
        logger.info("Data leakage check PASSED")
        return True
    else:
        logger.warning(f"Data leakage check: {issues_found} issues")
        return False


def check_feature_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: Optional[float] = None,
    output_path: Optional[str] = None,
) -> Tuple[List[Tuple], pd.DataFrame]:
    """Identifikovať dvojice vysoko korelovaných príznakov.

    Korelácia nad prahom indikuje potenciálnu multikolinearitu, ktorá
    môže destabilizovať interpretáciu dôležitosti príznakov.
    """
    threshold = threshold or CONFIG.CORRELATION_THRESHOLD

    valid_cols = [c for c in feature_cols if c in df.columns]
    corr_matrix = df[valid_cols].corr().abs()

    # Extrahujeme iba horný trojuholník (bez diagonály), aby každá
    # dvojica bola v zozname iba raz.
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    high_corr = []

    for col in upper.columns:
        for row in upper.index:
            if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > threshold:
                high_corr.append((col, row, upper.loc[row, col]))

    high_corr = sorted(high_corr, key=lambda x: x[2], reverse=True)

    if high_corr:
        logger.warning(f"Found {len(high_corr)} highly correlated feature pairs (>{threshold}):")
        for col1, col2, corr in high_corr[:5]:
            logger.warning(f"  {col1} <-> {col2}: {corr:.3f}")
    else:
        logger.info(f"No highly correlated features found (>{threshold})")

    return high_corr, corr_matrix
