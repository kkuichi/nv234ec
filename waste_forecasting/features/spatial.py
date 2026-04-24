"""Geolokačné príznaky odvodené z GPS súradníc kontajnerov.

Modul pridáva tri rodiny priestorových príznakov:

* Surové súradnice - zemepisná šírka a dĺžka ako samostatné
  numerické premenné.
* Euklidovská vzdialenosť od centra Prahy - približná vzdialenosť
  v kilometroch od referenčného bodu
  (CONFIG.PRAGUE_CENTER_LAT,
  CONFIG.PRAGUE_CENTER_LON). Výpočet používa
  aproximáciu 1° ≈ 111 km, ktorá je dostatočne presná v rozsahu
  jedného mesta.
* KMeans klaster - voliteľná kompaktná reprezentácia priestorovej
  oblasti. Klaster je natrénovaný nad unikátnymi kontajnermi
  a priradený každému záznamu.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import CONFIG

logger = logging.getLogger(__name__)

__all__ = ['add_geolocation_features']


def add_geolocation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pridať geolokačné príznaky na základe stĺpcov latitude a longitude.

    Ak niektorý zo stĺpcov súradníc chýba, vráti sa vstupný DataFrame
    bez zmeny (bez chyby). KMeans klastrovanie je voliteľné a aktivuje
    sa iba pri splnení dvoch podmienok: príznak je povolený cez
    CONFIG.GEO_ADD_CLUSTER_FEATURE a počet unikátnych
    kontajnerov so súradnicami je aspoň
    CONFIG.GEO_CLUSTER_MIN_CONTAINERS.
    """
    df = df.copy()

    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return df

    df['lat'] = df['latitude']
    df['lon'] = df['longitude']

    # Zaokrúhlené súradnice (binning) na tri desatinné miesta -
    # približne 100 m × 100 m mriežka, ktorá slúži ako robustná
    # kategorizácia priestoru pri zachovaní informácie o polohe.
    df['geo_bin_lat'] = (df['latitude'] * 1000).round() / 1000
    df['geo_bin_lon'] = (df['longitude'] * 1000).round() / 1000

    # Euklidovská aproximácia vzdialenosti v kilometroch.
    # Násobiteľ 111 km odpovedá dĺžke jedného stupňa na rovníku;
    # pre šírku Prahy (~50°) je chyba pri dĺžke rádu niekoľko percent,
    # čo je pre stromové modely úplne postačujúce.
    df['dist_from_center'] = np.sqrt(
        (df['latitude'] - CONFIG.PRAGUE_CENTER_LAT)**2 +
        (df['longitude'] - CONFIG.PRAGUE_CENTER_LON)**2
    ) * 111

    # Voliteľný KMeans klaster ako kompaktná reprezentácia polohy.
    if CONFIG.GEO_ADD_CLUSTER_FEATURE:
        coords = (
            df[['container_id', 'latitude', 'longitude']]
            .dropna(subset=['latitude', 'longitude'])
            .drop_duplicates('container_id')
        )
        if len(coords) >= CONFIG.GEO_CLUSTER_MIN_CONTAINERS:
            n_clusters = int(min(CONFIG.GEO_N_CLUSTERS, len(coords)))
            km = KMeans(n_clusters=n_clusters, random_state=CONFIG.SEED, n_init='auto')
            coords['geo_cluster'] = km.fit_predict(coords[['latitude', 'longitude']]).astype(int)
            df = df.merge(coords[['container_id', 'geo_cluster']], on='container_id', how='left')
        else:
            df['geo_cluster'] = np.nan
    else:
        df['geo_cluster'] = np.nan

    return df
