"""Fourierove príznaky pre reprezentáciu týždennej a dennej sezónnosti.

Cyklické premenné (deň v týždni, hodina v dni) sa v stromových modeloch
bežne kódujú ako jednoduché celé čísla. Takáto reprezentácia však
neodráža cyklickosť - napríklad 23. hodina je modelu "ďaleko" od 0.
hodiny, hoci v skutočnosti ide o bezprostredne nasledujúce časy.

Fourierove harmoniky vyjadrujú cyklickú premennú ako dvojicu
(sin(2πk·x/T), cos(2πk·x/T)), kde T je perióda (7 dní, 24 hodín)
a k index harmoniky. Táto reprezentácia je hladká a korektne
odráža blízkosť hodnôt na konci cyklu.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)

__all__ = ['add_fourier_features']


def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pridať Fourierove harmoniky pre týždennú a dennú sezónnosť.

    Počet harmoník je riadený parametrami
    CONFIG.WEEKLY_FOURIER_K (predvolene 3)
    a CONFIG.DAILY_FOURIER_K (predvolene 2).
    Funkcia očakáva prítomnosť stĺpcov day_of_week a hour;
    ak niektorý z nich chýba, príslušná rodina príznakov nie je
    vygenerovaná (bez chyby).
    """
    df = df.copy()

    # Týždenná sezónnosť (perióda = 7 dní).
    if 'day_of_week' in df.columns:
        dow_norm = df['day_of_week'] / 7.0
        for k in range(1, CONFIG.WEEKLY_FOURIER_K + 1):
            df[f'fourier_week_sin_{k}'] = np.sin(2 * np.pi * k * dow_norm)
            df[f'fourier_week_cos_{k}'] = np.cos(2 * np.pi * k * dow_norm)

    # Denná sezónnosť (perióda = 24 hodín).
    if 'hour' in df.columns:
        hour_norm = df['hour'] / 24.0
        for k in range(1, CONFIG.DAILY_FOURIER_K + 1):
            df[f'fourier_day_sin_{k}'] = np.sin(2 * np.pi * k * hour_norm)
            df[f'fourier_day_cos_{k}'] = np.cos(2 * np.pi * k * hour_norm)

    return df
