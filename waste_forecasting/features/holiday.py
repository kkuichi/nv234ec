"""České štátne sviatky ako kalendárne príznaky.

Modul generuje príznaky indikujúce, či daný dátum pripadá na český
štátny sviatok alebo je v jeho blízkosti. Veľká noc (pohyblivý
sviatok) sa vypočíta algoritmicky podľa tzv. Anonymous Gregorian
algoritmu (známeho aj ako Meeus-Jones-Butcher algoritmus); pevné
sviatky (Nový rok, Sviatok práce, Deň víťazstva a ďalšie) sú
pridané priamo ako fixné dátumy v roku.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ['get_czech_holidays', 'create_holiday_features']


def _compute_easter_sunday(year: int) -> datetime:
    """Vypočítať dátum Veľkonočnej nedele pre daný rok.

    Implementácia tzv. Anonymous Gregorian algoritmu
    (Meeus-Jones-Butcher). Algoritmus je čisto aritmetický,
    nevyžaduje tabuľky a je platný pre všetky roky v gregoriánskom
    kalendári.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day)


def get_czech_holidays(start_year: int, end_year: int) -> List[datetime]:
    """Vrátiť zoznam českých štátnych sviatkov pre zadaný rozsah rokov.

    Funkcia sa najprv pokúsi načítať sviatky z knižnice workalendar.
    Ak nie je nainštalovaná, použije sa vstavaná zoznam pevných
    sviatkov rozšírený o algoritmicky vypočítané Veľkonočné sviatky
    (Veľký piatok a Veľkonočný pondelok).

    Pevné sviatky zahŕňajú: Nový rok, Sviatok práce, Deň víťazstva,
    Deň slovanských vierozvestcov, Deň upálenia majstra Jána Husa,
    Deň českej štátnosti, Deň vzniku samostatného Československa,
    Deň boja za slobodu a demokraciu, Štedrý deň, 1. a 2. sviatok
    vianočný.
    """
    try:
        from workalendar.europe import CzechRepublic
        cal = CzechRepublic()
        holidays = []
        for year in range(start_year, end_year + 1):
            holidays.extend([datetime(h[0].year, h[0].month, h[0].day)
                           for h in cal.holidays(year)])
        logger.info(f"Načítaných sviatkov cez workalendar: {len(holidays)}")
        return holidays
    except ImportError:
        pass

    # Fallback: vstavaný zoznam pevných sviatkov + algoritmická Veľká noc.
    holidays = []
    for year in range(start_year, end_year + 1):
        fixed = [(1, 1), (5, 1), (5, 8), (7, 5), (7, 6), (9, 28),
                 (10, 28), (11, 17), (12, 24), (12, 25), (12, 26)]
        for m, d in fixed:
            holidays.append(datetime(year, m, d))

        easter = _compute_easter_sunday(year)
        # Veľký piatok (2 dni pred Veľkonočnou nedeľou).
        holidays.append(easter - timedelta(days=2))
        # Veľkonočný pondelok (1 deň po Veľkonočnej nedeli).
        holidays.append(easter + timedelta(days=1))

    logger.info(f"Algoritmicky vypočítaných sviatkov: {len(holidays)}")
    return sorted(holidays)


def create_holiday_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Odvodiť kalendárne príznaky súvisiace so sviatkami.

    Z dátumového stĺpca odvodí tri nové príznaky:

    * is_holiday - 1 ak daný dátum je sviatok, inak 0;
    * days_to_holiday - počet dní do najbližšieho sviatku (v oboch
      smeroch); hodnota 30 sa použije ako "far from any holiday";
    * is_near_holiday - 1 ak do najbližšieho sviatku ostávajú
      najviac 3 dni.
    """
    df = df.copy()

    if date_col not in df.columns:
        return df

    dates = pd.to_datetime(df[date_col])
    holidays = get_czech_holidays(dates.min().year, dates.max().year)
    holiday_dates = set(h.date() for h in holidays)
    holiday_ordinals = [h.toordinal() for h in holidays]

    date_values = dates.dt.date
    df['is_holiday'] = date_values.apply(lambda d: 1 if d in holiday_dates else 0)

    def days_to_nearest(d):
        """Vrátiť počet dní do najbližšieho sviatku (v oboch smeroch)."""
        if pd.isna(d) or not holiday_ordinals:
            return 30
        return min(abs(d.toordinal() - h) for h in holiday_ordinals)

    df['days_to_holiday'] = date_values.apply(days_to_nearest)
    df['is_near_holiday'] = (df['days_to_holiday'] <= 3).astype(int)

    return df
