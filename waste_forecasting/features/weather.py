"""Meteorologické príznaky pre modely plnosti.

Modul načíta lokálny Meteostat CSV, pripojí počasie k meraniam kontajnerov
a vytvorí príznaky z teploty, zrážok, vetra a tlaku. Pri chýbajúcom súbore
nepridáva žiadne meteorologické stĺpce.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)

__all__ = ['load_external_weather', 'merge_external_weather', 'ensure_weather_columns', 'add_temperature_features', 'add_precipitation_features', 'add_humidity_features', 'add_wind_features', 'add_pressure_features', 'add_comfort_index_features', 'add_weather_interactions', 'add_all_weather_features', 'get_weather_feature_names', 'analyze_weather_impact']


# Cache na úrovni modulu: načítanie meteorologického CSV prebehne najviac
# raz za beh pipeline, opakované volania vrátia uloženú verziu.
_WEATHER_CACHE: Dict[str, Any] = {"path": None, "df": None}


def _resolve_weather_path(weather_path: Optional[str]) -> Optional[str]:
    """Vyhľadať lokálny meteorologický CSV súbor.

    Funkcia hľadá IBA lokálne uložený súbor; žiadne dáta
    sa neťahajú z internetu ani externého API. Ak CSV nie je
    dostupné, vráti None a volajúci by mal meteorologickú
    časť pipeline celkom preskočiť.
    """
    candidates = []
    if weather_path:
        candidates.append(weather_path)
    candidates.extend([
        "weather.csv",
        "./weather.csv",
        os.path.join(os.getcwd(), "weather.csv"),
    ])
    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            continue
    return None


def load_external_weather(weather_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Načítať externé meteorologické dáta z CSV a namapovať na požadované stĺpce.

    Funkcia očakáva CSV súbor vo formáte Meteostat s nasledovnými
    stĺpcami (niektoré sú voliteľné):

    * time, date alebo datetime - časová pečiatka (povinné);
    * tavg alebo tmin/tmax - teplota v °C;
    * prcp - úhrn zrážok v mm;
    * wspd - rýchlosť vetra v km/h (konvertuje sa na m/s);
    * pres - atmosférický tlak v hPa;
    * snow, tsun - voliteľné sneh a slnečné žiarenie.

    Výsledok je kešovaný na úrovni modulu, takže opakované volanie
    pre rovnakú cestu nečíta súbor znova.
    """
    resolved = _resolve_weather_path(weather_path or CONFIG.WEATHER_DATA_PATH)
    if resolved is None:
        logger.warning(
            "Meteorologický CSV súbor nebol nájdený. Meteorologické príznaky sa preskočia."
        )
        return None

    if _WEATHER_CACHE["path"] == resolved and _WEATHER_CACHE["df"] is not None:
        return _WEATHER_CACHE["df"]

    w = pd.read_csv(resolved)

    # Detekcia dátumového stĺpca
    dt_col = None
    for c in ["time", "date", "datetime", "Date", "DATE"]:
        if c in w.columns:
            dt_col = c
            break
    if dt_col is None:
        logger.warning("Meteorologický CSV nemá časový stĺpec. Očakáva sa jeden z týchto: time, date, datetime.")
        return None

    w[dt_col] = pd.to_datetime(w[dt_col], errors="coerce", utc=True)
    w = w.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)
    w = w.rename(columns={dt_col: "weather_time"})

    # Premapovanie na očakávané názvy
    if "tavg" in w.columns:
        w["temperature"] = w["tavg"]
    elif "tmin" in w.columns and "tmax" in w.columns:
        w["temperature"] = (w["tmin"] + w["tmax"]) / 2.0
    elif "temperature" not in w.columns:
        w["temperature"] = np.nan

    if "tmin" in w.columns:
        w["temp_min"] = w["tmin"]
    if "tmax" in w.columns:
        w["temp_max"] = w["tmax"]
    if "temp_min" in w.columns and "temp_max" in w.columns:
        w["temp_daily_range_src"] = w["temp_max"] - w["temp_min"]

    if "prcp" in w.columns:
        w["precipitation"] = w["prcp"].fillna(0.0)
    elif "precipitation" not in w.columns:
        w["precipitation"] = np.nan

    # Meteostat uvádza wspd v km/h, prevádzame na m/s pre kompatibilitu s prahmi.
    if "wspd" in w.columns:
        w["wind_speed"] = w["wspd"] / 3.6
    elif "wind_speed" not in w.columns:
        w["wind_speed"] = np.nan

    if "pres" in w.columns:
        w["pressure"] = w["pres"]
    elif "pressure" not in w.columns:
        w["pressure"] = np.nan

    # Voliteľné stĺpce - ponecháme ich, ak existujú
    if "snow" in w.columns:
        w["snow"] = w["snow"]
    if "tsun" in w.columns:
        w["sunshine_min"] = w["tsun"]  # Meteostat uvádza tsun v minútach.

    # Necháme povinné stĺpce a k nim voliteľné extra, ak sú
    keep = ["weather_time", "temperature", "precipitation", "wind_speed", "pressure"]
    for extra in ["temp_min", "temp_max", "temp_daily_range_src", "snow", "sunshine_min"]:
        if extra in w.columns:
            keep.append(extra)

    w = w[keep].drop_duplicates(subset=["weather_time"], keep="last").reset_index(drop=True)

    _WEATHER_CACHE["path"] = resolved
    _WEATHER_CACHE["df"] = w

    logger.info(f"Načítané meteorologické dáta: {resolved}. Riadkov: {len(w):,}")
    return w


def merge_external_weather(df: pd.DataFrame, weather_path: Optional[str] = None) -> pd.DataFrame:
    """Pripojiť externé meteorologické dáta k meraniam.

    Funkcia najprv zistí, či sú meteorologické údaje denné (všetky
    pečiatky o 00:00:00) alebo v jemnejšom rozlíšení. Pre denné dáta
    používa jednoduchý join podľa dátumu; pre jemnejšie rozlíšenie
    použije pd.merge_asof smerom dozadu, aby sa k meraniu priradila
    najbližšia predchádzajúca meteorologická pečiatka. Tým sa zabráni
    použitiu budúcich informácií.

    Pri prekrývaní stĺpcov rozhoduje príznak
    CONFIG.WEATHER_OVERRIDE_EXISTING o tom, či
    externé hodnoty prepíšu existujúce, alebo ich iba doplnia.
    """
    if "measured_at_utc" not in df.columns:
        return df

    w = load_external_weather(weather_path)
    if w is None or len(w) == 0:
        return df

    df = df.copy()
    df["measured_at_utc"] = pd.to_datetime(df["measured_at_utc"], errors="coerce", utc=True)

    # Detekcia denného kroku (všetky hodnoty o 00:00:00 a zhruba 1-dňové intervaly)
    wt = w["weather_time"]
    is_daily = (
        wt.dt.hour.eq(0).all() and wt.dt.minute.eq(0).all() and wt.dt.second.eq(0).all()
    )

    if is_daily:
        w2 = w.copy()
        w2["weather_date"] = w2["weather_time"].dt.normalize()
        df["weather_date"] = df["measured_at_utc"].dt.normalize()

        df = df.merge(
            w2.drop(columns=["weather_time"]).rename(columns=lambda c: f"{c}_weather" if c != "weather_date" else c),
            on="weather_date",
            how="left"
        )
        df = df.drop(columns=["weather_date"])
    else:
        w2 = w.sort_values("weather_time").copy()
        df = df.sort_values("measured_at_utc").copy()
        df = pd.merge_asof(
            df,
            w2.rename(columns=lambda c: f"{c}_weather" if c != "weather_time" else c),
            left_on="measured_at_utc",
            right_on="weather_time",
            direction="backward"
        )
        df = df.drop(columns=["weather_time"], errors="ignore")

    # Buď nahradíme existujúce stĺpce, alebo ich len doplníme
    for base_col in ["temperature", "precipitation", "wind_speed", "pressure", "temp_min", "temp_max", "temp_daily_range_src", "snow", "sunshine_min"]:
        weather_col = f"{base_col}_weather"
        if weather_col not in df.columns:
            continue

        if base_col not in df.columns:
            df[base_col] = df[weather_col]
        else:
            if CONFIG.WEATHER_OVERRIDE_EXISTING:
                df[base_col] = df[weather_col]
            else:
                df[base_col] = df[base_col].combine_first(df[weather_col])

        df = df.drop(columns=[weather_col], errors="ignore")

    # Základná očista nereálnych teplôt
    if "temperature" in df.columns:
        df.loc[(df["temperature"] < -30) | (df["temperature"] > 50), "temperature"] = np.nan

    # Report pokrytia po spojení dát
    cov = {}
    for c in ["temperature", "precipitation", "wind_speed", "pressure"]:
        if c in df.columns:
            cov[c] = float(df[c].notna().mean())
    logger.info(f"Pokrytie po pripojení počasia: { {k: f'{v*100:.1f}%' for k, v in cov.items()} }")

    return df


def ensure_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Zaistiť prítomnosť meteorologických stĺpcov bez syntetizácie dát.

    Pipeline v tejto funkcii striktne dodržiava anti-leakage princípy:
    nevykonáva žiadnu náhodnú syntézu chýbajúcich hodnôt, ani
    spätné vypĺňanie (bfill), ktoré by využívalo informácie
    z budúcnosti. Dovolené je iba dopredné vypĺňanie (ffill)
    a explicitné nahradenie nuly pre chýbajúce zrážky.

    Pre každú meteorologickú premennú sa vytvorí aj binárny indikátor
    {col}_missing, ktorý stromovým modelom umožňuje rozlíšiť
    medzi skutočnou hodnotou a dopĺňaním.
    """
    logger.info("Kontrola meteorologických stĺpcov: bez syntetického dopĺňania, bez bfill, s indikátormi chýbajúcich hodnôt.")
    df = df.copy()

    if "measured_at_utc" not in df.columns:
        return df

    # Overenie existencie očakávaných stĺpcov
    required = ["temperature", "precipitation", "wind_speed", "pressure"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # Triedenie pred doplnením smerom dopredu.
    df["measured_at_utc"] = pd.to_datetime(df["measured_at_utc"], errors="coerce")
    df = df.sort_values("measured_at_utc")

    # Indikátory chýbajúcich hodnôt pred akýmkoľvek doplnením
    for col in required:
        df[f"{col}_missing"] = df[col].isna().astype(np.int8)

    # Stratégia dopĺňania (iba minulosťou)
    # Zrážky pri chýbajúcej hodnote sú zvyčajne 0, indikátor stále zachovávame
    df["precipitation"] = df["precipitation"].fillna(0.0)

    # Ostatné spojité premenné: iba doplnenie smerom dopredu, nikdy spätne.
    for col in ["temperature", "wind_speed", "pressure"]:
        df[col] = df[col].ffill()

    # Vlhkosť v zdrojovom meteorologickom datasete chýba, nepridáva sa

    return df


def add_temperature_features(df: pd.DataFrame, mode: str = "auto", step_hours: int = None) -> pd.DataFrame:
    """Odvodiť príznaky založené na teplote.

    Generuje kategorické indikátory (temp_freezing, temp_cold,
    temp_mild, temp_warm, temp_hot), kvadratickú transformáciu
    a odchýlku od referenčnej hodnoty 15 °C. V režime 'time' pridáva
    aj lagy a kĺzavé štatistiky.
    """
    df = df.copy()

    if "temperature" not in df.columns:
        return df

    if mode == "auto":
        mode = "time" if step_hours is not None else "event"

    temp = df["temperature"]

    df["temp_freezing"] = (temp <= CONFIG.TEMP_FREEZING_THRESHOLD).astype(np.int8)
    df["temp_cold"] = (temp < CONFIG.TEMP_COLD_THRESHOLD).astype(np.int8)
    df["temp_mild"] = ((temp >= CONFIG.TEMP_COLD_THRESHOLD) & (temp < CONFIG.TEMP_WARM_THRESHOLD)).astype(np.int8)
    df["temp_warm"] = ((temp >= CONFIG.TEMP_WARM_THRESHOLD) & (temp < CONFIG.TEMP_HOT_THRESHOLD)).astype(np.int8)
    df["temp_hot"] = (temp >= CONFIG.TEMP_HOT_THRESHOLD).astype(np.int8)

    df["temp_squared"] = temp ** 2
    df["temp_abs_deviation"] = np.abs(temp - 15)

    if "measured_at_utc" in df.columns:
        day_of_year = pd.to_datetime(df["measured_at_utc"]).dt.dayofyear
        seasonal_expected = 10 + 10 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        df["temp_anomaly"] = temp - seasonal_expected
        df["temp_above_normal"] = (df["temp_anomaly"] > 5).astype(np.int8)
        df["temp_below_normal"] = (df["temp_anomaly"] < -5).astype(np.int8)

    if "container_id" not in df.columns:
        return df

    g = df.groupby("container_id")["temperature"]

    if mode == "time":
        if step_hours is None:
            raise ValueError("step_hours must be provided for mode='time'")

        valid_lags_h = _valid_hour_windows(CONFIG.WEATHER_LAGS, step_hours)
        for lag_h in valid_lags_h:
            lag_s = _hours_to_steps(lag_h, step_hours)
            df[f"temp_lag_{lag_h}h"] = g.shift(lag_s)

        # Zmena cez 1 krok a 24 hodín, ak je k dispozícii
        df[f"temp_change_{step_hours}h"] = temp - g.shift(1)

        steps_24h = _hours_to_steps(24, step_hours)
        if steps_24h >= 1:
            df["temp_change_24h"] = temp - g.shift(steps_24h)

        for win_h in _valid_hour_windows(CONFIG.WEATHER_ROLL_WINDOWS, step_hours):
            win_s = _hours_to_steps(win_h, step_hours)
            df[f"temp_roll_mean_{win_h}h"] = g.transform(lambda x: x.shift(1).rolling(win_s, min_periods=1).mean())
            df[f"temp_roll_std_{win_h}h"] = g.transform(lambda x: x.shift(1).rolling(win_s, min_periods=1).std())
            df[f"temp_roll_max_{win_h}h"] = g.transform(lambda x: x.shift(1).rolling(win_s, min_periods=1).max())
            df[f"temp_roll_min_{win_h}h"] = g.transform(lambda x: x.shift(1).rolling(win_s, min_periods=1).min())

        if "temp_roll_max_24h" in df.columns and "temp_roll_min_24h" in df.columns:
            df["temp_daily_range"] = df["temp_roll_max_24h"] - df["temp_roll_min_24h"]

    else:
        # Režim vývozov: hodinové okná nedávajú zmysel, používame okná podľa vývozov.
        event_lags = [1, 2, 3]
        for lag_e in event_lags:
            df[f"temp_lag_{lag_e}_event"] = g.shift(lag_e)

        df["temp_change_1_event"] = temp - g.shift(1)

        for win_e in CONFIG.EVENT_ROLL_WINDOWS:
            df[f"temp_roll_mean_{win_e}_event"] = g.transform(lambda x: x.shift(1).rolling(win_e, min_periods=1).mean())
            df[f"temp_roll_std_{win_e}_event"] = g.transform(lambda x: x.shift(1).rolling(win_e, min_periods=1).std())
            df[f"temp_roll_max_{win_e}_event"] = g.transform(lambda x: x.shift(1).rolling(win_e, min_periods=1).max())
            df[f"temp_roll_min_{win_e}_event"] = g.transform(lambda x: x.shift(1).rolling(win_e, min_periods=1).min())

    return df


def add_precipitation_features(df: pd.DataFrame, mode: str = "auto", step_hours: int = None) -> pd.DataFrame:
    """Odvodiť príznaky založené na zrážkach.

    Generuje indikátory intenzity zrážok (rain_light, rain_moderate,
    rain_heavy), historické úhrny a indikátor suchého obdobia.
    V režime 'time' pridáva aj lagy a kĺzavé sumy.
    """
    df = df.copy()
    if "precipitation" not in df.columns:
        return df

    if mode == "auto":
        mode = "time" if step_hours is not None else "event"

    precip = df["precipitation"].fillna(0)
    df["is_raining"] = (precip > 0).astype(np.int8)
    df["rain_light"] = ((precip > 0) & (precip < CONFIG.PRECIP_LIGHT_THRESHOLD)).astype(np.int8)
    df["rain_moderate"] = ((precip >= CONFIG.PRECIP_LIGHT_THRESHOLD) & (precip < CONFIG.PRECIP_MODERATE_THRESHOLD)).astype(np.int8)
    df["rain_heavy"] = (precip >= CONFIG.PRECIP_MODERATE_THRESHOLD).astype(np.int8)

    if "container_id" not in df.columns:
        return df

    g_p = df.groupby("container_id")["precipitation"]
    g_r = df.groupby("container_id")["is_raining"]

    if mode == "time":
        if step_hours is None:
            raise ValueError("step_hours must be provided for mode='time'")

        valid_lags_h = _valid_hour_windows([1, 6, 12, 24], step_hours)
        for lag_h in valid_lags_h:
            lag_s = _hours_to_steps(lag_h, step_hours)
            df[f"precip_lag_{lag_h}h"] = g_p.shift(lag_s)

        for win_h in _valid_hour_windows([6, 12, 24, 48], step_hours):
            win_s = _hours_to_steps(win_h, step_hours)
            df[f"rained_last_{win_h}h"] = g_r.transform(
                lambda x: x.shift(1).rolling(win_s, min_periods=1).max()
            ).fillna(0).astype(np.int8)

        for win_h in _valid_hour_windows([6, 24, 48], step_hours):
            win_s = _hours_to_steps(win_h, step_hours)
            df[f"precip_sum_{win_h}h"] = g_p.transform(
                lambda x: x.shift(1).rolling(win_s, min_periods=1).sum()
            )

        dry_steps = g_r.transform(lambda x: (x == 0).groupby((x != 0).cumsum()).cumcount())
        df["dry_hours"] = (dry_steps * step_hours).clip(upper=168)

    else:
        event_lags = [1, 2, 3]
        for lag_e in event_lags:
            df[f"precip_lag_{lag_e}_event"] = g_p.shift(lag_e)

        for win_e in CONFIG.EVENT_ROLL_WINDOWS:
            df[f"rained_last_{win_e}_event"] = g_r.transform(
                lambda x: x.shift(1).rolling(win_e, min_periods=1).max()
            ).fillna(0).astype(np.int8)

        for win_e in CONFIG.EVENT_ROLL_WINDOWS:
            df[f"precip_sum_{win_e}_event"] = g_p.transform(
                lambda x: x.shift(1).rolling(win_e, min_periods=1).sum()
            )

        df["dry_events"] = g_r.transform(lambda x: (x == 0).groupby((x != 0).cumsum()).cumcount()).clip(upper=50)

    return df


def add_humidity_features(df: pd.DataFrame, mode: str = "auto", step_hours: int = None) -> pd.DataFrame:
    """Odvodiť príznaky založené na relatívnej vlhkosti vzduchu.

    Generuje indikátory komfortných rozsahov vlhkosti
    (humidity_low, humidity_comfortable, humidity_high).
    Pri prítomnosti stĺpca humidity pridá aj kĺzavé priemery
    a medzidennú zmenu; ak stĺpec chýba, vracia DataFrame nezmenený.
    """
    df = df.copy()
    if "humidity" not in df.columns:
        return df

    if mode == "auto":
        mode = "time" if step_hours is not None else "event"

    humid = df["humidity"]
    df["humidity_low"] = (humid < CONFIG.HUMIDITY_LOW_THRESHOLD).astype(np.int8)
    df["humidity_comfortable"] = ((humid >= CONFIG.HUMIDITY_LOW_THRESHOLD) & (humid <= CONFIG.HUMIDITY_HIGH_THRESHOLD)).astype(np.int8)
    df["humidity_high"] = (humid > CONFIG.HUMIDITY_HIGH_THRESHOLD).astype(np.int8)

    if "container_id" not in df.columns:
        return df

    g = df.groupby("container_id")["humidity"]

    if mode == "time":
        if step_hours is None:
            raise ValueError("step_hours must be provided for mode='time'")

        for win_h in _valid_hour_windows([6, 24], step_hours):
            win_s = _hours_to_steps(win_h, step_hours)
            df[f"humidity_roll_mean_{win_h}h"] = g.transform(lambda x: x.shift(1).rolling(win_s, min_periods=1).mean())

        steps_24h = _hours_to_steps(24, step_hours)
        if steps_24h >= 1:
            df["humidity_change_24h"] = humid - g.shift(steps_24h)

    else:
        for win_e in CONFIG.EVENT_ROLL_WINDOWS:
            df[f"humidity_roll_mean_{win_e}_event"] = g.transform(lambda x: x.shift(1).rolling(win_e, min_periods=1).mean())
        df["humidity_change_1_event"] = humid - g.shift(1)

    return df


def add_wind_features(df: pd.DataFrame, mode: str = "auto", step_hours: int = None) -> pd.DataFrame:
    """Odvodiť príznaky založené na rýchlosti vetra.

    Generuje indikátory sily vetra (wind_calm, wind_moderate,
    wind_strong) a kĺzavé štatistiky. Prahy pre kategorizáciu
    sú definované v CONFIG.
    """
    df = df.copy()
    if "wind_speed" not in df.columns:
        return df

    if mode == "auto":
        mode = "time" if step_hours is not None else "event"

    wind = df["wind_speed"]
    df["wind_calm"] = (wind < CONFIG.WIND_CALM_THRESHOLD).astype(np.int8)
    df["wind_moderate"] = ((wind >= CONFIG.WIND_CALM_THRESHOLD) & (wind < CONFIG.WIND_MODERATE_THRESHOLD)).astype(np.int8)
    df["wind_strong"] = (wind >= CONFIG.WIND_MODERATE_THRESHOLD).astype(np.int8)

    if "container_id" not in df.columns:
        return df

    g = df.groupby("container_id")["wind_speed"]

    if mode == "time":
        if step_hours is None:
            raise ValueError("step_hours must be provided for mode='time'")

        for win_h in _valid_hour_windows([6, 24], step_hours):
            win_s = _hours_to_steps(win_h, step_hours)
            df[f"wind_roll_mean_{win_h}h"] = g.transform(lambda x: x.shift(1).rolling(win_s, min_periods=1).mean())
            df[f"wind_roll_max_{win_h}h"] = g.transform(lambda x: x.shift(1).rolling(win_s, min_periods=1).max())

    else:
        for win_e in CONFIG.EVENT_ROLL_WINDOWS:
            df[f"wind_roll_mean_{win_e}_event"] = g.transform(lambda x: x.shift(1).rolling(win_e, min_periods=1).mean())
            df[f"wind_roll_max_{win_e}_event"] = g.transform(lambda x: x.shift(1).rolling(win_e, min_periods=1).max())

    return df


def add_pressure_features(df: pd.DataFrame, mode: str = "auto", step_hours: int = None) -> pd.DataFrame:
    """Odvodiť príznaky založené na atmosférickom tlaku.

    Generuje indikátory nízkeho a vysokého tlaku, zmenu tlaku
    za 3 a 24 hodín a indikátory stúpajúceho/klesajúceho tlaku
    (fyzikálne sú spojené s frontálnymi prechodmi a zmenami počasia).
    """
    df = df.copy()
    if "pressure" not in df.columns:
        return df

    if mode == "auto":
        mode = "time" if step_hours is not None else "event"

    press = df["pressure"]
    df["pressure_low"] = (press < CONFIG.PRESSURE_LOW_THRESHOLD).astype(np.int8)
    df["pressure_high"] = (press > CONFIG.PRESSURE_HIGH_THRESHOLD).astype(np.int8)

    if "container_id" not in df.columns:
        return df

    g = df.groupby("container_id")["pressure"]

    if mode == "time":
        if step_hours is None:
            raise ValueError("step_hours must be provided for mode='time'")

        # 3h okno má zmysel len ak krok delí 3, inak preskočíme
        if (3 % step_hours) == 0:
            s3 = _hours_to_steps(3, step_hours)
            if s3 >= 1:
                df["pressure_change_3h"] = press - g.shift(s3)
                df["pressure_falling"] = (df["pressure_change_3h"] < -3).astype(np.int8)
                df["pressure_rising"] = (df["pressure_change_3h"] > 3).astype(np.int8)

        s24 = _hours_to_steps(24, step_hours)
        if s24 >= 1:
            df["pressure_change_24h"] = press - g.shift(s24)

    else:
        df["pressure_change_1_event"] = press - g.shift(1)
        df["pressure_falling_event"] = (df["pressure_change_1_event"] < -3).astype(np.int8)
        df["pressure_rising_event"] = (df["pressure_change_1_event"] > 3).astype(np.int8)

    return df


def add_comfort_index_features(df: pd.DataFrame) -> pd.DataFrame:
    """Odvodiť indexy pocitovej teploty (Heat Index a Wind Chill).

    Heat Index modeluje pocitovú teplotu kombinovanou teplotou a
    vlhkosťou; relevantný je pri teplote nad 27 °C. Wind Chill
    modeluje ochladzovacie účinky vetra pri teplote pod 10 °C.
    Výsledný apparent_temp kombinuje oba indexy do jednej
    robustnej premennej. Ak nie sú k dispozícii potrebné
    vstupy (teplota, vlhkosť, vietor), príslušné stĺpce
    sa nevytvoria.
    """
    df = df.copy()
    
    has_temp = 'temperature' in df.columns
    has_humid = 'humidity' in df.columns
    has_wind = 'wind_speed' in df.columns
    
    if has_temp and has_humid:
        temp = df['temperature']
        humid = df['humidity']
        df['heat_index'] = np.where(temp >= 20,
            -8.78 + 1.61 * temp + 2.34 * humid - 0.09 * temp * humid, temp)
    
    if has_temp and has_wind:
        temp = df['temperature']
        wind_kmh = df['wind_speed'] * 3.6
        df['wind_chill'] = np.where((temp <= 10) & (wind_kmh > 4.8),
            13.12 + 0.6215 * temp - 11.37 * (wind_kmh ** 0.16) + 0.3965 * temp * (wind_kmh ** 0.16), temp)
    
    if 'heat_index' in df.columns and 'wind_chill' in df.columns:
        temp = df['temperature']
        df['apparent_temp'] = np.where(temp >= 20, df['heat_index'],
                                       np.where(temp <= 10, df['wind_chill'], temp))
    
    if has_temp and has_humid and has_wind:
        temp_score = 100 - np.minimum(np.abs(df['temperature'] - 21) * 4, 100)
        humid_score = 100 - np.minimum(np.abs(df['humidity'] - 50) * 1.5, 100)
        wind_score = 100 - np.minimum(np.abs(df['wind_speed'] - 2.5) * 15, 100)
        df['weather_comfort_score'] = np.clip((temp_score + humid_score + wind_score) / 3, 0, 100)
        df['weather_comfortable'] = (df['weather_comfort_score'] >= 60).astype(np.int8)
        df['weather_uncomfortable'] = (df['weather_comfort_score'] < 40).astype(np.int8)
    
    return df


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Odvodiť interakčné príznaky medzi počasím a ostatnými premennými.

    Zo základných meteorologických indikátorov a kalendárnych
    premenných vytvára krížové kombinácie, ktoré môžu zachytiť
    nelineárne vzťahy (napríklad odlišné správanie cez víkend pri
    peknom počasí, alebo vplyv horúčav na organický odpad).
    Vypočítava aj agregovaný indikátor kvality počasia
    weather_good / weather_bad a sezónne dummy premenné.
    """
    df = df.copy()
    
    if 'measured_at_utc' in df.columns and 'is_weekend' not in df.columns:
        df['is_weekend'] = (pd.to_datetime(df['measured_at_utc']).dt.dayofweek >= 5).astype(np.int8)
    
    # Indikátor kvality meteorologických dát
    good_conditions = []
    if 'temperature' in df.columns:
        good_conditions.append((df['temperature'] >= 15) & (df['temperature'] <= 28))
    if 'is_raining' in df.columns:
        good_conditions.append(df['is_raining'] == 0)
    if 'wind_strong' in df.columns:
        good_conditions.append(df['wind_strong'] == 0)
    
    if good_conditions:
        df['weather_good'] = np.all(good_conditions, axis=0).astype(np.int8)
        df['weather_bad'] = (~np.all(good_conditions, axis=0)).astype(np.int8)
    
    # Interakcie víkend + počasie
    if 'is_weekend' in df.columns:
        if 'weather_good' in df.columns:
            df['weekend_good_weather'] = (df['is_weekend'] & df['weather_good']).astype(np.int8)
        if 'is_raining' in df.columns:
            df['weekend_rainy'] = (df['is_weekend'] & df['is_raining']).astype(np.int8)
        if 'temp_warm' in df.columns:
            df['weekend_warm'] = (df['is_weekend'] & df['temp_warm']).astype(np.int8)
    
    # Interakcie typ odpadu + počasie
    if 'trash_type' in df.columns:
        is_organic = df['trash_type'].str.lower().str.contains('bio|organic|food|kompost', na=False)
        if 'temp_hot' in df.columns:
            df['organic_hot_weather'] = (is_organic & (df['temp_hot'] == 1)).astype(np.int8)
        if 'temp_cold' in df.columns:
            df['organic_cold_weather'] = (is_organic & (df['temp_cold'] == 1)).astype(np.int8)
    
    # Interakcie sviatok + počasie
    if 'is_holiday' in df.columns and 'weather_good' in df.columns:
        df['holiday_good_weather'] = (df['is_holiday'] & df['weather_good']).astype(np.int8)
    
    # Ročné obdobia
    if 'measured_at_utc' in df.columns:
        month = pd.to_datetime(df['measured_at_utc']).dt.month
        df['season_winter'] = month.isin([12, 1, 2]).astype(np.int8)
        df['season_spring'] = month.isin([3, 4, 5]).astype(np.int8)
        df['season_summer'] = month.isin([6, 7, 8]).astype(np.int8)
        df['season_autumn'] = month.isin([9, 10, 11]).astype(np.int8)
    
    return df


def _hours_to_steps(hours: int, step_hours: int) -> int:
    """Konvertovať počet hodín na počet krokov prevzorkovania.
    """
    if step_hours <= 0:
        return 0
    return hours // step_hours


def _valid_hour_windows(hour_list, step_hours):
    """Odfiltrovať hodinové okná, ktoré nie sú násobkom kroku prevzorkovania.

    Niektoré kombinácie (napr. 3-hodinové okno pri 6-hodinovom kroku)
    nemajú v rámci pravidelne prevzorkovanej série zmysel.
    """
    return [h for h in hour_list if (h % step_hours) == 0 and h >= step_hours]


def add_all_weather_features(df: pd.DataFrame, mode: str = "auto", step_hours: int = None) -> pd.DataFrame:
    """Zlúčiť, doplniť a odvodiť všetky meteorologické príznaky naraz.

    Funkcia pridá meteorologické príznaky v jednom kroku.
    Interne volá sekvenciu krokov:

    1. merge_external_weather - join s externým CSV;
    2. ensure_weather_columns - garancia stĺpcov bez leakage;
    3. add_temperature_features, add_precipitation_features,
       add_humidity_features, add_wind_features,
       add_pressure_features - rodiny príznakov pre jednotlivé veličiny;
    4. add_comfort_index_features - Heat Index a Wind Chill;
    5. add_weather_interactions - krížové interakcie.
    """
    if not CONFIG.ENABLE_WEATHER_FEATURES:
        logger.info("Meteorologické príznaky sú vypnuté")
        return df

    # Ak lokálny meteorologický CSV neexistuje, preskočíme celú
    # meteorologickú časť pipeline. Nepridajú sa ani NaN stĺpce,
    # ani indikátory chýbajúcich hodnôt, ani žiadne odvodené príznaky.
    # Dáta sa neťahajú z internetu.
    resolved = _resolve_weather_path(CONFIG.WEATHER_DATA_PATH)
    if resolved is None:
        logger.warning(
            "Meteorologický CSV (%s) nebol nájdený - preskakujem celú "
            "tvorbu meteorologických príznakov.",
            CONFIG.WEATHER_DATA_PATH,
        )
        return df

    if mode == "auto":
        mode = "time" if step_hours is not None else "event"

    logger.info(f"Pridávam meteorologické príznaky. režim={mode}, krok_hodiny={step_hours}")

    df = merge_external_weather(df)
    df = ensure_weather_columns(df)

    df = add_temperature_features(df, mode=mode, step_hours=step_hours)
    df = add_precipitation_features(df, mode=mode, step_hours=step_hours)
    df = add_humidity_features(df, mode=mode, step_hours=step_hours)
    df = add_wind_features(df, mode=mode, step_hours=step_hours)
    df = add_pressure_features(df, mode=mode, step_hours=step_hours)
    df = add_comfort_index_features(df)
    df = add_weather_interactions(df)

    return df


def get_weather_feature_names() -> List[str]:
    """Vrátiť kanonický zoznam názvov meteorologických príznakov.

    Používa sa najmä v ablačnej štúdii pre spoľahlivé oddelenie
    meteorologických príznakov od ostatných v trénovacej množine.
    """
    return [
        'temperature', 'temp_freezing', 'temp_cold', 'temp_mild', 'temp_warm', 'temp_hot',
        'temp_squared', 'temp_abs_deviation', 'temp_anomaly', 'temp_above_normal', 'temp_below_normal',
        'temp_change_1h', 'temp_change_24h', 'temp_daily_range',
        'precipitation', 'is_raining', 'rain_light', 'rain_moderate', 'rain_heavy',
        'rained_last_6h', 'rained_last_12h', 'rained_last_24h', 'rained_last_48h',
        'precip_sum_6h', 'precip_sum_24h', 'precip_sum_48h', 'dry_hours',
        'humidity', 'humidity_low', 'humidity_comfortable', 'humidity_high',
        'wind_speed', 'wind_calm', 'wind_moderate', 'wind_strong',
        'pressure', 'pressure_low', 'pressure_high', 'pressure_falling', 'pressure_rising',
        'heat_index', 'wind_chill', 'apparent_temp', 'weather_comfort_score',
        'weather_good', 'weather_bad', 'weekend_good_weather', 'weekend_rainy', 'weekend_warm',
        'organic_hot_weather', 'organic_cold_weather', 'holiday_good_weather',
        'season_winter', 'season_spring', 'season_summer', 'season_autumn',
    ]


def analyze_weather_impact(df: pd.DataFrame, target_col: str = 'target', output_dir: str = None) -> Dict[str, Any]:
    """Analyzovať vplyv meteorologických príznakov na plnosť kontajnerov.

    Funkcia vypočíta Pearsonove korelácie medzi každým meteorologickým
    príznakom a cieľovou premennou, identifikuje štatisticky
    významné vzťahy a voliteľne uloží výsledok do CSV súboru.
    """
    if not CONFIG.ENABLE_WEATHER_ANALYSIS:
        return {}
    
    logger.info("Analyzujem vplyv meteorologických príznakov...")
    results = {}
    
    if target_col not in df.columns:
        return results
    
    target = df[target_col]
    
    # Korelačná analýza
    weather_cols = [c for c in df.columns if any(w in c.lower() for w in 
                   ['temp', 'precip', 'rain', 'humid', 'wind', 'press', 'weather', 'comfort', 'season'])]
    weather_cols = [c for c in weather_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int8']]
    
    correlations = []
    for col in weather_cols:
        mask = df[col].notna() & target.notna()
        if mask.sum() > 100:
            try:
                from scipy.stats import pearsonr
                corr, p_val = pearsonr(df.loc[mask, col], target[mask])
                correlations.append({'feature': col, 'correlation': corr, 'p_value': p_val, 'significant': p_val < 0.05})
            except Exception:
                continue
    
    corr_df = pd.DataFrame(correlations)
    if len(corr_df) > 0:
        corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
        results['correlations'] = corr_df
        
        logger.info("Top 5 korelácií meteorologických príznakov:")
        for _, row in corr_df.head(5).iterrows():
            sig = "*" if row['significant'] else ""
            logger.info(f"  {row['feature']:30s}: r={row['correlation']:+.3f}{sig}")
    
    # Zoskupené štatistiky
    grouped = {}
    if 'temp_cold' in df.columns and 'temp_warm' in df.columns:
        for name, col in [('Cold', 'temp_cold'), ('Warm', 'temp_warm'), ('Hot', 'temp_hot')]:
            if col in df.columns:
                mask = df[col] == 1
                if mask.sum() > 0:
                    grouped[name] = {'mean': target[mask].mean(), 'std': target[mask].std(), 'n': mask.sum()}
        results['by_temperature'] = grouped
    
    if 'is_raining' in df.columns:
        rain_grouped = {}
        for name, val in [('Dry', 0), ('Rainy', 1)]:
            mask = df['is_raining'] == val
            if mask.sum() > 0:
                rain_grouped[name] = {'mean': target[mask].mean(), 'std': target[mask].std(), 'n': mask.sum()}
        results['by_rain'] = rain_grouped
    
    # Uloženie a vizualizácia
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        if len(corr_df) > 0:
            corr_df.to_csv(f'{output_dir}/weather_correlations.csv', index=False)
        
        with open(f'{output_dir}/weather_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
    
    return results

