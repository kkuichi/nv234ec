"""Centrálna konfigurácia projektu.

Trieda Config drží prahy, cesty, seedy, nastavenia validácie,
hyperparametre a prepínače pre voliteľné časti pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


__all__ = ["Config", "CONFIG"]


@dataclass
class Config:
    """Konfigurácia pipeline pre predikciu plnosti odpadových kontajnerov.

    Všetky atribúty sú ladiace parametre pipeline a sú organizované do
    logických skupín oddelených komentármi. Zmena hodnoty pred
    vytvorením inštancie (alebo na atribúte CONFIG pred spustením
    pipeline) priamo ovplyvní správanie príslušnej časti systému.
    """

    # Reprodukovateľnosť ------------------------------------------------
    SEED: int = 42
    STABILITY_SEEDS: List[int] = field(
        default_factory=lambda: [42, 123, 456, 789, 1011]
    )

    # Krížová validácia -------------------------------------------------
    N_FOLDS: int = 5
    OPTUNA_CV_FOLDS: int = 3
    CV_TYPE: str = "container_holdout"

    # Rozdelenie dát ----------------------------------------------------
    TEST_CONTAINER_FRACTION: float = 0.20
    VAL_FRACTION_PER_CONTAINER: float = 0.15

    # Filtrovanie dát ---------------------------------------------------
    MIN_COLLECTIONS_PER_CONTAINER: int = 10
    MIN_MEASUREMENTS_PER_CONTAINER: int = 100

    # Kapacitne segmentované modelovanie -------------------------------
    RUN_CAPACITY_SEGMENTED_MODELS: bool = True
    CAPACITY_SEGMENTS_TO_RUN: List[str] = field(
        default_factory=lambda: ["low", "high"]
    )

    # Exploratívna dátová analýza ---------------------------------------
    NEVER_FULL_MAX_PCT: float = 70.0

    # Geolokácia --------------------------------------------------------
    GEO_ADD_CLUSTER_FEATURE: bool = True
    GEO_N_CLUSTERS: int = 60
    GEO_CLUSTER_MIN_CONTAINERS: int = 200

    # Detekcia vývozov --------------------------------------------------
    # Tieto prahy používa heuristický detektor vývozov v module
    # data/preprocessing.py. Negatívny pokles plnosti signalizuje
    # potenciálny vývoz, ak sú súčasne splnené všetky podmienky.
    COLLECTION_DROP_THRESHOLD: float = -30.0
    LOW_AFTER_THRESHOLD: float = 50.0
    HIGH_BEFORE_THRESHOLD: float = 40.0
    MAX_HOURS_BETWEEN_MEASUREMENTS: float = 168.0
    USE_PCT_NEXT_IN_COLLECTION_DETECTION: bool = False

    # Mriežka pre analýzu citlivosti detektora vývozov
    SENSITIVITY_DROP_THRESHOLDS: List[float] = field(
        default_factory=lambda: [-20.0, -25.0, -30.0, -35.0, -40.0]
    )
    SENSITIVITY_LOW_AFTER: List[float] = field(
        default_factory=lambda: [40.0, 45.0, 50.0, 55.0, 60.0]
    )

    # Kapacitné prahy ---------------------------------------------------
    # Kontajner s nominálnou kapacitou <= LOW je klasifikovaný ako 'low',
    # >= HIGH ako 'high', medzi nimi ako 'medium'.
    CAPACITY_THRESHOLD_LOW: float = 2150.0
    CAPACITY_THRESHOLD_HIGH: float = 2500.0

    # Experiment C: čas do dosiahnutia prahu ----------------------------
    FILL_THRESHOLD: float = 85.0
    MAX_TTT_DAYS: int = 14

    # Tvorba príznakov --------------------------------------------------
    EVENT_LAGS: List[int] = field(default_factory=lambda: list(range(1, 11)))
    EVENT_ROLL_WINDOWS: List[int] = field(default_factory=lambda: [3, 5, 7])
    TIME_LAGS: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 48])
    TIME_ROLL_WINDOWS: List[int] = field(default_factory=lambda: [6, 12, 24])
    WEEKLY_FOURIER_K: int = 3
    DAILY_FOURIER_K: int = 2
    RESAMPLE_HOURS: int = 6

    # Štatistická validácia --------------------------------------------
    N_BOOTSTRAP: int = 1000
    CI_LEVEL: float = 0.95

    # Ladenie hyperparametrov ------------------------------------------
    OPTUNA_N_TRIALS: int = 50
    OPTUNA_TIMEOUT: int = 1800
    ENABLE_HYPERPARAMETER_TUNING: bool = True

    # Interpretovateľnosť ----------------------------------------------
    ENABLE_SHAP: bool = True
    SHAP_MAX_SAMPLES: int = 1000

    # Kódovanie kategoriálnych premenných ------------------------------
    LOW_CARDINALITY_THRESHOLD: int = 10
    TOP_K_CATEGORIES: int = 8
    CORRELATION_THRESHOLD: float = 0.95

    # Predvolené hyperparametre modelov --------------------------------
    # Tieto hodnoty sa použijú, ak je ladenie cez Optuna vypnuté
    # napríklad prepínačom --skip-tuning, alebo ako záložné hodnoty pri chybe.
    HIST_GB_PARAMS: Dict = field(
        default_factory=lambda: {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.1,
            "random_state": 42,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
        }
    )
    LGBM_PARAMS: Dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
    )
    XGB_PARAMS: Dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }
    )

    MODELS_TO_COMPARE: List[str] = field(
        default_factory=lambda: ["HistGradientBoosting", "LightGBM", "XGBoost"]
    )

    # Geolokácia - centrum Prahy ---------------------------------------
    PRAGUE_CENTER_LAT: float = 50.0812
    PRAGUE_CENTER_LON: float = 14.4269

    # Meteorologické dáta ----------------------------------------------
    # Meteorologické príznaky sa pridávajú IBA ak existuje lokálny CSV
    # súbor (predvolene weather.csv v pracovnom adresári). Ak súbor
    # chýba, celá meteorologická časť pipeline sa automaticky preskočí
    # a žiadne meteorologické stĺpce sa do datasetu nepridajú.
    # Údaje sa neťahajú zo žiadneho externého zdroja (internet, API).
    WEATHER_DATA_PATH: str = "weather.csv"
    WEATHER_OVERRIDE_EXISTING: bool = True
    ENABLE_WEATHER_FEATURES: bool = True
    ENABLE_WEATHER_ANALYSIS: bool = True

    # Teplotné prahy (°C) pre odvodené príznaky
    TEMP_COLD_THRESHOLD: float = 5.0
    TEMP_WARM_THRESHOLD: float = 20.0
    TEMP_HOT_THRESHOLD: float = 30.0
    TEMP_FREEZING_THRESHOLD: float = 0.0

    # Prahy pre zrážky (mm)
    PRECIP_LIGHT_THRESHOLD: float = 2.5
    PRECIP_MODERATE_THRESHOLD: float = 7.5
    PRECIP_HEAVY_THRESHOLD: float = 15.0

    # Prahy pre rýchlosť vetra (m/s)
    WIND_CALM_THRESHOLD: float = 2.0
    WIND_MODERATE_THRESHOLD: float = 8.0
    WIND_STRONG_THRESHOLD: float = 14.0

    # Prahy pre relatívnu vlhkosť (%)
    HUMIDITY_LOW_THRESHOLD: float = 40.0
    HUMIDITY_HIGH_THRESHOLD: float = 80.0

    # Prahy pre atmosférický tlak (hPa)
    PRESSURE_LOW_THRESHOLD: float = 1000.0
    PRESSURE_HIGH_THRESHOLD: float = 1020.0

    WEATHER_LAGS: List[int] = field(default_factory=lambda: [1, 6, 12, 24])
    WEATHER_ROLL_WINDOWS: List[int] = field(
        default_factory=lambda: [6, 12, 24, 48]
    )

    def to_dict(self) -> Dict:
        """Vrátiť konfiguráciu ako štandardný Python slovník.
        """
        return asdict(self)


# Zdieľaná inštancia singletonu používaná naprieč celým balíkom.
CONFIG = Config()
