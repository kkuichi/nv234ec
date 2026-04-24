"""Regresné metriky s bootstrap intervalmi spoľahlivosti.

Okrem bežných metrík RMSE, MAE, WAPE, SMAPE a R² ponúka modul aj
bootstrapové intervaly spoľahlivosti a párové štatistické testy
(Wilcoxonov signed-rank test a klasický párový t-test), ktoré
používam na porovnanie modelov.

Vstupy s NaN hodnotami sa pred výpočtom odfiltrujú, takže sa dajú
porovnávať aj modely, kde niektoré vzorky baseline nevie predpovedať.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import CONFIG

logger = logging.getLogger(__name__)

__all__ = [
    'rmse',
    'mae',
    'wape',
    'smape',
    'bootstrap_confidence_interval',
    'paired_statistical_test',
    'calculate_all_metrics',
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Druhá odmocnina priemernej kvadratickej chyby. Kvôli kvadratickej
    váhe silnejšie penalizuje veľké chyby než MAE.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Priemerná absolútna chyba. Robustnejšia voči výnimočným hodnotám
    ako RMSE.
    """
    return mean_absolute_error(y_true, y_pred)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error.

    Vážená percentuálna chyba: súčet absolútnych chýb delený súčtom
    absolútnych hodnôt skutočnej premennej. Pri nenulovej celkovej
    hodnote je škálovo invariantná a dobre interpretovateľná.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true) == 0:
        return np.nan

    total_actual = np.sum(np.abs(y_true))
    total_error = np.sum(np.abs(y_true - y_pred))

    if total_actual == 0:
        return 0.0 if total_error == 0 else np.nan

    return 100.0 * total_error / total_actual


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Symetrická percentuálna chyba s aditívnou konštantou epsilon
    pre numerickú stabilitu pri malých hodnotách. Je odolnejšia voči
    nule v skutočnej hodnote než klasické MAPE.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / denominator)


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable,
    n_bootstrap: Optional[int] = None,
    ci_level: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Bootstrap interval spoľahlivosti pre zadanú metriku.

    Neparametrická metóda založená na opakovanom výbere s vrátením.
    Pre každú iteráciu sa nanovo vypočíta metrika a z jej distribúcie
    sa vypočítajú kvantily zodpovedajúce ci_level.
    """
    n_bootstrap = n_bootstrap or CONFIG.N_BOOTSTRAP
    ci_level = ci_level or CONFIG.CI_LEVEL
    seed = seed or CONFIG.SEED

    rng = np.random.RandomState(seed)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = np.array(y_true)[mask], np.array(y_pred)[mask]

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

    point_estimate = metric_func(y_true, y_pred)
    n = len(y_true)

    bootstrap_values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            val = metric_func(y_true[idx], y_pred[idx])
            if not np.isnan(val):
                bootstrap_values.append(val)
        except Exception:
            continue

    if len(bootstrap_values) < 10:
        return point_estimate, np.nan, np.nan

    alpha = 1 - ci_level
    return (
        point_estimate,
        np.percentile(bootstrap_values, alpha / 2 * 100),
        np.percentile(bootstrap_values, (1 - alpha / 2) * 100),
    )


def paired_statistical_test(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    test_type: str = 'wilcoxon',
) -> Tuple[float, float, str]:
    """Párový štatistický test medzi modelom a baseline predikciou.

    Pre každú vzorku vypočíta absolútne chyby oboch prístupov a
    porovná ich párovým testom. Wilcoxonov signed-rank test je
    neparametrický a odporúča sa ak chyby nie sú normálne
    rozdelené; párový t-test je citlivejší pri platnosti normality.
    """
    errors_model = np.abs(y_true - y_pred_model)
    errors_baseline = np.abs(y_true - y_pred_baseline)

    mask = ~np.isnan(errors_model) & ~np.isnan(errors_baseline)
    errors_model, errors_baseline = errors_model[mask], errors_baseline[mask]

    if len(errors_model) < 10:
        return np.nan, np.nan, ""

    if test_type == 'wilcoxon':
        try:
            stat, p_value = wilcoxon(errors_model, errors_baseline, alternative='two-sided')
        except ValueError:
            # ValueError z Wilcoxon sa vyskytne, ak všetky rozdiely sú nulové.
            stat, p_value = 0, 1.0
    elif test_type == 'ttest':
        stat, p_value = ttest_rel(errors_model, errors_baseline)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    return stat, p_value, stars


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    with_ci: bool = True,
) -> Dict[str, Any]:
    """Vypočítať všetky regresné metriky naraz, voliteľne s CI.

    Štandardná agregačná funkcia používaná naprieč pipeline pre
    jednotné reportovanie metrík. Bootstrap CI sú vypočítané pre
    RMSE a MAE (ostatné metriky ich majú ako rozšírenie pri potrebe).
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = np.array(y_true)[mask], np.array(y_pred)[mask]

    if len(y_true) == 0:
        return {'n': 0, 'rmse': np.nan, 'mae': np.nan, 'wape': np.nan, 'smape': np.nan, 'r2': np.nan}

    result = {
        'n': len(y_true),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'wape': wape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

    if with_ci:
        for name, func in [('rmse', rmse), ('mae', mae)]:
            _, ci_l, ci_u = bootstrap_confidence_interval(y_true, y_pred, func)
            result[f'{name}_ci_lower'], result[f'{name}_ci_upper'] = ci_l, ci_u

    return result
