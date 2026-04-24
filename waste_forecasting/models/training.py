"""Tréning gradient boosting modelov a jednotný výber tréningovej funkcie.

Pre každý z troch porovnávaných modelov (HistGradientBoostingRegressor
zo scikit-learn, LightGBM a XGBoost) je v tomto module jedna tréningová
funkcia s rovnakou signatúrou: dostane trénovacie dáta a voliteľne
validačnú množinu, potom vráti trojicu (model, y_pred, info). Funkcia
train_model podľa textového identifikátora modelu zavolá správnu funkciu.

Jednotné správanie naprieč všetkými tromi: early stopping s toleranciou
20 iterácií bez zlepšenia, orezávanie predikcií do platného rozsahu
[0, 100] percent, a pri LightGBM a XGBoost ešte sanitizácia názvov
príznakov, pretože tieto knižnice nemajú radi medzery a diakritiku.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import CONFIG
from waste_forecasting.features.encoding import sanitize_feature_names

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logger = logging.getLogger(__name__)

__all__ = [
    'train_histgb_model',
    'train_lgbm_model',
    'train_xgb_model',
    'train_model',
    'predict_model',
    'get_default_params',
    'is_model_available',
]


def train_histgb_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    clip_range: Tuple[float, float] = (0, 100),
    seed: Optional[int] = None,
) -> Tuple[HistGradientBoostingRegressor, Optional[np.ndarray], Dict]:
    """Natrénovať HistGradientBoostingRegressor z knižnice scikit-learn.

    Ak sú zadané validačné dáta, funkcia vráti aj predikcie
    na validačnej množine orezané do intervalu clip_range.
    Early stopping sa riadi internou validation_fraction v rámci
    tréningových dát podľa parametrov v params.
    """
    params = (params or CONFIG.HIST_GB_PARAMS).copy()
    if seed is not None:
        params['random_state'] = seed

    model = HistGradientBoostingRegressor(**params)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    info = {
        'train_time': train_time,
        'n_iter': getattr(model, 'n_iter_', params.get('max_iter')),
        'params': params
    }

    # Ak early stopping zastavil tréning predčasne, zaznamenáme to.
    if hasattr(model, 'n_iter_'):
        max_iter = params.get('max_iter', 300)
        if model.n_iter_ < max_iter:
            logger.info(f"Predčasné zastavenie pri {model.n_iter_}/{max_iter}")

    logger.info(f"Čas tréningu: {train_time:.1f} s")

    if X_val is not None:
        y_pred = np.clip(model.predict(X_val), *clip_range)
        return model, y_pred, info

    return model, None, info


def train_lgbm_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    clip_range: Tuple[float, float] = (0, 100),
    seed: Optional[int] = None,
) -> Tuple[Any, Optional[np.ndarray], Dict]:
    """Natrénovať LightGBM regresný model.

    Funkcia vyžaduje nainštalovanú knižnicu lightgbm. Pre
    kompatibilitu sanitizuje názvy stĺpcov, ktoré LightGBM
    natívne nepodporuje (medzery, diakritika).

    Parameters a Returns sú rovnaké ako v train_histgb_model,
    okrem typu návratového modelu (lgb.LGBMRegressor).
    """
    if not HAS_LGBM:
        raise ImportError("LightGBM is not installed")

    params = (params or CONFIG.LGBM_PARAMS).copy()
    if seed is not None:
        params['random_state'] = seed

    callbacks = []
    fit_params: Dict = {}
    if X_val is not None and y_val is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))
        callbacks.append(lgb.log_evaluation(period=0))
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['callbacks'] = callbacks

    # Sanitizácia názvov stĺpcov kvôli kompatibilite s LightGBM.
    X_train_clean, _ = sanitize_feature_names(X_train)
    if X_val is not None and y_val is not None:
        X_val_clean, _ = sanitize_feature_names(X_val)
        fit_params['eval_set'] = [(X_val_clean, y_val)]

    model = lgb.LGBMRegressor(**params)

    start = time.time()
    model.fit(X_train_clean, y_train, **fit_params)
    train_time = time.time() - start

    info = {
        'train_time': train_time,
        'n_iter': model.n_estimators_ if hasattr(model, 'n_estimators_') else params.get('n_estimators'),
        'params': params
    }

    logger.info(f"Čas tréningu LightGBM: {train_time:.1f} s, iterácie: {info['n_iter']}")

    if X_val is not None:
        X_val_for_pred, _ = sanitize_feature_names(X_val)
        y_pred = np.clip(model.predict(X_val_for_pred), *clip_range)
        return model, y_pred, info

    return model, None, info


def train_xgb_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    clip_range: Tuple[float, float] = (0, 100),
    seed: Optional[int] = None,
) -> Tuple[Any, Optional[np.ndarray], Dict]:
    """Natrénovať XGBoost regresný model.

    Funkcia vyžaduje nainštalovanú knižnicu xgboost. Na rozdiel
    od LightGBM sa early stopping konfiguruje cez parameter
    early_stopping_rounds priamo v konštruktore.

    Parameters a Returns sú rovnaké ako v train_histgb_model,
    okrem typu návratového modelu (xgb.XGBRegressor).
    """
    if not HAS_XGB:
        raise ImportError("XGBoost is not installed")

    params = (params or CONFIG.XGB_PARAMS).copy()
    if seed is not None:
        params['random_state'] = seed

    fit_params: Dict = {}
    if X_val is not None and y_val is not None:
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['verbose'] = False

    # XGBRegressor prijíma early_stopping_rounds cez konštruktor.
    early_stop = 20 if X_val is not None else None

    # Sanitizácia názvov stĺpcov pre kompatibilitu s XGBoost.
    X_train_clean, _ = sanitize_feature_names(X_train)
    if X_val is not None and y_val is not None:
        X_val_clean, _ = sanitize_feature_names(X_val)
        fit_params['eval_set'] = [(X_val_clean, y_val)]

    model = xgb.XGBRegressor(**params, early_stopping_rounds=early_stop)

    start = time.time()
    model.fit(X_train_clean, y_train, **fit_params)
    train_time = time.time() - start

    n_iter = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else params.get('n_estimators')

    info = {
        'train_time': train_time,
        'n_iter': n_iter,
        'params': params
    }

    logger.info(f"Čas tréningu XGBoost: {train_time:.1f} s, iterácie: {info['n_iter']}")

    if X_val is not None:
        X_val_for_pred, _ = sanitize_feature_names(X_val)
        y_pred = np.clip(model.predict(X_val_for_pred), *clip_range)
        return model, y_pred, info

    return model, None, info


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    clip_range: Tuple[float, float] = (0, 100),
    seed: Optional[int] = None,
) -> Tuple[Any, Optional[np.ndarray], Dict]:
    """Vybrať tréningovú funkciu podľa názvu modelu.

    Experimentálny kód preto nepotrebuje vedieť, ktorá konkrétna knižnica
    je použitá.
    """
    if model_name == 'HistGradientBoosting':
        return train_histgb_model(X_train, y_train, X_val, y_val, params, clip_range, seed)
    elif model_name == 'LightGBM':
        return train_lgbm_model(X_train, y_train, X_val, y_val, params, clip_range, seed)
    elif model_name == 'XGBoost':
        return train_xgb_model(X_train, y_train, X_val, y_val, params, clip_range, seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def predict_model(
    model: Any,
    X: pd.DataFrame,
    model_name: str,
    clip_range: Tuple[float, float] = (0, 100),
) -> np.ndarray:
    """Vykonať predikciu s korektným spracovaním názvov príznakov.

    Pre LightGBM a XGBoost je potrebná sanitizácia názvov
    stĺpcov tak, ako sa vykonala pri tréningu.
    """
    if model_name in ('LightGBM', 'XGBoost'):
        X_clean, _ = sanitize_feature_names(X)
        return np.clip(model.predict(X_clean), *clip_range)
    return np.clip(model.predict(X), *clip_range)


def get_default_params(model_name: str) -> Dict:
    """Vrátiť predvolené hyperparametre pre zadaný model.
    """
    if model_name == 'HistGradientBoosting':
        return CONFIG.HIST_GB_PARAMS.copy()
    elif model_name == 'LightGBM':
        return CONFIG.LGBM_PARAMS.copy()
    elif model_name == 'XGBoost':
        return CONFIG.XGB_PARAMS.copy()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def is_model_available(model_name: str) -> bool:
    """Overiť, či je knižnica pre zadaný model nainštalovaná.

    HistGradientBoostingRegressor je súčasťou scikit-learn a je teda
    vždy dostupný; LightGBM a XGBoost môžu chýbať v závislosti od
    stavu prostredia.
    """
    if model_name == 'HistGradientBoosting':
        return True  # Súčasť scikit-learn - vždy dostupný.
    elif model_name == 'LightGBM':
        return HAS_LGBM
    elif model_name == 'XGBoost':
        return HAS_XGB
    return False
