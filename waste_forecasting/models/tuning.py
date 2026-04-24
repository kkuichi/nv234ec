"""Ladenie hyperparametrov cez Optuna.

Modul pripravuje parametrické priestory pre podporované modely a vyhodnocuje
ich pomocou temporálnej krížovej validácie.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import CONFIG
from waste_forecasting.data.splitting import temporal_train_val_split_per_container
from waste_forecasting.models.metrics import rmse
from waste_forecasting.models.training import (
    get_default_params,
    predict_model,
    train_model,
)

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

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

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)

__all__ = ['TemporalHyperparameterTuner', 'MultiModelTuner']


class TemporalHyperparameterTuner:
    """Ladenie hyperparametrov s temporálnou krížovou validáciou (HistGB).

    Zachovaná pôvodná implementácia pre spätnú kompatibilitu. Nový kód
    by mal používať MultiModelTuner, ktorá je zovšeobecnená
    pre viacero modelov.
    """

    def __init__(self, n_trials=None, timeout=None, cv_folds=None, seed=None):
        self.n_trials = n_trials or CONFIG.OPTUNA_N_TRIALS
        self.timeout = timeout or CONFIG.OPTUNA_TIMEOUT
        self.cv_folds = cv_folds or CONFIG.OPTUNA_CV_FOLDS
        self.seed = seed if seed is not None else CONFIG.SEED
        self.best_params: Optional[Dict] = None
        self.study = None

    def _create_splits(self, df: pd.DataFrame, feature_cols: List[str]):
        """Zostaviť temporálne CV foldy pre objektívnu funkciu.

        Kontajnery sú deterministicky zamiešané a rozdelené do
        cv_folds skupín. Validačná množina každého foldu sa
        vytvorí temporálnym rozdelením kontajnerov pridelených
        danému foldu.
        """
        containers = df['container_id'].unique()
        rng = np.random.RandomState(self.seed)
        containers = rng.permutation(containers)

        container_folds = np.array_split(containers, self.cv_folds)

        splits = []
        for fold_idx in range(self.cv_folds):
            val_containers = set(container_folds[fold_idx])
            train_containers = set(containers) - val_containers

            train_df = df[df['container_id'].isin(train_containers)]

            val_all = df[df['container_id'].isin(val_containers)]
            _, val_df = temporal_train_val_split_per_container(val_all, val_fraction=0.5)

            X_train = train_df[feature_cols]
            y_train = train_df['target'].values
            X_val = val_df[feature_cols]
            y_val = val_df['target'].values

            splits.append((X_train, y_train, X_val, y_val))

        return splits

    def objective(self, trial, splits):
        """Optuna objektívna funkcia: priemerný RMSE cez CV foldy."""
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'random_state': self.seed,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 15,
        }

        cv_scores = []
        for X_train, y_train, X_val, y_val in splits:
            model = HistGradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            cv_scores.append(rmse(y_val, y_pred))

        return np.mean(cv_scores)

    def tune(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Spustiť Optuna hľadanie a vrátiť najlepšie parametre."""
        if not HAS_OPTUNA:
            logger.warning("Optuna nie je dostupná, používam predvolené hodnoty")
            return CONFIG.HIST_GB_PARAMS

        logger.info(f"Spúšťam Optuna ladenie ({self.n_trials} pokusov, {self.cv_folds}-fold temporálna CV)")

        splits = self._create_splits(df, feature_cols)

        sampler = TPESampler(seed=self.seed)
        self.study = optuna.create_study(direction='minimize', sampler=sampler)

        self.study.optimize(
            lambda trial: self.objective(trial, splits),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=HAS_TQDM,
        )

        self.best_params = self.study.best_params.copy()
        self.best_params.update({
            'random_state': self.seed,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 15,
        })

        logger.info(f"Najlepšie temporálne CV RMSE: {self.study.best_value:.4f}")

        return self.best_params


class MultiModelTuner:
    """Ladenie hyperparametrov pre ľubovoľný z troch podporovaných modelov.

    Zovšeobecnená verzia pôvodného ladiča. Podľa parametra
    model_name zostaví parametrický priestor a objektívnu
    funkciu volá s wrapperom pre daný model.

    Parametrické priestory:

    * HistGradientBoosting: learning_rate ∈ [0.01, 0.3] (log),
      max_iter ∈ [100, 500], max_leaf_nodes ∈ [15, 63],
      min_samples_leaf ∈ [5, 50], l2_regularization ∈ [0, 1],
      max_depth ∈ [3, 12].
    * LightGBM: n_estimators ∈ [100, 500],
      learning_rate ∈ [0.01, 0.3] (log), max_depth ∈ [3, 12],
      num_leaves ∈ [15, 63], min_child_samples ∈ [5, 50],
      subsample ∈ [0.5, 1.0], colsample_bytree ∈ [0.5, 1.0],
      reg_alpha a reg_lambda ∈ [1e-8, 1] (log).
    * XGBoost: analogický priestor ako LightGBM s parametrom
      min_child_weight namiesto min_child_samples.
    """

    def __init__(
        self,
        model_name: str = 'HistGradientBoosting',
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        cv_folds: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.n_trials = n_trials or CONFIG.OPTUNA_N_TRIALS
        self.timeout = timeout or CONFIG.OPTUNA_TIMEOUT
        self.cv_folds = cv_folds or CONFIG.OPTUNA_CV_FOLDS
        self.seed = seed if seed is not None else CONFIG.SEED
        self.best_params: Optional[Dict] = None
        self.study = None

    def _create_splits(self, df: pd.DataFrame, feature_cols: List[str]):
        """Identická stratégia rozdeľovania ako v TemporalHyperparameterTuner."""
        containers = df['container_id'].unique()
        rng = np.random.RandomState(self.seed)
        containers = rng.permutation(containers)
        container_folds = np.array_split(containers, self.cv_folds)

        splits = []
        for fold_idx in range(self.cv_folds):
            val_containers = set(container_folds[fold_idx])
            train_containers = set(containers) - val_containers

            train_df = df[df['container_id'].isin(train_containers)]
            val_all = df[df['container_id'].isin(val_containers)]
            _, val_df = temporal_train_val_split_per_container(val_all, val_fraction=0.5)

            X_train = train_df[feature_cols]
            y_train = train_df['target'].values
            X_val = val_df[feature_cols]
            y_val = val_df['target'].values

            splits.append((X_train, y_train, X_val, y_val))

        return splits

    def _get_histgb_params(self, trial) -> Dict:
        """Parametrický priestor pre HistGradientBoostingRegressor."""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'random_state': self.seed,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 15,
        }

    def _get_lgbm_params(self, trial) -> Dict:
        """Parametrický priestor pre LightGBM."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': self.seed,
            'verbose': -1,
            'n_jobs': -1,
        }

    def _get_xgb_params(self, trial) -> Dict:
        """Parametrický priestor pre XGBoost."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': self.seed,
            'verbosity': 0,
            'n_jobs': -1,
        }

    def objective(self, trial, splits) -> float:
        """Objektívna funkcia: priemerný RMSE cez temporálne CV foldy.

        Podľa self.model_name vyberie parametrický priestor a každý
        fold natrénuje cez jednotný dispatcher train_model z modulu
        training.py.
        """
        if self.model_name == 'HistGradientBoosting':
            params = self._get_histgb_params(trial)
        elif self.model_name == 'LightGBM':
            params = self._get_lgbm_params(trial)
        elif self.model_name == 'XGBoost':
            params = self._get_xgb_params(trial)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        cv_scores = []
        for X_train, y_train, X_val, y_val in splits:
            model, y_pred, _ = train_model(
                self.model_name, X_train, y_train, X_val, y_val,
                params=params, clip_range=(0, 100)
            )
            if y_pred is not None:
                cv_scores.append(rmse(y_val, y_pred))
            else:
                # Záložný postup: ak obalová funkcia nevrátila predikciu, dopočítame ju ručne.
                y_pred_fallback = predict_model(model, X_val, self.model_name, (0, 100))
                cv_scores.append(rmse(y_val, y_pred_fallback))

        return np.mean(cv_scores)

    def tune(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Spustiť Optuna hľadanie a doplniť fixné parametre k nájdenému optimu.

        Po skončení hľadania sa k najlepším suggest-ovaným parametrom
        pripoja aj fixné parametre (random_state, verbose, n_jobs),
        ktoré nie sú súčasťou priestoru, ale model ich potrebuje.
        """
        if not HAS_OPTUNA:
            logger.warning("Optuna not available, using defaults")
            return get_default_params(self.model_name)

        logger.info(f"Spúšťam Optuna ladenie pre {self.model_name} ({self.n_trials} pokusov, {self.cv_folds}-fold)")

        splits = self._create_splits(df, feature_cols)

        sampler = TPESampler(seed=self.seed)
        self.study = optuna.create_study(direction='minimize', sampler=sampler)

        self.study.optimize(
            lambda trial: self.objective(trial, splits),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=HAS_TQDM,
        )

        self.best_params = self.study.best_params.copy()

        # Pridanie fixných parametrov, ktoré nie sú súčasťou
        # hľadania, no sú potrebné pre reprodukovateľnosť a výkon.
        if self.model_name == 'HistGradientBoosting':
            self.best_params.update({
                'random_state': self.seed,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 15,
            })
        elif self.model_name == 'LightGBM':
            self.best_params.update({
                'random_state': self.seed,
                'verbose': -1,
                'n_jobs': -1,
            })
        elif self.model_name == 'XGBoost':
            self.best_params.update({
                'random_state': self.seed,
                'verbosity': 0,
                'n_jobs': -1,
            })

        logger.info(f"Najlepšie CV RMSE pre {self.model_name}: {self.study.best_value:.4f}")

        return self.best_params
