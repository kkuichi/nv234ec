"""Podbalík waste_forecasting.models - modely, ladenie a metriky.

Obsahuje tri moduly: training s wrappermi pre HistGradientBoostingRegressor,
LightGBM a XGBoost (jednotný dispatcher cez train_model), tuning
s Optuna-založeným ladením hyperparametrov nad temporálnou krížovou validáciou
a metrics s regresnými metrikami (RMSE, MAE, WAPE, SMAPE, R²), bootstrap
intervalmi a párovými štatistickými testami.
"""
