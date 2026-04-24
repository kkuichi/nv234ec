"""Podbalík waste_forecasting.models - modely, ladenie a metriky.

Obsahuje tri moduly: training s tréningovými funkciami pre
HistGradientBoostingRegressor, LightGBM a XGBoost, tuning s Optuna ladením
hyperparametrov nad časovou krížovou validáciou a metrics s regresnými
metrikami (RMSE, MAE, WAPE, SMAPE, R²), bootstrap intervalmi a párovými
štatistickými testami.
"""
