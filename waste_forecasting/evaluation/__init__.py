"""Podbalík waste_forecasting.evaluation - evaluácia a reporting.

Zhromažďuje všetko, čo nasleduje po tréningu modelov: krížovú validáciu
(cross_validation), stabilitu pri viacerých seedoch (stability), SHAP
interpretáciu (interpretability), ablačnú štúdiu meteorologických príznakov
(weather_ablation), klasické referenčné modely (baselines), survival model
pre Experiment C (survival_baseline) a generovanie reportov
(reporting, visualization).
"""
