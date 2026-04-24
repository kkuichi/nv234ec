"""Podbalík waste_forecasting.evaluation - evaluácia a reporting.

Zhromažďuje všetko, čo nasleduje po tréningu modelov: krížovú validáciu
(cross_validation), multi-seed stabilitu (stability), SHAP interpretáciu
(interpretability), ablačnú štúdiu meteorologických príznakov
(weather_ablation), klasické baseline modely (baselines), survival
baseline pre Experiment C (survival_baseline) a generovanie finálnych
reportov (reporting, visualization).
"""
