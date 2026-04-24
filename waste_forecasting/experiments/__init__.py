"""Podbalík waste_forecasting.experiments - tri hlavné experimenty.

Experiment A robí event-based predikciu plnosti v momente detegovaného
vývozu (cieľ je plnosť pred vývozom), Experiment B rieši 24-hodinovú
predikciu na pravidelne prevzorkovanej časovej sérii a Experiment C
odhaduje počet dní do dosiahnutia 85 % plnosti (time-to-threshold).

Každý experiment trénuje rovnakú trojicu modelov (HistGradientBoosting,
LightGBM, XGBoost) na identickej sade príznakov a tom istom splite. Víťaz
sa určuje podľa najnižšieho testovacieho RMSE (pre C MAE) a následne sa na
neho aplikuje SHAP aj permutačná dôležitosť.
"""
