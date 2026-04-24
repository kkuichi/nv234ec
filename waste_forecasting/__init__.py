"""Balík waste_forecasting - prediktívne modelovanie plnosti kontajnerov.

Hlavný balík obsahuje pipeline pre predikciu plnosti komunálnych odpadových
kontajnerov v troch úlohách: predikcia plnosti v momente vývozu
(Experiment A), 24-hodinová predikcia na prevzorkovanej časovej sérii
(Experiment B) a odhad počtu dní do dosiahnutia 85 % plnosti (Experiment C).

Kód je rozdelený do podbalíkov podľa zodpovednosti: data pre načítanie
a predspracovanie, features pre tvorbu príznakov, models pre tréning
a ladenie, experiments pre tri hlavné úlohy a evaluation pre krížovú
validáciu, stabilitu, SHAP, referenčné modely a reporty.
"""
