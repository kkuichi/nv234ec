"""Balík waste_forecasting - prediktívne modelovanie plnosti kontajnerov.

Hlavný balík obsahuje pipeline pre predikciu plnosti komunálnych odpadových
kontajnerov v troch formuláciách úlohy: event-based predikcia v momente
vývozu (Experiment A), 24-hodinová predikcia na prevzorkovanej časovej sérii
(Experiment B) a odhad počtu dní do dosiahnutia 85 % plnosti (Experiment C).

Kód je rozdelený do podbalíkov podľa zodpovednosti: data pre načítanie
a predspracovanie, features pre feature engineering, models pre tréning
a ladenie, experiments pre tri hlavné úlohy a evaluation pre krížovú
validáciu, stabilitu, SHAP, baselines a reporting.
"""

__version__ = "1.0.0"
__author__ = "Bc. Natanael Varjan"
