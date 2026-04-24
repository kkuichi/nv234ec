"""Podbalík waste_forecasting.features - feature engineering.

Každá rodina príznakov má vlastný modul: temporal (lag, diferencie, kĺzavé
štatistiky a event/time datasety), fourier (cyklické sínusové a kosínusové
harmoniky), holiday (české štátne sviatky), spatial (GPS-odvodené príznaky
ako vzdialenosť od centra Prahy a KMeans klaster), weather (meteorologické
príznaky) a encoding (kategoriálne kódovanie).
"""
