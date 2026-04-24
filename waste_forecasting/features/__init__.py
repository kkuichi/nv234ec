"""Podbalík waste_forecasting.features - tvorba príznakov.

Každá rodina príznakov má vlastný modul: temporal (lag, diferencie, kĺzavé
štatistiky a datasety pre vývozy aj časovú predikciu), fourier (cyklické sínusové a kosínusové
harmoniky), holiday (české štátne sviatky), spatial (GPS-odvodené príznaky
ako vzdialenosť od centra Prahy a KMeans klaster), weather (meteorologické
príznaky) a encoding (kategoriálne kódovanie).
"""
