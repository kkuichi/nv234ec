"""Podbalík waste_forecasting.data - načítanie a predspracovanie dát.

Modul loading sa stará o načítanie surového CSV, typovú normalizáciu
a pamäťovú optimalizáciu. Modul preprocessing implementuje heuristickú
detekciu vývozov, prevzorkovanie časovej série a kontroly proti data leakage.
Modul splitting poskytuje container holdout, temporálne foldy a kapacitne
stratifikovaný výber testovacích kontajnerov.
"""
