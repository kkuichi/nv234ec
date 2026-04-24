"""Jednotkové testy pre balík waste_forecasting.

Test_features kryje feature engineering - Fourierove harmoniky, detekciu
českých sviatkov a sanitizáciu názvov stĺpcov. Test_metrics overuje
výpočet regresných metrík (RMSE, MAE, calculate_all_metrics).

Testy su zámerne malé a rýchle, nepracujú s reálnymi dátami. Spúšťajú sa
z koreňa projektu:

    python -m unittest discover tests
"""
