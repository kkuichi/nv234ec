"""Testy pre tvorbu príznakov.

Overujú Fourierove príznaky, sviatky a kódovanie kategoriálnych premenných
na malých syntetických dátach.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestFourierFeatures(unittest.TestCase):
    """Testy pre waste_forecasting.features.fourier.add_fourier_features."""

    def test_fourier_adds_expected_columns(self) -> None:
        """Overiť, že funkcia pridá sin/cos stĺpce pre týždeň aj deň.

        Očakávame, že pre prvú harmoniku (k=1) vzniknú štyri
        stĺpce: fourier_week_sin_1, fourier_week_cos_1,
        fourier_day_sin_1, fourier_day_cos_1.
        """
        from waste_forecasting.features.fourier import add_fourier_features

        df = pd.DataFrame(
            {
                "measured_at_utc": pd.date_range(
                    "2019-01-01", periods=24, freq="h", tz="UTC"
                ),
                "percent_calculated": np.arange(24, dtype=float),
            }
        )
        # add_fourier_features očakáva stĺpce 'day_of_week' a 'hour'.
        df["day_of_week"] = df["measured_at_utc"].dt.dayofweek
        df["hour"] = df["measured_at_utc"].dt.hour

        out = add_fourier_features(df)

        self.assertIn("fourier_week_sin_1", out.columns)
        self.assertIn("fourier_week_cos_1", out.columns)
        self.assertIn("fourier_day_sin_1", out.columns)
        self.assertIn("fourier_day_cos_1", out.columns)

    def test_fourier_values_in_range(self) -> None:
        """Overiť, že vygenerované hodnoty ležia v intervale [-1, 1].

        Sínus a kosínus musia byť z definície ohraničené. Ak by
        výstup prekročil tento rozsah, znamenalo by to chybu
        v normalizácii periódy (deľba T).
        """
        from waste_forecasting.features.fourier import add_fourier_features

        df = pd.DataFrame(
            {
                "measured_at_utc": pd.date_range(
                    "2019-01-01", periods=100, freq="h", tz="UTC"
                ),
                "percent_calculated": np.zeros(100),
            }
        )
        df["day_of_week"] = df["measured_at_utc"].dt.dayofweek
        df["hour"] = df["measured_at_utc"].dt.hour

        out = add_fourier_features(df)

        for col in ["fourier_week_sin_1", "fourier_day_sin_1"]:
            self.assertTrue((out[col] >= -1).all() and (out[col] <= 1).all())


class TestHolidayFeatures(unittest.TestCase):
    """Testy pre detekciu českých štátnych sviatkov."""

    def test_easter_known_years(self) -> None:
        """Overiť výpočet Veľkonočnej nedele pre roky 2019 a 2020.

        Dátumy sú overené proti kalendáru: 21. apríl 2019 a
        12. apríl 2020. Test odhalí chyby v Gaussovom algoritme
        (napr. nesprávne modulá alebo prehodené operandy).
        """
        from waste_forecasting.features.holiday import _compute_easter_sunday

        self.assertEqual(
            _compute_easter_sunday(2019), pd.Timestamp("2019-04-21")
        )
        self.assertEqual(
            _compute_easter_sunday(2020), pd.Timestamp("2020-04-12")
        )

    def test_holiday_detection_new_year(self) -> None:
        """Overiť, že is_holiday správne označí 1. január.

        Porovnáva sa s bežným dňom v strede roka (15. jún), ktorý
        nesmie byť označený ako sviatok.
        """
        from waste_forecasting.features.holiday import create_holiday_features

        df = pd.DataFrame(
            {
                "measured_at_utc": [
                    pd.Timestamp("2019-01-01", tz="UTC"),
                    pd.Timestamp("2019-06-15", tz="UTC"),
                ]
            }
        )
        out = create_holiday_features(df, date_col="measured_at_utc")
        self.assertIn("is_holiday", out.columns)
        self.assertEqual(out["is_holiday"].iloc[0], 1)
        self.assertEqual(out["is_holiday"].iloc[1], 0)


class TestCategoricalEncoding(unittest.TestCase):
    """Testy pre sanitizáciu názvov príznakov a kategoriálne kódovanie."""

    def test_sanitize_feature_names_removes_special_chars(self) -> None:
        """Overiť, že sanitizácia odstráni medzery a špeciálne znaky.

        XGBoost a LightGBM netolerujú v názvoch stĺpcov medzery a
        niektoré znaky ([, ], <). Po sanitizácii musí
        platiť: (1) žiadny stĺpec neobsahuje medzeru; (2) reverzná
        mapa umožňuje rekonštruovať pôvodné názvy pre final report.
        """
        from waste_forecasting.features.encoding import sanitize_feature_names

        df = pd.DataFrame(
            {
                "trash_type_Papír": [1, 0, 1],
                "trash_type_Nápojové kartóny": [0, 1, 0],
                "current_fill": [45.0, 67.3, 23.1],
            }
        )
        sanitized, reverse_map = sanitize_feature_names(df)

        for col in sanitized.columns:
            self.assertNotIn(" ", col)

        for sanitized_col, orig_col in reverse_map.items():
            self.assertIn(orig_col, df.columns)


if __name__ == "__main__":
    unittest.main()
