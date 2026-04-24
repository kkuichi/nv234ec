"""Testy pre regresné metriky.

Pokrývajú RMSE, MAE a agregovaný výstup calculate_all_metrics.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestMetrics(unittest.TestCase):
    """Testy pre základné regresné metriky."""

    def test_rmse_perfect_prediction(self) -> None:
        """RMSE pre identický vektor y_true == y_pred musí byť 0."""
        from waste_forecasting.models.metrics import rmse

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(rmse(y, y), 0.0, places=6)

    def test_rmse_constant_error(self) -> None:
        """RMSE pri konštantnej chybe +1 na všetkých vzorkách musí byť 1.

        Ak je každá predikcia posunutá o rovnakú hodnotu, RMSE sa
        musí rovnať tejto hodnote (druhá odmocnina z priemeru
        kvadrátov rovnakých čísel = to isté číslo).
        """
        from waste_forecasting.models.metrics import rmse

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # posun o +1
        self.assertAlmostEqual(rmse(y_true, y_pred), 1.0, places=6)

    def test_mae_handles_mixed_signs(self) -> None:
        """MAE musí správne spriemerovať absolútne hodnoty chýb.

        Pre chyby [-1, +1, -1] je MAE rovné 1 (priemer absolútnych
        hodnôt). Test odhalí chybu, keby sa namiesto abs použila
        obyčajná suma (dostali by sme -1/3).
        """
        from waste_forecasting.models.metrics import mae

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.0, 3.0, 2.0])  # chyby: -1, +1, -1
        self.assertAlmostEqual(mae(y_true, y_pred), 1.0, places=6)

    def test_calculate_all_metrics_returns_expected_keys(self) -> None:
        """Overiť, že agregovaná funkcia vracia očakávané metriky.

        Syntetické dáta: normálne rozdelenie s priemerom 50 a std 15,
        predikcia je skutočná hodnota + malý Gaussov šum (std 3).
        Požiadavky:

        * Slovník obsahuje všetky očakávané kľúče.
        * n zodpovedá veľkosti vstupu (200).
        * R² je > 0.5 (s takýmto malým šumom by mal byť blízko 1,
          prahová hodnota 0.5 je konzervatívna).
        """
        from waste_forecasting.models.metrics import calculate_all_metrics

        y_true = np.random.RandomState(42).normal(50, 15, size=200)
        y_pred = y_true + np.random.RandomState(1).normal(0, 3, size=200)

        metrics = calculate_all_metrics(y_true, y_pred, with_ci=False)

        for key in ("rmse", "mae", "wape", "smape", "r2", "n"):
            self.assertIn(key, metrics)

        self.assertEqual(metrics["n"], 200)
        self.assertGreater(metrics["r2"], 0.5)


if __name__ == "__main__":
    unittest.main()
