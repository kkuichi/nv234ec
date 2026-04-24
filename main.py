"""Vstupný bod pipeline pre prediktívne modelovanie plnosti kontajnerov.

Skript spracuje argumenty príkazového riadka, spustí vybrané experimenty
a uloží reporty, tabuľky a vizualizácie do výstupného priečinka.

Príklad:
    python main.py --data merged_dataset.csv --weather weather.csv --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import CONFIG
from waste_forecasting.data.loading import load_and_preprocess_data
from waste_forecasting.data.preprocessing import sensitivity_analysis_collections
from waste_forecasting.data.splitting import (
    get_test_containers_for_capacity,
    get_unified_test_containers,
)
from waste_forecasting.evaluation.baselines import (
    compare_all_baselines,
    run_arima_prophet_baseline_expB,
)
from waste_forecasting.evaluation.cross_validation import run_cross_validation
from waste_forecasting.evaluation.reporting import (
    export_fixed_table16,
    export_hyperparameter_tables,
    export_results_table,
    generate_final_report,
    save_capacity_comparison_table,
)
from waste_forecasting.evaluation.stability import run_stability_analysis
from waste_forecasting.evaluation.survival_baseline import run_survival_baseline_expC
from waste_forecasting.evaluation.weather_ablation import (
    run_weather_ablation_expB,
    run_weather_ablation_study,
)
from waste_forecasting.evaluation.visualization import generate_all_figures
from waste_forecasting.experiments.experiment_a import run_experiment_A
from waste_forecasting.experiments.experiment_b import run_experiment_B
from waste_forecasting.experiments.experiment_c import run_experiment_C

logger = logging.getLogger(__name__)


def _setup_logging(verbosity: int = 0) -> None:
    """Nakonfigurovať root logger pre celú pipeline.

    Predvolená úroveň je INFO. Pri dvoch prepínačoch -vv sa zapne
    DEBUG. Externé knižnice s verbóznymi výstupmi sa ponechávajú
    na úrovni WARNING.
    """
    level = logging.DEBUG if verbosity >= 2 else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    """Spracovať argumenty príkazového riadka pomocou argparse.
    """
    parser = argparse.ArgumentParser(
        description="Spustenie pipeline pre prediktívne modelovanie plnosti kontajnerov."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="merged_dataset.csv",
        help="Cesta k vstupnému CSV súboru (predvolene: merged_dataset.csv).",
    )
    parser.add_argument(
        "--weather",
        type=str,
        default="weather.csv",
        help="Cesta k meteorologickému CSV súboru (predvolene: weather.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Výstupný priečinok (predvolene: output/).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["A", "B", "C"],
        default=["A", "B", "C"],
        help="Zoznam experimentov na spustenie (predvolene: všetky).",
    )
    parser.add_argument(
        "--skip-capacity",
        action="store_true",
        help="Preskočiť kapacitne segmentované modelovanie.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Preskočiť referenčné modely (ARIMA, Prophet, survival).",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Preskočiť Optuna ladenie hyperparametrov (použijú sa predvolené hodnoty).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Zvýšiť verbozitu (predvolene INFO; -vv aktivuje DEBUG).",
    )
    return parser.parse_args()


def run_pipeline(
    data_path: str,
    output_dir: str,
    experiments: List[str],
    skip_capacity: bool = False,
    skip_baselines: bool = False,
    skip_tuning: bool = False,
) -> Dict:
    """Spustiť celý výpočtový tok.

    Funkcia načíta dáta, pripraví testovacie kontajnery, spustí vybrané
    experimenty a uloží finálne výstupy. Doplnkové analýzy sú ošetrené tak,
    aby ich zlyhanie nezastavilo hlavný beh.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dočasná modifikácia singletonu CONFIG pre tento beh pipeline.
    if skip_tuning:
        CONFIG.ENABLE_HYPERPARAMETER_TUNING = False

    logger.info("=" * 70)
    logger.info("Spúšťam pipeline predikcie plnosti kontajnerov")
    logger.info("=" * 70)
    logger.info("Dáta: %s", data_path)
    logger.info("Výstupný priečinok: %s", output_dir)
    logger.info("Experimenty: %s", ", ".join(experiments))

    # --- Načítanie a predspracovanie dát -----------------------------
    df = load_and_preprocess_data(data_path)
    test_containers = get_unified_test_containers(df)

    results: Dict = {}

    # --- Prípravné analýzy ------------------------------------------
    sensitivity_analysis_collections(df.copy(), output_dir)
    results["weather_ablation"] = run_weather_ablation_study(
        df.copy(), test_containers, output_dir
    )

    if not skip_baselines:
        results["baselines"] = compare_all_baselines(
            df.copy(), test_containers, output_dir
        )

    results["cv"] = run_cross_validation(df.copy(), test_containers, output_dir)

    # --- Hlavné experimenty -----------------------------------------
    if "A" in experiments:
        results["exp_A"] = run_experiment_A(
            df.copy(), test_containers, output_dir
        )
    if "B" in experiments:
        results["exp_B"] = run_experiment_B(
            df.copy(), test_containers, output_dir
        )
    if "C" in experiments:
        results["exp_C"] = run_experiment_C(
            df.copy(), test_containers, output_dir
        )

    results["stability"] = run_stability_analysis(df.copy(), output_dir)

    # --- Voliteľné doplnkové analýzy --------------------------------
    if not skip_baselines:
        if "B" in experiments:
            try:
                results["weather_ablation_B"] = run_weather_ablation_expB(
                    df.copy(), test_containers, output_dir
                )
            except Exception as exc:
                logger.warning("Ablácia počasia pre Experiment B zlyhala: %s", exc)

            try:
                results["arima_prophet_B"] = run_arima_prophet_baseline_expB(
                    df.copy(), test_containers, output_dir
                )
            except Exception as exc:
                logger.warning("Referenčné modely ARIMA/Prophet pre Experiment B zlyhali: %s", exc)

        if "C" in experiments:
            try:
                results["survival_C"] = run_survival_baseline_expC(
                    df.copy(), test_containers, output_dir
                )
            except Exception as exc:
                logger.warning("Survival referenčný model pre Exp C zlyhal: %s", exc)

    # --- Kapacitne segmentované modely ------------------------------
    if (
        not skip_capacity
        and CONFIG.RUN_CAPACITY_SEGMENTED_MODELS
        and "A" in experiments
    ):
        capacity_results = _run_capacity_segmented(
            df, experiments, output_dir, skip_baselines=skip_baselines
        )
        results["capacity_models"] = capacity_results

    # --- Finálny reporting ------------------------------------------
    generate_final_report(results, output_dir)
    export_results_table(results, output_dir)

    _export_optuna_trials(results, output_dir)
    _export_summary(results, output_dir)

    # --- Doplnkové CSV analýzy po hlavnom behu ----------------------
    try:
        export_hyperparameter_tables(output_dir)
        export_fixed_table16(output_dir)

        logger.info("Generujem vizualizácie použité v diplomovej práci...")
        generate_all_figures(df_raw=df, output_dir=output_dir)
    except Exception as exc:
        logger.warning("Doplnkové analýzy zlyhali: %s", exc)

    logger.info("=" * 70)
    logger.info("Hotovo. Výstupy sú v %s/", output_dir)
    logger.info("=" * 70)

    return results


def _run_capacity_segmented(
    df: pd.DataFrame,
    experiments: List[str],
    output_dir: str,
    skip_baselines: bool = False,
) -> Dict:
    """Spustiť rovnakú pipeline zvlášť pre každý kapacitný segment.

    Pre každý segment ('low', 'high') vyfiltruje podmnožinu
    kontajnerov, zvolí nezávislú testovaciu množinu a rekurzívne
    spustí hlavné experimenty a evaluačné kroky. Segmenty s menej
    ako 10 kontajnermi sú preskočené s varovaním.
    """
    # Lokálny import zamedzí cyklickej závislosti pri viacnásobnom volaní.
    from waste_forecasting.evaluation.cross_validation import run_cross_validation
    from waste_forecasting.evaluation.reporting import generate_final_report

    capacity_results: Dict = {}
    for cap in CONFIG.CAPACITY_SEGMENTS_TO_RUN:
        df_cap = df[df["capacity_class"] == cap].copy()
        n_cap = df_cap["container_id"].nunique()
        if n_cap < 10:
            logger.warning(
                "Kapacitný segment %s preskakujem pre nízky počet kontajnerov (n=%d)",
                cap,
                n_cap,
            )
            continue

        cap_dir = Path(output_dir) / f"capacity_{cap}"
        cap_dir.mkdir(parents=True, exist_ok=True)

        test_cap = get_test_containers_for_capacity(
            df_cap, cap, seed=CONFIG.SEED
        )
        cap_res: Dict = {}
        if not skip_baselines:
            cap_res["baselines"] = compare_all_baselines(
                df_cap.copy(), test_cap, str(cap_dir)
            )
        else:
            cap_res["baselines"] = None
        cap_res["cv"] = run_cross_validation(
            df_cap.copy(), test_cap, str(cap_dir)
        )
        if "A" in experiments:
            cap_res["exp_A"] = run_experiment_A(
                df_cap.copy(), test_cap, str(cap_dir)
            )
        if "B" in experiments:
            cap_res["exp_B"] = run_experiment_B(
                df_cap.copy(), test_cap, str(cap_dir)
            )
        if "C" in experiments:
            cap_res["exp_C"] = run_experiment_C(
                df_cap.copy(), test_cap, str(cap_dir)
            )

        generate_final_report(cap_res, str(cap_dir))
        export_results_table(cap_res, str(cap_dir))
        capacity_results[cap] = cap_res

    save_capacity_comparison_table(capacity_results, output_dir)
    return capacity_results


def _export_optuna_trials(results: Dict, output_dir: str) -> None:
    """Exportovať Optuna trial tabuľky a najlepšie hyperparametre.

    Prejde výsledky všetkých experimentov a pre každý zachytený
    optuna.Study uloží tabuľku so všetkými pokusmi do CSV súboru.
    Agregovaný JSON s najlepšími parametrami je zapísaný do
    optuna_best_params.json. Funkcia nevykonáva žiadnu činnosť,
    ak je Optuna tuning vypnutý.
    """
    if not CONFIG.ENABLE_HYPERPARAMETER_TUNING:
        return

    trials_dir = Path(output_dir) / "optuna_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    all_best_params: Dict = {}
    for exp_name in ["exp_A", "exp_B", "exp_C"]:
        exp_res = results.get(exp_name, {})
        model_results = exp_res.get("all_model_results", {})
        for model_name, model_res in model_results.items():
            bp = model_res.get("best_params", {})
            # Do JSON ukladáme iba jednoducho serializovateľné typy.
            bp_clean = {
                k: v
                for k, v in bp.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            all_best_params[f"{exp_name}_{model_name}"] = bp_clean

            ti = model_res.get("training_info", {})
            study = ti.get("optuna_study")
            if study is not None and hasattr(study, "trials_dataframe"):
                try:
                    trials_df = study.trials_dataframe()
                    trials_df.to_csv(
                        trials_dir / f"{exp_name}_{model_name}_trials.csv",
                        index=False,
                    )
                except Exception as exc:
                    logger.warning(
                        "Nepodarilo sa exportovať Optuna pokusy pre %s/%s: %s",
                        exp_name,
                        model_name,
                        exc,
                    )

    with open(Path(output_dir) / "optuna_best_params.json", "w") as f:
        json.dump(all_best_params, f, indent=2, default=str)


def _export_summary(results: Dict, output_dir: str) -> None:
    """Vytvoriť strojovo čitateľné zhrnutie výsledkov.

    Generuje súbor summary.json, ktorý obsahuje verziu formátu,
    časovú pečiatku behu, použitý typ krížovej validácie a testovacie
    metriky troch hlavných experimentov. Súbor je vhodný pre
    automatizované porovnávanie behov alebo import do dashboardu.
    """
    summary = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "cv_type": CONFIG.CV_TYPE,
        "results": {
            key: results.get(key, {}).get("test_metrics")
            for key in ("exp_A", "exp_B", "exp_C")
        },
    }
    with open(Path(output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main() -> int:
    """Hlavný entry point pri spustení skriptu z príkazového riadka.

    Spracuje argumenty, nakonfiguruje logovanie a zavolá
    run_pipeline. Zachytí chýbajúci vstupný súbor (exit 2)
    a generické výnimky (exit 1). Úspešný beh vracia 0.
    """
    args = _parse_args()
    CONFIG.WEATHER_DATA_PATH = args.weather
    _setup_logging(args.verbose)

    try:
        run_pipeline(
            data_path=args.data,
            output_dir=args.output,
            experiments=args.experiments,
            skip_capacity=args.skip_capacity,
            skip_baselines=args.skip_baselines,
            skip_tuning=args.skip_tuning,
        )
    except FileNotFoundError as exc:
        logger.error("Vstupný súbor sa nenašiel: %s", exc)
        return 2
    except Exception as exc:
        logger.exception("Pipeline zlyhala: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
