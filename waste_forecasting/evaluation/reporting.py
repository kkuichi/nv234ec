"""Generovanie reportov a CSV tabuliek.

Modul skladá textový report, súhrnné tabuľky výsledkov, tabuľky
hyperparametrov a výstupy stability pre diplomovú prácu.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import CONFIG

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = logging.getLogger(__name__)

__all__ = ['generate_final_report', 'export_results_table', 'save_capacity_comparison_table', 'export_hyperparameter_tables', 'export_fixed_table16']


def generate_final_report(all_results: Dict, output_dir: str):
    """Zostaví textový report zo všetkých experimentov.

    Prejde slovník all_results a pre každý dostupný experiment
    pripojí blok s metrikami (RMSE, MAE, R2, prípadne 95% CI).
    Kľúče, ktoré v slovníku nie sú alebo sú None, jednoducho
    preskočí. Výstup zapíše do output_dir/FINAL_REPORT.txt a
    zároveň vypíše do konzoly.
    """
    report = ""
    
    if 'exp_A' in all_results and all_results['exp_A'] is not None:
        m = all_results['exp_A']['test_metrics']
        b = all_results['exp_A']['baseline_metrics']
        pname = all_results['exp_A'].get('primary_model_name', 'N/A')
        report += f"""
Experiment A: predikcia plnosti pri vývoze
------------------------------------------
  Najlepší model: {pname}
  Test RMSE: {m['rmse']:.2f} [{m.get('rmse_ci_lower', 0):.2f}, {m.get('rmse_ci_upper', 0):.2f}] (95 % CI)
  Test MAE:  {m['mae']:.2f}
  Test R2:   {m['r2']:.3f}
  
  Referenčný model (posledná hodnota): RMSE={b['rmse']:.2f}, R2={b['r2']:.3f}
"""
    
    if 'baselines' in all_results and all_results['baselines'] is not None:
        report += "\nPorovnanie referenčných modelov\n" + "-" * 50 + "\n"
        for _, row in all_results['baselines'].iterrows():
            r2_val = row['r2'] if not pd.isna(row['r2']) else 0
            report += f"  {row['model']:30s}: RMSE={row['rmse']:.2f}, R2={r2_val:.3f}\n"
    
    if 'exp_B' in all_results and all_results['exp_B'] is not None:
        m = all_results['exp_B']['test_metrics']
        pname = all_results['exp_B'].get('primary_model_name', 'N/A')
        report += f"""
Experiment B: predikcia plnosti 24 hodín dopredu
------------------------------------------------
  Najlepší model: {pname}
  Test RMSE: {m['rmse']:.2f}
  Test MAE:  {m.get('mae', 0):.2f}
  Test R2:   {m['r2']:.3f}
"""
    
    if 'exp_C' in all_results and all_results['exp_C'] is not None:
        m = all_results['exp_C']['test_metrics']
        sb = all_results['exp_C'].get('selection_bias_info', {})
        pname = all_results['exp_C'].get('primary_model_name', 'N/A')
        report += f"""
Experiment C: čas do dosiahnutia 85 % plnosti
---------------------------------------------
  Najlepší model: {pname}
  Test RMSE: {m['rmse']:.2f} dňa
  Test MAE:  {m['mae']:.2f} dňa
  Test R2:   {m['r2']:.3f}
  
  Selekčné skreslenie:
    Pokrytie: {sb.get('n_containers_with_ttt', 0)}/{sb.get('n_containers_total', 0)} ({sb.get('coverage_pct', 0):.1f} %)
"""
    
    if 'stability' in all_results and all_results['stability'] is not None:
        s = all_results['stability']
        report += f"""
Analýza stability
-----------------
  RMSE: {s['rmse'].mean():.2f} +/- {s['rmse'].std():.2f}
  MAE:  {s['mae'].mean():.2f} +/- {s['mae'].std():.2f}
  R2:   {s['r2'].mean():.3f} +/- {s['r2'].std():.3f}
"""
    
    if 'weather_ablation' in all_results and all_results['weather_ablation'] is not None:
        wa = all_results['weather_ablation']
        imp = wa.get('improvement', {})
        report += f"""
Ablácia meteorologických príznakov
----------------------------------
  Bez počasia: RMSE={wa['WITHOUT_WEATHER']['rmse']:.2f}, R2={wa['WITHOUT_WEATHER']['r2']:.3f}
  S počasím:  RMSE={wa['WITH_WEATHER']['rmse']:.2f}, R2={wa['WITH_WEATHER']['r2']:.3f}
  
  Rozdiel:
    Zníženie RMSE: {imp.get('rmse_reduction', 0):.2f} ({imp.get('rmse_reduction_pct', 0):+.1f} %)
    Počet pridaných príznakov: {imp.get('feature_increase', 0)}
"""
    
    if 'cv' in all_results and all_results['cv'] is not None:
        cv = all_results['cv']
        report += f"""
Krížová validácia ({CONFIG.N_FOLDS}-fold, {CONFIG.CV_TYPE})
-----------------------------------------------------------
  RMSE: {cv['rmse'].mean():.2f} +/- {cv['rmse'].std():.2f}
  R2:   {cv['r2'].mean():.3f} +/- {cv['r2'].std():.3f}
"""
    
    
    with open(f'{output_dir}/FINAL_REPORT.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report uložený do {output_dir}/FINAL_REPORT.txt")
    print(report)


def export_results_table(all_results: Dict, output_dir: str) -> pd.DataFrame:
    """Uloží súhrnnú tabuľku výsledkov hlavných experimentov.

    Výstup results_table.csv obsahuje výsledky Experimentu A, B a C,
    ak sú dostupné. Pre každý experiment pridáva hlavný model a jeho
    interný baseline. Ak boli spustené dodatočné baseline modely,
    pridá ich ako baseline pre Experiment A, pretože compare_all_baselines()
    pracuje nad event-based datasetom.

    Tabuľka sa neradí globálne podľa RMSE, pretože Experiment C má inú
    jednotku cieľovej premennej. V experimentoch A a B sú chyby v
    percentuálnych bodoch, zatiaľ čo v Experimente C sú chyby v dňoch.
    Preto tabuľka obsahuje aj stĺpce metric_unit, primary_metric a
    primary_metric_value.
    """
    rows = []

    experiment_meta = {
        "exp_A": {
            "experiment": "Experiment A",
            "task": "Predikcia plnosti pri vývoze",
            "target": "Plnosť pred detegovaným vývozom",
            "metric_unit": "percentuálne body",
            "primary_metric": "RMSE",
            "baseline_name": "Last Value referenčný model",
            "order": 1,
        },
        "exp_B": {
            "experiment": "Experiment B",
            "task": "Predikcia plnosti 24 hodín dopredu",
            "target": "Plnosť o 24 hodín",
            "metric_unit": "percentuálne body",
            "primary_metric": "RMSE",
            "baseline_name": "Seasonal referenčný model",
            "order": 2,
        },
        "exp_C": {
            "experiment": "Experiment C",
            "task": "Čas do dosiahnutia prahu",
            "target": "Počet dní do dosiahnutia 85 % plnosti",
            "metric_unit": "days",
            "primary_metric": "MAE",
            "baseline_name": "Lineárna extrapolácia",
            "order": 3,
        },
    }

    def _metric_value(metrics: Dict, key: str):
        """Bezpečne vytiahne metriku zo slovníka."""
        if not isinstance(metrics, dict):
            return None
        value = metrics.get(key)
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return value

    def _add_row(
        exp_key: str,
        model_name: str,
        model_type: str,
        metrics: Dict,
        notes: str = "",
    ):
        """Pridá jeden riadok do výsledkovej tabuľky."""
        meta = experiment_meta[exp_key]
        primary_metric_key = meta["primary_metric"].lower()

        rows.append({
            "experiment_order": meta["order"],
            "experiment": meta["experiment"],
            "task": meta["task"],
            "target": meta["target"],
            "model_type": model_type,
            "model": model_name,
            "n": _metric_value(metrics, "n"),
            "RMSE": _metric_value(metrics, "rmse"),
            "MAE": _metric_value(metrics, "mae"),
            "WAPE": _metric_value(metrics, "wape"),
            "SMAPE": _metric_value(metrics, "smape"),
            "R2": _metric_value(metrics, "r2"),
            "metric_unit": meta["metric_unit"],
            "primary_metric": meta["primary_metric"],
            "primary_metric_value": _metric_value(metrics, primary_metric_key),
            "notes": notes,
        })

    for exp_key in ["exp_A", "exp_B", "exp_C"]:
        exp_result = all_results.get(exp_key)

        if not exp_result:
            continue

        meta = experiment_meta[exp_key]

        model_name = exp_result.get("primary_model_name", "Model")
        test_metrics = exp_result.get("test_metrics", {})

        if test_metrics:
            _add_row(
                exp_key=exp_key,
                model_name=model_name,
                model_type="main_model",
                metrics=test_metrics,
                notes="Najlepší model vybraný v rámci experimentu",
            )

        baseline_metrics = exp_result.get("baseline_metrics", {})
        if baseline_metrics:
            _add_row(
                exp_key=exp_key,
                model_name=meta["baseline_name"],
                model_type="internal_baseline",
                metrics=baseline_metrics,
                notes="Referenčný model vypočítaný v rámci experimentu",
            )

    baselines = all_results.get("baselines")
    if baselines is not None and isinstance(baselines, pd.DataFrame) and not baselines.empty:
        existing_models = {row["model"] for row in rows}

        for _, baseline_row in baselines.iterrows():
            model_name = baseline_row.get("model", "Baseline")

            if model_name in existing_models:
                continue

            metrics = {
                "n": baseline_row.get("n", None),
                "rmse": baseline_row.get("rmse", None),
                "mae": baseline_row.get("mae", None),
                "wape": baseline_row.get("wape", None),
                "smape": baseline_row.get("smape", None),
                "r2": baseline_row.get("r2", None),
            }

            _add_row(
                exp_key="exp_A",
                model_name=model_name,
                model_type="additional_baseline",
                metrics=metrics,
                notes="Doplnkový referenčný model z compare_all_baselines",
            )

            existing_models.add(model_name)

    if not rows:
        logger.warning("Pre results_table.csv nie sú dostupné žiadne výsledky")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["experiment_order", "model_type", "primary_metric_value"],
        na_position="last"
    ).reset_index(drop=True)

    df_public = df.drop(columns=["experiment_order"])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results_table.csv")
    df_public.to_csv(output_path, index=False)

    print("\n" + "=" * 100)
    print("SÚHRNNÁ TABUĽKA VÝSLEDKOV")
    print("=" * 100)
    print(df_public.to_string(index=False))
    print("=" * 100 + "\n")

    logger.info("Tabuľka výsledkov uložená do %s", output_path)

    return df_public


def save_capacity_comparison_table(capacity_results: Dict[str, Dict[str, Any]], output_dir: str) -> pd.DataFrame:
    """Uloží porovnanie modelov po kapacitných triedach.

    Pri samostatnom trénovaní modelov pre kapacitné segmenty je potrebné
    porovnať výsledky naprieč segmentmi a experimentmi. Funkcia z vnoreného
    slovníka capacity_results vyberie metriky pre každú kombináciu
    (kapacitná trieda, experiment) a uloží ich ako
    capacity_models_comparison.csv.

    Vracia dlhú tabuľku s jedným riadkom na dvojicu
    (kapacitná trieda, experiment).
    """
    rows = []
    for cap, res in capacity_results.items():
        for exp_key in ['exp_A', 'exp_B', 'exp_C']:
            if exp_key not in res or not res[exp_key]:
                continue
            m = res[exp_key].get('test_metrics', {})
            rows.append({
                'capacity_class': cap,
                'experiment': exp_key,
                'n': m.get('n', None),
                'rmse': m.get('rmse', None),
                'mae': m.get('mae', None),
                'wape': m.get('wape', m.get('smape', None)),
                'r2': m.get('r2', None),
            })
    df_out = pd.DataFrame(rows)
    if len(df_out) > 0:
        df_out.to_csv(os.path.join(output_dir, 'capacity_models_comparison.csv'), index=False)
    return df_out


def export_hyperparameter_tables(output_dir: str = 'output'):
    """Exportuje najlepšie hyperparametre z Optuny do CSV.

    Načíta optuna_best_params.json, ktorý sa generuje po Optuna ladení
    v hlavnej pipeline, a rozdelí ho do dvoch formátov:

    - hyperparams_exp_{A,B,C}.csv - dlhý formát, jeden súbor
      na experiment, stĺpce Model, Parameter, Hodnota
    - hyperparams_all_experiments.csv - wide formát, pre každú
      dvojicu (Model, Parameter) osobitný stĺpec pre každý
      experiment, čo umožňuje porovnať zmeny parametrov medzi úlohami

    Parametre sú zoradené tak, aby boli vhodné pre tabuľkové výstupy
    v texte diplomovej práce. Float hodnoty sú formátované
    podľa veľkosti - vedecká notácia pre malé, inak pevný desatinný.

    Vracia wide tabuľku, alebo None ak vstupný JSON nenájde.
    """
    params_path = f'{output_dir}/optuna_best_params.json'
    if not os.path.exists(params_path):
        logger.info(
            "Export hyperparametrov preskočený, súbor %s neexistuje.",
            params_path,
        )
        return None
    
    with open(params_path) as f:
        all_params = json.load(f)
    
    histgbr_params = ['learning_rate', 'max_iter', 'max_leaf_nodes', 'min_samples_leaf',
                      'l2_regularization', 'max_depth']
    lgbm_params = ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves',
                   'min_child_samples', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']
    xgb_params = ['n_estimators', 'learning_rate', 'max_depth', 'min_child_weight',
                  'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']
    
    for exp_label in ['A', 'B', 'C']:
        exp_key = f'exp_{exp_label}'
        rows = []
        for model_name, param_order in [('HistGradientBoosting', histgbr_params),
                                         ('LightGBM', lgbm_params),
                                         ('XGBoost', xgb_params)]:
            key = f'{exp_key}_{model_name}'
            if key not in all_params:
                continue
            params = all_params[key]
            for p in param_order:
                val = params.get(p)
                if val is not None:
                    if isinstance(val, float):
                        val_str = f'{val:.2e}' if abs(val) < 0.001 and val != 0 else (
                            f'{val:.6f}' if abs(val) < 1 else f'{val:.4f}')
                    else:
                        val_str = str(val)
                    rows.append({'Model': model_name, 'Parameter': p, 'Hodnota': val_str})
        
        pd.DataFrame(rows).to_csv(f'{output_dir}/hyperparams_exp_{exp_label}.csv', index=False)
        logger.info(f"Exportované hyperparametre Exp {exp_label}")
    
    # Súhrnná tabuľka v širokom formáte
    summary_rows = []
    for model_name, param_order in [('HistGradientBoosting', histgbr_params),
                                     ('LightGBM', lgbm_params),
                                     ('XGBoost', xgb_params)]:
        for p in param_order:
            row = {'Model': model_name, 'Parameter': p}
            for exp_label in ['A', 'B', 'C']:
                val = all_params.get(f'exp_{exp_label}_{model_name}', {}).get(p)
                if val is not None:
                    if isinstance(val, float):
                        row[f'Exp {exp_label}'] = f'{val:.2e}' if abs(val) < 0.001 and val != 0 else (
                            f'{val:.4f}' if abs(val) < 1 else f'{val:.2f}')
                    else:
                        row[f'Exp {exp_label}'] = str(val)
                else:
                    row[f'Exp {exp_label}'] = '-'
            summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f'{output_dir}/hyperparams_all_experiments.csv', index=False)
    logger.info(f"Súhrnná tabuľka: {output_dir}/hyperparams_all_experiments.csv")
    return df_summary


def export_fixed_table16(output_dir: str = 'output'):
    """Preformátuje multi-seed stability výsledky na per-model tabuľku.

    Pôvodne je stability_results.csv dlhá tabuľka - jeden riadok je
    (seed, model). Pre potreby tabuľkového výstupu sa exportuje
    v dvoch podobách:

    1. pivot - riadky sú seedy, stĺpce kombinácie {metric}_{model},
       uloží sa do table16_per_model_multiseed.csv
    2. súhrn - jeden riadok na model s priemerom a std RMSE/MAE/R2
       cez seedy, plus riadok 'Priemer (všetky modely)', uloží sa
       do table16_summary.csv

    Súhrn sa vypíše aj do konzoly, aby bol výsledok dostupný hneď
    po skončení behu.
    """
    stab_path = f'{output_dir}/stability/stability_results.csv'
    if not os.path.exists(stab_path):
        logger.error(f"Súbor {stab_path} neexistuje.")
        return None
    
    df = pd.read_csv(stab_path)
    
    # Výsledky po modeloch a seedoch
    pivot = df.pivot_table(index='seed', columns='model', values=['rmse', 'mae', 'r2'], aggfunc='first')
    pivot.columns = [f'{metric}_{model}' for metric, model in pivot.columns]
    pivot.reset_index().to_csv(f'{output_dir}/table16_per_model_multiseed.csv', index=False)
    
    # Súhrn
    summary_rows = []
    for model_name in df['model'].unique():
        s = df[df['model'] == model_name]
        summary_rows.append({
            'Model': model_name,
            'RMSE (priemer)': f"{s['rmse'].mean():.2f}", 'RMSE (std)': f"{s['rmse'].std():.2f}",
            'MAE (priemer)': f"{s['mae'].mean():.2f}", 'MAE (std)': f"{s['mae'].std():.2f}",
            'R² (priemer)': f"{s['r2'].mean():.4f}", 'R² (std)': f"{s['r2'].std():.4f}",
            'N seedov': len(s),
        })
    summary_rows.append({
        'Model': 'Priemer (všetky modely)',
        'RMSE (priemer)': f"{df['rmse'].mean():.2f}", 'RMSE (std)': f"{df['rmse'].std():.2f}",
        'MAE (priemer)': f"{df['mae'].mean():.2f}", 'MAE (std)': f"{df['mae'].std():.2f}",
        'R² (priemer)': f"{df['r2'].mean():.4f}", 'R² (std)': f"{df['r2'].std():.4f}",
        'N seedov': len(df),
    })
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f'{output_dir}/table16_summary.csv', index=False)
    
    print("\n" + "=" * 70)
    print("TABUĽKA 16 (po modeloch a seedoch)")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    return df_summary

