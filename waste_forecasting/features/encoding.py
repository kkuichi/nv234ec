"""Kódovanie kategoriálnych príznakov a sanitizácia názvov stĺpcov.

Modul poskytuje dve komponenty:

* SmartCategoricalEncoder - adaptívny kódovač, ktorý pre
  stĺpce s nízkou kardinalitou aplikuje úplné one-hot kódovanie
  a pre stĺpce s vysokou kardinalitou použije top-k + „other"
  stratégiu, ktorá zabraňuje explózii dimenzie a zachováva
  interpretovateľnosť.
* sanitize_feature_names - funkcia na očistu názvov
  stĺpcov od znakov, ktoré môžu spôsobovať problémy pri LightGBM
  a XGBoost modeloch (medzery, diakritika, špeciálne znaky).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from config import CONFIG

logger = logging.getLogger(__name__)

__all__ = ['SmartCategoricalEncoder', 'sanitize_feature_names']


class SmartCategoricalEncoder:
    """Adaptívny kódovač kategoriálnych premenných so spracovaním kardinality.

    Kódovač aplikuje rozdielnu stratégiu podľa počtu unikátnych hodnôt
    v stĺpci. Pri nízkej kardinalite (≤
    CONFIG.LOW_CARDINALITY_THRESHOLD, predvolene 10)
    vykoná klasické one-hot kódovanie. Pri vysokej kardinalite
    vyberie top-CONFIG.TOP_K_CATEGORIES (predvolene 8)
    najčastejších kategórií a ostatné nahradí špeciálnou hodnotou
    '_other_'.

    Po zavolaní fit_transform si kódovač zapamätá zvolenú
    stratégiu a top kategórie pre každý stĺpec, takže nasledujúce
    volania transform na testovej množine produkujú
    konzistentné stĺpce.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        low_cardinality_threshold: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        """Inicializácia kódovača.
        """
        self.categorical_columns = categorical_columns or [
            'trash_type', 'capacity_class', 'district', 'container_type', 'geo_cluster'
        ]
        self.low_cardinality_threshold = low_cardinality_threshold or CONFIG.LOW_CARDINALITY_THRESHOLD
        self.top_k = top_k or CONFIG.TOP_K_CATEGORIES

        self.encoders: Dict = {}
        self.top_categories: Dict = {}
        self.feature_names: Dict = {}
        self.cardinality_info: Dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Naučiť kódovač na trénovacej množine a transformovať dáta.
        """
        df = df.copy()

        for col in self.categorical_columns:
            if col not in df.columns:
                continue

            # Prevod na string a explicitné označenie chýbajúcich hodnôt.
            df[col] = df[col].fillna('_unknown_').astype(str)
            n_unique = df[col].nunique()

            self.cardinality_info[col] = {
                'n_unique': n_unique,
                'strategy': 'full_ohe' if n_unique <= self.low_cardinality_threshold else 'top_k_ohe'
            }

            if n_unique <= self.low_cardinality_threshold:
                # Nízka kardinalita: klasické one-hot kódovanie.
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
                encoded = encoder.fit_transform(df[[col]])
                feature_names = [f'{col}_{cat}' for cat in encoder.categories_[0]]

                self.encoders[col] = encoder
                self.top_categories[col] = None

            else:
                # Vysoká kardinalita: zachovanie iba top-k kategórií,
                # ostatné sú zlúčené do hodnoty '_other_'.
                value_counts = df[col].value_counts()
                top_cats = value_counts.head(self.top_k).index.tolist()
                self.top_categories[col] = set(top_cats)

                df[f'{col}_mapped'] = df[col].apply(
                    lambda x: x if x in self.top_categories[col] else '_other_'
                )

                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
                encoded = encoder.fit_transform(df[[f'{col}_mapped']])
                feature_names = [f'{col}_{cat}' for cat in encoder.categories_[0]]

                self.encoders[col] = encoder
                df = df.drop(columns=[f'{col}_mapped'])

                logger.info(f"  {col}: {n_unique} categories -> top-{self.top_k} + other")

            self.feature_names[col] = feature_names

            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index).astype(np.int8)
            df = pd.concat([df, encoded_df], axis=1)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplikovať naučený kódovač na nové dáta.

        Používa uložené top kategórie a fitnutý OneHotEncoder, takže
        trénovacia a testovacia množina majú identické sady stĺpcov
        (aj keď v testovej množine chýbajú niektoré kategórie).
        """
        df = df.copy()

        for col in self.categorical_columns:
            if col not in df.columns or col not in self.encoders:
                continue

            df[col] = df[col].fillna('_unknown_').astype(str)

            if self.top_categories[col] is not None:
                # Nové neznáme kategórie sú mapované na '_other_'.
                df[f'{col}_mapped'] = df[col].apply(
                    lambda x: x if x in self.top_categories[col] else '_other_'
                )
                encoded = self.encoders[col].transform(df[[f'{col}_mapped']])
                df = df.drop(columns=[f'{col}_mapped'])
            else:
                encoded = self.encoders[col].transform(df[[col]])

            encoded_df = pd.DataFrame(
                encoded, columns=self.feature_names[col], index=df.index
            ).astype(np.int8)
            df = pd.concat([df, encoded_df], axis=1)

        return df

    def get_all_feature_names(self) -> List[str]:
        """Vrátiť spoločný zoznam všetkých vytvorených stĺpcov.
        """
        all_names = []
        for names in self.feature_names.values():
            all_names.extend(names)
        return all_names

    def get_cardinality_report(self) -> str:
        """Vrátiť textový report o kardinalite a zvolenej stratégii.
        """
        lines = ["Categorical Encoding Strategy:"]
        for col, info in self.cardinality_info.items():
            lines.append(f"  {col}: {info['n_unique']} categories -> {info['strategy']}")
        return "\n".join(lines)


def sanitize_feature_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Očistiť názvy stĺpcov od znakov problematických pre LightGBM a XGBoost.

    LightGBM a XGBoost natívne nepodporujú v názvoch stĺpcov medzery,
    špeciálne znaky ani niektoré diakritické znaky - pri tréningu
    alebo predikcii môžu vyvolať výnimku alebo tichú chybu pri
    párovaní názvov. Funkcia nahradí všetky znaky, ktoré nie sú
    alfanumerické ani podčiarkovník, znakom _. Pri vzniku
    kolízie dopĺňa hash na zabezpečenie unikátnosti.
    """
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        clean = re.sub(r'[^A-Za-z0-9_]', '_', str(col))
        # Ak by rôzne stĺpce po očistení mali rovnaký názov,
        # pridáme hash pôvodného názvu, aby sme zachovali unikátnosť.
        if clean in rename_map.values():
            clean = clean + '_' + str(hash(col) % 10000)
        rename_map[col] = clean

    df_clean = df.rename(columns=rename_map)
    reverse_map = {v: k for k, v in rename_map.items()}
    return df_clean, reverse_map
