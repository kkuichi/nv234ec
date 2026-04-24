# Prediktívne modelovanie v oblasti odpadového hospodárstva

Tento repozitár obsahuje kód k diplomovej práci o predikcii plnosti komunálnych
odpadových kontajnerov v Prahe. Rieši tri úlohy: odhad plnosti pred vývozom,
predikciu plnosti 24 hodín dopredu a odhad počtu dní do dosiahnutia 85 % plnosti.

## Obsah repozitára

| Súbor / priečinok | Účel |
|---|---|
| `main.py` | Vstupný bod pipeline s CLI rozhraním. Spúšťa kroky od načítania dát po vytvorenie reportov. |
| `config.py` | Centrálna konfigurácia. Trieda `Config` ako `@dataclass` drží všetky laditeľné parametre. |
| `requirements.txt` | Zoznam Python závislostí s minimálnymi verziami. |
| `waste_forecasting/data/` | Načítanie CSV, heuristická detekcia vývozov, prevzorkovanie a rozdeľovanie dát. |
| `waste_forecasting/features/` | Šesť skupín príznakov: časové, Fourierove, sviatkové, geolokačné, meteorologické a kategoriálne. |
| `waste_forecasting/models/` | Jednotné rozhranie pre tri modely gradientného boostingu, ladenie cez Optuna a metriky s bootstrap intervalmi. |
| `waste_forecasting/experiments/` | Tri hlavné experimenty A, B, C. |
| `waste_forecasting/evaluation/` | Krížová validácia, stabilita pri viacerých seedoch, SHAP, ablačné štúdie, referenčné modely, reporty a vizualizácie. |
| `tests/` | Jednotkové testy (`unittest`). |
| `datasets/` | Použité datasety skomprimované vo formáte .zip. |

## Dátový model

Projekt pracuje s meraniami zo senzorov plnosti. Hlavný vstupný CSV musí
obsahovať aspoň tieto stĺpce:

| Stĺpec | Typ | Popis |
|---|---|---|
| `container_id` | int | Identifikátor kontajnera |
| `measured_at_utc` | timestamp | Čas merania |
| `percent_calculated` | float | Plnosť v percentách (0 - 100) |
| `trash_type` | str | Typ odpadu (Papír, Plast, Bio, ...) |

Voliteľné stĺpce rozširujú sadu príznakov a pipeline sa bez nich nezastaví:
`container_type`, `district`, `latitude`, `longitude`, `capacity`,
`temperature`, `firealarm`.

Meteorologické dáta sú samostatný CSV súbor. Časový stĺpec môže byť `date` alebo `time`.
Používané meteorologické stĺpce sú najmä `tavg`, `prcp`, `wspd` a `pres`.
Ak súbor chýba, pipeline pokračuje bez meteorologických príznakov.

## Vstupnné súbory

Pipeline očakáva dve vstupné CSV v pracovnom adresári:

- hlavný CSV s meraniami senzorov, predvolene `merged_dataset.csv`,
- meteorologický CSV, predvolene `weather.csv`.

Cesty k obom súborom sa dajú zmeniť cez CLI argumenty alebo cez
parametre triedy `Config`. Cesty a ostatné parametre sú zámerne
centralizované, takže kód jednotlivých modulov nemusí poznať reálnu
adresárovú štruktúru.

## Popis modulov

### Konfigurácia a spustenie

`config.py` obsahuje nastavenia pipeline v triede `Config`: prahy, seedy, počet foldov, podiel testovacích kontajnerov, limity pre Optuna ladenie a parametre tvorby príznakov. Ostatné moduly importujú zdieľanú inštanciu `CONFIG`.

`main.py` je vstupný bod z príkazového riadka. Spracuje CLI argumenty, nastaví logovanie, spustí vybrané experimenty a uloží výstupy do zadaného priečinka. Výpočtovo drahšie časti sa dajú vypnúť cez prepínače ako `--skip-tuning`, `--skip-baselines` alebo `--skip-capacity`.

### Predspracovanie dát

Moduly v `waste_forecasting/data/` načítajú vstupné CSV, odstránia neplatné záznamy, doplnia odvodené stĺpce a pripravia rozdelenie dát.

`loading.py` kontroluje povinné stĺpce, filtruje plnosti mimo intervalu 0 až 100, odvodzuje `capacity_class` a `district_num` a znižuje pamäťovú stopu numerických stĺpcov.

`preprocessing.py` deteguje vývozy podľa prudkého poklesu plnosti, prevzorkuje nepravidelnú časovú sériu na 6-hodinový krok a obsahuje kontroly proti úniku informácií z budúcnosti.

`splitting.py` pripravuje oddelenie testovacích kontajnerov, časový rez, päťfoldovú validáciu a výber podľa kapacitných segmentov. Testovacie kontajnery sa pri tomto delení nepoužijú v tréningu.

### Príznaky

Adresár `waste_forecasting/features/` pridáva príznaky používané v experimentoch A, B a C.

`temporal.py` vytvára lagy, kĺzavé štatistiky, intervaly medzi vývozmi a sezónne lagy. Pripravuje dáta pre všetky tri úlohy. Kĺzavé štatistiky používajú `shift(1)`, aby sa cieľová hodnota nedostala medzi vstupy.

`fourier.py` pridáva Fourierove harmoniky pre dennú a týždennú sezónnosť. Dvojice `sin` a `cos` zachovávajú cyklickú povahu času.

`holiday.py` počíta české štátne sviatky, Veľkú noc a odvodené stĺpce `is_holiday`, `days_to_holiday` a `is_near_holiday`.

`spatial.py` dopĺňa geolokačné príznaky, vzdialenosť od centra Prahy a voliteľný KMeans klaster nad kontajnermi.

`weather.py` pripája meteorologické dáta cez `merge_asof` bez použitia budúcich meraní. Pridáva príznaky z teploty, zrážok, vetra a tlaku, plus interakcie s víkendom, sviatkami a typom odpadu.

`encoding.py` pripravuje kategoriálne premenné pre gradient boosting modely. Malé kardinality kóduje cez one-hot, veľké redukuje na najčastejšie hodnoty a zvyšok spája do `_other_`.

### Modely a metriky

`waste_forecasting/models/training.py` obsahuje tréningové funkcie pre `HistGradientBoostingRegressor`, `LightGBM` a `XGBoost`. Všetky majú rovnaké rozhranie a predikcie orezávajú do intervalu 0 až 100.

`tuning.py` ladí hyperparametre cez Optuna. Používa deterministický `TPESampler(seed=CONFIG.SEED)` a cieľovú funkciu založenú na priemernom RMSE naprieč temporálnymi foldmi.

`metrics.py` počíta RMSE, MAE, WAPE, SMAPE, R², bootstrap intervaly spoľahlivosti a párové testy na porovnanie modelov.

### Experimenty

`experiment_a.py` odhaduje plnosť kontajnera tesne pred vývozom. Z detegovaných vývozov vytvorí dataset, pridá príznaky, natrénuje modely a pre víťazný model pripraví interpretáciu.

`experiment_b.py` rieši predikciu plnosti 24 hodín dopredu na pravidelne prevzorkovanej sérii. Používa 6-hodinový krok.

`experiment_c.py` odhaduje počet dní do dosiahnutia 85 % plnosti. Víťazný model sa vyberá podľa MAE v dňoch. Modul zároveň upozorňuje na selekčné skreslenie, pretože do datasetu vstupujú len kontajnery, ktoré aspoň raz dosiahli prah.

### Evaluácia a výstupy

`cross_validation.py` spúšťa päťfoldovú validáciu nad úlohou predikcie pri vývoze a kontroluje stabilitu metrík medzi foldmi.

`stability.py` opakuje predikciu pri vývoze s viacerými seedmi a meria, ako veľmi sa menia výsledky pri inom rozdelení dát.

`interpretability.py` počíta SHAP hodnoty pre víťazný model a exportuje najdôležitejšie príznaky do CSV aj obrázkov.

`baselines.py` obsahuje referenčné metódy Last-Value, Seasonal Naive, ARIMA a Prophet.

`survival_baseline.py` pridáva Kaplan-Meier a Cox Proportional Hazards modely pre Experiment C.

`reporting.py` zostavuje `FINAL_REPORT.txt`, CSV tabuľky s porovnaním modelov, Optuna hyperparametre a tabuľky stability.

`visualization.py` generuje obrázky pre diplomovú prácu a prílohy. Pri chýbajúcich vstupných CSV preskočí len príslušný graf.

`weather_ablation.py` porovnáva dva varianty pipeline - s meteorologickými príznakmi a bez nich

## Experimenty

Pipeline rieši tri nezávislé, ale metodicky prepojené úlohy. Každý
experiment má vlastný modul v `waste_forecasting/experiments/`.

- Experiment A predikuje plnosť kontajnera tesne pred detegovaným vývozom.
- Experiment B predikuje plnosť 24 hodín dopredu na pravidelne prevzorkovanej časovej sérii.
- Experiment C odhaduje počet dní do dosiahnutia 85 % plnosti.

Každý experiment porovnáva tri modely: `HistGradientBoostingRegressor`,
`LightGBM` a `XGBoost`. Víťaz sa určuje podľa RMSE (A, B) alebo MAE (C)
na testovacej množine.

## Ako na seba moduly nadväzujú

Bežný beh pipeline vyzerá takto:

1. `load_and_preprocess_data` načíta a vyčistí vstupný CSV.
2. `get_unified_test_containers` vyberie 20 % kontajnerov ako testovaciu množinu.
3. `sensitivity_analysis_collections` a `weather_ablation` spustia predbežné analýzy citlivosti.
4. `run_cross_validation` vytvorí kontrolné validačné výsledky. Pri plnom behu sa spúšťajú aj referenčné modely.
5. Experimenty A, B, C sa spustia postupne, každý s vlastnou tvorbou príznakov a Optuna ladením.
6. `run_stability_analysis` kontroluje variabilitu výsledkov naprieč seedmi.
7. Pri plnom behu sa doplnia referenčné modely ARIMA/Prophet pre Experiment B a model prežitia pre Experiment C.
8. Kapacitne segmentované modely opakujú hlavné experimenty pre segmenty `low` a `high`.
9. `generate_final_report` a `generate_all_figures` pripravia reporty, tabuľky a obrázky.

Referenčné modely sa dajú vypnúť cez `--skip-baselines` a kapacitne segmentované modely cez `--skip-capacity`.

## Spustenie prostredia

Kód bol vyvíjaný v Pythone 3.11.14. Odporúčaná verzia je Python 3.11. Pre reprodukovateľnú prácu je
vhodné použiť virtuálne prostredie so závislosťami z `requirements.txt`.

Postup:

```bash
conda create -n environment python=3.11
conda activate environment
pip install -r requirements.txt
```

## Spustenie pipeline

Všetky behy idú cez `main.py` z koreňa repozitára.

```bash
# Plný beh
python main.py --data merged_dataset.csv --weather weather.csv --output results/

# Rýchlejší overovací beh s Experimentom B
python main.py --experiments B --data merged_dataset.csv --weather weather.csv --output output/ --skip-tuning --skip-baselines --skip-capacity

# Plný zoznam argumentov
python main.py --help
```

CLI argumenty:

| Argument | Predvolené | Popis |
|---|---|---|
| `--data` | `merged_dataset.csv` | Vstupný CSV |
| `--weather` | `weather.csv` | Dataset s počasím |
| `--output` | `output/` | Výstupný priečinok |
| `--experiments` | `A B C` | Ktoré experimenty spustiť |
| `--skip-tuning` | vypnuté | Preskočiť Optuna ladenie |
| `--skip-baselines` | vypnuté | Preskočiť referenčné modely |
| `--skip-capacity` | vypnuté | Preskočiť kapacitné segmenty |
| `-v` | vypnuté | Zapnúť podrobnejšie logovanie na úrovni INFO |
| `-vv` | vypnuté | Zapnúť detailné logovanie na úrovni DEBUG |

## Poznámky k používaniu

- Pipeline predpokladá spúšťanie z koreňa repozitára, aby importy typu `from config import CONFIG` fungovali bez úprav `PYTHONPATH`.
- Optuna ladenie je najdlhší krok. Na rýchle overenie celého toku spusti beh s `--skip-tuning --skip-baselines --skip-capacity`.
- Natrénované modely sa do repozitára neukladajú. Pri zachovaní `SEED = 42` a verzií z `requirements.txt` sú výsledky reprodukovateľné opätovným spustením.
- Ak meteorologický CSV chýba, beh to zapíše do logu a pokračuje bez meteorologických príznakov.
