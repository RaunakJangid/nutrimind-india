"""Phase 0 preprocessing for NutriMind RAG v4.

Inputs:
- data/raw/ifct2017_compositions.csv from nodef/ifct2017
- optional local IFCT2017.pdf reference for provenance
- data/processed/icmr_rda.csv, until a real raw RDA file is supplied

Outputs:
- data/processed/ifct2017.csv in app lookup format
- data/processed/ifct2017_foods.csv for audit/search
- data/processed/ifct2017_source.json for provenance
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
RDA_PATH = PROCESSED_DIR / "icmr_rda.csv"
IFCT_PATH = PROCESSED_DIR / "ifct2017.csv"
IFCT_FOODS_PATH = PROCESSED_DIR / "ifct2017_foods.csv"
IFCT_SOURCE_PATH = PROCESSED_DIR / "ifct2017_source.json"
RAW_IFCT_COMPOSITIONS = RAW_DIR / "ifct2017_compositions.csv"
DEFAULT_IFCT_PDF_PATH = Path(os.getenv("IFCT_PDF_PATH", r"C:\Users\rauna\Desktop\Baby RAG\IFCT2017.pdf"))

NUTRIENT_COLUMNS = {
    "protein": {"column": "protcnt", "unit": "g", "factor": 1.0},
    "iron": {"column": "fe", "unit": "mg", "factor": 1000.0},
    "calcium": {"column": "ca", "unit": "mg", "factor": 1000.0},
    "vitamin_c": {"column": "vitc", "unit": "mg", "factor": 1000.0},
    "zinc": {"column": "zn", "unit": "mg", "factor": 1000.0},
    "folate": {"column": "folsum", "unit": "ug", "factor": 1_000_000.0},
    "vitamin_a": {"column": "vita", "unit": "ug", "factor": 1_000_000.0},
    "vitamin_d": {"column": "vitd", "unit": "ug", "factor": 1_000_000.0},
}

APP_FOOD_TO_IFCT_NAME = {
    "lentils": "Lentil, dal",
    "lentils_moong": "Green gram, dal",
    "lentils_toor": "Red gram, dal",
    "rice_polished": "Rice, raw, milled",
    "rice_brown": "Rice, raw, brown",
    "wheat_flour": "Wheat flour, atta",
    "milk_cow": "Milk, whole, Cow",
    "curd": "Milk, whole, Cow",
    "egg_whole": "Egg, poultry, whole, boiled",
    "apple_fruit": "Apple, big",
    "banana_ripe": "Banana, ripe, robusta",
    "spinach_leaves": "Spinach",
    "carrot_root": "Carrot, orange",
    "potato_tuber": "Potato, brown skin, big",
    "tomato_ripe": "Tomato, ripe, local",
    "onion_bulb": "Onion, big",
    "chicken_broiler": "Chicken, poultry, breast, skinless",
    "fish_rohu": "Rohu",
    "paneer": "Paneer",
    "ghee": "Ghee",
    "oil_groundnut": "Groundnut oil",
    "jaggery": "Jaggery, cane",
    "bengal_gram": "Bengal gram, dal",
    "rajma": "Rajmah, red",
    "soybean": "Soya bean, brown",
    "coconut": "Coconut, kernel, fresh",
    "brinjal": "Brinjal 1",
    "cabbage": "Cabbage, green",
    "capsicum": "Capsicum, green",
    "green_peas": "Peas, fresh",
    "french_beans": "French beans, country",
    "radish": "Radish, elongate, white skin",
    "sweet_potato": "Sweet potato, brown skin",
    "bajra": "Bajra",
    "jowar": "Jowar",
    "rice_flakes": "Rice, flakes",
    "semolina": "Wheat, semolina",
    "refined_flour": "Wheat flour, refined",
    "whole_wheat_flour": "Wheat flour, atta",
    # Ingredient aliases used by FOOD_MAP targets
    "amaranth_leaves": "Amaranth leaves, green",
    "butter": "Ghee",
    "gram_flour": "Bengal gram, dal",
    "groundnut": "Ground nut",
    "maize": "Maize, dry",
    "murmura": "Rice, puffed",
    "sago": "Tapioca",
    "sugar": "Jaggery, cane",
    "tamarind": "Tamarind, pulp",
    "turnip": "Radish, elongate, white skin",
    "vermicelli": "Wheat, vermicelli",
    "yam": "Yam, ordinary",
    "multigrain_flour": "Wheat flour, atta",
    "oats": "Wheat, whole",
    "muesli": "Wheat, whole",
    "corn_flakes": "Maize, dry",
    "wheat_flakes": "Wheat, whole",
    "puffed_wheat": "Wheat, bulgur",
    # Prepared-food proxies (primary ingredient IFCT backing)
    "idli": "Rice, raw, milled",
    "dosa": "Rice, raw, milled",
    "upma": "Wheat, semolina",
    "sambar": "Red gram, dal",
    "rasam": "Red gram, dal",
    "vegetable_curry": "Cabbage, green",
    "coconut_chutney": "Coconut, kernel, fresh",
    "mango_pickles": "Mango, ripe, banganapalli",
    "papad": "Bengal gram, dal",
    "sooji_halwa": "Wheat, semolina",
    "rice_kheer": "Rice, raw, milled",
    "besan_laddu": "Bengal gram, dal",
    "milk_barfi": "Milk, whole, Cow",
    "puri": "Wheat flour, refined",
    "bhatura": "Wheat flour, refined",
    "naan": "Wheat flour, atta",
    "kulcha": "Wheat flour, atta",
    "paratha": "Wheat flour, atta",
    "thepla": "Wheat flour, atta",
    "dhokla": "Bengal gram, dal",
    "khandvi": "Bengal gram, dal",
    "handvo": "Bengal gram, whole",
    "mutter_paneer": "Peas, fresh",
    "palak_paneer": "Spinach",
    "dal_makhani": "Lentil, dal",
    "kadhi": "Bengal gram, dal",
    "samosa": "Wheat flour, refined",
    "pakora": "Bengal gram, dal",
    "kachori": "Wheat flour, refined",
    "jalebi": "Wheat flour, refined",
    "gulab_jamun": "Milk, whole, Cow",
    "rasgulla": "Milk, whole, Cow",
    "sandesh": "Paneer",
    "mishti_doi": "Milk, whole, Cow",
    "payasam": "Rice, raw, milled",
    "kesari": "Wheat, semolina",
    "mysore_pak": "Bengal gram, dal",
    "obattu": "Wheat flour, atta",
    "holige": "Wheat flour, atta",
    "pongal": "Rice, raw, milled",
    "bisi_bele_bath": "Rice, raw, milled",
    "puliyogare": "Rice, raw, milled",
    "lemon_rice": "Rice, raw, milled",
    "tamarind_rice": "Rice, raw, milled",
    "tomato_rice": "Rice, raw, milled",
    "coconut_rice": "Rice, raw, milled",
    "curd_rice": "Rice, raw, milled",
    "vangi_bath": "Brinjal 1",
    "pulao": "Rice, raw, milled",
    "biryani": "Rice, raw, milled",
    "fried_rice": "Rice, raw, milled",
    "jeera_rice": "Rice, raw, milled",
    "ghee_rice": "Rice, raw, milled",
    "sambar_rice": "Rice, raw, milled",
    "rasam_rice": "Rice, raw, milled",
    "khichdi": "Rice, raw, milled",
    "phirni": "Rice, raw, milled",
    "seviyan": "Wheat, vermicelli",
    "sheer_korma": "Milk, whole, Cow",
    "double_ka_meetha": "Wheat flour, refined",
    "qubani_ka_meetha": "Jaggery, cane",
    "shahi_tukda": "Wheat flour, refined",
    "malpua": "Wheat flour, refined",
    "imarti": "Bengal gram, dal",
    "peda": "Milk, whole, Cow",
    "modak": "Rice, raw, milled",
    "karanji": "Wheat flour, refined",
    "chakli": "Bengal gram, dal",
    "shankarpali": "Wheat flour, refined",
    "chivda": "Rice, flakes",
    "laiyya": "Rice, puffed",
}


def normalize_food_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", key).strip("_")


def validate_processed_files() -> None:
    rda = pd.read_csv(RDA_PATH)
    ifct = pd.read_csv(IFCT_PATH)

    required_rda = {"age_group", "nutrient", "value", "unit", "source"}
    required_ifct = {"food_key", "nutrient", "value_per_100g", "unit", "source"}

    missing_rda = required_rda - set(rda.columns)
    missing_ifct = required_ifct - set(ifct.columns)
    if missing_rda:
        raise ValueError(f"RDA CSV missing columns: {sorted(missing_rda)}")
    if missing_ifct:
        raise ValueError(f"IFCT CSV missing columns: {sorted(missing_ifct)}")
    if rda[list(required_rda)].isna().any().any():
        raise ValueError("RDA CSV contains missing required values")
    if ifct[list(required_ifct)].isna().any().any():
        raise ValueError("IFCT CSV contains missing required values")


def build_ifct_processed() -> None:
    if not RAW_IFCT_COMPOSITIONS.exists():
        print("Raw IFCT composition CSV not found; validating existing processed IFCT only.")
        return

    raw = pd.read_csv(RAW_IFCT_COMPOSITIONS)
    if raw.shape[0] < 500:
        raise ValueError(f"IFCT raw file looks incomplete: expected 500+ foods, got {raw.shape[0]}")

    food_audit_rows = raw[["code", "name", "scie", "grup", "regn", "tags"]].copy()
    food_audit_rows["food_key"] = food_audit_rows["name"].map(normalize_food_key)
    food_audit_rows.to_csv(IFCT_FOODS_PATH, index=False)

    rows: list[dict[str, object]] = []
    source = "IFCT 2017 via nodef/ifct2017 compositions/index.csv"

    def add_food_rows(food_key: str, row: pd.Series) -> None:
        for nutrient, spec in NUTRIENT_COLUMNS.items():
            column = spec["column"]
            if column not in row or pd.isna(row[column]):
                continue
            rows.append(
                {
                    "food_key": food_key,
                    "nutrient": nutrient,
                    "value_per_100g": round(float(row[column]) * float(spec["factor"]), 6),
                    "unit": spec["unit"],
                    "source": source,
                    "ifct_code": row["code"],
                    "ifct_name": row["name"],
                }
            )

    for _, row in raw.iterrows():
        add_food_rows(normalize_food_key(str(row["name"])), row)

    for app_key, ifct_name in APP_FOOD_TO_IFCT_NAME.items():
        match = raw[raw["name"].str.lower() == ifct_name.lower()]
        if match.empty:
            raise ValueError(f"Missing IFCT mapping for {app_key}: {ifct_name}")
        add_food_rows(app_key, match.iloc[0])

    processed = pd.DataFrame(rows)
    processed.to_csv(IFCT_PATH, index=False)
    write_ifct_source_metadata(raw)


def write_ifct_source_metadata(raw: pd.DataFrame) -> None:
    pdf_info = {"path": str(DEFAULT_IFCT_PDF_PATH), "exists": DEFAULT_IFCT_PDF_PATH.exists(), "pages": None}
    if pdf_info["exists"]:
        try:
            from pypdf import PdfReader

            pdf_info["pages"] = len(PdfReader(str(DEFAULT_IFCT_PDF_PATH)).pages)
        except Exception as exc:
            pdf_info["read_warning"] = str(exc)

    metadata = {
        "structured_source": "https://github.com/nodef/ifct2017/blob/main/compositions/index.csv",
        "reference_pdf": pdf_info,
        "raw_food_rows": int(raw.shape[0]),
        "raw_columns": int(raw.shape[1]),
        "nutrients_exported": sorted(NUTRIENT_COLUMNS),
        "app_food_aliases_exported": sorted(APP_FOOD_TO_IFCT_NAME),
    }
    IFCT_SOURCE_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build_ifct_processed()
    validate_processed_files()
    print("Processed nutrition CSVs validated.")
