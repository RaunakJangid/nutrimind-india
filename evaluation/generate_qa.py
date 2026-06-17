from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

from core.bridge import calculate_gap
from core.food_map import FOOD_MAP
from core.models import QueryEntities

ROOT = Path(__file__).resolve().parents[1]
RDA_PATH = ROOT / "data" / "processed" / "icmr_rda.csv"
IFCT_PATH = ROOT / "data" / "processed" / "ifct2017.csv"
CHUNKS_PATH = ROOT / "data" / "processed" / "icmr_chunks.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "ground_truth_qa.json"

SEED = 42
RDA_SOURCE = "RDA_2020"
IFCT_SOURCE = "IFCT_2017"
DGI_SOURCE = "DGI_2024"

SUPPORTED_AGE_GROUPS = {
    "0-6_months": 6,
    "6-12_months": 9,
    "1-3_years": 24,
    "4-6_years": 60,
}


def normalize_age_group_label(group: str) -> str:
    return group.replace("_", " ")


def age_desc_from_months(age_months: int) -> str:
    if age_months < 12:
        return f"{age_months}-month-old"
    years = age_months // 12
    months = age_months % 12
    if months == 0:
        return f"{years}-year-old"
    return f"{years}-year-{months}-month-old"


def extract_first_sentence(text: str) -> str:
    if not text or not text.strip():
        return ""
    text = text.strip().replace("\n", " ").replace("  ", " ")
    match = re.search(r"(.+?[.!?])(\s|$)", text)
    if match:
        return match.group(1).strip()
    return text[:200].strip()


def build_rda_query(age_group: str, nutrient: str) -> str:
    return f"What is the daily {nutrient.replace('_', ' ')} requirement for the {normalize_age_group_label(age_group)} age group?"


def build_rda_answer(age_group: str, nutrient: str, value: float, unit: str) -> str:
    return f"The daily {nutrient.replace('_', ' ')} requirement for the {normalize_age_group_label(age_group)} age group is {value:g} {unit}."


def build_diet_query(age_months: int, food: str, nutrient: str) -> str:
    return (
        f"Does 1 serving of {food.replace('_', ' ')} meet the {nutrient.replace('_', ' ')} requirement for a {age_desc_from_months(age_months)}?"
    )


def build_diet_answer(result: dict) -> str:
    nutrient = result["nutrient"].replace("_", " ")
    age_desc = age_desc_from_months(result["age_months"])
    first_food = result["food_details"][0] if result["food_details"] else {}
    food = first_food.get("food", "food").replace("_", " ")
    required = result["required_value"]
    consumed = result["consumed_value"]
    gap = result["gap_value"]
    unit = result["required_unit"]
    if gap < 0:
        return (
            f"For a {age_desc}, 1 serving of {food} provides {consumed:.1f} {unit} of {nutrient}, exceeding the requirement of {required:.1f} {unit} by {abs(gap):.1f} {unit}."
        )
    return (
        f"For a {age_desc}, 1 serving of {food} provides {consumed:.1f} {unit} of {nutrient}, leaving a gap of {gap:.1f} {unit} from the daily requirement of {required:.1f} {unit}."
    )


def build_general_query(chunk: dict) -> str:
    topic = chunk.get("title") or chunk.get("section_title") or chunk.get("section") or chunk.get("chunk_id")
    if not topic:
        topic = "DGI 2024 guideline"
    return f"According to DGI 2024 guidance on {topic}, what is the recommendation?"


def build_general_answer(chunk: dict) -> str:
    text = chunk.get("text", "").strip()
    if not text:
        text = chunk.get("title") or chunk.get("section") or "The guideline provides DGI 2024 recommendations."
    return f"DGI 2024 guidance says: {text}" 


def choose_rda_pairs(rda: pd.DataFrame, count: int = 20) -> list[dict]:
    age_groups = set(SUPPORTED_AGE_GROUPS.keys())
    combinations = list(
        rda[rda["age_group"].isin(age_groups)][["age_group", "nutrient", "value", "unit"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    random.shuffle(combinations)
    return combinations[:count]


def choose_diet_pairs(rda: pd.DataFrame, ifct: pd.DataFrame, count: int = 20) -> list[dict]:
    valid_rda_keys = {
        (row["age_group"], row["nutrient"]) for row in rda[rda["age_group"].isin(SUPPORTED_AGE_GROUPS)][["age_group", "nutrient"]].to_dict(orient="records")
    }
    food_choices = set(ifct["food_key"]) & set(FOOD_MAP.keys())
    candidates = []
    for food_key in food_choices:
        food_rows = ifct[ifct["food_key"] == food_key]
        for nutrient in food_rows["nutrient"].unique():
            for age_group, age_months in SUPPORTED_AGE_GROUPS.items():
                if (age_group, nutrient) not in valid_rda_keys:
                    continue
                candidates.append({"food": food_key, "nutrient": nutrient, "age_group": age_group, "age_months": age_months})
    random.shuffle(candidates)
    return candidates[:count]


def choose_general_pairs(chunks: list[dict], count: int = 10) -> list[dict]:
    dgi_chunks = [chunk for chunk in chunks if str(chunk.get("source", "")).upper() == DGI_SOURCE]
    random.shuffle(dgi_chunks)
    return dgi_chunks[:count]


def validate_sources(item: dict) -> None:
    if not item["expected_answer"]:
        raise ValueError(f"Expected answer is empty for item {item['id']}")
    for source in item["expected_sources"]:
        if source not in {RDA_SOURCE, IFCT_SOURCE, DGI_SOURCE}:
            raise ValueError(f"Invalid source {source} in item {item['id']}")


def main() -> None:
    random.seed(SEED)

    rda = pd.read_csv(RDA_PATH)
    ifct = pd.read_csv(IFCT_PATH)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    items: list[dict] = []
    rda_pairs = choose_rda_pairs(rda, 20)
    diet_pairs = choose_diet_pairs(rda, ifct, 20)
    general_pairs = choose_general_pairs(chunks, 10)

    item_id = 1
    for row in rda_pairs:
        query = build_rda_query(row["age_group"], row["nutrient"])
        answer = build_rda_answer(row["age_group"], row["nutrient"], row["value"], row["unit"])
        item = {
            "id": f"qa_{item_id:03d}",
            "query": query,
            "expected_answer": answer,
            "expected_sources": [RDA_SOURCE],
            "nutrient": row["nutrient"],
            "age_months": None,
            "age_group": row["age_group"],
            "foods": [],
            "category": "rda_lookup",
        }
        validate_sources(item)
        items.append(item)
        item_id += 1

    for row in diet_pairs:
        result = calculate_gap(
            QueryEntities(
                nutrient=row["nutrient"],
                age_months=row["age_months"],
                foods=[row["food"]],
                servings={row["food"]: 1.0},
                intent="diet_check",
            )
        )
        answer = build_diet_answer(result.model_dump())
        item = {
            "id": f"qa_{item_id:03d}",
            "query": build_diet_query(row["age_months"], row["food"], row["nutrient"]),
            "expected_answer": answer,
            "expected_sources": [RDA_SOURCE, IFCT_SOURCE],
            "nutrient": row["nutrient"],
            "age_months": row["age_months"],
            "age_group": row["age_group"],
            "foods": [row["food"]],
            "category": "diet_check",
        }
        validate_sources(item)
        items.append(item)
        item_id += 1

    for chunk in general_pairs:
        query = build_general_query(chunk)
        answer = build_general_answer(chunk)
        item = {
            "id": f"qa_{item_id:03d}",
            "query": query,
            "expected_answer": answer,
            "expected_sources": [chunk.get("source", DGI_SOURCE)],
            "nutrient": None,
            "age_months": None,
            "age_group": None,
            "foods": [],
            "category": "general_question",
            "chunk_id": chunk.get("chunk_id"),
            "chunk_title": chunk.get("title") or chunk.get("section_title") or None,
        }
        validate_sources(item)
        items.append(item)
        item_id += 1

    OUTPUT_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(items)} QA items to {OUTPUT_PATH}")
    print("Sample items:")
    for sample in items[:3]:
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
