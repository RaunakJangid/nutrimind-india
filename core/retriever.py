from __future__ import annotations

import os
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - exercised only in lean local environments
    fuzz = None
    process = None
    from difflib import SequenceMatcher

from core.food_map import FOOD_MAP
from core.models import QueryEntities, RetrievalResult, SemanticChunk


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "processed"))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", ROOT / "data" / "indices" / "faiss_icmr.index"))
CHUNKS_PATH = DATA_DIR / "icmr_chunks.json"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

AGE_GROUP_MAP = {
    (0, 6): "0-6_months",
    (7, 12): "6-12_months",
    (13, 36): "1-3_years",
    (37, 60): "4-6_years",
}


def normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


@lru_cache(maxsize=1)
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    rda_path = DATA_DIR / "icmr_rda.csv"
    ifct_path = DATA_DIR / "ifct2017.csv"
    rda = pd.read_csv(rda_path)
    ifct = pd.read_csv(ifct_path)
    rda["nutrient"] = rda["nutrient"].map(normalize_key)
    ifct["nutrient"] = ifct["nutrient"].map(normalize_key)
    ifct["food_key"] = ifct["food_key"].map(normalize_key)
    return rda, ifct


def map_age_to_group(age_months: int) -> str:
    for (start, end), group in AGE_GROUP_MAP.items():
        if start <= age_months <= end:
            return group
    return "unsupported_age"


def map_food(food_name: str) -> str:
    key = normalize_key(food_name)
    if key in FOOD_MAP:
        return FOOD_MAP[key]

    choices = list(FOOD_MAP)
    if process is not None and fuzz is not None:
        match = process.extractOne(key, choices, scorer=fuzz.WRatio)
        if match and match[1] >= 80:
            return FOOD_MAP[match[0]]
    else:
        scored = [(choice, SequenceMatcher(None, key, choice).ratio() * 100) for choice in choices]
        match = max(scored, key=lambda item: item[1])
        if match[1] >= 75:
            return FOOD_MAP[match[0]]
    return "unknown_food"


def get_rda(nutrient: str, age_group: str) -> dict[str, Any]:
    rda, _ = load_datasets()
    nutrient_key = normalize_key(nutrient)
    row = rda[(rda["nutrient"] == nutrient_key) & (rda["age_group"] == age_group)]
    if row.empty:
        raise LookupError(f"RDA not found for {nutrient_key} and {age_group}")
    item = row.iloc[0]
    return {
        "value": float(item["value"]),
        "unit": str(item["unit"]),
        "source": str(item["source"]),
        "confidence": 1.0,
        "age_group": age_group,
        "nutrient": nutrient_key,
    }


def get_food_nutrients(food: str, nutrient: str) -> dict[str, Any]:
    _, ifct = load_datasets()
    food_key = normalize_key(food)
    nutrient_key = normalize_key(nutrient)
    row = ifct[(ifct["food_key"] == food_key) & (ifct["nutrient"] == nutrient_key)]
    if row.empty:
        raise LookupError(f"{nutrient_key} not found for {food_key}")
    item = row.iloc[0]
    return {
        "value_per_100g": float(item["value_per_100g"]),
        "unit": str(item["unit"]),
        "source": str(item["source"]),
        "confidence": 1.0,
        "food_key": food_key,
        "nutrient": nutrient_key,
    }


@lru_cache(maxsize=1)
def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        return []
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _load_faiss_assets() -> tuple[Any, Any] | None:
    if not FAISS_INDEX_PATH.exists():
        return None
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        index = faiss.read_index(str(FAISS_INDEX_PATH))
        model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        return index, model
    except Exception:
        return None


def _lexical_score(query: str, text: str) -> float:
    query_terms = {normalize_key(term) for term in re.findall(r"[A-Za-z_]+", query) if len(term) > 2}
    text_terms = {normalize_key(term) for term in re.findall(r"[A-Za-z_]+", text)}
    if not query_terms:
        return 0.0
    overlap = len(query_terms & text_terms) / len(query_terms)
    fuzzy_score = 0.0
    if process is not None and fuzz is not None:
        fuzzy_score = fuzz.partial_ratio(query.lower(), text.lower()) / 100
    else:
        fuzzy_score = SequenceMatcher(None, query.lower(), text.lower()).ratio()
    return max(overlap, fuzzy_score)


def semantic_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    chunks = load_chunks()
    if not chunks:
        return []

    faiss_assets = _load_faiss_assets()
    if faiss_assets is not None:
        index, model = faiss_assets
        try:
            embedding = model.encode([query], normalize_embeddings=True)
            scores, indexes = index.search(embedding, min(top_k, len(chunks)))
            results = []
            for score, idx in zip(scores[0], indexes[0]):
                if idx < 0 or idx >= len(chunks):
                    continue
                item = dict(chunks[idx])
                item["score"] = float(score)
                results.append(item)
            return results
        except Exception:
            pass

    scored = []
    for chunk in chunks:
        item = dict(chunk)
        item["score"] = _lexical_score(query, item.get("text", ""))
        scored.append(item)
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def retrieve_all(query_entities: QueryEntities, query_text: str = "") -> RetrievalResult:
    age_group = map_age_to_group(query_entities.age_months) if query_entities.age_months is not None else None
    result = RetrievalResult(age_group=age_group)

    if query_entities.nutrient and age_group and age_group != "unsupported_age":
        try:
            result.rda = get_rda(query_entities.nutrient, age_group)
        except LookupError as exc:
            result.errors.append(str(exc))

    for food in query_entities.foods:
        mapped = map_food(food)
        if mapped == "unknown_food":
            result.unknown_foods.append(food)
            continue
        if not query_entities.nutrient:
            continue
        try:
            nutrient = get_food_nutrients(mapped, query_entities.nutrient)
            nutrient["food"] = food
            nutrient["mapped_to"] = mapped
            result.foods.append(nutrient)
        except LookupError as exc:
            result.errors.append(str(exc))

    semantic_query = query_text or " ".join(filter(None, [query_entities.nutrient, age_group or "", *query_entities.foods]))
    result.semantic = [SemanticChunk.model_validate(item) for item in semantic_search(semantic_query, int(os.getenv("MAX_RETRIEVAL_CHUNKS", "5")))]
    return result
