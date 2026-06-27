from __future__ import annotations

import json
import os
import re

from pydantic import ValidationError

from core.models import QueryEntities
from core.retriever import FOOD_MAP, normalize_key


# NOTE: these must match exact column names in data/processed/icmr_rda.csv.
NUTRIENTS = {
    "iron", "protein", "calcium", "vit_c", "vit_d", "vit_a",
    "zinc", "folate", "vit_b6", "vit_b12", "riboflavin", "thiamine",
    "iodine", "niacin", "magnesium", "dietary_fiber",
}

NUTRIENT_ALIASES = {
    "vitamin c": "vit_c",    "vit c": "vit_c",
    "vitamin d": "vit_d",    "vit d": "vit_d",
    "vitamin a": "vit_a",    "vit a": "vit_a",
    "vitamin b6": "vit_b6",  "vit b6": "vit_b6",  "b6": "vit_b6",
    "vitamin b12": "vit_b12","vit b12": "vit_b12", "b12": "vit_b12",
    "vitamin b2": "riboflavin","vit b2": "riboflavin","riboflavin": "riboflavin",
    "vitamin b1": "thiamine","vit b1": "thiamine",  "thiamine": "thiamine",
    "vitamin b3": "niacin",  "vit b3": "niacin",    "niacin": "niacin",
    "iodine": "iodine",      "magnesium": "magnesium",
    "dietary fiber": "dietary_fiber","dietary fibre": "dietary_fiber",
    "fiber": "dietary_fiber","fibre": "dietary_fiber",
}

# Range form: "1-3 years", "6-12 months"
AGE_RANGE_RE = re.compile(r"(\d+)\s*-\s*(\d+)\s*(months?|years?|yrs?)", re.I)

# Single form — "old" is optional so "18 month baby", "18 months", "2-year-old" all match
AGE_SINGLE_RE = re.compile(r"(\d+)\s*-?\s*(months?|years?|yrs?)(?:\s*-?\s*old)?\b", re.I)

QTY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(bowl|cup|plate|piece|roti|chapati|glass|spoon|egg)s?", re.I)

# Keywords that signal a general information request.
# Covers source queries ("get","source"), safety ("avoid","safe"),
# feeding advice ("feed","eat"), and child-context words ("baby","toddler")
# so age-free but clearly child-nutrition queries still route to
# general_question rather than unknown.
_GQ_KEYWORDS = [
    # dietary advice
    "feed", "feeding", "diet", "meal", "nutrition", "balanced",
    "eat", "eating", "food", "foods",
    # source / availability
    "get", "source", "sources", "how to", "food for",
    "which food", "foods that", "rich in", "contain", "provide",
    # safety / avoidance
    "avoid", "safe", "good for", "bad for",
    "introduce", "start",
    # child-context (no age number but clearly infant/child topic)
    "baby", "babies", "infant", "infants",
    "toddler", "toddlers", "child", "children",
    # health / wellbeing
    "healthy", "growth", "development",
]


def _extract_age_months(text: str) -> int | None:
    match = AGE_RANGE_RE.search(text)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        unit = match.group(3).lower()
        mid = (start + end) / 2
        return int(mid) if "month" in unit else int(mid * 12)

    match = AGE_SINGLE_RE.search(text)
    if match:
        value = int(match.group(1))
        unit = match.group(2).lower()
        return value if "month" in unit else value * 12

    return None


def _extract_nutrient(text: str) -> str | None:
    normalized_text = text.lower().replace("-", " ")
    for alias in sorted(NUTRIENT_ALIASES, key=len, reverse=True):
        if alias in normalized_text:
            return NUTRIENT_ALIASES[alias]
    for nutrient in NUTRIENTS:
        if nutrient.replace("_", " ") in normalized_text or nutrient in normalized_text:
            return nutrient
    return None


def _extract_foods(text: str) -> list[str]:
    normalized_text = text.lower().replace("-", " ")
    found = []
    for food in FOOD_MAP:
        phrase = food.replace("_", " ")
        if re.search(rf"\b{re.escape(phrase)}\b", normalized_text):
            canonical = normalize_key(food)
            if canonical not in found:
                found.append(canonical)
    return found


def _extract_servings(text: str, foods: list[str]) -> dict[str, float]:
    servings = {food: 1.0 for food in foods}
    for match in QTY_RE.finditer(text):
        amount = float(match.group(1))
        unit = normalize_key(match.group(2))
        if unit in {"roti", "chapati", "egg"} and unit in foods:
            servings[unit] = amount
    for food in foods:
        pattern = re.compile(
            rf"(\d+(?:\.\d+)?)\s+(?:\w+\s+)?{re.escape(food.replace('_', ' '))}", re.I
        )
        match = pattern.search(text)
        if match:
            servings[food] = float(match.group(1))
    return servings


def _decompose_with_gemini(query_text: str) -> QueryEntities | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        prompt = (
            "Extract nutrition query entities as JSON with keys nutrient, age_months, "
            "foods, servings, intent. Intent is rda_lookup, diet_check, or unknown. "
            f"Query: {query_text}"
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=200),
        )
        raw = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        return QueryEntities.model_validate(json.loads(raw))
    except (Exception, ValidationError):
        return None


def decompose(query_text: str) -> QueryEntities:
    if not query_text.strip():
        return QueryEntities(intent="unknown", confidence=0.0)

    age_months  = _extract_age_months(query_text)
    nutrient    = _extract_nutrient(query_text)
    foods       = _extract_foods(query_text)
    servings    = _extract_servings(query_text, foods)
    query_lower = query_text.lower()

    # ── Priority cascade ───────────────────────────────────────────────────────
    # If age_months is extracted, the parent mentioned their child's age.
    # NutriMind MUST attempt a useful answer in every such case — never unknown.

    # RULE 1: age + nutrient + food → diet_check (full gap calculation)
    if age_months is not None and nutrient and foods:
        return QueryEntities(
            nutrient=nutrient, age_months=age_months,
            foods=foods, servings=servings,
            intent="diet_check", confidence=0.9,
        )

    # RULE 2: age + nutrient (no food) → rda_lookup
    if age_months is not None and nutrient:
        return QueryEntities(
            nutrient=nutrient, age_months=age_months,
            foods=[], servings={},
            intent="rda_lookup", confidence=0.8,
        )

    # RULE 3: age + food (no nutrient) → diet_check, default nutrient = protein
    # "is dal good for my 6 month old" — we know food + age, protein is the
    # most universal concern. Falls to FALLBACK gracefully if IFCT lookup fails.
    if age_months is not None and foods:
        return QueryEntities(
            nutrient="protein", age_months=age_months,
            foods=foods, servings=servings,
            intent="diet_check", confidence=0.7,
        )

    # RULE 4: age present, no nutrient, no food → general_question
    # "what should I feed my 9 month old" — DGI semantic search surfaces
    # age-appropriate feeding guidelines. Never return unknown for these.
    if age_months is not None:
        return QueryEntities(
            nutrient=nutrient, age_months=age_months,
            foods=foods, servings=servings,
            intent="general_question", confidence=0.6,
        )

    # ── No age extracted ───────────────────────────────────────────────────────

    # RULE 5: GQ keywords → general_question
    # Catches "healthy eating for toddlers", "is dal good for babies",
    # "how to get vitamin D", "foods rich in iron", etc.
    if any(kw in query_lower for kw in _GQ_KEYWORDS):
        return QueryEntities(
            nutrient=nutrient, age_months=age_months,
            foods=foods, servings=servings,
            intent="general_question", confidence=0.6,
        )

    # Gemini fallback (only when key is configured)
    fallback = _decompose_with_gemini(query_text)
    if fallback and (fallback.age_months or fallback.nutrient):
        return fallback

    # RDA keyword fallback ("daily requirement", "rda for iron")
    if nutrient and any(w in query_lower for w in ["requirement", "rda", "daily"]):
        return QueryEntities(
            nutrient=nutrient, age_months=age_months,
            foods=[], servings={},
            intent="rda_lookup", confidence=0.5,
        )

    # RULE 5b: nutrient extracted but no age → general info request
    # "how will he get vitamin D", "sources of calcium"
    if nutrient:
        return QueryEntities(
            nutrient=nutrient, age_months=None,
            foods=[], servings={},
            intent="general_question", confidence=0.5,
        )

    # RULE 6: food extracted but no age, no nutrient → general_question
    # "is dal good for babies" when "babies" keyword also missed
    if foods:
        return QueryEntities(
            nutrient=None, age_months=None,
            foods=foods, servings=servings,
            intent="general_question", confidence=0.5,
        )

    # RULE 7: truly nothing extracted — unrelated query (e.g. "hello nutri")
    # Only place where unknown is returned.
    return QueryEntities(
        nutrient=nutrient, age_months=age_months,
        foods=foods, servings=servings,
        intent="unknown", confidence=0.0,
    )