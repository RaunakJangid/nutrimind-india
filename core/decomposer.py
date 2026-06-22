from __future__ import annotations

import json
import os
import re

from pydantic import ValidationError

from core.models import QueryEntities
from core.retriever import FOOD_MAP, normalize_key


# NOTE: these must match exact column names in data/processed/icmr_rda.csv.
# Renamed vitamin_c/vitamin_d/vitamin_a -> vit_c/vit_d/vit_a to match CSV
# convention (vit_b6 was already correct). VERIFY against actual CSV headers.
NUTRIENTS = {
    "iron",
    "protein",
    "calcium",
    "vit_c",
    "vit_d",
    "vit_a",
    "zinc",
    "folate",
    "vit_b6",
    "vit_b12",
    "riboflavin",
    "thiamine",
    "iodine",
    "niacin",
    "magnesium",
    "dietary_fiber",
}

NUTRIENT_ALIASES = {
    "vitamin c": "vit_c",
    "vit c": "vit_c",
    "vitamin d": "vit_d",
    "vit d": "vit_d",
    "vitamin a": "vit_a",
    "vit a": "vit_a",
    "vitamin b6": "vit_b6",
    "vit b6": "vit_b6",
    "b6": "vit_b6",
    "vitamin b12": "vit_b12",
    "vit b12": "vit_b12",
    "b12": "vit_b12",
    "vitamin b2": "riboflavin",
    "vit b2": "riboflavin",
    "riboflavin": "riboflavin",
    "vitamin b1": "thiamine",
    "vit b1": "thiamine",
    "thiamine": "thiamine",
    "vitamin b3": "niacin",
    "vit b3": "niacin",
    "niacin": "niacin",
    "iodine": "iodine",
    "magnesium": "magnesium",
    "dietary fiber": "dietary_fiber",
    "dietary fibre": "dietary_fiber",
    "fiber": "dietary_fiber",
    "fibre": "dietary_fiber",
}

# Range form: "1-3 years", "6-12 months"
AGE_RANGE_RE = re.compile(r"(\d+)\s*-\s*(\d+)\s*(months?|years?|yrs?)", re.I)
# Single form: "2-year-old", "9-month-old", "6 month old"
AGE_SINGLE_RE = re.compile(r"(\d+)\s*-?\s*(months?|years?|yrs?)\s*-?\s*old", re.I)

QTY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(bowl|cup|plate|piece|roti|chapati|glass|spoon|egg)s?", re.I)


def _extract_age_months(text: str) -> int | None:
    # Try range form first ("1-3 years")
    match = AGE_RANGE_RE.search(text)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        unit = match.group(3).lower()
        mid = (start + end) / 2
        if "month" in unit:
            return int(mid)
        else:  # years
            return int(mid * 12)

    # Try single-age form ("2-year-old", "9-month-old")
    match = AGE_SINGLE_RE.search(text)
    if match:
        value = int(match.group(1))
        unit = match.group(2).lower()
        if "month" in unit:
            return value
        else:  # years
            return value * 12

    return None


def _extract_nutrient(text: str) -> str | None:
    normalized_text = text.lower().replace("-", " ")
    # Check multi-word aliases first (longest match wins to avoid "vit b1"
    # matching inside "vit b12" etc. -- sort by alias length descending)
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
        pattern = re.compile(rf"(\d+(?:\.\d+)?)\s+(?:\w+\s+)?{re.escape(food.replace('_', ' '))}", re.I)
        match = pattern.search(text)
        if match:
            servings[food] = float(match.group(1))
    return servings


def _decompose_with_gemini(query_text: str) -> QueryEntities | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Extract nutrition query entities as JSON with keys nutrient, age_months, "
            "foods, servings, intent. Intent is rda_lookup, diet_check, or unknown. "
            f"Query: {query_text}"
        )
        response = model.generate_content(prompt, request_options={"timeout": 5})
        raw = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        return QueryEntities.model_validate(json.loads(raw))
    except (Exception, ValidationError):
        return None


def decompose(query_text: str) -> QueryEntities:
    if not query_text.strip():
        return QueryEntities(intent="unknown", confidence=0.0)

    age_months = _extract_age_months(query_text)
    nutrient = _extract_nutrient(query_text)
    foods = _extract_foods(query_text)
    servings = _extract_servings(query_text, foods)

    if age_months is not None and nutrient and foods:
        return QueryEntities(
            nutrient=nutrient,
            age_months=age_months,
            foods=foods,
            servings=servings,
            intent="diet_check",
            confidence=0.9,
        )
    if age_months is not None and nutrient:
        return QueryEntities(
            nutrient=nutrient,
            age_months=age_months,
            foods=[],
            servings={},
            intent="rda_lookup",
            confidence=0.8,
        )

    if any(keyword in query_text.lower() for keyword in ["feed", "diet", "meal", "nutrition", "balanced"]):
        return QueryEntities(
            nutrient=nutrient,
            age_months=age_months,
            foods=foods,
            servings=servings,
            intent="general_question",
            confidence=0.6,
        )

    fallback = _decompose_with_gemini(query_text)
    # Only use Gemini if it gives something useful
    if fallback and (fallback.age_months or fallback.nutrient):
        return fallback

    query_lower = query_text.lower()

    if nutrient and any(word in query_lower for word in ["requirement", "rda", "daily"]):
        return QueryEntities(
            nutrient=nutrient,
            age_months=age_months,
            foods=[],
            servings={},
            intent="rda_lookup",
            confidence=0.5,
        )

    return QueryEntities(
        nutrient=nutrient,
        age_months=age_months,
        foods=foods,
        servings=servings,
        intent="unknown",
        confidence=0.0,
    )