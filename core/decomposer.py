from __future__ import annotations

import json
import os
import re

from pydantic import ValidationError

from core.models import QueryEntities
from core.retriever import FOOD_MAP, normalize_key


NUTRIENTS = {
    "iron",
    "protein",
    "calcium",
    "vitamin_c",
    "vitamin_d",
    "vitamin_a",
    "zinc",
    "folate",
}
NUTRIENT_ALIASES = {
    "vitamin c": "vitamin_c",
    "vit c": "vitamin_c",
    "vitamin d": "vitamin_d",
    "vitamin a": "vitamin_a",
}
AGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[- ]?\s*(year|years|yr|yrs|y|month|months|mo|mos|m)", re.I)
QTY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(bowl|cup|plate|piece|roti|chapati|glass|spoon|egg)s?", re.I)


def _extract_age_months(text: str) -> int | None:
    match = AGE_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith(("year", "yr")) or unit == "y":
        return int(round(value * 12))
    return int(round(value))


def _extract_nutrient(text: str) -> str | None:
    normalized_text = text.lower().replace("-", " ")
    for alias, nutrient in NUTRIENT_ALIASES.items():
        if alias in normalized_text:
            return nutrient
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
    if fallback:
        return fallback
    return QueryEntities(nutrient=nutrient, age_months=age_months, foods=foods, servings=servings, intent="unknown", confidence=0.0)
