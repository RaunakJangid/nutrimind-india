from __future__ import annotations

from core.models import CalculationResult, FoodContribution, MergedContext, QueryEntities
from core.retriever import get_food_nutrients, get_rda, map_age_to_group, map_food, normalize_key


SERVING_SIZES = {
    "dal": {"grams": 150, "description": "1 bowl cooked"},
    "moong_dal": {"grams": 150, "description": "1 bowl cooked"},
    "toor_dal": {"grams": 150, "description": "1 bowl cooked"},
    "rice": {"grams": 100, "description": "1 cup cooked"},
    "brown_rice": {"grams": 100, "description": "1 cup cooked"},
    "roti": {"grams": 30, "description": "1 medium roti"},
    "chapati": {"grams": 30, "description": "1 medium chapati"},
    "milk": {"grams": 250, "description": "1 glass"},
    "dahi": {"grams": 100, "description": "1 bowl"},
    "egg": {"grams": 50, "description": "1 whole egg"},
    "apple": {"grams": 150, "description": "1 medium"},
    "banana": {"grams": 100, "description": "1 medium"},
    "spinach": {"grams": 50, "description": "1 cup cooked"},
    "chicken": {"grams": 100, "description": "1 piece cooked"},
    "fish": {"grams": 100, "description": "1 piece cooked"},
}


def calculate_gap(query_entities: QueryEntities, merged_context: MergedContext | None = None) -> CalculationResult:
    if query_entities.nutrient is None:
        raise ValueError("A nutrient is required")
    if query_entities.age_months is None:
        raise ValueError("Age is required")

    nutrient = normalize_key(query_entities.nutrient)
    age_group = map_age_to_group(query_entities.age_months)
    if age_group == "unsupported_age":
        raise ValueError("Supported age range is 0 to 60 months")

    rda = merged_context.rda if merged_context and merged_context.rda else get_rda(nutrient, age_group)
    total_consumed = 0.0
    food_details: list[FoodContribution] = []
    unknown_foods: list[str] = []
    flags: list[str] = []

    for food in query_entities.foods:
        servings = query_entities.servings.get(food, query_entities.servings.get(normalize_key(food), 1.0))
        mapped_food = map_food(food)
        if mapped_food == "unknown_food":
            unknown_foods.append(food)
            flags.append("unknown_food")
            continue

        serving_info = SERVING_SIZES.get(normalize_key(food), {"grams": 100, "description": "100 g assumed"})
        if normalize_key(food) not in SERVING_SIZES:
            flags.append("assumed_default_serving")
        grams = float(servings) * float(serving_info["grams"])

        try:
            nutrient_info = get_food_nutrients(mapped_food, nutrient)
        except LookupError:
            flags.append("nutrient_not_available")
            continue

        amount = (nutrient_info["value_per_100g"] * grams) / 100
        total_consumed += amount
        food_details.append(
            FoodContribution(
                food=food,
                mapped_to=mapped_food,
                servings=float(servings),
                grams=grams,
                value_per_100g=nutrient_info["value_per_100g"],
                nutrient_amount=amount,
                unit=nutrient_info["unit"],
            source=nutrient_info["source"],
            confidence=float(nutrient_info.get("confidence", 1.0)),
            )
        )

    gap_value = rda["value"] - total_consumed
    gap_percent = (gap_value / rda["value"]) * 100 if rda["value"] else 0.0
    return CalculationResult(
        nutrient=nutrient,
        age_months=query_entities.age_months,
        age_group=age_group,
        required_value=rda["value"],
        required_unit=rda["unit"],
        consumed_value=total_consumed,
        consumed_unit=rda["unit"],
        gap_value=gap_value,
        gap_percent=gap_percent,
        food_details=food_details,
        unknown_foods=unknown_foods,
        flags=sorted(set(flags)),
        rda_source=rda,
        intent=query_entities.intent,
        conflicts=merged_context.conflicts if merged_context else [],
    )
