from core.food_map import FOOD_MAP, validate_food_map_targets
from core.models import QueryEntities
from core.retriever import (
    get_food_nutrients,
    get_rda,
    load_datasets,
    map_age_to_group,
    map_food,
    retrieve_all,
    semantic_search,
)


def test_age_mapping():
    assert map_age_to_group(6) == "0-6_months"
    assert map_age_to_group(24) == "1-3_years"
    assert map_age_to_group(72) == "unsupported_age"


def test_food_mapping_exact_and_fuzzy():
    assert map_food("dal") == "lentils"
    assert map_food("daal") == "lentils"
    assert map_food("chawal") == "rice_polished"
    assert map_food("palak") == "spinach_leaves"
    assert map_food("not_a_food") == "unknown_food"


def test_food_map_coverage():
    assert len(FOOD_MAP) >= 150
    assert validate_food_map_targets() == []


def test_dataset_lookup():
    rda, ifct = load_datasets()
    assert not rda.empty
    assert not ifct.empty
    assert get_rda("iron", "1-3_years")["value"] == 8.0
    assert get_food_nutrients("lentils", "protein")["value_per_100g"] == 24.35


def test_semantic_search_returns_relevant_chunks():
    results = semantic_search("iron for toddlers", 3)
    assert results
    assert len(results) <= 3
    assert all("text" in item and "score" in item for item in results)
    top_text = " ".join(item["text"].lower() for item in results)
    assert "iron" in top_text or "child" in top_text or "infant" in top_text


def test_retrieve_all_multi_source():
    entities = QueryEntities(
        nutrient="protein",
        age_months=24,
        foods=["dal", "rice"],
        intent="diet_check",
    )
    result = retrieve_all(entities, "Is dal and rice enough protein for 2 year old?")
    assert result.rda is not None
    assert result.rda["value"] > 0
    assert len(result.foods) == 2
    assert result.semantic
    assert result.age_group == "1-3_years"
    assert not result.unknown_foods
