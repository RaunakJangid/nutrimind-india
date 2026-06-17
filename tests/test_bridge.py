import pytest

from core.bridge import calculate_gap
from core.models import QueryEntities


def test_single_food_gap():
    result = calculate_gap(QueryEntities(nutrient="protein", age_months=24, foods=["dal"], servings={"dal": 1}, intent="diet_check"))
    assert result.consumed_value == pytest.approx(36.525)
    assert result.gap_value == pytest.approx(-24.025)


def test_multiple_food_gap():
    result = calculate_gap(
        QueryEntities(nutrient="protein", age_months=24, foods=["dal", "rice"], servings={"dal": 1, "rice": 1}, intent="diet_check")
    )
    assert result.consumed_value == pytest.approx(44.465)


def test_missing_food_is_flagged():
    result = calculate_gap(QueryEntities(nutrient="iron", age_months=24, foods=["xyz"], servings={"xyz": 1}, intent="diet_check"))
    assert "unknown_food" in result.flags
    assert result.consumed_value == 0


def test_out_of_range_age_raises():
    with pytest.raises(ValueError):
        calculate_gap(QueryEntities(nutrient="iron", age_months=90, foods=["dal"], servings={"dal": 1}, intent="diet_check"))
