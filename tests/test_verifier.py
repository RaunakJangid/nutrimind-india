from core.bridge import calculate_gap
from core.models import QueryEntities
from core.verifier import verify


def test_valid_calculation_verifies():
    result = calculate_gap(QueryEntities(nutrient="protein", age_months=24, foods=["dal"], servings={"dal": 1}, intent="diet_check"))
    assert verify(result).verified is True


def test_modified_rda_fails():
    result = calculate_gap(QueryEntities(nutrient="protein", age_months=24, foods=["dal"], servings={"dal": 1}, intent="diet_check"))
    result.required_value = 999
    assert verify(result).verified is False


def test_wrong_math_fails():
    result = calculate_gap(QueryEntities(nutrient="protein", age_months=24, foods=["dal"], servings={"dal": 1}, intent="diet_check"))
    result.gap_value = 123
    assert verify(result).verified is False
