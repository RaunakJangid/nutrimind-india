from core.bridge import calculate_gap
from core.models import QueryEntities
from core.synthesizer import render_response


def test_rda_template_renders():
    result = calculate_gap(QueryEntities(nutrient="iron", age_months=24, intent="rda_lookup"))
    assert "daily iron requirement is 8 mg" in render_response(result, "rda_lookup")


def test_diet_template_renders_gap():
    result = calculate_gap(QueryEntities(nutrient="protein", age_months=24, foods=["rice"], servings={"rice": 1}, intent="diet_check"))
    text = render_response(result, "diet_check")
    assert "remaining gap" in text
    assert "4.6 g" in text
