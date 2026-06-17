from core.decomposer import decompose


def test_rda_lookup_query():
    entities = decompose("Iron requirement for 2-year-old")
    assert entities.intent == "rda_lookup"
    assert entities.age_months == 24
    assert entities.nutrient == "iron"


def test_diet_check_query():
    entities = decompose("Is dal enough protein for 18 month baby")
    assert entities.intent == "diet_check"
    assert entities.age_months == 18
    assert entities.foods == ["dal"]


def test_quantities():
    entities = decompose("My 2 year child eats 2 roti and 1 bowl dal for protein")
    assert entities.servings["roti"] == 2
    assert entities.servings["dal"] == 1


def test_unknown_query():
    assert decompose("What should I feed?").intent == "general_question"
