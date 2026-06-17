from core.llm_backends import DeterministicBackend, get_backend


def test_deterministic_backend_generates_citations():
    answer = DeterministicBackend().generate(
        "",
        {
            "calculation": {
                "nutrient": "protein",
                "required_value": 13,
                "required_unit": "g",
                "consumed_value": 13.5,
                "gap_value": -0.5,
            }
        },
    )
    assert "[Source: ICMR RDA]" in answer
    assert "[Source: IFCT]" in answer


def test_backend_switching():
    assert get_backend("deterministic").name == "deterministic"
