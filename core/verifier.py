from __future__ import annotations

import math

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    from difflib import SequenceMatcher

    class _FuzzFallback:
        @staticmethod
        def partial_ratio(left: str, right: str) -> float:
            return SequenceMatcher(None, left, right).ratio() * 100

    fuzz = _FuzzFallback()

from core.models import CalculationResult, CheckResult, MergedContext, VerificationResult
from core.retriever import get_food_nutrients, get_rda


TOLERANCE = 0.001


def check_llm_grounding(answer: str, context: MergedContext, min_score: float = 0.55) -> bool:
    if not answer:
        return False
    source_text = context.context_text.lower()
    if not source_text:
        return False
    answer_sentences = [sentence.strip() for sentence in answer.split(".") if len(sentence.strip()) > 15]
    if not answer_sentences:
        return True
    grounded = 0
    for sentence in answer_sentences:
        score = fuzz.partial_ratio(sentence.lower(), source_text) / 100
        has_citation = "[source:" in sentence.lower() or any(token in source_text for token in sentence.lower().split() if len(token) > 4)
        if score >= min_score or has_citation:
            grounded += 1
    return grounded / len(answer_sentences) >= 0.7


def verify(calculation_result: CalculationResult, answer: str = "", context: MergedContext | None = None) -> VerificationResult:
    checks: list[CheckResult] = []

    try:
        rda = get_rda(calculation_result.nutrient, calculation_result.age_group)
        rda_ok = math.isclose(rda["value"], calculation_result.required_value, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
    except LookupError as exc:
        rda_ok = False
        rda_detail = str(exc)
    else:
        rda_detail = "RDA value matches dataset" if rda_ok else "RDA value differs from dataset"
    checks.append(CheckResult(name="rda_exact_match", passed=rda_ok, detail=rda_detail))

    foods_ok = True
    for detail in calculation_result.food_details:
        try:
            nutrient = get_food_nutrients(detail.mapped_to, calculation_result.nutrient)
            value_ok = math.isclose(nutrient["value_per_100g"], detail.value_per_100g, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
        except LookupError:
            value_ok = False
        foods_ok = foods_ok and value_ok
    checks.append(CheckResult(name="food_values_match", passed=foods_ok, detail="Food nutrient values match dataset"))

    expected_consumed = sum(item.nutrient_amount for item in calculation_result.food_details)
    consumed_ok = math.isclose(expected_consumed, calculation_result.consumed_value, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
    expected_gap = calculation_result.required_value - calculation_result.consumed_value
    gap_ok = math.isclose(expected_gap, calculation_result.gap_value, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
    expected_percent = (expected_gap / calculation_result.required_value) * 100 if calculation_result.required_value else 0.0
    percent_ok = math.isclose(expected_percent, calculation_result.gap_percent, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
    checks.append(CheckResult(name="math_check", passed=consumed_ok and gap_ok and percent_ok, detail="Derived math is consistent"))

    conflicts_ok = not calculation_result.conflicts
    checks.append(CheckResult(name="conflict_check", passed=conflicts_ok, detail="No unresolved source conflicts"))

    if context is not None and answer:
        grounded_ok = check_llm_grounding(answer, context)
        checks.append(CheckResult(name="llm_grounding", passed=grounded_ok, detail="LLM answer is grounded in retrieved context"))

    verified = all(check.passed for check in checks)
    fail_reason = None if verified else "; ".join(check.detail for check in checks if not check.passed)
    return VerificationResult(verified=verified, checks=checks, fail_reason=fail_reason)
