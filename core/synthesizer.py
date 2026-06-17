from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.llm_backends import DeterministicBackend, get_backend
from core.models import CalculationResult, MergedContext, SynthesisResult


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape())


def age_desc(age_months: int) -> str:
    if age_months < 12:
        return f"{age_months}-month-old"
    years = age_months // 12
    months = age_months % 12
    if months == 0:
        return f"{years}-year-old"
    return f"{years}-year-{months}-month-old"


def render_response(calculation_result: CalculationResult, intent: str | None = None) -> str:
    template_name = "rda_lookup.j2" if (intent or calculation_result.intent) == "rda_lookup" else "diet_check.j2"
    template = env.get_template(template_name)
    return template.render(
        result=calculation_result,
        nutrient=calculation_result.nutrient.replace("_", " "),
        age_desc=age_desc(calculation_result.age_months),
        required_value=f"{calculation_result.required_value:.1f}".rstrip("0").rstrip("."),
        consumed_value=f"{calculation_result.consumed_value:.1f}".rstrip("0").rstrip("."),
        gap_value=f"{calculation_result.gap_value:.1f}".rstrip("0").rstrip("."),
        gap_percent=f"{calculation_result.gap_percent:.0f}",
    )


def build_prompt(query: str, merged_context: MergedContext, calculation_result: CalculationResult | None = None) -> str:
    prompt_template = (TEMPLATE_DIR / "prompts" / "v1_rag_prompt.txt").read_text(encoding="utf-8")
    rda_context = merged_context.rda or {}
    ifct_context = "\n".join(str(food) for food in merged_context.foods) or "No IFCT food context retrieved."
    semantic_context = "\n\n".join(chunk.text for chunk in merged_context.semantic) or "No semantic context retrieved."
    conflicts = "\n".join(conflict.description for conflict in merged_context.conflicts) or "None."
    return prompt_template.format(
        icmr_rda_context=rda_context,
        ifct_context=ifct_context,
        semantic_context=semantic_context,
        conflicts=conflicts,
        calculation=calculation_result.model_dump() if calculation_result else {},
        query=query,
    )


def synthesize_answer(
    query: str,
    merged_context: MergedContext,
    calculation_result: CalculationResult | None,
    model_backend: str,
) -> SynthesisResult:
    prompt = build_prompt(query, merged_context, calculation_result)
    context = {"merged_context": merged_context.model_dump(), "calculation": calculation_result.model_dump() if calculation_result else None}
    backend = get_backend(model_backend)
    try:
        answer = backend.generate(prompt, context)
        used_backend = backend.name
    except Exception:
        answer = DeterministicBackend().generate(prompt, context)
        used_backend = f"{backend.name}->deterministic"

    citations = sorted(set(re.findall(r"\[Source:\s*([^\]]+)\]", answer)))
    return SynthesisResult(answer=answer, citations=citations, model_backend=used_backend, prompt_version="v1")
