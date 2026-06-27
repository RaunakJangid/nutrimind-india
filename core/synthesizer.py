from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.llm_backends import DeterministicBackend, get_backend
from core.models import CalculationResult, MergedContext, SynthesisResult


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape())

# Dedicated prompt for general_question intent.
# {age_context} is injected when age can be extracted from the query
# ("For a 9-month-old child.") so the LLM interprets DGI chunks in the
# right developmental context and the answer feels personal to that parent.
# Only top-2 chunks are passed (see synthesize_answer) to keep it focused.
_GQ_PROMPT = (
    "You are a helpful nutrition advisor for Indian parents.\n"
    "{age_context}"
    "Based on the following ICMR-NIN dietary guidelines, answer this question "
    "for a parent asking about infant or child nutrition.\n\n"
    "Question: {query}\n\n"
    "Guidelines:\n{context}\n\n"
    "Give a clear, concise, parent-friendly answer in 2-3 sentences. "
    "Use simple language and avoid jargon. "
    "If the guidelines do not cover the question, say so honestly."
)

# Same pattern as decomposer.AGE_SINGLE_RE — duplicated here to avoid
# circular import. Used only for age-context injection in the GQ prompt.
_AGE_RE = re.compile(r"(\d+)\s*-?\s*(months?|years?|yrs?)(?:\s*-?\s*old)?\b", re.I)


def _age_context_str(query: str) -> str:
    """Return 'For a X-month-old child. ' if age is detectable, else ''."""
    m = _AGE_RE.search(query)
    if not m:
        return ""
    v, u = int(m.group(1)), m.group(2).lower()
    am = v if "month" in u else v * 12
    if am < 12:
        desc = f"{am}-month-old"
    else:
        y, mo = am // 12, am % 12
        desc = f"{y}-year-old" if mo == 0 else f"{y} year {mo} month old"
    return f"For a {desc} child. "


def age_desc(age_months: int) -> str:
    if age_months < 12:
        return f"{age_months}-month-old"
    years = age_months // 12
    months = age_months % 12
    if months == 0:
        return f"{years}-year-old"
    return f"{years}-year-{months}-month-old"


def render_response(calculation_result: CalculationResult, intent: str | None = None) -> str:
    template_name = (
        "rda_lookup.j2"
        if (intent or calculation_result.intent) == "rda_lookup"
        else "diet_check.j2"
    )
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


def build_prompt(
    query: str,
    merged_context: MergedContext,
    calculation_result: CalculationResult | None = None,
) -> str:
    prompt_template = (
        TEMPLATE_DIR / "prompts" / "v1_rag_prompt.txt"
    ).read_text(encoding="utf-8")

    rda_context = merged_context.rda or {}
    ifct_context = (
        "\n".join(str(food) for food in merged_context.foods)
        or "No IFCT food context retrieved."
    )
    semantic_context = (
        "\n\n".join(chunk.text for chunk in merged_context.semantic)
        or "No semantic context retrieved."
    )
    conflicts = (
        "\n".join(conflict.description for conflict in merged_context.conflicts)
        or "None."
    )

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

    # ─────────────────────────────────────────────────────────────────────────
    # STRICT MODE: rda_lookup — deterministic, no LLM
    # ─────────────────────────────────────────────────────────────────────────
    if calculation_result and calculation_result.intent == "rda_lookup":
        value   = f"{calculation_result.required_value:.1f}".rstrip("0").rstrip(".")
        unit    = calculation_result.required_unit
        nutrient = calculation_result.nutrient.replace("_", " ")
        age_group = calculation_result.age_group.replace("_", " ")

        return SynthesisResult(
            answer=(
                f"The daily {nutrient} requirement for the {age_group} age group "
                f"is {value} {unit}."
            ),
            citations=[],
            model_backend="deterministic_rda",
            prompt_version="v1",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STRICT MODE: diet_check — deterministic template, no LLM (Fix 1)
    # ─────────────────────────────────────────────────────────────────────────
    if calculation_result and calculation_result.intent == "diet_check":
        return SynthesisResult(
            answer=render_response(calculation_result, intent="diet_check"),
            citations=[],
            model_backend="deterministic_diet_check",
            prompt_version="v1",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LLM MODE: general_question (calculation_result is None)
    # ─────────────────────────────────────────────────────────────────────────
    # Top-2 chunks only — retrieved_contexts still holds all chunks for RAGAS
    # recall measurement; we narrow here at synthesis time only.
    top_chunks = merged_context.semantic[:2]
    ctx_text = (
        "\n\n".join(chunk.text for chunk in top_chunks)
        if top_chunks else "No relevant ICMR-NIN guidelines retrieved."
    )

    # Inject age context so the LLM interprets DGI content correctly
    # ("For a 9-month-old child.") — makes the answer feel personal.
    age_ctx = _age_context_str(query)

    gq_prompt = _GQ_PROMPT.format(
        age_context=age_ctx,
        query=query,
        context=ctx_text,
    )

    backend = get_backend(model_backend)
    print(f"[GQ] backend={backend.name!r}  chunks={len(top_chunks)}  age_ctx={age_ctx!r}")

    try:
        answer = backend.generate(gq_prompt, {})
        print(f"[GQ] OK  len={len(answer)}  preview={answer[:120].replace(chr(10),' ')!r}")
        used_backend = backend.name

    except Exception as _exc:
        # Print exact exception so we can diagnose the real failure cause
        print(f"GQ LLM ERROR: {type(_exc).__name__}: {_exc}")
        if top_chunks:
            prefix = f"Based on ICMR-NIN dietary guidelines{' for a ' + age_ctx.rstrip('. ') + ':' if age_ctx else ':'} "
            answer = prefix + " ".join(chunk.text for chunk in top_chunks)
        else:
            answer = (
                "I don't have specific ICMR-NIN guidelines for this question. "
                "Please consult a pediatric nutritionist."
            )
        used_backend = f"{backend.name}->chunk_fallback"

    citations = sorted(set(re.findall(r"\[Source:\s*([^\]]+)\]", answer)))
    return SynthesisResult(
        answer=answer,
        citations=citations,
        model_backend=used_backend,
        prompt_version="v1",
    )