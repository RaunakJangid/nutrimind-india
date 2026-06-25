from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.llm_backends import DeterministicBackend, get_backend
from core.models import CalculationResult, MergedContext, SynthesisResult


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape())

# Dedicated prompt for general_question intent.
# Intentionally NOT a file template — it's short, stable, and keeping it
# inline makes the parent-friendly tone constraint explicit and reviewable.
# Top-2 chunks only are passed in (see synthesize_answer) to avoid walls
# of text that confuse both the LLM and the reader.
_GQ_PROMPT = (
    "You are a helpful nutrition advisor for Indian parents.\n"
    "Based on the following ICMR-NIN dietary guidelines, answer this question "
    "for a parent asking about infant or child nutrition.\n\n"
    "Question: {query}\n\n"
    "Guidelines:\n{context}\n\n"
    "Give a clear, concise, parent-friendly answer in 2-3 sentences. "
    "Use simple language and avoid jargon. "
    "If the guidelines do not cover the question, say so honestly."
)


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

    # =========================
    # STRICT MODE: rda_lookup
    # =========================
    if calculation_result and calculation_result.intent == "rda_lookup":
        value = f"{calculation_result.required_value:.1f}".rstrip("0").rstrip(".")
        unit = calculation_result.required_unit
        nutrient = calculation_result.nutrient.replace("_", " ")
        age_group = calculation_result.age_group.replace("_", " ")

        answer = (
            f"The daily {nutrient} requirement for the {age_group} age group "
            f"is {value} {unit}."
        )

        return SynthesisResult(
            answer=answer,
            citations=[],
            model_backend="deterministic_rda",
            prompt_version="v1"
        )

    # =========================
    # FIX 1: STRICT diet_check
    # =========================
    if calculation_result and calculation_result.intent == "diet_check":
        answer = render_response(calculation_result, intent="diet_check")

        return SynthesisResult(
            answer=answer,
            citations=[],
            model_backend="deterministic_diet_check",
            prompt_version="v1"
        )

    # =========================
    # LLM MODE: general_question
    # =========================
    # Uses a dedicated parent-friendly prompt with the top-2 most relevant
    # FAISS chunks only. Rationale for top-2:
    #   - merged_context.semantic is already ranked by similarity score
    #   - dumping 5–7 chunks produces walls of text that degrade LLM output
    #   - retrieval still fetches 7 chunks (for RAGAS recall measurement);
    #     we only narrow here at synthesis time, not at retrieval time
    top_chunks = merged_context.semantic[:2]
    ctx_text = (
        "\n\n".join(chunk.text for chunk in top_chunks)
        if top_chunks else "No relevant ICMR-NIN guidelines retrieved."
    )
    gq_prompt = _GQ_PROMPT.format(query=query, context=ctx_text)

    backend = get_backend(model_backend)
    print(f"[GQ] backend={backend.name!r}  chunks_used={len(top_chunks)}")

    try:
        answer = backend.generate(gq_prompt, {})
        # Diagnostic: confirm LLM returned real content, not a canned stub
        print(f"[GQ] backend.generate() OK  len={len(answer)}  "
              f"preview={answer[:120].replace(chr(10),' ')!r}")
        used_backend = backend.name

    except Exception as _exc:
        # backend.generate() raised — most common cause: no API key set,
        # so get_backend() returned DeterministicBackend which has no LLM
        # and its generate() returns a canned message unrelated to the query.
        # Instead, build a readable answer directly from the retrieved chunks
        # so the user sees real guideline text, not a confusing stub.
        print(f"[GQ] backend.generate() FAILED: {_exc!r}  — using chunk fallback")
        if top_chunks:
            answer = (
                "Based on ICMR-NIN dietary guidelines: "
                + " ".join(chunk.text for chunk in top_chunks)
            )
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
        prompt_version="v1"
    )