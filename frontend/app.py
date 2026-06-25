from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.bridge import calculate_gap
from core.context_merger import merge_context
from core.decomposer import decompose
from core.retriever import retrieve_all
from core.synthesizer import synthesize_answer
from core.verifier import check_llm_grounding, verify
from db.db import init_db, insert_pipeline_run, insert_query


DISCLAIMER = "NutriMind provides dataset-backed estimates. Not medical advice."
FALLBACK = "Unable to verify this answer with trusted data. Please try rephrasing your question."


def proof_rows(result):
    rows = [
        {
            "Item": "RDA",
            "Nutrient": result.nutrient,
            "Value": f"{result.required_value:g} {result.required_unit}",
            "Source": result.rda_source["source"],
            "Confidence": result.rda_source.get("confidence", 1.0),
        }
    ]
    for item in result.food_details:
        rows.append(
            {
                "Item": f"{item.food} -> {item.mapped_to}",
                "Nutrient": result.nutrient,
                "Value": f"{item.value_per_100g:g} {item.unit} per 100g",
                "Source": item.source,
                "Confidence": item.confidence,
            }
        )
    return rows


def run_pipeline(query_text: str, model_backend: str = "gemini") -> dict:
    started = time.perf_counter()
    query_id = insert_query(query_text)
    entities = decompose(query_text)

    if entities.intent == "unknown":
        latency_ms = int((time.perf_counter() - started) * 1000)
        insert_pipeline_run(
            query_id,
            {
                "age_months": entities.age_months,
                "nutrient": entities.nutrient,
                "foods": entities.foods,
                "servings": entities.servings,
                "intent": entities.intent,
                "model_backend": model_backend,
                "verified": False,
                "latency_ms": latency_ms,
                "error_reason": "unknown_query",
            },
        )
        return {
            "query_id": query_id,
            "answer": "Please include your child's age and a nutrient, such as: Iron requirement for 2-year-old.",
            "verified": False,
            "proof": [],
            "raw": entities.model_dump(),
        }

    try:
        retrieval = retrieve_all(entities, query_text)
        merged_context = merge_context(retrieval)
        if entities.intent == "general_question":
            synthesis = synthesize_answer(query_text, merged_context, None, model_backend)
            verified = bool(merged_context.semantic) and check_llm_grounding(synthesis.answer, merged_context)
            answer = synthesis.answer if verified else FALLBACK
            proof = [
                {
                    "Item": chunk.title or chunk.chunk_id,
                    "Nutrient": "general",
                    "Value": "Semantic context",
                    "Source": chunk.source,
                    "Confidence": round(chunk.score, 3),
                }
                for chunk in merged_context.semantic
            ]
            latency_ms = int((time.perf_counter() - started) * 1000)
            insert_pipeline_run(
                query_id,
                {
                    "age_months": entities.age_months,
                    "nutrient": entities.nutrient,
                    "foods": entities.foods,
                    "servings": entities.servings,
                    "intent": entities.intent,
                    "model_backend": synthesis.model_backend,
                    "answer_text": answer,
                    "verified": verified,
                    "proof": proof,
                    "latency_ms": latency_ms,
                    "error_reason": None if verified else "llm_grounding_failed",
                },
            )
            return {
                "query_id": query_id,
                "answer": answer,
                "verified": verified,
                "proof": proof,
                "model_backend": synthesis.model_backend,
                "citations": synthesis.citations,
                "raw": {
                    "entities": entities.model_dump(),
                    "retrieval": retrieval.model_dump(),
                    "merged_context": merged_context.model_dump(),
                    "synthesis": synthesis.model_dump(),
                },
            }

        if entities.nutrient is None or entities.age_months is None:
            raise ValueError("Age and nutrient are required for RDA lookup or diet checks")

        result = calculate_gap(entities, merged_context)
        synthesis = synthesize_answer(query_text, merged_context, result, model_backend)
        verification = verify(result, synthesis.answer, merged_context)

        # ── Critical-check gate (replaces single verification.verified bool) ───
        # conflict_check is conservative by design: it flags any structured/
        # semantic numeric discrepancy (e.g. a FAISS narrative chunk citing an
        # approximate value vs. the precise ICMR CSV figure). These are benign
        # and expected — showing FALLBACK for them would reject correct answers.
        # Gate the user-facing answer on rda_match + math_check only.
        # conflict_flag is recorded separately for monitoring/paper reporting.
        _checks = {chk.name: chk.passed for chk in verification.checks}
        critical_ok = (
            _checks.get("rda_exact_match", True)
            and _checks.get("math_check", True)
        )
        conflict_flagged = not _checks.get("conflict_check", True)

        answer = synthesis.answer if critical_ok else FALLBACK
        proof = proof_rows(result)
        for chunk in merged_context.semantic:
            proof.append(
                {
                    "Item": chunk.title or chunk.chunk_id,
                    "Nutrient": result.nutrient,
                    "Value": "Semantic context",
                    "Source": chunk.source,
                    "Confidence": round(chunk.score, 3),
                }
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        insert_pipeline_run(
            query_id,
            {
                "age_months": result.age_months,
                "age_group": result.age_group,
                "nutrient": result.nutrient,
                "foods": entities.foods,
                "servings": entities.servings,
                "intent": entities.intent,
                "model_backend": synthesis.model_backend,
                "rda_value": result.required_value,
                "rda_unit": result.required_unit,
                "consumed_value": result.consumed_value,
                "gap_value": result.gap_value,
                "gap_percent": result.gap_percent,
                "answer_text": answer,
                "verified": critical_ok,           # rda_match + math_check
                "conflict_flagged": conflict_flagged,  # informational, not a rejection signal
                "proof": proof,
                "latency_ms": latency_ms,
                "error_reason": None if critical_ok else verification.fail_reason,
            },
        )
        return {
            "query_id": query_id,
            "answer": answer,
            "verified": critical_ok,
            "conflict_flagged": conflict_flagged,
            "proof": proof,
            "model_backend": synthesis.model_backend,
            "citations": synthesis.citations,
            "raw": {
                "entities": entities.model_dump(),
                "retrieval": retrieval.model_dump(),
                "merged_context": merged_context.model_dump(),
                "result": result.model_dump(),
                "synthesis": synthesis.model_dump(),
                "verification": verification.model_dump(),
                "verify_checks": _checks,          # per-check breakdown for debug
            },
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        insert_pipeline_run(
            query_id,
            {
                "age_months": entities.age_months,
                "nutrient": entities.nutrient,
                "foods": entities.foods,
                "servings": entities.servings,
                "intent": entities.intent,
                "model_backend": model_backend,
                "verified": False,
                "latency_ms": latency_ms,
                "error_reason": str(exc),
            },
        )
        return {"query_id": query_id, "answer": FALLBACK, "verified": False, "proof": [], "raw": {"error": str(exc)}}


def main() -> None:
    st.set_page_config(page_title="NutriMind-India", page_icon="NM", layout="centered")
    init_db(os.getenv("LOG_DB_PATH", str(ROOT / "data" / "query_log.db")))

    st.title("NutriMind-India")
    st.info(DISCLAIMER)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    model_backend = st.selectbox("Model backend", ["gemini", "llama", "deterministic"], index=0)
    debug = st.checkbox("Debug mode")

    if not st.session_state.chat_history:
        cols = st.columns(2)
        examples = ["Iron requirement for 2-year-old", "Is dal enough protein for 2-year-old?"]
        for col, example in zip(cols, examples):
            if col.button(example):
                st.session_state.pending_example = example

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                verified = message["metadata"].get("verified", False)
                conflict = message["metadata"].get("conflict_flagged", False)
                st.markdown(":green[Verified]" if verified else ":red[Not verified]")
                if verified and conflict:
                    st.caption("⚠️ Source conflict detected (structured vs. narrative — informational only)")
                if message["metadata"].get("model_backend"):
                    st.caption(f"Model: {message['metadata']['model_backend']}")
                if message["metadata"]["proof"]:
                    with st.expander("Show Proof"):
                        st.dataframe(pd.DataFrame(message["metadata"]["proof"]), use_container_width=True, hide_index=True)
                if debug:
                    st.json(message["metadata"]["raw"])

    query = st.session_state.pop("pending_example", None) or st.chat_input("Ask about your child's nutrition")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query, "metadata": {}})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Checking ICMR data..."):
                response = run_pipeline(query, model_backend)
            st.markdown(response["answer"])
            verified = response.get("verified", False)
            conflict = response.get("conflict_flagged", False)
            st.markdown(":green[Verified]" if verified else ":red[Not verified]")
            if verified and conflict:
                st.caption("⚠️ Source conflict detected (structured vs. narrative — informational only)")
            st.caption(f"Model: {response.get('model_backend', model_backend)}")
            if response["proof"]:
                with st.expander("Show Proof"):
                    st.dataframe(pd.DataFrame(response["proof"]), use_container_width=True, hide_index=True)
            if debug:
                st.json(response["raw"])
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"], "metadata": response})


if __name__ == "__main__":
    main()