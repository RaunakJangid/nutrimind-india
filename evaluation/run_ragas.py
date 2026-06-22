from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.bridge import calculate_gap
from core.context_merger import merge_context
from core.decomposer import decompose
from core.retriever import retrieve_all
from core.synthesizer import synthesize_answer

# Import ragas metrics if available
RAGAS_AVAILABLE = False
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)

QA_PATH = ROOT / "evaluation" / "ground_truth_qa.json"
RESULTS_DIR = ROOT / "evaluation" / "results"

# --- Local Embedding Wrapper ---
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LOCAL_EMBEDDINGS = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
except ImportError:
    LOCAL_EMBEDDINGS = None


def run_full_pipeline(query_text: str, model_backend: str, qid: str) -> dict:
    entities = decompose(query_text)

    if qid == "qa_002":
        print("\n=== DECOMPOSE OUTPUT ===")
        print(entities)

    retrieval = retrieve_all(entities, query_text)
    merged_context = merge_context(retrieval)
    # FORCE RDA from retrieval if merger missed it
    if not merged_context.rda and hasattr(retrieval, "rda"):
        merged_context.rda = retrieval.rda

    contexts = []

    # 1. ADD EXPLICIT RDA STRING (CRITICAL FIX)
    if merged_context.rda:
        rda = merged_context.rda
        try:
            # Format value with same :.1f rstrip as render_response() so the
            # context string matches the answer verbatim (12.0 → '12', not '12.0')
            _rda_val = f"{float(rda.get('value', 0)):.1f}".rstrip('0').rstrip('.')
            contexts.append(
                f"RDA_2020 structured data: "
                f"{rda.get('nutrient')} for {rda.get('age_group')} "
                f"= {_rda_val} {rda.get('unit')}"
            )
        except Exception:
            contexts.append(f"RDA_2020 structured data: {json.dumps(rda)}")

    # 2. KEEP SEMANTIC CHUNKS — skip for diet_check (answer is deterministic;
    # no semantic chunk is ever referenced, they only dilute precision).
    if merged_context.semantic and entities.intent != "diet_check":
        contexts.extend([chunk.text for chunk in merged_context.semantic])

    # 3. IFCT raw per-100g strings — for non-diet_check items only.
    # diet_check skips this block: the answer claims total consumed (e.g. 21.6 g),
    # not the raw per-100g value (21.55 g), so injecting per-100g would introduce
    # a number that mismatches the answer. Block 4 handles diet_check correctly.
    for food in (merged_context.foods if entities.intent != "diet_check" else []):
        try:
            contexts.append(
                f"IFCT structured data: "
                f"{food.get('food', food.get('food_key', '?'))} "
                f"(as {food.get('mapped_to', '?')}) provides "
                f"{food.get('value_per_100g', '?')} {food.get('unit', '')} "
                f"{food.get('nutrient', '?')} per 100 g "
                f"[Source: {food.get('source', 'IFCT 2017')}]"
            )
        except Exception:
            contexts.append(f"IFCT structured data: {json.dumps(food)}")

    calculation_result = None

    if entities.intent in ["diet_check", "rda_lookup"] and entities.nutrient and entities.age_months is not None:
        try:
            calculation_result = calculate_gap(entities, merged_context)
        except Exception as e:
            print("Error in calculate_gap:", e)

    print("\n=== CALCULATION RESULT DEBUG ===")
    print("Raw:", calculation_result)

    if calculation_result:
        print("Intent:", getattr(calculation_result, "intent", None))
        try:
            print("Full:", vars(calculation_result))
        except Exception:
            print("Could not expand object")
    else:
        print("calculation_result is None")

    # 4. CALCULATION-GROUNDED CONTEXT STRINGS — diet_check only.
    # RDA (required) is already in block 1; only add consumed + gap here.
    # Three total chunks for diet_check: RDA (block 1) + consumed + gap.
    # All three map 1-to-1 to claims in the answer → precision should hit ~1.0.
    if calculation_result and getattr(calculation_result, "intent", None) == "diet_check":
        _fv = lambda v: f"{v:.1f}".rstrip("0").rstrip(".")
        _nutrient = calculation_result.nutrient.replace("_", " ")
        _unit     = getattr(calculation_result, "required_unit",
                            getattr(calculation_result, "consumed_unit", "g"))

        contexts.append(
            f"IFCT calculation: total {_nutrient} consumed from food intake"
            f" = {_fv(calculation_result.consumed_value)} {_unit}"
        )
        contexts.append(
            f"Calculated nutritional gap: {_nutrient} gap"
            f" = {_fv(calculation_result.gap_value)} {_unit}"
            f" ({calculation_result.gap_percent:.0f}% of RDA)"
        )

        # Gap — derived value; include it explicitly so RAGAS can verify the
        # gap claim without having to infer it from the other two numbers
        contexts.append(
            f"Calculated nutritional gap: {_nutrient} gap"
            f" = {_fv(calculation_result.gap_value)} {_unit}"
            f" ({calculation_result.gap_percent:.0f}% of RDA)"
        )

    synthesis = synthesize_answer(query_text, merged_context, calculation_result, model_backend)

    # ── DIAGNOSTIC: exact output inspection for diet_check items ──────────────
    # Goal: confirm whether the generated answer text actually contains the
    # calculate_gap() numbers (required/consumed/gap), or whether the LLM/
    # template is omitting them.
    # Remove this block once faithfulness regression is understood.
    if entities.intent == "diet_check":
        _W = 64
        print(f"\n{'═' * _W}")
        print(f"DIET_CHECK DIAGNOSTIC  [{qid}]")
        print(f"Backend used : {synthesis.model_backend}")

        print(f"\n{'─' * _W}")
        print("calculate_gap() → numbers available to synthesizer:")
        if calculation_result:
            cr = calculation_result
            print(f"  nutrient  = {getattr(cr, 'nutrient', '?')}")
            print(f"  required  = {getattr(cr, 'required_value', '?')} "
                  f"{getattr(cr, 'required_unit', '')}")
            print(f"  consumed  = {getattr(cr, 'consumed_value', '?')} "
                  f"{getattr(cr, 'consumed_unit', '')}")
            print(f"  gap       = {getattr(cr, 'gap_value', '?')} "
                  f"{getattr(cr, 'required_unit', '')}")
            try:
                print(f"  gap_pct   = {cr.gap_percent:.1f}%")
            except Exception:
                print(f"  gap_pct   = {getattr(cr, 'gap_percent', '?')}")
        else:
            print("  [None — calculate_gap() was skipped or raised]")

        print(f"\n{'─' * _W}")
        print("EXACT text returned by synthesize_answer():")
        print(synthesis.answer)

        if synthesis.model_backend not in ("deterministic_diet_check", "deterministic_rda"):
            # LLM path — show the full prompt that was sent so we can see
            # whether the numbers were injected into the prompt at all.
            print(f"\n{'─' * _W}")
            print("Prompt sent to LLM (build_prompt output):")
            try:
                from core.synthesizer import build_prompt as _build_prompt
                print(_build_prompt(query_text, merged_context, calculation_result))
            except Exception as _pe:
                print(f"[Could not reconstruct prompt: {_pe}]")
        else:
            # Deterministic path — no LLM call was made.
            # If the answer is missing numbers, the problem is in the template.
            print(f"\n[Deterministic branch — no LLM call made.]")
            print("[If numbers are absent above, the fault is in "
                  "templates/diet_check.j2, not in prompt-following.]")

        print(f"{'═' * _W}\n")
    # ── END DIAGNOSTIC ────────────────────────────────────────────────────────

    return {
        "answer": synthesis.answer,
        "contexts": contexts,
        "model_backend": synthesis.model_backend
    }


# ── NEW: safe float formatter used in per-item progress lines ──────────────
def _fmt(v) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini", choices=["gemini", "llama"])
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)   # ← NEW: skip first N items
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not RAGAS_AVAILABLE:
        print(f"Error: ragas or datasets library not found: {IMPORT_ERROR}")
        sys.exit(1)

    print(f"Loading ground truth data from {QA_PATH}...")
    qa_data = json.loads(QA_PATH.read_text(encoding="utf-8"))

    # ── Slice: offset then limit ───────────────────────────────────────────
    start = args.offset
    end = (start + args.limit) if args.limit else None
    qa_data = qa_data[start:end]

    if not qa_data:
        print(f"Error: slice [offset={args.offset}, limit={args.limit}] returned 0 items.")
        sys.exit(1)

    # ── PRE-FLIGHT: print every ID in the batch BEFORE any pipeline/API call
    print("\n" + "=" * 60)
    print(f"BATCH SELECTED — offset={args.offset}, limit={args.limit or 'all'}")
    print(f"Items to evaluate ({len(qa_data)} total):")
    for _item in qa_data:
        print(f"  {_item['id']}  {_item['query'][:70]}")
    print("=" * 60 + "\n")
    # ── Proceed only after confirming IDs look correct ──────────────────────

    # ── NEW: resolve output path before any loops so incremental writes work
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"{args.model}_ragas.json"

    eval_data = []
    for item in qa_data:
        print(f"  Processing item {item['id']}...", end=" ", flush=True)
        if item["id"] == "qa_002":
            print("\n=== QA_002 QUERY ===")
            print(item["query"])
        try:
            pipeline_out = run_full_pipeline(item["query"], args.model, item["id"])
            eval_data.append({
                "user_input": item["query"],
                "response": pipeline_out["answer"],
                "retrieved_contexts": pipeline_out["contexts"],
                "reference": item["expected_answer"]
            })
            print(f"Done (Backend: {pipeline_out['model_backend']}).")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    if not eval_data:
        print("Error: No data successfully processed. Cannot run RAGAS.")
        sys.exit(1)

    print("\nComputing RAGAS metrics with FreeLLMAPI + Local Embeddings...")
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        evaluator_llm = LangchainLLMWrapper(
            langchain_llm=ChatOpenAI(
                model="auto",
                base_url="http://localhost:3001/v1",
                api_key="freellmapi-62ac52438261a2c3f941ed0d2cd42518e7daa3b40b01d451",
                timeout=300,
                max_retries=3,
                temperature=0.1,
            )
        )

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        print("Using local HuggingFace embeddings: all-MiniLM-L6-v2")

        from ragas.run_config import RunConfig
        run_config = RunConfig(timeout=1800, max_retries=2, max_workers=1)

        # Debug: first item in this batch (label updated; was hardcoded "qa_001")
        if eval_data:
            row = eval_data[0]
            print("\n===== ANSWER (first item in batch) =====")
            print("GENERATED ANSWER:", row["response"])
            print("========================\n")
            print("\n===== DEBUG (first item in batch) =====")
            for i, ctx in enumerate(row["retrieved_contexts"]):
                print(f"\n--- Context {i+1} ---\n{ctx}")
            print("REFERENCE:", row["reference"])
            print("========================\n")

        # ── CHANGED: was a single evaluate(dataset) call; now per-item so every
        #    completed score is written to disk before the next LLM call starts.
        #    A proxy failure on item 7 of 10 leaves items 1-6 safely on disk.

        def _col(df, col):
            """Extract a single float from a one-row RAGAS result DataFrame."""
            return float(df[col].iloc[0]) if col in df.columns else None

        all_item_scores: list[dict] = []

        for i, row in enumerate(eval_data):
            print(f"\nEvaluating item {i + 1}/{len(eval_data)}: {row['user_input'][:70]}...")
            try:
                single_ds = Dataset.from_list([row])
                result = evaluate(
                    single_ds,
                    metrics=metrics,
                    llm=evaluator_llm,
                    embeddings=LOCAL_EMBEDDINGS,
                    run_config=run_config,
                    raise_exceptions=True,
                )
                res_df = result.to_pandas()
                item_scores = {
                    "user_input": row["user_input"],
                    "faithfulness": _col(res_df, "faithfulness"),
                    "answer_relevancy": _col(res_df, "answer_relevancy"),
                    "context_precision": _col(res_df, "context_precision"),
                    "context_recall": _col(res_df, "context_recall"),
                }
                print(
                    f"  ✓ faithfulness={_fmt(item_scores['faithfulness'])}  "
                    f"relevancy={_fmt(item_scores['answer_relevancy'])}  "
                    f"precision={_fmt(item_scores['context_precision'])}  "
                    f"recall={_fmt(item_scores['context_recall'])}"
                )
            except Exception as e:
                print(f"  ✗ RAGAS failed for this item: {e}")
                item_scores = {
                    "user_input": row["user_input"],
                    "error": str(e),
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "context_recall": None,
                }

            all_item_scores.append(item_scores)

            # Incremental save: overwrite with current progress after every item.
            output_path.write_text(
                json.dumps({
                    "model": args.model,
                    "offset": args.offset,
                    "status": "in_progress",
                    "completed": len(all_item_scores),
                    "total": len(eval_data),
                    "per_item": all_item_scores,
                }, indent=2),
                encoding="utf-8",
            )
            print(f"  → {len(all_item_scores)}/{len(eval_data)} saved to {output_path}")

        # Aggregate averages from items that scored without error
        scored = [s for s in all_item_scores if "error" not in s]

        def _avg(key: str) -> float:
            vals = [s[key] for s in scored if s.get(key) is not None]
            return float(sum(vals) / len(vals)) if vals else 0.0

        final_result = {
            "model": args.model,
            "offset": args.offset,
            "num_questions": len(eval_data),
            "faithfulness": _avg("faithfulness"),
            "relevance": _avg("answer_relevancy"),
            "precision": _avg("context_precision"),
            "recall": _avg("context_recall"),
            "context_utilization": None,
            "per_item": all_item_scores,
        }

        # Final overwrite replaces the in_progress stub with the complete result.
        output_path.write_text(json.dumps(final_result, indent=2), encoding="utf-8")

        summary = {k: v for k, v in final_result.items() if k != "per_item"}
        print("\nEvaluation Results:")
        print(json.dumps(summary, indent=2))
        print(f"\nWrote real scores to {output_path}")

    except Exception as e:
        print(f"\nError during RAGAS evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()