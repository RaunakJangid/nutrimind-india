from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.bridge import calculate_gap
from core.context_merger import merge_context
from core.decomposer import decompose
from core.models import MergedContext
from core.retriever import retrieve_all
from core.synthesizer import synthesize_answer
from core.verifier import verify as run_verify

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
except ImportError as exc:
    IMPORT_ERROR = str(exc)

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LOCAL_EMBEDDINGS = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
except ImportError:
    LOCAL_EMBEDDINGS = None

QA_PATH     = ROOT / "evaluation" / "ground_truth_qa.json"
RESULTS_DIR = ROOT / "evaluation" / "results"

VARIANTS = ["full", "no_semantic", "no_structured", "no_merger", "no_verifier"]

# What each variant disables — printed in the header so runs are self-documenting
VARIANT_DESC = {
    "full":          "complete pipeline (baseline)",
    "no_semantic":   "FAISS/semantic_search() disabled — structured lookups only",
    "no_structured": "get_rda()/get_food_nutrients() disabled — semantic chunks only",
    "no_merger":     "context_merger.merge() bypassed — raw retrieval passthrough",
    "no_verifier":   "verifier.verify() skipped — answer returned without math check",
}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    return f"{v:.3f}" if v is not None else "  N/A"


def _col(df, col):
    return float(df[col].iloc[0]) if col in df.columns else None


def _avg(items: list[dict], key: str) -> float | None:
    vals = [s[key] for s in items if s.get(key) is not None]
    return float(sum(vals) / len(vals)) if vals else None


# ── Pipeline variant ───────────────────────────────────────────────────────────

def run_pipeline_variant(
    query_text: str,
    model_backend: str,
    qid: str,
    variant: str,
) -> dict:
    """
    Single pipeline that implements all 5 variants via a parameter.

    Variant     What changes
    ----------  ---------------------------------------------------------------
    full        Nothing — identical to run_ragas.py's run_full_pipeline()
    no_semantic retrieval.semantic cleared before merging; blocks 2 skipped
    no_structured
                retrieval.rda + retrieval.foods cleared; blocks 1/3/4 skipped
    no_merger   MergedContext built directly from retrieval (no conflict
                detection, no confidence scoring, no context_text formatting)
    no_verifier verify() call omitted; raw synthesis answer returned as-is
    """
    entities  = decompose(query_text)
    retrieval = retrieve_all(entities, query_text)

    # ── Variant: disable component at retrieval level (before merge) ───────────
    if variant == "no_semantic":
        retrieval.semantic = []

    elif variant == "no_structured":
        retrieval.rda   = None
        retrieval.foods = []

    # ── Merge (or bypass) ─────────────────────────────────────────────────────
    if variant == "no_merger":
        # Skip ContextMerger entirely — raw passthrough with no conflict detection,
        # no priority weighting, no formatted context_text for the verifier.
        _ctx_lines: list[str] = []
        if retrieval.rda:
            _ctx_lines.append(f"RDA: {retrieval.rda}")
        for _food in retrieval.foods:
            _ctx_lines.append(f"IFCT: {_food}")
        for _chunk in retrieval.semantic:
            _ctx_lines.append(_chunk.text)
        merged_context = MergedContext(
            rda=retrieval.rda,
            foods=retrieval.foods,
            semantic=retrieval.semantic,
            conflicts=[],
            context_text="\n".join(_ctx_lines),
            confidence=1.0,
        )
    else:
        merged_context = merge_context(retrieval)
        # Guard: force RDA if merger dropped it
        if not merged_context.rda and getattr(retrieval, "rda", None):
            merged_context.rda = retrieval.rda

    # ── Build contexts (RAGAS retrieved_contexts) ──────────────────────────────
    # Logic mirrors run_ragas.py exactly for the `full` variant.
    # Each variant flag gates the appropriate block(s).
    contexts: list[str] = []

    # Block 1: RDA structured string
    # Skipped for no_structured (retrieval.rda is already None).
    if merged_context.rda:
        rda = merged_context.rda
        try:
            _rda_val = f"{float(rda.get('value', 0)):.1f}".rstrip("0").rstrip(".")
            contexts.append(
                f"RDA_2020 structured data: "
                f"{rda.get('nutrient')} for {rda.get('age_group')} "
                f"= {_rda_val} {rda.get('unit')}"
            )
        except Exception:
            contexts.append(f"RDA_2020 structured data: {json.dumps(rda)}")

    # Block 2: Semantic chunks
    # Skipped for no_semantic (retrieval.semantic is already []).
    # Also skipped for diet_check on all variants (block 4 handles that intent).
    if merged_context.semantic and entities.intent != "diet_check":
        contexts.extend([chunk.text for chunk in merged_context.semantic])

    # Block 3: IFCT raw per-100g strings (non-diet_check only)
    # Skipped for no_structured (retrieval.foods is already []).
    # Skipped for diet_check on all variants (block 4 uses calculation values).
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

    # ── calculate_gap ──────────────────────────────────────────────────────────
    calculation_result = None
    if (entities.intent in ("diet_check", "rda_lookup")
            and entities.nutrient
            and entities.age_months is not None):
        try:
            calculation_result = calculate_gap(entities, merged_context)
        except Exception as exc:
            print(f"    [calculate_gap error: {exc}]")

    # Block 4: Calculation-grounded context strings — diet_check only.
    # Skipped for no_structured: no RDA or IFCT data means no valid calculation,
    # and injecting these strings would fabricate grounding that doesn't exist.
    if (calculation_result
            and getattr(calculation_result, "intent", None) == "diet_check"
            and variant != "no_structured"):
        _fv       = lambda v: f"{v:.1f}".rstrip("0").rstrip(".")
        _nutrient = calculation_result.nutrient.replace("_", " ")
        _unit     = getattr(calculation_result, "required_unit",
                            getattr(calculation_result, "consumed_unit", "g"))
        _age_grp  = getattr(calculation_result, "age_group", "")

        contexts.append(
            f"RDA_2020 structured data: {_nutrient} requirement"
            f"{' for ' + _age_grp if _age_grp else ''} "
            f"= {_fv(calculation_result.required_value)} {_unit}"
        )
        contexts.append(
            f"IFCT calculation: total {_nutrient} consumed from food intake"
            f" = {_fv(calculation_result.consumed_value)} {_unit}"
        )
        contexts.append(
            f"Calculated nutritional gap: {_nutrient} gap"
            f" = {_fv(calculation_result.gap_value)} {_unit}"
            f" ({calculation_result.gap_percent:.0f}% of RDA)"
        )

    # ── Synthesize ─────────────────────────────────────────────────────────────
    synthesis = synthesize_answer(query_text, merged_context, calculation_result, model_backend)

    # ── Verify (skipped for no_verifier) ──────────────────────────────────────
    verification_passed: bool | None = None

    if variant == "no_verifier":
        # Intentional skip — N/A in results is correct, not a silent crash.
        # Confirmed path: this branch is taken, verification_passed stays None.
        if calculation_result and entities.intent == "diet_check":
            print(f"    [no_verifier] verify() intentionally skipped for {qid}")

    elif calculation_result:
        try:
            import traceback as _tb
            vr = run_verify(calculation_result, synthesis.answer, merged_context)
            verification_passed = vr.verified

            # ── DIAGNOSTIC: show per-check breakdown for diet_check items ────────
            # Runs on every diet_check item so we can see WHICH check fails and WHY.
            # Remove once the 0%-pass-rate contradiction is resolved.
            if entities.intent == "diet_check":
                _W = 56
                print("\n    " + "─"*_W)
                print(f"    VERIFY [{qid}]  overall={'PASS' if vr.verified else 'FAIL'}")

                # Show food_details so we can check if it's empty (likely math_check culprit)
                _fd = getattr(calculation_result, "food_details", None)
                print(f"    food_details present: {_fd is not None}, "
                      f"len: {len(_fd) if _fd is not None else 'N/A'}")
                if _fd:
                    for _d in _fd:
                        print(f"      detail: mapped_to={getattr(_d,'mapped_to','?')}  "
                              f"value_per_100g={getattr(_d,'value_per_100g','?')}  "
                              f"nutrient_amount={getattr(_d,'nutrient_amount','?')}")

                print(f"    calculation_result fields:")
                print(f"      required_value={getattr(calculation_result,'required_value','?')}")
                print(f"      consumed_value={getattr(calculation_result,'consumed_value','?')}")
                print(f"      gap_value     ={getattr(calculation_result,'gap_value','?')}")
                print(f"      gap_percent   ={getattr(calculation_result,'gap_percent','?')}")
                print(f"      conflicts     ={getattr(calculation_result,'conflicts',[])}")

                for chk in vr.checks:
                    icon = "✓" if chk.passed else "✗"
                    print(f"      {icon} {chk.name:<22} {chk.detail}")

                if vr.fail_reason:
                    print(f"    FAIL REASON: {vr.fail_reason}")
                print(f"    {'─'*_W}")

        except Exception as exc:
            # If verify() itself crashes, print the full traceback so we know
            # this is a bug (not an intentional skip like no_verifier).
            print(f"    [verify EXCEPTION for {qid}]: {exc}")
            _tb.print_exc()

    return {
        "answer":               synthesis.answer,
        "contexts":             contexts,
        "model_backend":        synthesis.model_backend,
        "verification_passed":  verification_passed,
    }


# ── Per-variant RAGAS evaluation ───────────────────────────────────────────────

def evaluate_variant(
    variant:       str,
    qa_data:       list[dict],
    model_backend: str,
    evaluator_llm,
    run_config,
    metrics:       list,
    output_path:   Path,
) -> dict:
    """Run pipeline + RAGAS evaluation for one variant. Returns summary dict."""
    _W = 64
    print(f"\n{'═' * _W}")
    print(f"VARIANT : {variant.upper()}")
    print(f"DESC    : {VARIANT_DESC[variant]}")
    print(f"ITEMS   : {len(qa_data)}")
    print(f"{'═' * _W}")

    # ── Pipeline pass ──────────────────────────────────────────────────────────
    eval_data: list[dict] = []
    for item in qa_data:
        print(f"  [{variant}] {item['id']}...", end=" ", flush=True)
        try:
            out = run_pipeline_variant(
                item["query"], model_backend, item["id"], variant
            )
            eval_data.append({
                "user_input":          item["query"],
                "response":            out["answer"],
                "retrieved_contexts":  out["contexts"],
                "reference":           item["expected_answer"],
                # Private fields (stripped before RAGAS, kept for reporting)
                "_backend":            out["model_backend"],
                "_verification_passed": out["verification_passed"],
            })
            print(f"ok  [{out['model_backend']}]")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not eval_data:
        result = {"variant": variant, "num_questions": 0, "error": "all items failed"}
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    # ── RAGAS per-item ─────────────────────────────────────────────────────────
    print(f"\n  Scoring {len(eval_data)} items with RAGAS...")
    all_item_scores: list[dict] = []

    for i, row in enumerate(eval_data):
        ragas_row = {k: v for k, v in row.items() if not k.startswith("_")}
        print(
            f"    [{i+1:>2}/{len(eval_data)}] {row['user_input'][:50]!r:<52}",
            end=" ",
            flush=True,
        )
        try:
            result = evaluate(
                Dataset.from_list([ragas_row]),
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=LOCAL_EMBEDDINGS,
                run_config=run_config,
                raise_exceptions=True,
            )
            res_df = result.to_pandas()
            item_scores: dict = {
                "user_input":          row["user_input"],
                "variant":             variant,
                "backend":             row["_backend"],
                "verification_passed": row["_verification_passed"],
                "faithfulness":        _col(res_df, "faithfulness"),
                "answer_relevancy":    _col(res_df, "answer_relevancy"),
                "context_precision":   _col(res_df, "context_precision"),
                "context_recall":      _col(res_df, "context_recall"),
            }
            print(
                f"F={_fmt(item_scores['faithfulness'])} "
                f"P={_fmt(item_scores['context_precision'])} "
                f"R={_fmt(item_scores['context_recall'])}"
            )
        except Exception as exc:
            print(f"RAGAS error: {exc}")
            item_scores = {
                "user_input":          row["user_input"],
                "variant":             variant,
                "error":               str(exc),
                "faithfulness":        None,
                "answer_relevancy":    None,
                "context_precision":   None,
                "context_recall":      None,
                "verification_passed": row["_verification_passed"],
            }

        all_item_scores.append(item_scores)

        # Incremental save after every item — crash-safe
        output_path.write_text(
            json.dumps({
                "variant":    variant,
                "status":     "in_progress",
                "completed":  len(all_item_scores),
                "total":      len(eval_data),
                "per_item":   all_item_scores,
            }, indent=2),
            encoding="utf-8",
        )

    # ── Aggregate ──────────────────────────────────────────────────────────────
    scored = [s for s in all_item_scores if "error" not in s]

    verified_items = [s for s in all_item_scores if s.get("verification_passed") is not None]
    verify_rate = (
        sum(1 for s in verified_items if s["verification_passed"]) / len(verified_items)
        if verified_items else None
    )

    summary = {
        "variant":               variant,
        "description":           VARIANT_DESC[variant],
        "status":                "complete",
        "num_questions":         len(eval_data),
        "faithfulness":          _avg(scored, "faithfulness"),
        "relevance":             _avg(scored, "answer_relevancy"),
        "precision":             _avg(scored, "context_precision"),
        "recall":                _avg(scored, "context_recall"),
        "verification_pass_rate": verify_rate,
        "per_item":              all_item_scores,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  ✓ Saved {output_path.name}  "
          f"(F={_fmt(summary['faithfulness'])} "
          f"P={_fmt(summary['precision'])} "
          f"R={_fmt(summary['recall'])})")
    return summary


# ── Comparison table ───────────────────────────────────────────────────────────

def build_comparison(results_dir: Path) -> dict:
    """Scan completed ablation_*.json files and return comparison dict."""
    comparison: dict = {}
    for variant in VARIANTS:
        path = results_dir / f"ablation_{variant}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") != "complete":
            continue
        comparison[variant] = {
            "faithfulness":           data.get("faithfulness"),
            "relevance":              data.get("relevance"),
            "precision":              data.get("precision"),
            "recall":                 data.get("recall"),
            "verification_pass_rate": data.get("verification_pass_rate"),
            "num_questions":          data.get("num_questions"),
        }
    return comparison


def print_comparison_table(comparison: dict) -> None:
    if not comparison:
        print("  (no completed variants to compare yet)")
        return

    col_w = 9
    header = (
        f"{'Variant':<18}"
        f"{'Faith':>{col_w}}"
        f"{'Relev':>{col_w}}"
        f"{'Prec':>{col_w}}"
        f"{'Recall':>{col_w}}"
        f"{'Verif%':>{col_w}}"
        f"{'N':>5}"
    )
    sep = "═" * len(header)

    print(f"\n{sep}")
    print("ABLATION COMPARISON")
    print(sep)
    print(header)
    print("─" * len(header))

    for variant in VARIANTS:
        if variant not in comparison:
            continue
        r   = comparison[variant]
        vp  = (f"{r['verification_pass_rate']:.0%}"
               if r.get("verification_pass_rate") is not None else "  N/A")
        n   = r.get("num_questions", "?")
        print(
            f"{variant:<18}"
            f"{_fmt(r.get('faithfulness')):>{col_w}}"
            f"{_fmt(r.get('relevance')):>{col_w}}"
            f"{_fmt(r.get('precision')):>{col_w}}"
            f"{_fmt(r.get('recall')):>{col_w}}"
            f"{vp:>{col_w}}"
            f"{n:>5}"
        )

    print(sep)
    print("Δ from full (lower = component matters more):")
    print("─" * len(header))

    full = comparison.get("full", {})
    for variant in VARIANTS:
        if variant == "full" or variant not in comparison:
            continue
        r = comparison[variant]

        def delta(key: str) -> str:
            base = full.get(key)
            cur  = r.get(key)
            if base is None or cur is None:
                return f"{'—':>{col_w}}"
            d    = cur - base
            return f"{d:>+{col_w}.3f}"

        print(
            f"{variant:<18}"
            f"{delta('faithfulness')}"
            f"{delta('relevance')}"
            f"{delta('precision')}"
            f"{delta('recall')}"
            f"{'—':>{col_w}}"
            f"{'':>5}"
        )
    print(sep)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation evaluation — run the full pipeline with one component "
            "disabled at a time and measure RAGAS metric impact."
        )
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS + ["all"],
        default="full",
        help=(
            "Component to disable. 'all' runs every variant sequentially "
            "(caution: 5 × limit × RAGAS calls)."
        ),
    )
    parser.add_argument("--model",  default="gemini", choices=["gemini", "llama"])
    parser.add_argument("--limit",  type=int, default=10,
                        help="Max items to evaluate (default 10; 0 = no limit).")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N items in ground_truth_qa.json.")
    args = parser.parse_args()

    if not RAGAS_AVAILABLE:
        print(f"Error: ragas/datasets not available — {IMPORT_ERROR}")
        sys.exit(1)

    # ── Load and slice QA data ─────────────────────────────────────────────────
    print(f"Loading {QA_PATH}...")
    qa_data = json.loads(QA_PATH.read_text(encoding="utf-8"))

    start   = args.offset
    end     = (start + args.limit) if args.limit else None
    qa_data = qa_data[start:end]

    if not qa_data:
        print(f"Error: slice [offset={args.offset}, limit={args.limit}] returned 0 items.")
        sys.exit(1)

    print(f"\nBatch: items {start+1}–{start+len(qa_data)} "
          f"(offset={args.offset}, limit={args.limit or 'all'})")
    print("IDs:", [item["id"] for item in qa_data])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── RAGAS infrastructure (shared across variants) ──────────────────────────
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

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
    ragas_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    run_config    = RunConfig(timeout=1800, max_retries=2, max_workers=1)

    # ── Run variant(s) ─────────────────────────────────────────────────────────
    variants_to_run = VARIANTS if args.variant == "all" else [args.variant]

    for variant in variants_to_run:
        evaluate_variant(
            variant       = variant,
            qa_data       = qa_data,
            model_backend = args.model,
            evaluator_llm = evaluator_llm,
            run_config    = run_config,
            metrics       = ragas_metrics,
            output_path   = RESULTS_DIR / f"ablation_{variant}.json",
        )

    # ── Comparison table (rebuilt from all completed result files) ─────────────
    comparison = build_comparison(RESULTS_DIR)
    print_comparison_table(comparison)

    if comparison:
        comp_path = RESULTS_DIR / "ablation_comparison.json"
        comp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"\nComparison saved → {comp_path}")


if __name__ == "__main__":
    main()