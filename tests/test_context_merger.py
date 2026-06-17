from core.context_merger import ContextMerger
from core.models import RetrievalResult, SemanticChunk


def test_merge_without_conflicts():
    retrieval = RetrievalResult(
        rda={"nutrient": "protein", "age_group": "1-3_years", "value": 13, "unit": "g", "source": "ICMR RDA", "confidence": 1.0},
        semantic=[SemanticChunk(chunk_id="a", text="Protein guidance for children.", source="ICMR-NIN", score=0.8)],
    )
    merged = ContextMerger().merge(retrieval)
    assert merged.conflicts == []
    assert "ICMR RDA" in merged.context_text


def test_structured_vs_semantic_conflict():
    retrieval = RetrievalResult(
        rda={"nutrient": "iron", "age_group": "1-3_years", "value": 9, "unit": "mg", "source": "ICMR RDA", "confidence": 1.0},
        semantic=[SemanticChunk(chunk_id="a", text="Iron for children is 20 mg daily.", source="ICMR-NIN", score=0.8)],
    )
    merged = ContextMerger().merge(retrieval)
    assert merged.conflicts
    assert "Structured RDA value is used" in merged.conflicts[0].resolution
