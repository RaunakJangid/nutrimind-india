#!/usr/bin/env python
"""Validate semantic search queries and audit RDA iron discrepancy."""

import sys
import json
from pathlib import Path
import fitz  # PyMuPDF

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.retriever import semantic_search

RDA_PDF_PATH = Path(r"C:\Users\rauna\Desktop\Baby RAG\RDA 2020.pdf")

def run_semantic_queries():
    """Run 4 semantic search queries and validate results."""
    queries = [
        "iron requirement for toddlers",
        "breastfeeding duration recommendation",
        "balanced diet for pregnant woman",
        "vitamin D sunlight",
    ]

    print("=" * 80)
    print("PART 1: SEMANTIC SEARCH VALIDATION")
    print("=" * 80)

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] '{query}'")
        results = semantic_search(query, top_k=3)
        
        for rank, result in enumerate(results[:2], 1):
            chunk_id = result.get("chunk_id", "N/A")
            source = result.get("source", "N/A")
            page = result.get("page", "N/A")
            text_preview = result.get("text", "")[:100].replace("\n", " ")
            score = result.get("score", "N/A")
            print(f"  Top-{rank}: [{chunk_id}] source={source} page={page} score={score:.4f}")
            print(f"    Text: {text_preview}...")
        
        # If top result may not be topically relevant, show top-3 for diagnosis
        if results and results[0].get("score", 0) < 0.5:
            print(f"  ⚠️  WARNING: Top score < 0.5, showing top-3 for diagnosis:")
            for rank, result in enumerate(results[:3], 1):
                chunk_id = result.get("chunk_id", "N/A")
                score = result.get("score", "N/A")
                text_preview = result.get("text", "")[:80].replace("\n", " ")
                print(f"    [{rank}] {chunk_id} (score={score:.4f}): {text_preview}...")


def extract_rda_table_page_16():
    """Extract RDA table from page 16 of RDA_2020.pdf."""
    if not RDA_PDF_PATH.exists():
        print(f"\n❌ PDF not found: {RDA_PDF_PATH}")
        return None

    doc = fitz.open(RDA_PDF_PATH)
    page = doc[15]  # 0-indexed, so page 16 is index 15
    text = page.get_text("text")
    
    print("\n" + "=" * 80)
    print("PART 2: RDA IRON VALUE AUDIT")
    print("=" * 80)
    print(f"\nPage 16 text (first 2000 chars):")
    print(text[:2000])
    
    # Find the 1-3 years row and Iron column
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "1-3" in line or "1-3 years" in line:
            print(f"\n  Line {i}: {line}")
    
    return text


def check_current_rda_value():
    """Check what get_rda() returns for iron, 1-3_years."""
    from core.retriever import get_rda
    
    try:
        result = get_rda("iron", "1-3_years")
        print(f"\nCurrent get_rda('iron', '1-3_years'):")
        print(f"  Value: {result['value']} {result['unit']}")
        print(f"  Source: {result['source']}")
    except Exception as e:
        print(f"\n❌ Error calling get_rda: {e}")


def main():
    run_semantic_queries()
    extract_rda_table_page_16()
    check_current_rda_value()


if __name__ == "__main__":
    main()
