#!/usr/bin/env python
"""Extract complete RDA/EAR tables from RDA_2020.pdf pages 15-16."""

import fitz
import json
from pathlib import Path

RDA_PDF_PATH = Path(r"C:\Users\rauna\Desktop\Baby RAG\RDA 2020.pdf")

def extract_rda_tables():
    """Extract EAR (page 15) and RDA (page 16) tables from PDF."""
    
    if not RDA_PDF_PATH.exists():
        print(f"ERROR: PDF not found at {RDA_PDF_PATH}")
        return None
    
    doc = fitz.open(RDA_PDF_PATH)
    
    # Page 15 = index 14 (EAR table)
    # Page 16 = index 15 (RDA table)
    
    print("=" * 100)
    print("PAGE 15 - SUMMARY OF EAR FOR INDIANS")
    print("=" * 100)
    page_15_text = doc[14].get_text("text")
    print(page_15_text)
    
    print("\n" + "=" * 100)
    print("PAGE 16 - SUMMARY OF RDA FOR INDIANS")
    print("=" * 100)
    page_16_text = doc[15].get_text("text")
    print(page_16_text)
    
    # Save to files for manual inspection
    with open("page_15_ear.txt", "w", encoding="utf-8") as f:
        f.write(page_15_text)
    with open("page_16_rda.txt", "w", encoding="utf-8") as f:
        f.write(page_16_text)
    
    print("\n" + "=" * 100)
    print("Raw text saved to page_15_ear.txt and page_16_rda.txt for inspection")
    print("=" * 100)

if __name__ == "__main__":
    extract_rda_tables()
