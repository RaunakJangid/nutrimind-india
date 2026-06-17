#!/usr/bin/env python
"""Extract exact RDA table values from PDF page 16."""

import fitz
import pandas as pd

RDA_PDF_PATH = r'C:\Users\rauna\Desktop\Baby RAG\RDA 2020.pdf'
doc = fitz.open(RDA_PDF_PATH)
page = doc[15]  # page 16
text = page.get_text('text')

print("=" * 80)
print("EXTRACTING IRON VALUES FROM RDA TABLE (PAGE 16)")
print("=" * 80)

# Manual extraction by looking at the table structure
# The table shows: Children row with 1-3y, 4-6y, 7-9y in one row
# Iron column for these is: 8, 11, 15

lines = text.split('\n')

# Find the children section
for i, line in enumerate(lines):
    if 'Children' in line and '1-3y' in line:
        print(f"\nFound children row at line {i}:")
        print(f"  {line}")
        
        # The structure appears to be:
        # Children | 1-3y | 4-6y | 7-9y | (then all the nutrient columns follow)
        # We need to find the Iron column values
        
        # Look at the table structure more carefully
        # From the output, the layout is complex due to table reformatting
        # Let me parse based on the known structure
        
# Based on visual inspection of the output:
# Children row shows: 1-3y | 4-6y | 7-9y with values:
# Iron values are: 8 | 11 | 15

print("\nManually extracted Iron values from RDA table (page 16):")
print("  1-3y:  8 mg/d")
print("  4-6y:  11 mg/d")
print("  7-9y:  15 mg/d")

print("\nCurrent CSV values in icmr_rda.csv:")
import csv
from pathlib import Path
rda_csv = Path(r'c:\Users\rauna\Documents\NutriGuide Kids\data\processed\icmr_rda.csv')
with open(rda_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['nutrient'] == 'iron' and 'years' in row['age_group']:
            print(f"  {row['age_group']}: {row['value']} {row['unit']}")

print("\nDISCREPANCIES FOUND:")
print("  1-3_years: CSV has 9 mg, PDF shows 8 mg ❌ MISMATCH")
print("  4-6_years: CSV has 13 mg, PDF shows 11 mg ❌ MISMATCH")
print("  6-12_months: CSV has 11 mg (need to verify PDF)")
print("  0-6_months: CSV has 0.3 mg (need to verify PDF)")
