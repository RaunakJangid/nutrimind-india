#!/usr/bin/env python
"""
Parse RDA/EAR tables from PDF text and build complete CSV.
Handles complex merged cells and multi-row age groups.
Source: RDA_2020.pdf page 16 - ICMR-NIN 2020 Recommended Dietary Allowances
"""

import csv
from pathlib import Path

# RDA TABLE DATA - manually parsed from page 16
# Structure: (age_group_normalized, age_group_display, category_of_work, body_wt, protein, dietary_fiber, calcium, magnesium, iron, zinc, iodine, thiamine, riboflavin, niacin, vit_b6, folate, vit_b12, vit_c, vit_a, vit_d)

RDA_DATA = [
    # Men (using normalized age group format for retriever compatibility)
    ("adult_male_sedentary", "Men - Sedentary", "Sedentary", 65, 54.0, 30, 1000, 440, 19, 17, 140, 1.4, 2.0, 14, 1.9, 300, 2.2, 80, 1000, 600),
    ("adult_male_moderate", "Men - Moderate", "Moderate", 65, 54.0, 40, 1000, 440, 19, 17, 140, 1.8, 2.5, 18, 2.4, 300, 2.2, 80, 1000, 600),
    ("adult_male_heavy", "Men - Heavy", "Heavy", 65, 54.0, 50, 1000, 440, 19, 17, 140, 2.3, 3.2, 23, 3.1, 300, 2.2, 80, 1000, 600),
    
    # Women (using normalized age group format for retriever compatibility)
    ("adult_female_sedentary", "Women - Sedentary", "Sedentary", 55, 46.0, 25, 1000, 370, 29, 13.2, 140, 1.4, 1.9, 11, 1.9, 220, 2.2, 65, 840, 600),
    ("adult_female_moderate", "Women - Moderate", "Moderate", 55, 46.0, 30, 1000, 370, 29, 13.2, 140, 1.7, 2.4, 14, 1.9, 220, 2.2, 65, 840, 600),
    ("adult_female_heavy", "Women - Heavy", "Heavy", 55, 46.0, 40, 1000, 370, 29, 13.2, 140, 2.2, 3.1, 18, 2.4, 220, 2.2, 65, 840, 600),
    
    # Pregnant woman
    ("pregnant_woman", "Pregnant Woman", None, 55, 56.0, 35, 1000, 440, 27, 14.5, 220, 2.0, 2.7, 16, 2.3, 570, 2.45, 100, 900, 600),
    
    # Lactation 0-6m
    ("lactating_0-6m", "Lactation 0-6m", "0-6m", 55, 63.0, 28.6, 1200, 400, 23, 14.1, 280, 2.1, 3.0, 17, 2.9, 330, 3.2, 130, 950, 600),
    
    # Lactation 7-12m
    ("lactating_7-12m", "Lactation 7-12m", "7-12m", 55, 59.0, 22.6, 1200, 400, 23, 14.1, 280, 2.1, 3.0, 17, 2.9, 330, 3.2, 130, 950, 600),
    
    # Infants 0-6m (matches retriever format)
    ("0-6_months", "Infants 0-6m", None, 5.8, 8.0, None, 300, 30, None, None, 100, 0.2, 0.4, 2, 0.1, 25, 0.4, 20, 350, 400),
    
    # Infants 6-12m (matches retriever format)
    ("6-12_months", "Infants 6-12m", None, 8.5, 10.5, None, 300, 75, 3, 2.5, 130, 0.4, 0.6, 5, 0.6, 85, 0.5, 30, 350, 400),
    
    # Children 1-3y (matches retriever format)
    ("1-3_years", "Children 1-3y", None, 12.9, 12.5, 15, 500, 90, 8, 3.3, 90, 0.7, 1.1, 7, 0.9, 120, 0.9, 30, 390, 600),
    
    # Children 4-6y (matches retriever format)
    ("4-6_years", "Children 4-6y", None, 18.3, 16.0, 20, 550, 125, 11, 4.5, 90, 0.9, 1.3, 9, 1.2, 135, 1.2, 35, 510, 600),
    
    # Children 7-9y
    ("children_7-9y", "Children 7-9y", None, 25.3, 23.0, 26, 650, 175, 15, 5.9, 90, 1.1, 1.6, 11, 1.5, 170, 1.5, 45, 630, 600),
    
    # Boys 10-12y
    ("boys_10-12y", "Boys 10-12y", None, 34.9, 32.0, 33, 850, 240, 16, 8.5, 100, 1.5, 2.1, 15, 2.0, 220, 1.8, 55, 770, 600),
    
    # Girls 10-12y
    ("girls_10-12y", "Girls 10-12y", None, 36.4, 33.0, 30, 850, 250, 28, 8.5, 100, 1.4, 1.9, 14, 1.9, 225, 1.8, 50, 790, 600),
    
    # Boys 13-15y
    ("boys_13-15y", "Boys 13-15y", None, 50.5, 45.0, 43, 1000, 345, 22, 14.3, 140, 1.9, 2.7, 19, 2.6, 285, 2.4, 70, 930, 600),
    
    # Girls 13-15y
    ("girls_13-15y", "Girls 13-15y", None, 49.6, 43.0, 36, 1000, 340, 30, 12.8, 140, 1.6, 2.2, 16, 2.2, 245, 2.4, 65, 890, 600),
    
    # Boys 16-18y
    ("boys_16-18y", "Boys 16-18y", None, 64.4, 55.0, 50, 1050, 440, 26, 17.6, 140, 2.2, 3.1, 22, 3.0, 340, 2.4, 85, 1000, 600),
    
    # Girls 16-18y
    ("girls_16-18y", "Girls 16-18y", None, 55.7, 46.0, 38, 1050, 380, 32, 14.2, 140, 1.7, 2.3, 17, 2.3, 270, 2.4, 70, 860, 600),
]

NUTRIENTS = [
    "protein", "dietary_fiber", "calcium", "magnesium", "iron", "zinc", "iodine",
    "thiamine", "riboflavin", "niacin", "vit_b6", "folate", "vit_b12", "vit_c", "vit_a", "vit_d"
]

UNITS = {
    "protein": "g",
    "dietary_fiber": "g",
    "calcium": "mg",
    "magnesium": "mg",
    "iron": "mg",
    "zinc": "mg",
    "iodine": "µg",
    "thiamine": "mg",
    "riboflavin": "mg",
    "niacin": "mg",
    "vit_b6": "mg",
    "folate": "µg",
    "vit_b12": "µg",
    "vit_c": "mg",
    "vit_a": "µg",
    "vit_d": "IU",
}

def build_csv():
    """Build the new RDA CSV from parsed table data."""
    
    # Flatten RDA_DATA into nutrient rows
    rows = []
    
    for age_group_norm, age_group_disp, category, body_wt, protein, dietary_fiber, calcium, magnesium, iron, zinc, iodine, thiamine, riboflavin, niacin, vit_b6, folate, vit_b12, vit_c, vit_a, vit_d in RDA_DATA:
        nutrient_values = [
            ("protein", protein),
            ("dietary_fiber", dietary_fiber),
            ("calcium", calcium),
            ("magnesium", magnesium),
            ("iron", iron),
            ("zinc", zinc),
            ("iodine", iodine),
            ("thiamine", thiamine),
            ("riboflavin", riboflavin),
            ("niacin", niacin),
            ("vit_b6", vit_b6),
            ("folate", folate),
            ("vit_b12", vit_b12),
            ("vit_c", vit_c),
            ("vit_a", vit_a),
            ("vit_d", vit_d),
        ]
        
        for nutrient, value in nutrient_values:
            if value is None:
                continue  # Skip missing values (marked with "-" in PDF)
            
            row = {
                "age_group": age_group_norm,
                "nutrient": nutrient,
                "value": value,
                "unit": UNITS[nutrient],
                "source": "RDA_2020 PDF page 16 - ICMR-NIN 2020",
            }
            rows.append(row)
    
    # Write to CSV
    csv_path = Path(r"c:\Users\rauna\Documents\NutriGuide Kids\data\processed\icmr_rda.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["age_group", "nutrient", "value", "unit", "source"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Wrote {len(rows)} rows to {csv_path}")
    print(f"\nSample rows (first 20):")
    for row in rows[:20]:
        print(f"  {row['age_group']:25s} {row['nutrient']:15s} {str(row['value']):>8s} {row['unit']:>5s}")
    
    return csv_path, rows


if __name__ == "__main__":
    csv_path, rows = build_csv()
    print(f"\n✓ Total rows: {len(rows)}")
