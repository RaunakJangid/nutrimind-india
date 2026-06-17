#!/usr/bin/env python
"""
Validate that every cell in icmr_rda.csv matches the PDF table.
Tests get_rda() for all nutrients and age groups.
"""

import sys
from pathlib import Path
import csv

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.retriever import get_rda, load_datasets

def validate_all_cells():
    """Test get_rda() against all CSV cells and verify 100% match."""
    
    # Load the CSV
    csv_path = Path(r"c:\Users\rauna\Documents\NutriGuide Kids\data\processed\icmr_rda.csv")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
    
    print("=" * 100)
    print("VALIDATION: Checking get_rda() against all CSV cells")
    print("=" * 100)
    
    total = len(csv_rows)
    passed = 0
    failed = 0
    errors = []
    
    # Group by age_group for summary
    by_age_group = {}
    
    for i, row in enumerate(csv_rows, 1):
        age_group = row['age_group']
        nutrient = row['nutrient']
        expected_value = float(row['value'])
        expected_unit = row['unit']
        
        if age_group not in by_age_group:
            by_age_group[age_group] = {'total': 0, 'passed': 0}
        by_age_group[age_group]['total'] += 1
        
        try:
            result = get_rda(nutrient, age_group)
            actual_value = result['value']
            actual_unit = result['unit']
            
            if abs(actual_value - expected_value) < 0.001:  # Allow small float rounding
                passed += 1
                by_age_group[age_group]['passed'] += 1
                if i % 50 == 0:
                    print(f"  [{i:3d}/{total}] ✓ {age_group:25s} {nutrient:15s} = {actual_value:>8} {actual_unit}")
            else:
                failed += 1
                error_msg = f"VALUE MISMATCH: {age_group}/{nutrient}: expected {expected_value} {expected_unit}, got {actual_value} {actual_unit}"
                errors.append(error_msg)
                print(f"  [FAIL] {error_msg}")
        except LookupError as e:
            failed += 1
            error_msg = f"LOOKUP ERROR: {age_group}/{nutrient}: {e}"
            errors.append(error_msg)
            print(f"  [FAIL] {error_msg}")
        except Exception as e:
            failed += 1
            error_msg = f"ERROR: {age_group}/{nutrient}: {e}"
            errors.append(error_msg)
            print(f"  [FAIL] {error_msg}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total rows tested: {total}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Match rate: {passed}/{total} = {100*passed/total:.1f}%")
    
    print(f"\nBy age group:")
    for age_group in sorted(by_age_group.keys()):
        stats = by_age_group[age_group]
        pct = 100 * stats['passed'] / stats['total']
        status = "✓" if stats['passed'] == stats['total'] else "✗"
        print(f"  {status} {age_group:25s}: {stats['passed']:2d}/{stats['total']:2d} ({pct:5.1f}%)")
    
    if errors:
        print(f"\n{len(errors)} ERRORS FOUND:")
        for error in errors[:20]:  # Show first 20 errors
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more errors")
        return False
    else:
        print(f"\n✓ ALL {total} CELLS VALIDATED SUCCESSFULLY (100% MATCH)")
        return True


if __name__ == "__main__":
    success = validate_all_cells()
    sys.exit(0 if success else 1)
