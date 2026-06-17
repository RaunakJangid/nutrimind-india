from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
RDA_PDF_PATH = Path(r"C:\Users\rauna\Desktop\Baby RAG\RDA 2020.pdf")
DGI_PDF_PATH = Path(r"C:\Users\rauna\Desktop\Baby RAG\DGI_2024.pdf")
OUT_PATH = ROOT / "data" / "processed" / "icmr_chunks.json"

UPPER_HEADINGS = {
    "SUMMARY OF RECOMMENDATIONS",
    "REFERENCE BODY WEIGHT",
    "ENERGY",
    "PROTEIN",
    "FATS AND OILS",
    "DIETARY FIBER",
    "CARBOHYDRATES",
    "MINERALS",
    "VITAMINS",
    "WATER",
    "ANTIOXIDANTS",
    "RDA FOR ELDERLY",
    "FAT SOLUBLE VITAMINS",
}

NUTRIENT_HEADINGS = [
    "Iron",
    "Zinc",
    "Copper, Chromium and Manganese",
    "Selenium",
    "Iodine",
    "Thiamine B1",
    "Riboflavin B2",
    "Niacin B3",
    "Pantothenic acid B5",
    "Pyridoxine B6",
    "Biotin B7",
    "Folate B9",
    "Cyanocobalamin B12",
    "Ascorbic acid (Vitamin C)",
    "Vitamin A",
    "Vitamin D",
    "Vitamin E & K",
]
NUTRIENT_RE = re.compile(
    r"^(?P<heading>Iron|Zinc|Copper, Chromium and Manganese|Selenium|Iodine|Thiamine B1|Riboflavin B2|Niacin B3|Pantothenic acid B5|Pyridoxine B6|Biotin B7|Folate B9|Cyanocobalamin B12|Ascorbic acid \(Vitamin C\)|Vitamin A|Vitamin D|Vitamin E & K)\b",
    re.IGNORECASE,
)

DGI_HEADING_RE = re.compile(r"^GUIDELINE\s*\x01*([0-9]{1,2})(?:\s|\x01|$)", re.IGNORECASE)

DGI_TITLE_MAP = {
    1: "Eat a variety of foods to ensure a balanced diet",
    2: "Ensure provision of extra food and healthcare during pregnancy and lactation",
    3: "Ensure exclusive breastfeeding for the first six months and continue breastfeeding till two years and beyond",
    4: "Start feeding homemade semi-solid complementary foods to the infant soon after six months of age",
    5: "Ensure adequate and appropriate diets for children and adolescents both in health and sickness",
    6: "Eat plenty of vegetables and legumes",
    7: "Use oils/fats in moderation; choose a variety of oil seeds, nuts, nutricereals, and legumes to meet daily needs of fats and essential fatty acids (EFA)",
    8: "Obtain good quality proteins and essential amino acids (EAA) through appropriate combination of foods and avoid protein supplements to build muscle mass",
    9: "Adopt a healthy lifestyle to prevent abdominal obesity, overweight and overall obesity",
    10: "Be physically active and exercise regularly to maintain good health",
    11: "Restrict salt intake",
    12: "Consume safe and clean foods",
    13: "Adopt appropriate pre-cooking and cooking methods",
    14: "Drink adequate quantity of water",
    15: "Minimize the consumption of high fat, sugar, salt (HFSS) and ultra-processed foods (UPFs)",
    16: "Include nutrient-rich foods in the diets of the elderly for health and wellness",
    17: "Read information on food labels to make informed and healthy food choices",
}


def normalize_space(text: str) -> str:
    text = text.replace("\x01", " ")
    text = re.sub(r"[ \t\r\n]+", " ", text)
    return text.strip()


def clean_line(line: str) -> str:
    line = line.replace("\x01", " ")
    line = re.sub(r"[ \t\r]+", " ", line).strip()
    if line.isdigit():
        return ""
    if line.strip().upper() == "DIETARY GUIDELINES FOR INDIANS":
        return ""
    return line


def clean_chunk_text(lines: list[str]) -> str:
    cleaned = [clean_line(line) for line in lines]
    cleaned = [line for line in cleaned if line]
    text = " ".join(cleaned)
    return normalize_space(text)


def clean_table_text(lines: list[str]) -> str:
    cleaned_lines = []
    for line in lines:
        line = line.replace("\x01", " ")
        line = line.strip()
        if not line or line.isdigit() or line.upper() == "DIETARY GUIDELINES FOR INDIANS":
            continue
        line = re.sub(r" {2,}", " | ", line)
        cleaned_lines.append(line)
    return normalize_space(" \n".join(cleaned_lines))


def is_rda_heading(line: str) -> bool:
    if not line or line.isdigit():
        return False
    normalized = line.strip()[:120]
    if any(normalized.upper().startswith(h) for h in UPPER_HEADINGS):
        return True
    if NUTRIENT_RE.match(normalized):
        return True
    return False


def build_rda_narrative_chunks() -> list[dict[str, object]]:
    doc = fitz.open(RDA_PDF_PATH)
    pages = [doc[i].get_text("text") for i in range(3, 14)]
    lines: list[tuple[int, str]] = []
    for page_idx, page_text in enumerate(pages, start=4):
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if line == "":
                continue
            lines.append((page_idx, line))

    chunks = []
    current_lines: list[str] = []
    current_title = "Summary of Recommendations"
    current_page = 4
    for page, line in lines:
        if is_rda_heading(line):
            if current_lines:
                chunks.append({
                    "chunk_id": f"rda_narrative_{len(chunks)+1:03d}",
                    "text": clean_chunk_text(current_lines),
                    "source": "RDA_2020",
                    "chapter": "RDA_2020",
                    "section": normalize_space(current_title).lower().replace(" ", "_"),
                    "page": current_page,
                    "title": current_title,
                    "section_title": current_title,
                    "chunk_type": "narrative",
                })
                current_lines = []
            current_title = normalize_space(line.rstrip(":"))
            current_page = page
        current_lines.append(line)
    if current_lines:
        chunks.append({
            "chunk_id": f"rda_narrative_{len(chunks)+1:03d}",
            "text": clean_chunk_text(current_lines),
            "source": "RDA_2020",
            "chapter": "RDA_2020",
            "section": normalize_space(current_title).lower().replace(" ", "_"),
            "page": current_page,
            "title": current_title,
            "section_title": current_title,
            "chunk_type": "narrative",
        })
    return chunks


def build_rda_table_chunks() -> list[dict[str, object]]:
    doc = fitz.open(RDA_PDF_PATH)
    page_map = [
        (15, "Summary of EAR for Indians"),
        (16, "Summary of RDA for Indians"),
        (17, "Recommendations for Dietary Fat Intake in Indians"),
        (18, "Daily Nutrient Recommendations for the Elderly in India"),
        (19, "Acceptable Macronutrient Distribution Range (AMDR) by Age and Physiological Groups"),
        (19.5, "Summary of Recommended Intakes for Other Minerals and Trace Elements in Adults"),
        (20, "Tolerable Upper Limit (TUL) for Nutrients"),
        (21, "Balanced Diet for Moderate Active Man"),
        (22, "Balanced Diet for Sedentary Man"),
        (23, "Balanced Diet for Moderate Active Woman"),
        (24, "Balanced Diet for Sedentary Woman"),
        (25, "Balanced Diet for Pregnant Woman"),
        (26, "Key Micronutrients in Different Food Groups"),
    ]
    chunks: list[dict[str, object]] = []
    count = 1

    def add_chunk(page_num: int | float, title: str, lines: list[str], page: int):
        nonlocal count
        chunks.append({
            "chunk_id": f"rda_table_{count:03d}",
            "text": clean_table_text(lines),
            "source": "RDA_2020",
            "chapter": "RDA_2020",
            "section": normalize_space(title).lower().replace(" ", "_"),
            "page": page,
            "title": title,
            "section_title": title,
            "chunk_type": "table",
        })
        count += 1

    for page_key, title in page_map:
        if page_key == 19.5:
            page = 19
            page_text = doc[18].get_text("text")
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]
            # Page text sometimes splits the header across two lines, e.g.
            # "SUMMARY OF RECOMMENDED" on one line and "INTAKES FOR..." on the next.
            # Match either the full phrase or the start fragment to be robust.
            split_index = next(
                (
                    idx
                    for idx, line in enumerate(lines)
                    if ("SUMMARY OF RECOMMENDED INTAKES" in line.strip().upper())
                    or (line.strip().upper().startswith("SUMMARY OF RECOMMENDED"))
                ),
                None,
            )
            if split_index is None:
                raise ValueError("Could not find AMDR split on page 19")
            chunk1_lines = lines[:split_index]
            chunk2_lines = lines[split_index:]
            add_chunk(19, page_map[4][1], chunk1_lines, 19)
            add_chunk(19, page_map[5][1], chunk2_lines, 19)
            continue
        if isinstance(page_key, int):
            page_text = doc[page_key - 1].get_text("text")
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]
            add_chunk(page_key, title, lines, page_key)
    return chunks


def parse_dgi_heading_title(lines: list[str], idx: int) -> str:
    # Normalize around the heading line to capture title after it, even if split across multiple lines.
    title_lines: list[str] = []
    for offset in range(1, 10):
        if idx + offset >= len(lines):
            break
        cand = clean_line(lines[idx + offset])
        if not cand:
            continue
        if cand.upper().startswith("RATIONALE") or cand.upper().startswith("DIETARY GUIDELINES"):
            break
        if cand.upper().startswith("GUIDELINE"):
            break
        if cand.upper().startswith("POINTS TO REGISTER"):
            break
        if cand.upper().startswith("WHAT ") or cand.upper().startswith("WHY ") or cand.upper().startswith("HOW ") or cand.upper().startswith("WHEN ") or cand.upper().startswith("WHICH ") or cand.upper().startswith("WHERE ") or cand.upper().startswith("WHO "):
            break
        title_lines.append(cand)
    title = normalize_space(" ".join(title_lines))
    if len(title.split()) >= 4:
        return title

    # Fallback: look backward for title lines before heading.
    prev_lines: list[str] = []
    for offset in range(1, 6):
        if idx - offset < 0:
            break
        cand = clean_line(lines[idx - offset])
        if not cand or cand.isdigit() or cand.upper().startswith("DIETARY GUIDELINES"):
            break
        if cand.upper().startswith("POINTS TO REGISTER"):
            break
        prev_lines.insert(0, cand)
    title = normalize_space(" ".join(prev_lines))
    if len(title.split()) >= 4 and not title.upper().startswith("RATIONALE"):
        return title
    return ""


def build_dgi_guideline_chunks() -> list[dict[str, object]]:
    doc = fitz.open(DGI_PDF_PATH)
    page_texts = [doc[i].get_text("text") for i in range(17, len(doc))]
    lines: list[tuple[int, str]] = []
    for page_offset, page_text in enumerate(page_texts, start=18):
        for raw_line in page_text.splitlines():
            lines.append((page_offset, raw_line))

    heading_positions: dict[int, int] = {}
    headings: dict[int, dict[str, object]] = {}
    for idx, (page, raw_line) in enumerate(lines):
        text = raw_line.strip()
        if not text:
            continue
        match = DGI_HEADING_RE.match(text)
        if not match:
            continue
        num = int(match.group(1))
        if num in heading_positions:
            continue
        title = parse_dgi_heading_title([line for _, line in lines], idx)
        heading_positions[num] = idx
        headings[num] = {
            "page": page,
            "line_idx": idx,
            "title": title or f"Guideline {num}",
        }

    if len(heading_positions) != 17:
        raise ValueError(f"Expected 17 DGI guideline starts, found {len(heading_positions)}")

    ordered = sorted(headings.items(), key=lambda item: item[1]["line_idx"])
    chunks: list[dict[str, object]] = []
    for index, (guideline_number, meta) in enumerate(ordered):
        start_idx = meta["line_idx"]
        next_idx = ordered[index + 1][1]["line_idx"] if index + 1 < len(ordered) else len(lines)
        
        # Use the correct title from the map
        correct_title = DGI_TITLE_MAP.get(guideline_number, f"Guideline {guideline_number}")
        
        block_lines = [lines[i][1] for i in range(start_idx, next_idx)]
        cleaned_lines = []
        for raw_line in block_lines:
            line = clean_line(raw_line)
            if not line:
                continue
            if DGI_HEADING_RE.match(line):
                continue
            if re.match(r"^GUIDELINE\s*[0-9]{1,2}$", line, re.IGNORECASE):
                continue
            if line.upper().startswith("DIETARY GUIDELINES FOR INDIANS"):
                continue
            cleaned_lines.append(line)
        text = normalize_space(" ".join(cleaned_lines))
        text = f"Guideline {guideline_number}: {correct_title}. {text}"
        chunks.append({
            "chunk_id": f"dgi_guideline_{guideline_number:02d}",
            "text": text,
            "source": "DGI_2024",
            "chapter": "DGI_2024",
            "section": f"guideline_{guideline_number}",
            "page": meta["page"],
            "title": correct_title,
            "section_title": correct_title,
            "chunk_type": "guideline",
            "guideline_number": guideline_number,
        })
    return chunks


def main() -> None:
    if not RDA_PDF_PATH.exists():
        raise FileNotFoundError(f"Missing RDA PDF: {RDA_PDF_PATH}")
    if not DGI_PDF_PATH.exists():
        raise FileNotFoundError(f"Missing DGI PDF: {DGI_PDF_PATH}")

    chunks = []
    chunks.extend(build_rda_narrative_chunks())
    chunks.extend(build_rda_table_chunks())
    chunks.extend(build_dgi_guideline_chunks())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH} with {len(chunks)} chunks")


if __name__ == "__main__":
    main()
