---
title: NutriMind India
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# NutriMind-India

markdown# NutriMind-India

**A verified RAG system for infant and child nutrition queries (ages 0–5), 
grounded in official ICMR-NIN dietary guidelines.**

> Dataset-backed estimates only. Not medical advice.

---

## What it does

NutriMind answers three types of parent queries about child nutrition:

- **RDA lookup** — "What is the iron requirement for a 2-year-old?"  
  → Deterministic answer from ICMR-NIN RDA 2020 tables. No LLM involved.

- **Diet check** — "Is dal enough protein for an 18-month-old?"  
  → Calculates nutrient gap from IFCT 2017 food data vs. RDA requirement.  
  → LLM generates explanation; verifier checks math and grounding.

- **General guidance** — "How will my child get vitamin D?"  
  → Semantic retrieval from Dietary Guidelines for Indians 2024 (DGI 2024).  
  → LLM synthesizes a parent-friendly answer from relevant guidelines.

Every answer shows a **Verified / Not verified** badge and an 
expandable **Show Proof** table citing the source data.

---

## Data sources

All three sources are real, official ICMR-NIN publications:

| Source | Coverage |
|--------|----------|
| ICMR-NIN RDA & EAR 2020 (Short Report) | RDA/EAR/TUL values for 20 age/gender groups, 16 nutrients |
| IFCT 2017 (Indian Food Composition Tables) | 4,648 nutrient rows across 542 foods |
| Dietary Guidelines for Indians 2024 (DGI 2024) | All 17 guidelines, chunked for semantic retrieval |

---

## Architecture
Query

└── Decomposer (Tier 1: regex / Tier 2: Gemini fallback)

└── Retriever (structured RDA + IFCT + FAISS semantic)

└── Context Merger (conflict detection, priority rules)

└── Bridge (calculate_gap — deterministic math)

└── Synthesizer (intent-dependent strategy)

└── Verifier (RDA match, math check,

conflict flag, LLM grounding)

└── UI (Streamlit)

**Synthesis strategy by intent:**
- `rda_lookup` — deterministic template, zero LLM calls, zero hallucination risk
- `diet_check` — LLM generates explanation around deterministic gap numbers; 
  verifier checks math consistency
- `general_question` — LLM synthesizes from top-2 DGI semantic chunks

---

## Setup

**Prerequisites:** Python 3.11+, pip

```bash
git clone <your-repo-url>
cd nutrimind-india

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**Environment:**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**Data preprocessing (required on first run):**
```bash
python scripts/preprocess_data.py
python scripts/build_faiss_index.py
```

**Run the app:**
```bash
streamlit run frontend/app.py
```

**Optional — Llama backend (local inference via Ollama):**
```bash
ollama pull llama3.1:8b
# Set MODEL_BACKEND=llama in .env
```

---

## Tests

```bash
pytest tests/ -v --tb=short
```

23 tests covering decomposer, retriever, bridge, context merger, 
synthesizer, verifier, and LLM backends.

---

## Evaluation

Ground-truth dataset: `evaluation/ground_truth_qa.json`  
50 QA pairs (20 rda_lookup, 20 diet_check, 10 general_question),  
programmatically generated from real ICMR-NIN data and verified.

**RAGAS evaluation** (faithfulness, relevance, context precision/recall):
```bash
python evaluation/run_ragas.py --model gemini --limit 50
# Or in batches to avoid rate limits:
python evaluation/run_ragas.py --model gemini --offset 0 --limit 10
python evaluation/run_ragas.py --model gemini --offset 10 --limit 10
# ...
```

**Ablation study** (5 variants: full, no_semantic, no_structured, 
no_merger, no_verifier):
```bash
python evaluation/run_ablation.py --variant all --limit 10
python evaluation/run_ablation.py --variant no_structured --limit 50
```

**Key ablation findings:**
- Removing structured data (`no_structured`): faithfulness 0.90 → 0.0 
  on diet_check — structured RDA/IFCT data is load-bearing
- Removing semantic search (`no_semantic`): faithfulness 0.85 → 0.0 
  on general_question — semantic retrieval is load-bearing for DGI queries
- Both sources are necessary; neither alone covers the full query scope

**Verification breakdown** (reported separately from RAGAS):
- RDA match rate: 100%
- Math check rate: 100%  
- Conflict flag rate: reported per run (conservative — flags any 
  structured/semantic discrepancy)
- LLM grounding rate: ~75% on diet_check

> **Note on evaluation reproducibility:** RAGAS judge LLM calls were 
> served via FreeLLMAPI (a free-tier aggregation proxy with automatic 
> provider fallover across Gemini, Groq, Cerebras, etc.). Judge-model 
> identity varies per call based on real-time provider availability. 
> Final paper-reported metrics should use a pinned paid judge model 
> (e.g. GPT-4o-mini) for full reproducibility.

---

## Project structure
nutrimind-india/

├── core/

│   ├── decomposer.py      # Query parsing (regex + Gemini fallback)

│   ├── retriever.py       # Structured + semantic retrieval

│   ├── context_merger.py  # Conflict detection, priority rules

│   ├── bridge.py          # Nutrient gap calculation

│   ├── synthesizer.py     # Intent-dependent answer generation

│   ├── verifier.py        # Answer verification layer

│   ├── llm_backends.py    # Gemini, Llama, Deterministic backends

│   ├── food_map.py        # 198 food aliases (Hindi/English)

│   └── models.py          # Pydantic data models

├── data/

│   ├── processed/         # icmr_rda.csv, ifct2017.csv, icmr_chunks.json

│   └── indices/           # faiss_icmr.index

├── evaluation/

│   ├── ground_truth_qa.json

│   ├── run_ragas.py

│   ├── run_ablation.py

│   └── results/

├── frontend/

│   └── app.py             # Streamlit UI

├── scripts/

│   ├── preprocess_data.py

│   └── build_faiss_index.py

├── templates/prompts/     # Versioned LLM prompt templates

├── tests/                 # 23 unit tests

└── db/db.py               # Query logging (SQLite)

---

## Citation

Data sources:
- ICMR-NIN Expert Group. *Recommended Dietary Allowances and Estimated 
  Average Requirements (RDA & EAR) — 2020.* ICMR-National Institute of 
  Nutrition, Hyderabad.
- National Institute of Nutrition. *Indian Food Composition Tables 2017 
  (IFCT 2017).* ICMR-NIN, Hyderabad.
- National Institute of Nutrition. *Dietary Guidelines for Indians 2024 
  (DGI 2024).* ICMR-NIN, Hyderabad.

---

## Disclaimer

NutriMind provides estimates based on ICMR-NIN published data. 
It is not a substitute for medical advice. Always consult a qualified 
healthcare provider for your child's specific nutritional needs.
