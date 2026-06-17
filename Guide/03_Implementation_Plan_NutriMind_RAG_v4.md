# Implementation Plan — NutriMind India (RAG Paper v4)

## Overview
- Duration: 14 days (expanded for RAG paper requirements)
- Team: 1 developer (solo)
- Output: Working Streamlit app + publishable RAG evaluation

## Execution Order (Non-negotiable)

DATA PREP -> FAISS INDEX -> DB -> RETRIEVER -> CONTEXT MERGER -> BRIDGE -> DECOMPOSER -> LLM BACKENDS -> SYNTHESIZER -> VERIFIER -> UI -> EVALUATION DATASET -> RAGAS + BASELINES -> ABLATION -> TEST -> DEPLOY

---

## Phase 0: Data Preprocessing (Day 0-1)

**Goal:** Clean all datasets, build FAISS index

### Tasks
1. Run `scripts/preprocess_data.py`
   - Clean ICMR RDA CSV
   - Clean IFCT 2017 CSV
   - Standardize units, food names, age groups
   - Validate completeness

2. Run `scripts/build_faiss_index.py` (NEW)
   - Parse ICMR-NIN text chapters (PDF/TXT)
   - Chunk: 512 tokens, 64 overlap
   - Embed with all-MiniLM-L6-v2
   - Build FAISS IVF index
   - Save: `data/indices/faiss_icmr.index` + `data/processed/icmr_chunks.json`

3. Validate all data assets
   - All FOOD_MAP foods in IFCT
   - All age groups in RDA
   - FAISS index loads correctly
   - Semantic search returns relevant chunks

**Success Criteria:**
- All preprocessing scripts run without errors
- FAISS index builds and loads
- Sample semantic queries return relevant chunks
- All data committed to repo

---

## Phase 1: Setup (Day 2)

**Goal:** Working dev environment + runnable skeleton

### Tasks
1. Initialize repo + folder structure
   ```bash
   mkdir -p frontend core db data/processed data/indices scripts templates/prompts tests evaluation/results .streamlit
   ```

2. Create `requirements.txt`
   ```
   streamlit>=1.28
   pandas>=2.0
   numpy>=1.24
   pydantic>=2.0
   python-dotenv>=1.0
   rapidfuzz>=3.0
   jinja2>=3.1
   sentence-transformers>=2.2
   faiss-cpu>=1.7
   google-generativeai>=0.3
   ragas>=0.1
   pytest>=7.0
   ```

3. Create `.env.example`
   ```bash
   GEMINI_API_KEY=your_key_here
   MODEL_BACKEND=gemini
   LLAMA_BASE_URL=http://localhost:11434
   DATA_DIR=./data/processed
   FAISS_INDEX_PATH=./data/indices/faiss_icmr.index
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   LOG_DB_PATH=./data/query_log.db
   APP_ENV=development
   LOG_LEVEL=INFO
   GEMINI_DAILY_LIMIT=1500
   ```

4. Create `.streamlit/config.toml`
   ```toml
   [theme]
   primaryColor = "#0F2444"
   backgroundColor = "#F1F5F9"
   secondaryBackgroundColor = "#FFFFFF"
   textColor = "#1E293B"
   font = "sans serif"
   ```

5. Create `Dockerfile`
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   RUN apt-get update && apt-get install -y libomp-dev && rm -rf /var/lib/apt/lists/*
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
   COPY . .
   EXPOSE 7860
   CMD ["streamlit", "run", "frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
   ```

6. Create skeleton `frontend/app.py`

7. Test
   ```bash
   streamlit run frontend/app.py
   ```

**Success Criteria:**
- App opens without errors
- Theme applied
- Docker builds successfully

---

## Phase 2: Database (Day 2-3)

**Goal:** Logging + evaluation tables

### Tasks
1. Implement `db/db.py`
   - 4 tables: queries, pipeline_runs, evaluation_runs, evaluation_results
   - JSON serialization for complex fields
   - All CRUD operations

2. Test DB
   ```bash
   python -c "from db.db import init_db; init_db('./data/query_log.db')"
   ```

**Success Criteria:**
- All 4 tables create without errors
- Insert + retrieve roundtrip works
- JSON fields serialize correctly

---

## Phase 3: Data + Retrieval Layer (Day 3-4)

**Goal:** Multi-source retrieval pipeline

### Tasks
1. Implement `core/models.py`
   - QueryEntities, CalculationResult, VerificationResult, MergedContext, Conflict

2. Implement `core/retriever.py`
   - `load_datasets()` — load CSVs + FAISS index at startup
   - `get_rda()` — structured lookup
   - `get_food_nutrients()` — structured lookup
   - `semantic_search()` — FAISS vector search
   - `map_age_to_group()` — age mapping
   - `map_food()` — FOOD_MAP + rapidfuzz

3. Implement FOOD_MAP (150+ foods)

4. Test all retrieval paths
   ```bash
   python -c "from core.retriever import semantic_search; print(semantic_search('iron for toddlers', 3))"
   ```

**Success Criteria:**
- All three retrieval paths work
- FAISS returns relevant chunks
- Structured lookups are exact
- Fuzzy matching finds variants

---

## Phase 4: Context Merger + Conflict Detector (Day 5)

**Goal:** Merge multi-source retrieval, detect conflicts

### Tasks
1. Implement `core/context_merger.py` (NEW)
   - `merge()` — combine RDA + IFCT + semantic results
   - `detect_conflicts()` — flag contradictions
   - Priority rules: structured > semantic

2. Unit test
   ```bash
   pytest tests/test_context_merger.py -v
   ```

**Test Cases:**
- No conflicts -> merged context
- Structured vs semantic conflict -> flag, structured wins
- Semantic vs semantic conflict -> flag, higher score wins
- Missing source -> handle gracefully

**Success Criteria:**
- Conflicts detected correctly
- Priority rules applied
- Merged context is coherent

---

## Phase 5: Nutrient-Gap Bridge (Day 5-6)

**Goal:** Core calculation engine

### Tasks
1. Implement `core/bridge.py`
   - `calculate_gap()` with multi-food support
   - SERVING_SIZES (150+ foods)
   - Conflict flags from context merger

2. Unit test
   ```bash
   pytest tests/test_bridge.py -v
   ```

**Test Cases:**
- Single food, single nutrient
- Multiple foods, single nutrient
- Missing food -> flag
- Missing serving -> default 100g
- Conflict flagged -> note in result

**Success Criteria:**
- All tests pass
- Manual test returns sensible numbers

---

## Phase 6: Decomposer (Day 6-7)

**Goal:** Query parsing

### Tasks
1. Implement Tier 1: Pattern matching
   - Age regex, nutrient keywords, food keywords, quantity regex

2. Implement Tier 2: Gemini fallback
   - Structured prompt, JSON parsing, 1 retry

3. Unit test
   ```bash
   pytest tests/test_decomposer.py -v
   ```

**Test Cases:**
- "Iron requirement for 2-year-old" -> rda_lookup
- "Is dal enough protein for 18 month baby" -> diet_check
- "My child eats 2 roti and 1 bowl dal daily" -> foods + servings
- "What should I feed?" -> unknown

**Success Criteria:**
- Tier 1 handles 80%+ without Gemini
- Tier 2 catches edge cases
- Unknown triggers clarification

---

## Phase 7: LLM Backends (Day 7-8)

**Goal:** Dual backend support (Gemini + Llama)

### Tasks
1. Implement `core/llm_backends.py` (NEW)
   - `LLMBackend` abstract class
   - `GeminiBackend` — google-generativeai
   - `LlamaBackend` — Ollama API

2. Implement `core/synthesizer.py`
   - Prompt template loading from `templates/prompts/`
   - Context injection from merged retrieval
   - Citation extraction
   - Backend switching

3. Create prompt templates
   - `templates/prompts/v1_rag_prompt.txt`
   - `templates/prompts/v1_rda_prompt.txt`

4. Unit test
   ```bash
   pytest tests/test_llm_backends.py -v
   pytest tests/test_synthesizer.py -v
   ```

**Test Cases:**
- Gemini generates answer with citations
- Llama generates answer with citations
- Backend switching works
- Prompt template loads correctly

**Success Criteria:**
- Both backends functional
- Answers include citations
- Prompts are versioned

---

## Phase 8: Verifier (Day 8-9)

**Goal:** Prevent hallucinations

### Tasks
1. Implement `core/verifier.py`
   - RDA exact match
   - IFCT exact match
   - LLM grounding check (claims vs context)
   - Math verification

2. Unit test
   ```bash
   pytest tests/test_verifier.py -v
   ```

**Test Cases:**
- All correct -> verified=True
- Modified RDA -> verified=False
- LLM hallucination -> verified=False
- Wrong math -> verified=False

**Success Criteria:**
- All manipulation caught
- Legitimate calculations pass
- LLM grounding checked

---

## Phase 9: UI (Streamlit) (Day 9-10)

**Goal:** Usable product

### Tasks
1. Build chat interface
2. Build answer card with verified badge + model indicator
3. Build proof section with citations
4. Build loading states
5. Build error states
6. Add disclaimer
7. Add example queries
8. Add model selector (Gemini vs Llama)
9. Add debug mode (sidebar)

**Success Criteria:**
- Clean interface
- All features accessible
- Model switcher works
- Proof shows citations

---

## Phase 10: Integration + End-to-End (Day 10-11)

**Goal:** Full pipeline works

### Tasks
1. Wire all modules
   ```
   query -> decompose -> retrieve -> merge -> bridge -> synthesize -> verify -> display
   ```

2. Add timing (latency_ms)
3. Add pipeline logging
4. Test complete flows
   - Journey 1: RDA lookup
   - Journey 2: Diet check
   - Edge: Unknown food
   - Edge: Both backends

**Success Criteria:**
- Latency < 3 sec (Tier 1), < 5 sec (Tier 2 + LLM)
- All journeys correct
- Logs capture full trace

---

## Phase 11: Evaluation Dataset (Day 11-12)

**Goal:** Curated ground-truth QA pairs

### Tasks
1. Create `evaluation/ground_truth_qa.json`
   - 50 QA pairs minimum
   - 20 RDA lookup, 20 diet check, 10 general
   - Each with expected_answer + expected_sources

2. Validate dataset
   - All answers verifiable against datasets
   - Sources correctly identified
   - Coverage across age groups and nutrients

**Success Criteria:**
- 50 QA pairs created
- All answers verified manually
- JSON schema validated

---

## Phase 12: RAGAS + Baselines (Day 12-13)

**Goal:** Mandatory evaluation

### Tasks
1. Implement `evaluation/run_ragas.py`
   - Faithfulness, Relevance, Precision, Recall, Context Utilization
   - Run against ground_truth_qa.json
   - Output: results/gemini_ragas.json, results/llama_ragas.json

2. Implement `evaluation/run_baseline.py`
   - Vanilla LLM baseline
   - Pure Semantic RAG baseline
   - NutriMind full system

3. Implement `evaluation/run_model_comparison.py`
   - Gemini vs Llama 3.1 8B

4. Run evaluations
   ```bash
   python evaluation/run_ragas.py --model gemini
   python evaluation/run_ragas.py --model llama
   python evaluation/run_baseline.py --baseline vanilla
   python evaluation/run_baseline.py --baseline semantic
   python evaluation/run_baseline.py --baseline full
   ```

**Success Criteria:**
- RAGAS metrics calculated
- Baselines compared
- Results saved to evaluation/results/

---

## Phase 13: Ablation Study (Day 13)

**Goal:** Prove multi-source value

### Tasks
1. Implement `evaluation/run_ablation.py`
   - Full system
   - No semantic search
   - No structured data
   - No context merger
   - No verifier

2. Run ablation
   ```bash
   python evaluation/run_ablation.py --variant no_semantic
   python evaluation/run_ablation.py --variant no_structured
   python evaluation/run_ablation.py --variant no_merger
   python evaluation/run_ablation.py --variant no_verifier
   ```

3. Generate comparison table

**Success Criteria:**
- Ablation results show multi-source > single-source
- Verifier impact quantified
- Results saved

---

## Phase 14: Testing (Day 13-14)

**Goal:** Reliability + safety

### Tasks
1. Run all unit tests
   ```bash
   pytest tests/ -v --tb=short
   ```

2. Manual testing matrix
   | Query | Age | Foods | Backend | Expected |
   |-------|-----|-------|---------|----------|
   | RDA lookup | 6mo | none | Gemini | RDA value |
   | RDA lookup | 2yr | none | Llama | RDA value |
   | Diet check | 2yr | dal | Gemini | Gap calc |
   | Diet check | 2yr | dal + rice | Llama | Multi-food |
   | Unknown food | 2yr | xyz | Gemini | Flag |
   | General | none | none | Both | Semantic answer |

3. Failure testing
   - No internet -> Tier 1 works, Tier 2 fails gracefully
   - Invalid API key -> Tier 2 fails, Tier 1 works
   - Corrupt CSV -> clear error
   - Empty query -> validation error

4. Performance testing
   - 10 consecutive queries per backend
   - Verify latency targets

**Success Criteria:**
- All unit tests pass
- Manual matrix complete
- No crashes
- Latency targets met

---

## Phase 15: Deployment (Day 14)

**Goal:** Live demo + reproducible evaluation

### Tasks
1. Create `README.md`
   - Project description
   - Setup instructions (including Ollama)
   - Usage examples
   - Evaluation instructions

2. Create `.gitignore`

3. Commit to GitHub

4. Deploy on HuggingFace Spaces
   - Create Space (Streamlit + Docker)
   - Add GEMINI_API_KEY to Secrets
   - Verify build

5. Smoke test
   - Public URL loads
   - Example queries work
   - Both backends tested (Gemini on HF, Llama locally)

6. Evaluation reproducibility
   - Include evaluation/ground_truth_qa.json in repo
   - Include evaluation scripts
   - Document how to run RAGAS + baselines + ablation

**Success Criteria:**
- Public URL works
- Evaluation reproducible by others
- Clean install from README

---

## Done Criteria (RAG Paper Ready)

### Functional
- User asks: "Is dal enough protein for 2-year-old?"
- System returns: Answer + Gap + Proof table + Citations + Verified = TRUE

### Research
- RAGAS evaluation complete with 50 QA pairs
- Baseline comparison: Vanilla LLM vs Pure Semantic vs NutriMind
- Ablation study: multi-source vs single-source
- Model comparison: Gemini vs Llama 3.1 8B
- All results saved in evaluation/results/

### Safety
- No hallucinated numbers
- Verifier blocks invalid outputs
- LLM grounding checked

### Performance
- Tier 1 (pattern match): < 3 sec
- Tier 2 (Gemini + synthesis): < 5 sec
- Llama (local): < 5 sec

### Deployment
- Public URL works
- Evaluation reproducible
- Clean install from README

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| ICMR/IFCT data not available | HIGH | Preprocess BEFORE Day 1. Placeholder data for dev if needed. |
| FAISS build fails | MEDIUM | Test on target environment early. Use faiss-cpu (not GPU). |
| Gemini quota exceeded | MEDIUM | Tier 1 handles 80%. Add quota check. |
| Ollama setup complex | MEDIUM | Document setup. Provide alternative: only Gemini on HF. |
| RAGAS installation issues | LOW | Pin versions. Test in clean environment. |
| Single developer bottleneck | MEDIUM | 14-day plan with buffer. Cut V2 features if behind. |

---

## Appendix: Daily Standup Checklist

**Day 1:** Data preprocessing + FAISS index built?
**Day 2:** Setup + DB working?
**Day 3-4:** Retriever loads all three sources?
**Day 5:** Context merger detects conflicts?
**Day 5-6:** Bridge calculates correctly?
**Day 6-7:** Decomposer parses queries?
**Day 7-8:** Both LLM backends functional?
**Day 8-9:** Verifier catches hallucinations?
**Day 9-10:** UI complete with model switcher?
**Day 10-11:** End-to-end pipeline works?
**Day 11-12:** Ground-truth QA dataset created?
**Day 12-13:** RAGAS + baselines run?
**Day 13:** Ablation study complete?
**Day 14:** Deployed + smoke tested?
