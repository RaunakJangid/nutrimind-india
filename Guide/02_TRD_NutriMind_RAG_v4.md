# Technical Requirements Document (TRD) — NutriMind India (RAG Paper v4)

## 1. Frontend

### Framework
**Streamlit (Python)** — single-process, no separate backend API.

### State Management
`st.session_state` stores:
- `chat_history`: list of {role, content, metadata, sources}
- `last_query_id`: UUID
- `ui_states`: expander open/close, debug mode
- `model_backend`: "gemini" | "llama"

### UI Components
| Component | Streamlit API | Purpose |
|-----------|--------------|---------|
| Chat input | `st.chat_input()` | User query entry |
| Chat messages | `st.chat_message()` | Conversation display |
| Proof section | `st.expander("Show Proof")` | Collapsible evidence |
| Loading | `st.spinner("Checking ICMR data...")` | Processing feedback |
| Verified badge | `st.badge()` or markdown | Trust indicator |
| Model selector | `st.selectbox()` | Gemini vs Llama toggle |
| Disclaimer | Fixed banner/footer | Legal safety |

### Mandatory UI Elements
- Verified badge (green if verified, red if not)
- Proof table with source references + confidence scores
- Model indicator (which backend generated the answer)
- Disclaimer: "NutriMind provides dataset-backed estimates. Not medical advice."

## 2. Backend Architecture

### Pattern
Single-process Python. Streamlit UI imports core modules directly.

```
User Input
    |
Decomposer (keyword + LLM fallback)
    |
Multi-Source Retriever
    |-- Structured: ICMR RDA CSV
    |-- Structured: IFCT 2017 CSV
    |-- Semantic: ICMR-NIN text chapters (FAISS)
    |
Context Merger + Conflict Detector
    |
LLM Synthesis (Prompt Engineering)
    |-- Backend: Gemini API
    |-- Backend: Llama 3.1 8B (Ollama)
    |
Verifier (strict validation)
    |
UI Output + Proof + Citations
```

### Core Modules

#### 2.1 decomposer.py
**Purpose:** Convert free-text query to structured entities

**Approach:** Two-tier
1. **Tier 1 — Pattern matching:**
   - Age regex: (\d+)\s*(year|yr|month|mo|y|m) -> age_months
   - Nutrient keywords: iron, protein, calcium, vitamin_a, etc.
   - Food keywords: FOOD_MAP keys + rapidfuzz
   - Quantity regex: (\d+)\s*(bowl|cup|plate|piece|roti|glass|spoon)
2. **Tier 2 — LLM fallback:**
   - Gemini with structured prompt
   - JSON output parsed with Pydantic
   - 1 retry on failure

**Pydantic Model:**
```python
class QueryEntities(BaseModel):
    nutrient: str | None
    age_months: int | None
    foods: list[str]
    servings: dict[str, float]
    intent: str  # "rda_lookup" | "diet_check" | "general_question" | "unknown"
```

**Age Parsing:**
- Extract numeric + unit -> convert to months
- Store: age_months (int)

**Age Group Mapping (ICMR):**
```python
AGE_GROUP_MAP = {
    (0, 6):   "0-6_months",
    (7, 12):  "6-12_months",
    (13, 36): "1-3_years",
    (37, 60): "4-6_years",
    (61, 84): "7-9_years",
}
```

#### 2.2 retriever.py — Multi-Source Retrieval

**Purpose:** Fetch data from three sources simultaneously

**Data Sources:**
1. data/processed/icmr_rda.csv — Structured RDA values
2. data/processed/ifct2017.csv — Structured food nutrients
3. data/indices/faiss_icmr.index + data/indices/icmr_chunks.json — Semantic search over ICMR-NIN text

**Functions:**
```python
def retrieve_all(query_entities: QueryEntities) -> RetrievalResult:
    # Parallel retrieval from all three sources

def get_rda(nutrient: str, age_group: str) -> dict:
    # Returns: {value, unit, source, confidence: 1.0}

def get_food_nutrients(food: str, nutrient: str) -> dict:
    # Returns: {value_per_100g, unit, source, confidence: 1.0}

def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    # Returns: [{text, source_chapter, score, chunk_id}]

def map_age_to_group(age_months: int) -> str:
    # Maps to ICMR age group

def map_food(food_name: str) -> str:
    # FOOD_MAP + rapidfuzz fallback (threshold >= 80)
```

**Food Mapping:**
```python
FOOD_MAP = {
    "dal": "lentils", "moong_dal": "lentils_moong", "toor_dal": "lentils_toor",
    "rice": "rice_polished", "brown_rice": "rice_brown",
    "roti": "wheat_flour", "chapati": "wheat_flour", "paratha": "wheat_flour",
    "milk": "milk_cow", "dahi": "curd", "yogurt": "curd",
    "egg": "egg_whole", "apple": "apple_fruit", "banana": "banana_ripe",
    "spinach": "spinach_leaves", "carrot": "carrot_root", "potato": "potato_tuber",
    "tomato": "tomato_ripe", "onion": "onion_bulb", "chicken": "chicken_broiler",
    "fish": "fish_rohu", "paneer": "paneer", "ghee": "ghee", "butter": "butter",
    "oil": "oil_groundnut", "sugar": "sugar", "jaggery": "jaggery",
    "gram": "bengal_gram", "rajma": "rajma", "soybean": "soybean",
    "groundnut": "groundnut", "coconut": "coconut", "tamarind": "tamarind",
    "green_leafy": "amaranth_leaves", "beetroot": "beetroot", "cabbage": "cabbage",
    "cauliflower": "cauliflower", "brinjal": "brinjal", "ladies_finger": "ladies_finger",
    "peas": "green_peas", "beans": "french_beans", "capsicum": "capsicum",
    "radish": "radish", "turnip": "turnip", "sweet_potato": "sweet_potato",
    "yam": "yam", "tapioca": "tapioca", "maize": "maize", "ragi": "ragi",
    "bajra": "bajra", "jowar": "jowar", "wheat": "wheat_flour",
    "poha": "rice_flakes", "idli": "idli", "dosa": "dosa", "upma": "upma",
    "sambar": "sambar", "rasam": "rasam", "curry": "vegetable_curry",
    "chutney": "coconut_chutney", "pickle": "mango_pickles", "papad": "papad",
    "halwa": "sooji_halwa", "kheer": "rice_kheer", "laddu": "besan_laddu",
    "barfi": "milk_barfi", "puri": "puri", "bhatura": "bhatura",
    "naan": "naan", "kulcha": "kulcha", "parantha": "paratha",
    "thepla": "thepla", "dhokla": "dhokla", "khandvi": "khandvi",
    "handvo": "handvo", "mutter_paneer": "mutter_paneer",
    "palak_paneer": "palak_paneer", "dal_makhani": "dal_makhani",
    "chole": "chole", "rajma_chawal": "rajma", "kadhi": "kadhi",
    "samosa": "samosa", "pakora": "pakora", "kachori": "kachori",
    "jalebi": "jalebi", "gulab_jamun": "gulab_jamun", "rasgulla": "rasgulla",
    "sandesh": "sandesh", "mishti_doi": "mishti_doi", "payasam": "payasam",
    "kesari": "kesari", "mysore_pak": "mysore_pak", "obattu": "obattu",
    "holige": "holige", "pongal": "pongal", "bisi_bele_bath": "bisi_bele_bath",
    "puliyogare": "puliyogare", "lemon_rice": "lemon_rice", "tamarind_rice": "tamarind_rice",
    "tomato_rice": "tomato_rice", "coconut_rice": "coconut_rice", "curd_rice": "curd_rice",
    "bisibelebath": "bisi_bele_bath", "vangi_bath": "vangi_bath", "pulao": "pulao",
    "biryani": "biryani", "fried_rice": "fried_rice", "jeera_rice": "jeera_rice",
    "ghee_rice": "ghee_rice", "sambar_rice": "sambar_rice", "rasam_rice": "rasam_rice",
    "khichdi": "khichdi", "phirni": "phirni", "seviyan": "seviyan", "sheer_korma": "sheer_korma",
    "double_ka_meetha": "double_ka_meetha", "qubani_ka_meetha": "qubani_ka_meetha",
    "shahi_tukda": "shahi_tukda", "malpua": "malpua", "imarti": "imarti",
    "peda": "peda", "modak": "modak", "karanji": "karanji", "chakli": "chakli",
    "shankarpali": "shankarpali", "chivda": "chivda", "laiyya": "laiyya",
    "murmura": "murmura", "puffed_rice": "murmura", "flattened_rice": "rice_flakes",
    "sabudana": "sago", "sago": "sago", "vermicelli": "vermicelli",
    "semiya": "vermicelli", "rava": "semolina", "sooji": "semolina", "semolina": "semolina",
    "besan": "gram_flour", "gram_flour": "gram_flour", "maida": "refined_flour",
    "refined_flour": "refined_flour", "atta": "whole_wheat_flour",
    "whole_wheat_flour": "whole_wheat_flour", "multigrain_flour": "multigrain_flour",
    "oats": "oats", "muesli": "muesli", "corn_flakes": "corn_flakes",
    "wheat_flakes": "wheat_flakes", "rice_flakes": "rice_flakes", "puffed_wheat": "puffed_wheat",
}
# 150+ foods mapped — full IFCT coverage
# Fuzzy matching (rapidfuzz) for variants not in FOOD_MAP
# Threshold: score >= 80 to accept, else "unknown_food"
```

**FAISS Semantic Search:**
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- Index: FAISS IVF index for fast approximate search
- Chunks: 512 tokens with 64 token overlap
- Metadata per chunk: source_chapter, page_number, section_title
- Top-k: 5 chunks retrieved per query

#### 2.3 context_merger.py
**Purpose:** Merge multi-source retrieval + detect conflicts

**NEW MODULE — RAG Paper Requirement**

```python
class ContextMerger:
    def merge(self, rda_result, ifct_results, semantic_results) -> MergedContext:
        # 1. Combine all sources
        # 2. Detect contradictions (e.g., RDA says 9mg iron, text says 8mg)
        # 3. Priority: structured data > semantic text
        # 4. Confidence scoring per source

    def detect_conflicts(self, sources: list) -> list[Conflict]:
        # Flag when structured data contradicts text
        # Flag when multiple text chunks disagree
        # Return: [{type, sources, resolution}]
```

**Conflict Resolution Rules:**
1. RDA CSV vs IFCT CSV: Both structured — flag for user, use RDA for calculation
2. Structured vs Semantic: Structured wins, semantic provides context
3. Semantic vs Semantic: Higher confidence score wins, flag disagreement

#### 2.4 bridge.py
**Purpose:** Nutrient gap calculation

**Serving Sizes (cooked/ready-to-eat grams):**
```python
SERVING_SIZES = {
    "dal": {"grams": 150, "description": "1 bowl cooked"},
    "moong_dal": {"grams": 150, "description": "1 bowl cooked"},
    "toor_dal": {"grams": 150, "description": "1 bowl cooked"},
    "rice": {"grams": 100, "description": "1 cup cooked"},
    "brown_rice": {"grams": 100, "description": "1 cup cooked"},
    "roti": {"grams": 30, "description": "1 medium roti"},
    "chapati": {"grams": 30, "description": "1 medium chapati"},
    "paratha": {"grams": 40, "description": "1 medium paratha"},
    "milk": {"grams": 250, "description": "1 glass"},
    "dahi": {"grams": 100, "description": "1 bowl"},
    "yogurt": {"grams": 100, "description": "1 bowl"},
    "egg": {"grams": 50, "description": "1 whole egg"},
    "apple": {"grams": 150, "description": "1 medium"},
    "banana": {"grams": 100, "description": "1 medium"},
    "spinach": {"grams": 50, "description": "1 cup cooked"},
    "carrot": {"grams": 50, "description": "1 medium"},
    "potato": {"grams": 100, "description": "1 medium boiled"},
    "tomato": {"grams": 80, "description": "1 medium"},
    "onion": {"grams": 50, "description": "1 medium"},
    "chicken": {"grams": 100, "description": "1 piece cooked"},
    "fish": {"grams": 100, "description": "1 piece cooked"},
    "paneer": {"grams": 50, "description": "1/2 cup cubed"},
    "ghee": {"grams": 10, "description": "2 teaspoons"},
    "butter": {"grams": 10, "description": "2 teaspoons"},
    "oil": {"grams": 15, "description": "1 tablespoon"},
    "sugar": {"grams": 10, "description": "2 teaspoons"},
    "jaggery": {"grams": 20, "description": "1 piece"},
    "gram": {"grams": 100, "description": "1/2 cup cooked"},
    "rajma": {"grams": 100, "description": "1/2 cup cooked"},
    "soybean": {"grams": 50, "description": "1/4 cup cooked"},
    "groundnut": {"grams": 30, "description": "small handful"},
    "coconut": {"grams": 30, "description": "1/4 cup grated"},
    "tamarind": {"grams": 20, "description": "1 tablespoon pulp"},
    "green_leafy": {"grams": 50, "description": "1 cup cooked"},
    "beetroot": {"grams": 100, "description": "1 medium"},
    "cabbage": {"grams": 100, "description": "1 cup shredded"},
    "cauliflower": {"grams": 100, "description": "1 cup florets"},
    "brinjal": {"grams": 100, "description": "1 medium"},
    "ladies_finger": {"grams": 100, "description": "10-12 pieces"},
    "peas": {"grams": 100, "description": "1/2 cup"},
    "beans": {"grams": 100, "description": "10-12 pieces"},
    "capsicum": {"grams": 80, "description": "1 medium"},
    "radish": {"grams": 100, "description": "1 medium"},
    "turnip": {"grams": 100, "description": "1 medium"},
    "sweet_potato": {"grams": 150, "description": "1 medium"},
    "yam": {"grams": 100, "description": "1/2 cup cooked"},
    "tapioca": {"grams": 100, "description": "1/2 cup cooked"},
    "maize": {"grams": 100, "description": "1/2 cup cooked"},
    "ragi": {"grams": 100, "description": "1/2 cup cooked"},
    "bajra": {"grams": 100, "description": "1/2 cup cooked"},
    "jowar": {"grams": 100, "description": "1/2 cup cooked"},
    "wheat": {"grams": 100, "description": "1/2 cup cooked"},
    "poha": {"grams": 50, "description": "1/2 cup dry"},
    "idli": {"grams": 80, "description": "2 medium"},
    "dosa": {"grams": 100, "description": "1 medium"},
    "upma": {"grams": 150, "description": "1 bowl"},
    "sambar": {"grams": 150, "description": "1 bowl"},
    "rasam": {"grams": 100, "description": "1 bowl"},
    "curry": {"grams": 100, "description": "1/2 cup"},
    "chutney": {"grams": 30, "description": "2 tablespoons"},
    "pickle": {"grams": 15, "description": "1 tablespoon"},
    "papad": {"grams": 10, "description": "1 piece"},
    "halwa": {"grams": 100, "description": "1/2 cup"},
    "kheer": {"grams": 150, "description": "1 bowl"},
    "laddu": {"grams": 50, "description": "1 piece"},
    "barfi": {"grams": 50, "description": "1 piece"},
    "puri": {"grams": 30, "description": "2 pieces"},
    "bhatura": {"grams": 50, "description": "1 piece"},
    "naan": {"grams": 80, "description": "1 piece"},
    "kulcha": {"grams": 80, "description": "1 piece"},
    "parantha": {"grams": 40, "description": "1 medium"},
    "thepla": {"grams": 40, "description": "1 medium"},
    "dhokla": {"grams": 100, "description": "2 pieces"},
    "khandvi": {"grams": 100, "description": "4-5 rolls"},
    "handvo": {"grams": 150, "description": "1 piece"},
    "mutter_paneer": {"grams": 150, "description": "1 bowl"},
    "palak_paneer": {"grams": 150, "description": "1 bowl"},
    "dal_makhani": {"grams": 150, "description": "1 bowl"},
    "chole": {"grams": 150, "description": "1 bowl"},
    "rajma_chawal": {"grams": 200, "description": "1 plate"},
    "kadhi": {"grams": 150, "description": "1 bowl"},
    "samosa": {"grams": 50, "description": "1 piece"},
    "pakora": {"grams": 50, "description": "3-4 pieces"},
    "kachori": {"grams": 50, "description": "1 piece"},
    "jalebi": {"grams": 50, "description": "2 pieces"},
    "gulab_jamun": {"grams": 50, "description": "2 pieces"},
    "rasgulla": {"grams": 50, "description": "2 pieces"},
    "sandesh": {"grams": 30, "description": "1 piece"},
    "mishti_doi": {"grams": 100, "description": "1 bowl"},
    "payasam": {"grams": 150, "description": "1 bowl"},
    "kesari": {"grams": 100, "description": "1/2 cup"},
    "mysore_pak": {"grams": 50, "description": "1 piece"},
    "obattu": {"grams": 100, "description": "1 piece"},
    "holige": {"grams": 100, "description": "1 piece"},
    "pongal": {"grams": 150, "description": "1 bowl"},
    "bisi_bele_bath": {"grams": 200, "description": "1 bowl"},
    "puliyogare": {"grams": 150, "description": "1 bowl"},
    "lemon_rice": {"grams": 150, "description": "1 bowl"},
    "tamarind_rice": {"grams": 150, "description": "1 bowl"},
    "tomato_rice": {"grams": 150, "description": "1 bowl"},
    "coconut_rice": {"grams": 150, "description": "1 bowl"},
    "curd_rice": {"grams": 200, "description": "1 bowl"},
    "bisibelebath": {"grams": 200, "description": "1 bowl"},
    "vangi_bath": {"grams": 150, "description": "1 bowl"},
    "pulao": {"grams": 150, "description": "1 bowl"},
    "biryani": {"grams": 200, "description": "1 plate"},
    "fried_rice": {"grams": 150, "description": "1 bowl"},
    "jeera_rice": {"grams": 150, "description": "1 bowl"},
    "ghee_rice": {"grams": 150, "description": "1 bowl"},
    "sambar_rice": {"grams": 200, "description": "1 bowl"},
    "rasam_rice": {"grams": 200, "description": "1 bowl"},
    "khichdi": {"grams": 200, "description": "1 bowl"},
    "phirni": {"grams": 150, "description": "1 bowl"},
    "seviyan": {"grams": 150, "description": "1 bowl"},
    "sheer_korma": {"grams": 150, "description": "1 bowl"},
    "double_ka_meetha": {"grams": 100, "description": "1/2 cup"},
    "qubani_ka_meetha": {"grams": 100, "description": "1/2 cup"},
    "shahi_tukda": {"grams": 100, "description": "1 piece"},
    "malpua": {"grams": 50, "description": "1 piece"},
    "imarti": {"grams": 50, "description": "1 piece"},
    "peda": {"grams": 30, "description": "1 piece"},
    "modak": {"grams": 50, "description": "1 piece"},
    "karanji": {"grams": 50, "description": "1 piece"},
    "chakli": {"grams": 30, "description": "5-6 pieces"},
    "shankarpali": {"grams": 30, "description": "5-6 pieces"},
    "chivda": {"grams": 30, "description": "1/4 cup"},
    "laiyya": {"grams": 20, "description": "1/4 cup"},
    "murmura": {"grams": 20, "description": "1 cup puffed"},
    "puffed_rice": {"grams": 20, "description": "1 cup puffed"},
    "flattened_rice": {"grams": 50, "description": "1/2 cup dry"},
    "sabudana": {"grams": 100, "description": "1/2 cup cooked"},
    "sago": {"grams": 100, "description": "1/2 cup cooked"},
    "vermicelli": {"grams": 100, "description": "1/2 cup cooked"},
    "semiya": {"grams": 100, "description": "1/2 cup cooked"},
    "rava": {"grams": 100, "description": "1/2 cup cooked"},
    "sooji": {"grams": 100, "description": "1/2 cup cooked"},
    "semolina": {"grams": 100, "description": "1/2 cup cooked"},
    "besan": {"grams": 100, "description": "1/2 cup cooked"},
    "gram_flour": {"grams": 100, "description": "1/2 cup cooked"},
    "maida": {"grams": 100, "description": "1/2 cup cooked"},
    "refined_flour": {"grams": 100, "description": "1/2 cup cooked"},
    "atta": {"grams": 100, "description": "1/2 cup cooked"},
    "whole_wheat_flour": {"grams": 100, "description": "1/2 cup cooked"},
    "multigrain_flour": {"grams": 100, "description": "1/2 cup cooked"},
    "oats": {"grams": 50, "description": "1/2 cup cooked"},
    "muesli": {"grams": 50, "description": "1/2 cup"},
    "corn_flakes": {"grams": 30, "description": "1 cup"},
    "wheat_flakes": {"grams": 30, "description": "1 cup"},
    "rice_flakes": {"grams": 50, "description": "1/2 cup dry"},
    "puffed_wheat": {"grams": 20, "description": "1 cup puffed"},
}
# 150+ foods mapped — full IFCT coverage
```

**Calculation Logic:** Same as v3 but with multi-food aggregation and conflict flags from context merger.

#### 2.5 synthesizer.py — LLM Synthesis with Prompt Engineering

**RESTORED — RAG Paper Requirement**

**Purpose:** Generate natural-language answers from retrieved context

**Approach:**
1. Build context-rich prompt from merged retrieval results
2. Include citations to sources
3. Call LLM backend
4. Parse response, extract answer + citations

**Prompt Template (versioned):**
```
You are a nutrition expert answering questions for Indian parents.
Use ONLY the provided context to answer. Do not make up information.

Context from ICMR RDA:
{icmr_rda_context}

Context from IFCT 2017:
{ifct_context}

Context from ICMR-NIN Guidelines:
{semantic_context}

Conflicts detected:
{conflicts}

User Question: {query}

Answer the question in simple language. Include specific numbers from the context.
Cite your sources using [Source: ICMR RDA], [Source: IFCT], [Source: ICMR-NIN Text].
If there are conflicts, mention them clearly.

Answer:
```

**Backends:**
```python
class LLMBackend(ABC):
    def generate(self, prompt: str, context: dict) -> str: ...

class GeminiBackend(LLMBackend):
    # Uses google-generativeai
    # Model: gemini-1.5-flash
    # Temperature: 0.1
    # Max tokens: 512

class LlamaBackend(LLMBackend):
    # Uses Ollama local server
    # Model: llama3.1:8b
    # Temperature: 0.1
    # Max tokens: 512
```

**Switching:**
```python
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "gemini")  # gemini | llama
```

**Prompt Engineering:**
- Prompt templates stored in templates/prompts/ with version control
- A/B testing framework for prompt variants
- Temperature 0.1 for factual consistency
- Max tokens 512 to prevent rambling

#### 2.6 verifier.py

**Purpose:** Prevent hallucinations

**Verification Rules:**
1. RDA values match dataset (exact, within float tolerance)
2. IFCT values match dataset (exact)
3. LLM output is grounded in retrieved context (RAGAS faithfulness check)
4. Derived values (gap, %) are mathematically correct
5. No contradictions unresolved

**LLM Grounding Check:**
```python
def check_llm_grounding(answer: str, context: dict) -> bool:
    # Extract claims from answer
    # Check each claim exists in context
    # Use rapidfuzz for fuzzy matching of claims to context chunks
    # Return: percentage of claims grounded
```

**Critical Behavior:**
- If verified == False -> block answer, show fallback
- Log failure reason for debugging

#### 2.7 db.py

**Purpose:** Pipeline traceability + evaluation support

**Schema (4 tables):**
```sql
-- queries
CREATE TABLE queries (
    id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- pipeline_runs
CREATE TABLE pipeline_runs (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    model_backend TEXT,
    age_months INTEGER,
    age_group TEXT,
    nutrient TEXT,
    foods_json TEXT,
    servings_json TEXT,
    intent TEXT,
    rda_value REAL,
    rda_unit TEXT,
    consumed_value REAL,
    gap_value REAL,
    gap_percent REAL,
    answer_text TEXT,
    verified BOOLEAN,
    proof_json TEXT,
    latency_ms INTEGER,
    error_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES queries(id)
);

-- evaluation_runs (NEW)
CREATE TABLE evaluation_runs (
    id TEXT PRIMARY KEY,
    run_name TEXT NOT NULL,
    model_backend TEXT,
    ragas_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- evaluation_results (NEW)
CREATE TABLE evaluation_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    query_text TEXT,
    expected_answer TEXT,
    generated_answer TEXT,
    faithfulness REAL,
    relevance REAL,
    precision REAL,
    recall REAL,
    context_utilization REAL,
    FOREIGN KEY (run_id) REFERENCES evaluation_runs(id),
    FOREIGN KEY (query_id) REFERENCES queries(id)
);

CREATE INDEX idx_pipeline_query ON pipeline_runs(query_id);
CREATE INDEX idx_pipeline_time ON pipeline_runs(created_at);
CREATE INDEX idx_eval_run ON evaluation_results(run_id);
```

## 3. Authentication

**MVP Decision:** No authentication
- No login, no signup, no sessions

## 4. Hosting & Deployment

**Platform:** HuggingFace Spaces (Docker)

**Why Docker:**
- FAISS requires C++ libraries (faiss-cpu)
- Ollama requires separate container or external service
- Dockerfile gives control over system dependencies

**Dockerfile:**
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

**Ollama Setup:**
- Option A: Separate HF Space (Ollama)
- Option B: Local development only
- Option C: RunPod / Vast.ai GPU instance

## 5. Third-Party APIs

### Google Gemini API
- Purpose: Decomposer Tier 2 + LLM Synthesis
- Model: gemini-1.5-flash
- Cost: Free tier (1,500 requests/day)

### Ollama (Llama 3.1 8B)
- Purpose: Local LLM backend for comparison
- Model: llama3.1:8b
- Cost: Free (runs locally)
- Setup: ollama pull llama3.1:8b

### HuggingFace Hub
- Purpose: Download embedding models
- Model: sentence-transformers/all-MiniLM-L6-v2

## 6. Key Libraries

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

## 7. Environment Variables

```bash
GEMINI_API_KEY=                    # HF Spaces Secrets
MODEL_BACKEND=gemini               # gemini | llama
LLAMA_BASE_URL=http://localhost:11434
DATA_DIR=./data/processed
FAISS_INDEX_PATH=./data/indices/faiss_icmr.index
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOG_DB_PATH=./data/query_log.db
APP_ENV=development
LOG_LEVEL=INFO
GEMINI_DAILY_LIMIT=1500
MAX_RETRIEVAL_CHUNKS=5
MIN_VERIFIER_MATCH_SCORE=0.95
```

## 8. Folder Structure

```
nutrimind-india/
|
├── frontend/
│   └── app.py
|
├── core/
│   ├── __init__.py
│   ├── models.py
│   ├── decomposer.py
│   ├── retriever.py
│   ├── context_merger.py          # NEW
│   ├── bridge.py
│   ├── synthesizer.py             # LLM synthesis
│   ├── verifier.py
│   └── llm_backends.py            # NEW
|
├── db/
│   └── db.py
|
├── data/
│   ├── processed/
│   │   ├── icmr_rda.csv
│   │   ├── ifct2017.csv
│   │   └── icmr_chunks.json       # NEW
│   ├── indices/
│   │   └── faiss_icmr.index       # NEW
│   └── query_log.db
|
├── scripts/
│   ├── preprocess_data.py
│   ├── build_faiss_index.py       # NEW
│   └── evaluate.py                # NEW
|
├── templates/
│   ├── prompts/
│   │   ├── v1_rag_prompt.txt      # NEW
│   │   └── v1_rda_prompt.txt
│   ├── rda_lookup.j2
│   └── diet_check.j2
|
├── evaluation/                      # NEW
│   ├── ground_truth_qa.json
│   ├── run_ragas.py
│   ├── run_ablation.py
│   ├── run_baseline.py
│   └── results/
|
├── tests/
│   ├── test_decomposer.py
│   ├── test_retriever.py
│   ├── test_context_merger.py     # NEW
│   ├── test_bridge.py
│   ├── test_synthesizer.py
│   ├── test_verifier.py
│   └── test_llm_backends.py       # NEW
|
├── .streamlit/
│   └── config.toml
|
├── .env.example
├── Dockerfile                     # NEW
├── requirements.txt
└── README.md
```

## 9. Data Preprocessing

### 9.1 CSV Preprocessing
**Script:** scripts/preprocess_data.py
- Clean ICMR RDA + IFCT 2017
- Standardize units, food names, age groups
- Output: data/processed/*.csv

### 9.2 Text Chunking + FAISS Index
**Script:** scripts/build_faiss_index.py (NEW)

**Input:** ICMR-NIN text chapters (PDF/TXT)
**Output:** data/indices/faiss_icmr.index + data/processed/icmr_chunks.json

**Steps:**
1. Parse PDF/TXT to plain text
2. Split into chunks: 512 tokens, 64 overlap
3. Embed with all-MiniLM-L6-v2
4. Build FAISS IVF index (nlist=100)
5. Save index + chunk metadata

**Chunk Metadata:**
```json
{
  "chunk_id": "icmr_ch1_sec3_001",
  "text": "...",
  "source": "ICMR-NIN Dietary Guidelines for Indians",
  "chapter": "1",
  "section": "3.2",
  "page": 45,
  "title": "Protein Requirements for Children"
}
```

## 10. Evaluation System (Mandatory)

### 10.1 Ground Truth QA Dataset
**File:** evaluation/ground_truth_qa.json

**Format:**
```json
[
  {
    "id": "qa_001",
    "query": "Iron requirement for 2-year-old",
    "expected_answer": "A 2-year-old needs about 9 mg of iron daily (ICMR RDA 2017).",
    "expected_sources": ["icmr_rda"],
    "nutrient": "iron",
    "age_months": 24,
    "foods": [],
    "category": "rda_lookup"
  },
  {
    "id": "qa_002",
    "query": "Is dal enough protein for 2-year-old?",
    "expected_answer": "Dal provides about X g protein per 100g. A 2-year-old needs Y g daily. One bowl of dal provides Z% of daily needs.",
    "expected_sources": ["icmr_rda", "ifct"],
    "nutrient": "protein",
    "age_months": 24,
    "foods": ["dal"],
    "category": "diet_check"
  }
]
```

**Minimum size:** 50 QA pairs
- RDA lookup: 20 pairs
- Diet check: 20 pairs
- General questions: 10 pairs

### 10.2 RAGAS Evaluation
**Script:** evaluation/run_ragas.py

**Metrics:**
- Faithfulness
- Relevance
- Precision
- Recall
- Context Utilization

**Usage:**
```bash
python evaluation/run_ragas.py --model gemini --output results/gemini_ragas.json
python evaluation/run_ragas.py --model llama --output results/llama_ragas.json
```

### 10.3 Baseline Comparison
**Script:** evaluation/run_baseline.py

**Baselines:**
1. Vanilla LLM: Query -> Gemini/Llama directly (no retrieval)
2. Pure Semantic RAG: Query -> FAISS only -> LLM (no structured data)
3. NutriMind (Full): Query -> Multi-source -> LLM

### 10.4 Ablation Study
**Script:** evaluation/run_ablation.py

**Variants:**
1. Full system (all three sources)
2. No semantic search (CSV only)
3. No structured data (semantic only)
4. No context merger (raw retrieval -> LLM)
5. No verifier (LLM output unfiltered)

### 10.5 Model Comparison
**Script:** evaluation/run_model_comparison.py

**Compare:** Gemini 1.5 Flash vs Llama 3.1 8B
**Metrics:** RAGAS + latency + cost

## 11. Constraints

| Category | Constraint |
|----------|-----------|
| Cost | $0 budget, free APIs + local models |
| Architecture | 100% Python, single process |
| Performance | < 3 sec (Tier 1), < 5 sec (Tier 2 + LLM) |
| Data | Static datasets + pre-built FAISS index |
| UI | Desktop-first, basic mobile |
| Safety | Verifier blocks all unverified outputs |
| Evaluation | Mandatory RAGAS + baselines + ablation |

## 12. Core Principle

> Deterministic data first. LLM for synthesis. Verifier always last. Evaluation proves everything.

---
