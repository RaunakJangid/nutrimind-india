# Product Requirements Document (PRD) — NutriMind India (RAG Paper v4)

## 1. App Identity
- **Name:** NutriMind-India
- **Tagline:** "Ask. Verify. Feed right — with trusted Indian nutrition data."
- **Research Goal:** Publishable RAG system with verified nutrition reasoning

## 2. Problem Statement

Indian parents (Tier 2/3 cities, mothers 25–35) cannot get reliable, India-specific answers to child nutrition questions. Current options fail: Google is inconsistent, AI tools hallucinate, doctors are expensive, and nutrition apps use Western datasets.

**Research Gap:** No existing RAG system combines structured nutrition tables (ICMR RDA, IFCT), unstructured text chapters (ICMR-NIN), and a verification layer for child nutrition in the Indian context.

## 3. Target User

**Primary:** Mother, 25–35, Tier 2/3 India, smartphone user, child 6 months – 5 years.

**Secondary:** Researchers evaluating RAG systems for health domains.

## 4. Core Value Proposition

NutriMind delivers **verifiable nutrition answers** through a multi-source RAG pipeline:
- Structured data: ICMR RDA + IFCT 2017 (CSV)
- Unstructured text: ICMR-NIN dietary guidelines (vector search)
- LLM synthesis with citation grounding
- Verification engine preventing hallucinations

## 5. Core Features (Must-Have — RAG Paper)

### F1. Chat-based Query Interface
- Free-text input with example queries
- Multi-turn context retention

### F2. Multi-Source Retrieval (RAG Core)
**Three retrieval paths:**
1. **Structured Lookup:** ICMR RDA CSV (exact match: nutrient + age_group)
2. **Structured Lookup:** IFCT 2017 CSV (exact match: food + nutrient)
3. **Semantic Search:** ICMR-NIN text chapters via FAISS/ChromaDB (vector similarity)

### F3. Context Merger + Conflict Detector
- Merge outputs from all three retrieval paths
- Detect contradictions between sources
- Flag conflicts for user awareness
- Priority: structured data > semantic text when conflict exists

### F4. LLM Synthesis with Prompt Engineering
- Generate natural-language answers from retrieved context
- Include citations to sources
- Prompt template versioning for reproducibility
- Two backends: Gemini (cloud) + Llama 3.1 8B (local via Ollama)

### F5. Nutrient Gap Calculation
- Required vs consumed intake
- Gap (absolute + percentage)
- Multi-food aggregation

### F6. Proof Table (Trust Layer)
- Every answer shows: nutrient values, sources, food mappings, confidence scores

### F7. Verification Engine
- Validates: RDA values match dataset, IFCT values match dataset
- Validates: LLM output is grounded in retrieved context
- Validates: derived values (gap, %) are mathematically correct
- If verification fails: block answer, show fallback

### F8. Evaluation System (Mandatory — Paper Requirement)
- Ground-truth QA dataset (curated from ICMR-NIN + IFCT)
- RAGAS metrics: Faithfulness, Relevance, Precision, Recall
- Baseline comparison: Vanilla LLM vs Pure Semantic RAG vs NutriMind
- Ablation study: remove each retrieval path, measure impact
- Model comparison: Gemini vs Llama 3.1 8B

## 6. Nice to Have (V2)
- Hindi language support
- Voice input
- Meal planning suggestions
- Child growth tracking
- Personal profiles

## 7. Out of Scope (Explicit)
- ❌ User login/signup (MVP)
- ❌ Payments/subscriptions
- ❌ Medical diagnosis or prescriptions
- ❌ Wearable/device integrations
- ❌ Real-time diet tracking
- ❌ Native mobile app (web-only via Streamlit)
- ❌ Social/community features

## 8. User Stories

1. As a mother, I want to know my child's daily nutrient requirement so that I can ensure proper nutrition.
2. As a caregiver, I want to check if dal + rice is enough so that I can adjust meals if needed.
3. As a user, I want to see the source of the answer so that I can trust it.
4. As a first-time parent, I want simple explanations so that I can understand without medical knowledge.
5. As a cautious user, I want the system to avoid wrong answers so that I don't harm my child.
6. As a researcher, I want reproducible RAG evaluation so that I can compare system variants.

## 9. Success Metrics

### Product Metrics
| Metric | Target |
|--------|--------|
| Verification Pass Rate | ≥ 95% |
| Response Time (Tier 1) | < 3 sec |
| Response Time (Tier 2) | < 5 sec |
| Query Success Rate | ≥ 85% |
| Trust Engagement | ≥ 40% |

### Research Metrics (Mandatory)
| Metric | Target | Baseline Comparison |
|--------|--------|---------------------|
| RAGAS Faithfulness | ≥ 0.85 | vs Vanilla LLM ≥ 0.60 |
| RAGAS Relevance | ≥ 0.80 | vs Vanilla LLM ≥ 0.55 |
| RAGAS Precision | ≥ 0.80 | vs Pure Semantic RAG ≥ 0.65 |
| RAGAS Recall | ≥ 0.75 | vs Pure Semantic RAG ≥ 0.60 |
| Context Utilization | ≥ 0.70 | — |
| Llama 3.1 8B vs Gemini | Within 10% | — |

## 10. Product Principle (Non-negotiable)

> "Never return an answer that cannot be verified against trusted data."

## 11. MVP Done Criteria (RAG Paper Ready)

System is DONE when:
- User asks: "Is dal enough protein for 2-year-old?"
- System returns: Answer + Gap + Proof table + Verified = TRUE
- RAGAS evaluation complete with baselines
- Ablation study shows multi-source retrieval outperforms single-source
- Both Gemini and Llama 3.1 8B backends functional

---
