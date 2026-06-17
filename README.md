# NutriMind-India RAG v4

NutriMind-India is a Streamlit prototype for verified child-nutrition answers in the Indian context.

The v4 architecture combines:
- Structured ICMR RDA lookup
- Structured IFCT 2017 food nutrient lookup
- Semantic ICMR-NIN text retrieval
- Context merging and conflict detection
- Gemini or Llama synthesis, with deterministic fallback for local development
- Verification before answers are shown
- Evaluation scaffolding for RAGAS, baselines, ablation, and model comparison

## Setup

```bash
python -m pip install -r requirements.txt
python scripts/preprocess_data.py
streamlit run frontend/app.py
```

Optional FAISS index build:

```bash
python scripts/build_faiss_index.py
```

Optional Llama backend:

```bash
ollama pull llama3.1:8b
```

## Run Tests

```bash
pytest tests/ -v --tb=short
```

## Evaluation

The starter ground-truth file is `evaluation/ground_truth_qa.json`. Placeholder runners are included so the workflow is reproducible while real ICMR-NIN chunks and final metrics are being prepared.

```bash
python evaluation/run_ragas.py --model gemini
python evaluation/run_baseline.py --baseline full
python evaluation/run_ablation.py --variant full
python evaluation/run_model_comparison.py
```

## Data Note

The checked-in CSV and text files are placeholder development data. Replace them with real processed ICMR RDA, IFCT 2017, and ICMR-NIN chunks before research evaluation or deployment.
