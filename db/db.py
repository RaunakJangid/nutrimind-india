from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
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

CREATE INDEX IF NOT EXISTS idx_pipeline_query ON pipeline_runs(query_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_time ON pipeline_runs(created_at);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    run_name TEXT NOT NULL,
    model_backend TEXT,
    ragas_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluation_results (
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

CREATE INDEX IF NOT EXISTS idx_eval_run ON evaluation_results(run_id);
"""

_db_path: str | None = None


def init_db(path: str) -> None:
    global _db_path
    _db_path = path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()}
        if "model_backend" not in existing_columns:
            conn.execute("ALTER TABLE pipeline_runs ADD COLUMN model_backend TEXT")


def _connect(path: str | None = None) -> sqlite3.Connection:
    resolved = path or _db_path or "./data/query_log.db"
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    return conn


def insert_query(query_text: str) -> str:
    query_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute("INSERT INTO queries (id, query_text) VALUES (?, ?)", (query_id, query_text))
    return query_id


def insert_pipeline_run(query_id: str, run_data: dict[str, Any]) -> str:
    run_id = str(uuid.uuid4())
    values = {
        "id": run_id,
        "query_id": query_id,
        "model_backend": run_data.get("model_backend"),
        "age_months": run_data.get("age_months"),
        "age_group": run_data.get("age_group"),
        "nutrient": run_data.get("nutrient"),
        "foods_json": json.dumps(run_data.get("foods", [])),
        "servings_json": json.dumps(run_data.get("servings", {})),
        "intent": run_data.get("intent"),
        "rda_value": run_data.get("rda_value"),
        "rda_unit": run_data.get("rda_unit"),
        "consumed_value": run_data.get("consumed_value"),
        "gap_value": run_data.get("gap_value"),
        "gap_percent": run_data.get("gap_percent"),
        "answer_text": run_data.get("answer_text"),
        "verified": bool(run_data.get("verified", False)),
        "proof_json": json.dumps(run_data.get("proof", [])),
        "latency_ms": run_data.get("latency_ms"),
        "error_reason": run_data.get("error_reason"),
    }
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    with _connect() as conn:
        conn.execute(f"INSERT INTO pipeline_runs ({columns}) VALUES ({placeholders})", tuple(values.values()))
    return run_id


def _decode_run(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["foods"] = json.loads(item.pop("foods_json") or "[]")
    item["servings"] = json.loads(item.pop("servings_json") or "{}")
    item["proof"] = json.loads(item.pop("proof_json") or "[]")
    item["verified"] = bool(item["verified"])
    return item


def get_pipeline_run(query_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE query_id = ? ORDER BY created_at DESC LIMIT 1",
            (query_id,),
        ).fetchone()
    return _decode_run(row) if row else None


def get_recent_queries(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT q.id, q.query_text, q.created_at, p.intent, p.verified, p.error_reason
            FROM queries q
            LEFT JOIN pipeline_runs p ON p.query_id = q.id
            ORDER BY q.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def insert_evaluation_run(run_name: str, model_backend: str, ragas_version: str | None = None) -> str:
    run_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO evaluation_runs (id, run_name, model_backend, ragas_version) VALUES (?, ?, ?, ?)",
            (run_id, run_name, model_backend, ragas_version),
        )
    return run_id


def insert_evaluation_result(run_id: str, query_id: str, result: dict[str, Any]) -> str:
    result_id = str(uuid.uuid4())
    values = {
        "id": result_id,
        "run_id": run_id,
        "query_id": query_id,
        "query_text": result.get("query_text"),
        "expected_answer": result.get("expected_answer"),
        "generated_answer": result.get("generated_answer"),
        "faithfulness": result.get("faithfulness"),
        "relevance": result.get("relevance"),
        "precision": result.get("precision"),
        "recall": result.get("recall"),
        "context_utilization": result.get("context_utilization"),
    }
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    with _connect() as conn:
        conn.execute(f"INSERT INTO evaluation_results ({columns}) VALUES ({placeholders})", tuple(values.values()))
    return result_id
