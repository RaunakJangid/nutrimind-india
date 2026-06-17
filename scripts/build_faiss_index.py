"""Build a FAISS index for ICMR-NIN text chunks.

This script uses data/processed/icmr_chunks.json as input. In development, the
checked-in placeholder chunks keep the pipeline runnable. Replace that file with
real ICMR-NIN chunks before paper evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "icmr_chunks.json"
INDEX_PATH = ROOT / "data" / "indices" / "faiss_icmr.index"


def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("No chunks found")
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers and faiss-cpu to build the FAISS index") from exc

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    embeddings = model.encode([chunk["text"] for chunk in chunks], normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Wrote {INDEX_PATH} with {len(chunks)} chunks")


if __name__ == "__main__":
    main()
