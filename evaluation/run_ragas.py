from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QA_PATH = ROOT / "evaluation" / "ground_truth_qa.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    result = {
        "model": args.model,
        "note": "Development placeholder metrics. Run with real RAGAS dependencies and real data for paper results.",
        "num_questions": len(qa),
        "faithfulness": None,
        "relevance": None,
        "precision": None,
        "recall": None,
        "context_utilization": None,
    }
    output = Path(args.output) if args.output else ROOT / "evaluation" / "results" / f"{args.model}_ragas.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
