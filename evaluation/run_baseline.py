from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["vanilla", "semantic", "full"], default="full")
    args = parser.parse_args()
    output = ROOT / "evaluation" / "results" / f"baseline_{args.baseline}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"baseline": args.baseline, "status": "placeholder"}, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
