"""Archive small, real lab evidence without model weights or credentials.

Run after each GPU stage: python scripts/export_submission.py
Download the executed Colab notebook separately (File > Download > .ipynb).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATTERNS = (
    "submission/**/*.md", "submission/screenshots/*.png",
    "notebooks/*.ipynb", "data/pref/*.parquet", "data/pref/stats.json", "data/pref/samples.json",
    "data/eval/*.json", "data/eval/*.jsonl",
    "adapters/**/adapter_config.json", "adapters/**/dpo_metrics.json",
    "adapters/**/split.json", "adapters/**/trainer_history.json",
    "adapters/**/variants_summary.json", "adapters/**/grpo_metrics.json",
    "bonus/data/*.jsonl", "bonus/data/*.parquet", "bonus/data/review.json",
    "bonus/results/*.json", "bonus/results/*.jsonl",
    "bonus/**/*.md",
)


def export(root: Path, output: Path) -> dict:
    paths = sorted({p for pattern in PATTERNS for p in root.glob(pattern) if p.is_file()})
    manifest = {
        "files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "note": "Evidence only. Save the executed Colab notebook separately. Weights remain in the GPU runtime.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("evidence_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "submission" / "lab22-evidence.zip")
    args = parser.parse_args()
    result = export(REPO, args.output)
    print(f"Saved {len(result['files'])} evidence files to {args.output}")
    print(result["note"])


if __name__ == "__main__":
    main()
