"""Train Python Mentor VN on student-reviewed preferences after NB1.

python bonus/train.py --data bonus/data/reviewed_pairs.jsonl [--export-gguf]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def reviewed_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 200:
        raise ValueError("Challenge requires at least 200 preferences.")
    if not all(r.get("reviewed") is True for r in rows):
        raise ValueError("Review/edit every pair first; reviewed=true must reflect actual student review.")
    if sum(r.get("human_authored") is True for r in rows) < 100:
        raise ValueError("Challenge requires >=100 pairs actually written/rewritten by the student.")
    for r in rows:
        for key in ("prompt", "chosen", "rejected", "topic"):
            if not isinstance(r.get(key), str) or not r[key].strip():
                raise ValueError(f"Invalid {key} in {r.get('id')}")
        if r["chosen"].strip() == r["rejected"].strip():
            raise ValueError(f"Identical responses in {r.get('id')}")
    if len({r["prompt"].strip().casefold() for r in rows}) != len(rows):
        raise ValueError("Duplicate prompts: keep one preference per prompt.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "bonus/data/reviewed_pairs.jsonl")
    parser.add_argument("--export-gguf", action="store_true")
    args = parser.parse_args()
    rows = reviewed_rows(args.data)

    import unsloth  # noqa: F401
    from datasets import Dataset
    from trl import DPOTrainer

    from lab22 import config as C
    from lab22 import data as D
    from lab22 import modeling as MD

    assert C.SFT_MERGED.exists(), "Run NB1 first: reference must be merged SFT."
    model, tokenizer = MD.load_model(C.SFT_MERGED)
    # Hold out entire misconception families: numeric variants of the same
    # exercise must not appear on both sides of this evaluation.
    topics = sorted({r["topic"] for r in rows})
    if len(topics) < 3:
        raise ValueError("Use >=3 topic families so entire topics can be held out.")
    held_topics = set(topics[-max(1, len(topics) // 5):])
    train, held = [], []
    for row in rows:
        pair = D.to_conversational(row)
        if not D.fits_budget(pair, tokenizer, C.MAX_LEN, template_kwargs=C.CHAT_TEMPLATE_KWARGS):
            raise ValueError(f"Pair {row.get('id')} exceeds MAX_LEN; shorten it or explicitly increase the budget.")
        (held if row["topic"] in held_topics else train).append(pair)
    D.assert_disjoint(train, held)
    pref = REPO / "bonus/data"
    pref.mkdir(parents=True, exist_ok=True)
    train_ds, held_ds = Dataset.from_list(train), Dataset.from_list(held)
    train_ds.to_parquet(str(pref / "train.parquet"))
    held_ds.to_parquet(str(pref / "eval.parquet"))
    output = C.ADAPTERS / "python-mentor"
    model = MD.add_lora(model)
    trainer = DPOTrainer(model=model, ref_model=None, processing_class=tokenizer,
                         train_dataset=train_ds, eval_dataset=held_ds,
                         args=MD.dpo_config(output.parent / "python-mentor-checkpoints"))
    result = trainer.train()
    ev = trainer.evaluate()
    trainer.model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    D.save_split_fingerprint(pref, output)
    results = REPO / "bonus/results"
    results.mkdir(parents=True, exist_ok=True)
    meta = {"n_train": len(train), "n_eval": len(held), "held_topics": sorted(held_topics),
            "human_authored": sum(r["human_authored"] is True for r in rows),
            "train_metrics": result.metrics, "eval_metrics": ev,
            "history": trainer.state.log_history, "reference": str(C.SFT_MERGED),
            "beta": C.DPO_BETA, "lr": C.DPO_LR, "seed": C.SEED}
    (results / "training.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.export_gguf:
        del trainer, model
        MD.cleanup()
        model, tokenizer = MD.load_model(output, load_in_4bit=False)
        model.save_pretrained_gguf(str(REPO / "gguf-python-mentor"), tokenizer, quantization_method="q4_k_m")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
