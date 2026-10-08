"""Check the completed student loss and safe evidence handoff."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.export_submission import export
from bonus.build_data import build_rows
from bonus.train import reviewed_rows


def student_loss():
    source = REPO / "notebooks/00_dpo_loss_from_scratch.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "my_dpo_loss")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["my_dpo_loss"]


def test_student_loss_has_correct_gradient_and_is_stable():
    loss = student_loss()
    pc, pr = torch.tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)
    ref = torch.tensor([0.0])
    value = loss(pc, pr, ref, ref)
    assert value.item() == pytest.approx(0.69314718)
    value.backward()
    assert pc.grad.item() < 0 and pr.grad.item() > 0
    for margin in (-10000.0, 10000.0):
        assert torch.isfinite(loss(torch.tensor([margin]), ref, ref, ref, beta=1.0))


def test_student_loss_demonstrates_displacement():
    loss = student_loss()
    c, r = torch.tensor([-20.0]), torch.tensor([-22.0])
    assert torch.allclose(loss(c + 1, r - 1, c, r), loss(c - 3, r - 5, c, r))


def test_export_includes_evidence_and_excludes_weights_and_secrets(tmp_path):
    for name, content in {
        "data/eval/judge_summary.json": '{"n": 50}',
        "adapters/dpo/adapter_config.json": '{}',
        "adapters/dpo/adapter_model.safetensors": 'weights',
        "submission/screenshots/02-sft-loss.png": 'image fixture',
        ".env": 'API_KEY=secret', "models/config.json": '{}',
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    archive_path = tmp_path / "evidence.zip"
    manifest = export(tmp_path, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "data/eval/judge_summary.json", "adapters/dpo/adapter_config.json",
            "submission/screenshots/02-sft-loss.png", "evidence_manifest.json",
        }
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest


def test_bonus_drafts_do_not_claim_student_review_or_authorship(tmp_path):
    rows = build_rows()
    assert len(rows) == 200 and len({r['prompt'] for r in rows}) == 200
    assert not any(r["reviewed"] or r["human_authored"] for r in rows)
    path = tmp_path / "draft.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Review/edit"):
        reviewed_rows(path)
    for row in rows:
        row["reviewed"] = True
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="actually written"):
        reviewed_rows(path)


def test_colab_can_verify_and_back_up_before_bonus():
    import nbformat
    from build_colab import render

    for tier in ("T4", "BIGGPU"):
        raw = render(tier)
        nb = nbformat.reads(json.dumps(raw), as_version=4)
        nbformat.validate(nb)
        sources = ["".join(c["source"]) for c in raw["cells"]]
        core = next(i for i, s in enumerate(sources) if s.startswith("---\n# ⏵ `notebooks/04_"))
        bonus = next(i for i, s in enumerate(sources) if s.startswith("---\n# ⏵ `notebooks/03b_"))
        assert core < bonus
        assert any("scripts/export_submission.py" in s for s in sources[core:bonus])
        assert any(s.startswith("%%writefile /content/lab22/scripts/verify.py") for s in sources)
        assert any(s.startswith("%%writefile /content/lab22/notebooks/03_dpo_train.py") for s in sources)
