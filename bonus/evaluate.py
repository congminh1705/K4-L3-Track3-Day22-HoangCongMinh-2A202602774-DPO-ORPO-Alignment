"""Save five real, unseen SFT vs Python Mentor comparisons on the GPU runtime."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
PROMPTS = [
    "Mình có danh sách [4, 7, 10] và muốn tính tổng bằng vòng lặp. Chỉ cho một gợi ý để mình tự làm.",
    "Trong Python, vì sao `if diem = 8:` báo SyntaxError? Gợi ý cách phân biệt phép gán với so sánh nhé.",
    "Mình viết `for x in a: a.remove(x)` để xoá số âm, nhưng bị bỏ sót. Cho mình cách tự kiểm tra nguyên nhân.",
    "Mình đã tự giải bài bình phương: `def square(x): return x*x`. Kiểm tra giúp và cho bản hoàn chỉnh để đối chiếu.",
    "Code của mình báo IndexError khi lấy a[len(a)]. Mình chưa hiểu chỉ số, đừng đưa ngay toàn bộ lời giải.",
]


def main() -> None:
    import unsloth  # noqa: F401
    from lab22 import config as C
    from lab22 import modeling as MD

    output = REPO / "bonus/results"
    output.mkdir(parents=True, exist_ok=True)
    responses = {}
    for label, checkpoint in (("sft", C.SFT_MERGED), ("mentor", C.ADAPTERS / "python-mentor")):
        if not checkpoint.exists():
            raise RuntimeError(f"Missing {checkpoint}: run NB1 and bonus/train.py first.")
        model, tokenizer = MD.load_model(checkpoint)
        responses[label] = MD.generate(model, tokenizer, PROMPTS, max_new_tokens=256, batch_size=1)
        del model, tokenizer
        MD.cleanup()
    rows = [{"id": f"mentor-test-{i + 1}", "prompt": prompt,
             "sft": responses["sft"][i], "mentor": responses["mentor"][i]}
            for i, prompt in enumerate(PROMPTS)]
    (output / "five_comparisons.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    parts = ["# Năm tương tác thật — SFT và Python Mentor VN\n",
             "Sinh greedy trên cùng prompt và chat template; cần người học chấm mù tính đúng và mức gợi mở.\n"]
    for row in rows:
        parts.append(f"## {row['id']}\n\n**Câu hỏi:** {row['prompt']}\n\n"
                     f"**SFT:**\n\n{row['sft']}\n\n**Python Mentor:**\n\n{row['mentor']}\n")
    (REPO / "bonus/demo/5-samples.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved five real comparisons to {output}")


if __name__ == "__main__":
    main()
