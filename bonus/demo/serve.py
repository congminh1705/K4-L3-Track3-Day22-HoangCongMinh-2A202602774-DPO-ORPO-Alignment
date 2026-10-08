"""Local interactive GGUF demo: python bonus/demo/serve.py --model <file.gguf>."""
from __future__ import annotations

import argparse
from pathlib import Path

SYSTEM = (
    "Bạn là gia sư Python bằng tiếng Việt cho người mới học. Giải thích chính xác, ngắn gọn. "
    "Khi người học xin gợi ý, nêu nguyên nhân hoặc bước kiểm tra rồi hỏi một câu gợi mở; "
    "không đưa ngay cả lời giải. Khi họ đã tự thử và muốn đối chiếu, đưa lời giải kèm giải thích. "
    "Không khẳng định đã chạy code nếu chưa chạy; không yêu cầu mật khẩu hoặc dữ liệu riêng tư."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpu-layers", type=int, default=0, help="0 uses CPU; -1 offloads all layers")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"GGUF not found: {args.model}")
    from llama_cpp import Llama

    llm = Llama(model_path=str(args.model), n_ctx=2048, n_gpu_layers=args.gpu_layers, verbose=False)
    print("Python Mentor VN (experimental). Nhập /quit để thoát. Mỗi lượt là một bài độc lập.")
    try:
        while True:
            try:
                prompt = input("\nBạn: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt == "/quit":
                break
            if not prompt:
                continue
            result = llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=384,
            )
            print("Gia sư:", result["choices"][0]["message"]["content"])
    finally:
        llm.close()


if __name__ == "__main__":
    main()
