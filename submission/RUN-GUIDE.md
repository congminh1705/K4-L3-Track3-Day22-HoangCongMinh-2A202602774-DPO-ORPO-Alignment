# Chạy phần GPU và hoàn tất bằng chứng

Máy đã kiểm tra: RTX 3050 laptop 4 GB, Windows, Python 3.13. NB0 chạy trên CPU. Cấu hình T4 của lab cần khoảng 9–12 GB VRAM khi DPO và GPU 16 GB được khuyến nghị. Chưa thực hiện NB1–NB7; chưa có adapter, ảnh huấn luyện, điểm judge hay benchmark.

## Colab T4

1. Tải `colab/Lab22_DPO_T4.ipynb` lên Google Colab, chọn T4 GPU.
2. Chạy setup rồi NB0 → NB1 → NB2 → NB3 → NB4. Không giảm số mẫu mặc định khi lấy bằng chứng để nộp: SFT 1.000; preference train 800, held-out 100; judge ≥50 câu held-out và 8 câu cố định.
3. Sau NB4, tải `submission/lab22-evidence.zip` từ bảng Files và tải notebook bằng **File → Download → .ipynb**. Lưu bản có output vào `notebooks/Lab22_DPO_T4_executed.ipynb` trong repo này; giữ file Colab gốc được sinh từ source.
4. Chạy tiếp NB3b, NB5, NB6, NB7 và β-sweep nếu còn thời gian GPU. Các stage tự giải phóng model của stage trước. ZIP bằng chứng được cập nhật sau từng stage; tải lại bản cuối và notebook giữ output.
5. Giải nén ZIP vào gốc repo, giữ đúng các đường dẫn. ZIP chỉ chứa file nhỏ, không có trọng số, API key hoặc `.env`. Giữ model/adapter/GGUF trong runtime hoặc lưu riêng nếu cần chạy lại; ZIP không giúp khôi phục huấn luyện sau khi Colab xoá phiên.

## Bài phản tư và kiểm tra

- §1–§2: `data/pref/stats.json`, `adapters/dpo/dpo_metrics.json`.
- §3: đồ thị reward riêng chosen/rejected của train và held-out, cùng `trainer_history.json`. Viết ≥100 từ dựa trên diễn biến thực tế.
- §4: `data/eval/judge_summary.json`, `side_by_side.jsonl`. So CI với 0.5; đọc sanity, per_judge, độ dài và chọn một ví dụ helpfulness, một safety.
- §6: viết ≥150 từ về một quyết định thật, phương án thay thế và kết quả quan sát.
- Bonus: variants summary, deploy meta, benchmark results, GRPO metrics và các thư mục dpo-b* tương ứng. Không đánh dấu đã hoàn thành khi chỉ có code.
- Kiểm tra đầy đủ ngay trong runtime GPU bằng `python scripts/verify.py` sau khi điền REFLECTION. Lệnh cần model SFT đã gộp và dữ liệu thật. Sau khi chỉ nhập ZIP về máy, verifier vẫn có thể báo thiếu model vì trọng số được loại khỏi ZIP/Git theo yêu cầu.

## Bonus khác

Chấm chéo API cần tài khoản/key và tên model do bạn chọn. Sau khi RM đã chấm, chỉ chạy lại cell §3–§4 NB4 trên cùng outputs: đặt `C.JUDGE_PROVIDER`, `C.JUDGE_MODEL` và lấy key từ Colab secrets. Không sinh lại outputs nếu muốn giữ chính xác cùng hash. Summary sẽ ghi cross_judge nếu có kết quả cùng dữ liệu. API có chi phí riêng; chưa chạy bonus này.

HF Hub cần tài khoản, tên repo và token có quyền ghi. Không đăng bản nháp model card hoặc adapter chưa huấn luyện; rà soát giấy phép dữ liệu SFT trước khi phát hành. Chưa upload model.

Python Mentor VN: xem `bonus/README.md`. Có 200 nháp AI, chưa phải ≥100 cặp do sinh viên tự tạo. Tự viết/viết lại và duyệt dữ liệu trước khi chạy `bonus/train.py`; sau đó `bonus/evaluate.py` sinh năm tương tác thật. Có thể chạy các script này trong cùng runtime Colab sau NB1.

## Kiểm tra CPU trên Windows

```powershell
.\.venv\Scripts\python.exe -m pytest -q scripts/ --basetemp=.venv/pytest-next-run
.\.venv\Scripts\python.exe scripts/execute_nb0.py
.\.venv\Scripts\python.exe -X utf8 scripts/verify.py
```

Trên máy này môi trường CPU được cài riêng, không cài toàn bộ dependencies CUDA vào Windows. `verify` phải tiếp tục báo các bằng chứng GPU còn thiếu cho đến khi chúng được tạo thật. Khi commit, đưa notebook có output, screenshots, JSON/JSONL, Parquet và REFLECTION vào Git; không đưa `.venv`, weights, GGUF hoặc token vào Git. Source không quy định số commit hay tên commit cố định.
