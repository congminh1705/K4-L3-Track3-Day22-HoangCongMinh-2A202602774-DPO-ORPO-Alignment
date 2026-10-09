# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Hoàng Công Minh — 2A202602774  
**Khoá:** K4 · Track 3  
**Tier đã chạy:** Colab T4  
**Ngày:** 2026-10-09

**Trạng thái:** Đã có bằng chứng NB0–NB4 trong notebook và ZIP. Các mục bonus NB3b, NB5–NB7, β-sweep chưa có kết quả trong gói evidence này.

## 0. Kết quả NB0 đã chạy

Notebook `notebooks/00_dpo_loss_from_scratch.ipynb` giữ output của 9 cell code chạy trên CPU:

| Kiểm tra | Kết quả thật |
|---|---:|
| `my_dpo_loss` khớp bản tham chiếu, sai số ≤1e-6 | 0.6981 |
| Policy = reference | 0.6931 ≈ ln(2) |
| Margin +2, chosen tăng 1 / rejected giảm 1 | loss 0.127; reward +1 / −1 |
| Margin +2, chosen giảm 3 / rejected giảm 5 | loss 0.127; reward −3 / −5 |
| RPO trên hai kịch bản trên | 2.027 / 2.427 |

Margin tăng trong khi xác suất chosen giảm vì DPO tối ưu hiệu giữa hai log-ratio. Nếu rejected giảm nhanh hơn chosen, hiệu vẫn tăng. Với β=1, chosen giảm 3 nat và rejected giảm 5 nat làm margin tăng 2 nat, bằng kịch bản chosen tăng 1 nat và rejected giảm 1 nat. DPO cho cùng loss dù hướng thay đổi xác suất chosen khác nhau. RPO thêm NLL(chosen), nên phạt kịch bản chosen giảm mạnh hơn. Đây là số liệu đồ chơi của NB0, chưa phải chẩn đoán mô hình đã huấn luyện.

> Các số liệu NB1–NB4 dưới đây được lấy từ ZIP evidence: `dpo_metrics.json`, `stats.json`, `judge_summary.json` và `side_by_side.jsonl`.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4; VRAM cao nhất ghi nhận 6.93 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (tiếng Việt), 800 train / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.875% |
| DPO: β / tốc độ học / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | Panel reward model Skywork Qwen3-4B và Llama-3.2-3B |
| Chi phí | Không có số liệu chi phí trong evidence; chạy trên Google Colab |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 1,895 giây (~31.6 phút) |
| VRAM cao nhất | 6.93 GB |
| Reward gap cuối trên train (chosen − rejected) | 0.0911 |
| Reward accuracy trên held-out | 0.73 |
| Reward margin trên held-out | 0.0831 |
| Chẩn đoán tự động | INTENDED |
| Độ dài trung bình NB4, SFT → DPO | 594.6 → 586.7 ký tự (toàn bộ 58 mục) |

---

## 3. Đọc đường reward

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên tập train, reward cuối của chosen là 0.391 và rejected là 0.300, nên khoảng cách đạt 0.091. Trên held-out, chosen đạt 0.403 còn rejected đạt 0.320, margin là 0.083. Cả hai reward đều dương vì policy đã dịch chuyển so với reference theo hướng tăng xác suất tương đối; chosen cao hơn rejected ở cả train lẫn held-out. Khoảng cách held-out gần với train, và reward accuracy held-out là 0.73, nên kết quả không chỉ xuất hiện trên tập huấn luyện. Loss ghi nhận giảm từ 0.692 ở log đầu xuống 0.675 cuối. Các số liệu này phù hợp với chẩn đoán tự động INTENDED: mô hình ưu tiên chosen hơn rejected, mà held-out đi cùng chiều với train. Tuy vậy, reward margin đo mức phân biệt cặp theo mục tiêu huấn luyện, không tự chứng minh câu trả lời tốt hơn với người dùng. NB4 là phép đánh giá riêng cho chất lượng đầu ra; kết quả của NB4 gần ngang SFT và chưa chứng minh cải thiện có ý nghĩa thống kê.

---

## 4. So sánh SFT và SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate DPO (CI 95%) | Win rate cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 9 | 9 | 32 | 0.500 (0.420–0.580) | 0.489 | 0.556 |
| hữu ích — helpfulness | 4 | 2 | 0 | 2 | 0.750 (0.500–1.000) | 0.750 | 0.500 |
| an toàn — safety | 4 | 0 | 1 | 3 | 0.375 (0.125–0.500) | 0.500 | 1.000 |

Tổng cả 8 câu cố định và 50 câu held-out có 58 mục: DPO thắng 11, SFT thắng 10, hoà 37; win rate DPO 0.509, CI 95% 0.431–0.586. CI held-out chứa 0.5 nên chưa có bằng chứng DPO tốt hơn SFT. Hai mô hình trả lời dài trung bình gần bằng nhau; DPO còn ngắn hơn một chút, và win rate trên cặp có độ dài gần nhau là 0.489. Vì vậy kết quả không gợi ý DPO thắng nhờ chỉ viết dài hơn.

Panel có `sanity_accuracy` tổng hợp 1.0 và không có cặp lỗi định dạng. Tuy nhiên, kết quả sanity theo từng judge là Qwen3 0.667 và Llama 1.0; cần thận trọng với đánh giá tiếng Việt của judge Qwen3. Win rate held-out từng judge là 0.46 (Qwen3) và 0.50 (Llama), khá gần nhau; mức đồng thuận ghi nhận là 0.828. Tương quan điểm với độ dài thấp (Spearman 0.126 và 0.007), cũng không cho thấy điểm chủ yếu dựa vào độ dài.

Ví dụ hữu ích `h1` hỏi cách quicksort hoạt động: cả hai câu trả lời giải thích pivot và chia danh sách tương tự nhau; judge chọn DPO, nhưng khác biệt nhỏ nên không đủ kết luận rộng. Ví dụ an toàn `s1` hỏi công thức chế tạo chất nổ: cả hai mô hình đều từ chối và khuyên tránh hành vi nguy hiểm; đây là một tie, phù hợp với việc DPO chưa cải thiện rõ nhóm safety. Các câu này là ví dụ minh hoạ; tổng hợp điểm vẫn cho thấy đa số là hoà.

---

## 5. Đánh đổi theo β (bonus)

Chưa chạy β-sweep; không có số liệu để so sánh. Dự đoán: β nhỏ hơn có thể cho policy đi xa reference hơn, nên margin tăng nhanh nhưng rủi ro làm giảm chất lượng ngôn ngữ gốc. β lớn hơn thường kìm thay đổi và có thể giữ câu trả lời gần SFT hơn. Cần chạy cùng split và seed để kiểm tra giả thuyết.

---

## 6. Một quyết định quan trọng nhất

Tôi chọn β=0.1 cho lần chạy DPO. Phương án thay thế hợp lý là β=0.05 hoặc β=0.5. β nhỏ hơn có thể làm mô hình đổi mạnh hơn theo preference, nhưng cũng dễ đi quá xa mô hình reference; β lớn hơn có thể giữ ổn định hơn nhưng học preference chậm hoặc margin nhỏ. Tôi giữ 0.1 theo cấu hình T4 của lab để có một lần chạy vừa sức, rồi đánh giá riêng train, held-out và đầu ra NB4.

Kết quả xác nhận DPO đã học được tín hiệu preference trong phạm vi bài lab: reward gap cuối trên train là 0.091, held-out gap là 0.083, reward accuracy held-out là 0.73, và chẩn đoán tự động là INTENDED. Điều tôi thấy đáng chú ý là bằng chứng này chưa chuyển thành thắng lợi rõ ở đánh giá câu trả lời: held-out NB4 có win rate 0.50 với khoảng tin cậy 0.42–0.58, chứa mức ngang bằng 0.5. Hai reward model cũng cho win rate gần nhau, 0.46 và 0.50, nhưng mẫu chỉ có 50 câu held-out nên độ bất định còn lớn. Nếu làm lại, tôi sẽ giữ nguyên β trong lần lặp đầu để so sánh công bằng, tăng số lượng câu đánh giá độc lập, kiểm tra thủ công thêm các trường hợp hoà và theo dõi chất lượng tiếng Việt cùng format đầu ra. Sau đó mới chạy sweep β trên cùng split để biết margin cao hơn có thực sự tạo câu trả lời hữu ích và an toàn hơn hay không.

---

## 7. Bộ đo chuẩn (bonus NB6)

Chưa chạy NB6; không có kết quả IFEval, GSM8K hoặc Global-MMLU-vi trong ZIP.

---

## 8. Biến thể loss (bonus NB3b)

Chưa chạy NB3b; không có số liệu so sánh DPO, RPO, DPO-norm, LD-DPO và ORPO.

---

## 9. GRPO (bonus NB7)

Chưa chạy NB7; chưa có kết quả độ chính xác hoặc reward GRPO.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng API khác (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)

---

## Điều bất ngờ nhất

Reward margin held-out đi cùng chiều với train, nhưng NB4 vẫn cho kết quả gần hoà. Điều đó nhắc tôi rằng học đúng cặp preference trong tập dữ liệu chưa đồng nghĩa với cải thiện rõ trên đánh giá đầu ra độc lập.

