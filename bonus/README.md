# Python Mentor VN — gia sư Python gợi mở (experimental v0)

**Người thực hiện:** Hoàng Công Minh · 2A202602774 · K4, Track 3.

**Trạng thái:** đã chuẩn bị mã, dữ liệu nháp và quy trình đánh giá; chưa huấn luyện adapter, chưa có GGUF hoặc tương tác từ mô hình. Không coi câu trả lời trong dữ liệu nháp là kết quả thực nghiệm.

## Người dùng

Sản phẩm dành cho người Việt mới học Python, đặc biệt sinh viên đang làm bài tập đầu tiên về biến, điều kiện, vòng lặp, danh sách và hàm. Người học thường có đoạn mã ngắn nhưng chưa hiểu thông báo lỗi, hoặc biết đáp án mong muốn mà chưa biết cách tự kiểm tra. Một chatbot đưa ngay lời giải hoàn chỉnh có thể giúp nộp bài nhanh nhưng khiến người học bỏ qua nguyên nhân. Python Mentor VN hướng đến một phiên học ngắn: người học đưa câu hỏi, nhận một gợi ý cụ thể, thử sửa rồi quay lại đối chiếu. Demo hiện hỗ trợ từng câu hỏi độc lập; chưa lưu lịch sử nhiều lượt.

## Kiến thức chuyên môn và dữ liệu

Các cặp ưu tiên tập trung vào hiểu nhầm có thể kiểm tra bằng Python: input trả về chuỗi; range loại cận cuối; append trả về None; hai biến có thể trỏ cùng danh sách; giá trị mặc định khả biến được dùng lại; phép in trên dictionary kiểm tra khoá; except quá rộng che lỗi; phép chia nguyên làm tròn xuống; và sắp xếp chuỗi số cần xác định khoá so sánh. Dữ liệu có thêm tình huống người học đã tự giải và xin đáp án để đối chiếu. Tình huống này giúp tránh dạy mô hình từ chối đưa lời giải trong mọi trường hợp.

File `data/draft_pairs.jsonl` chứa 200 bản nháp có AI hỗ trợ, gồm mười nhóm chủ đề với hai mươi tình huống mỗi nhóm. Những tình huống cùng nhóm khá giống nhau; đây là hạn chế về đa dạng, cần biên tập sâu trước khi huấn luyện. Chưa có cặp nào được đánh dấu là do sinh viên tự viết. Theo BONUS-CHALLENGE, sinh viên phải thực sự tự viết hoặc viết lại ít nhất 100 cặp, kiểm tra cả hai phản hồi, rồi tạo `data/reviewed_pairs.jsonl`. Chỉ đặt `human_authored=true` khi việc viết lại đã diễn ra và `reviewed=true` sau khi đã kiểm tra. Không đổi cờ hàng loạt để coi nháp AI là dữ liệu chuyên môn của mình.

## Mục tiêu ứng dụng

Khi người học yêu cầu một gợi ý, phản hồi được chọn phải nêu đúng nguyên nhân hoặc đưa một phép kiểm tra nhỏ, sau đó mời người học tự thử. Phản hồi bị loại có thể đưa ngay cả đáp án, dùng quá nhiều tiếng Anh, hoặc giải thích sai. Khi người học đã làm xong và muốn đối chiếu, phản hồi được chọn có thể cung cấp code hoàn chỉnh ngắn gọn. Vì vậy mục tiêu là chọn mức hỗ trợ theo ngữ cảnh, không chỉ làm câu trả lời ngắn đi. Trong đánh giá, ưu tiên tính đúng trước mức gợi mở: một gợi ý thân thiện nhưng sai kiến thức vẫn là thất bại.

## Sản phẩm và cách chạy

Luồng huấn luyện dùng mô hình SFT đã gộp của NB1 làm reference, gắn LoRA mới và dùng cùng cấu hình DPO của lab. Dữ liệu được tách theo cả nhóm chủ đề để những biến thể chỉ đổi số của cùng bài không xuất hiện trên cả train và held-out. Script từ chối dữ liệu vượt ngân sách token thay vì âm thầm làm mất phần trả lời. Sau huấn luyện, lưu metrics, lịch sử trainer, adapter và có thể xuất GGUF Q4_K_M. Trọng số không đưa vào Git.

Trong môi trường GPU đã chạy NB1, từ gốc repo:

```bash
python bonus/train.py --data bonus/data/reviewed_pairs.jsonl --export-gguf
python bonus/evaluate.py
```

Sau khi tải GGUF về laptop và cài `llama-cpp-python`, chạy một lệnh để trò chuyện:

```bash
python bonus/demo/serve.py --model gguf-python-mentor/<file>.gguf
```

Demo mặc định dùng CPU để phù hợp laptop có VRAM thấp. GGUF dùng chat template chứa trong file; chưa xác nhận chất lượng định dạng và tốc độ cho đến khi có file xuất thật. Không tự thực thi đoạn code do người dùng hoặc mô hình đưa ra.

## Năm tình huống kiểm tra độc lập

Script `evaluate.py` dùng năm câu hỏi không nằm trong dữ liệu nháp: gợi ý tính tổng bằng vòng lặp; phân biệt gán với so sánh; xoá phần tử trong lúc duyệt danh sách; đối chiếu hàm bình phương đã làm xong; và sửa lỗi chỉ số bằng độ dài danh sách. Script lưu câu trả lời thật của SFT và adapter vào `results/five_comparisons.jsonl`, rồi ghi `demo/5-samples.md`. Hiện chưa có đầu ra thật, vì vậy không đưa câu trả lời giả vào báo cáo. Khi có kết quả, nhờ người học chấm mù từng cặp theo tính đúng, mức hỗ trợ phù hợp và khả năng tự làm bước tiếp theo; ghi cả phản hồi tiêu cực.

## Hạn chế trung thực

Chưa thử với người học thật, chưa có bằng chứng DPO cải thiện chất lượng dạy kèm và chưa đo tốc độ CPU. Các nháp có cấu trúc lặp lại; giữ nguyên có thể khiến mô hình học văn mẫu. Reward accuracy không thay thế đánh giá sư phạm. Dữ liệu SFT của lab mang giấy phép CC BY-NC nên cần xét đầy đủ giấy phép của mô hình, dữ liệu và adapter trước khi sử dụng thương mại hoặc đăng Hub. Demo không lưu hội thoại, nhưng người dùng vẫn nên tránh nhập mật khẩu, dữ liệu cá nhân và bài kiểm tra có quy định cấm trợ giúp. Model card ghi rõ phạm vi, nguồn dữ liệu và trạng thái thực nghiệm.
