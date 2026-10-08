# Python Mentor VN experimental v0

## Trạng thái

Thiết kế và code; chưa huấn luyện, chưa phát hành trọng số. Không có số đo chất lượng hoặc tốc độ.

## Dùng để làm gì

Hỗ trợ người mới học Python bằng tiếng Việt: giải thích lỗi, gợi mở từng bước và đối chiếu bài đã tự làm. Phản hồi đúng mức hỗ trợ mà người học yêu cầu.

## Không dùng để làm gì

Không chấm điểm chính thức, không thay giáo viên, không bảo đảm code đúng hoặc an toàn. Không tự chạy code, không xử lý bí mật, không tự nhận đã kiểm chứng bằng thực thi.

## Mô hình và dữ liệu

- Base theo `lab22/config.py`; reference là `models/sft-merged` sau NB1.
- Adapter DPO mới: `adapters/python-mentor`; β và lr theo cấu hình lab.
- 200 cặp nháp do AI hỗ trợ; hiện 0 cặp sinh viên xác nhận tự viết, 0 cặp đã duyệt.
- Phải tự viết/viết lại ≥100 cặp và duyệt tất cả trước khi huấn luyện.
- Held-out theo nhóm chủ đề; năm prompt kiểm tra riêng trong `bonus/evaluate.py`.
- Chưa có quyền phát hành thương mại được xác nhận. SFT gốc CC BY-NC; không gắn một giấy phép rộng hơn cho adapter khi chưa xét nguồn.

## Đánh giá và hạn chế

Chưa có kết quả. Sau huấn luyện, đọc `bonus/results/training.json`, chấm mù năm cặp đầu ra và thử với người học thật. Không suy ra năng lực từ các câu chosen do người soạn dữ liệu viết. Các biến thể chỉ đổi số có độ đa dạng thấp; khả năng tổng quát ngoài Python cơ bản chưa được kiểm chứng. GGUF có thể thay đổi chất lượng sau lượng tử hoá; cần so đầu ra với bản HF.

## Nhật ký dùng AI

Yêu cầu hữu ích là diễn đạt hành vi theo tình huống cụ thể: khi người học xin một gợi ý, chosen nêu nguyên nhân và một phép kiểm tra; khi người học đã tự giải và xin đối chiếu, chosen được đưa code hoàn chỉnh. Điều này biến “gia sư tốt” thành các tiêu chí có thể rà soát. Chưa có một prompt thất bại được ghi nhận trong thực nghiệm mô hình; không bịa ra lịch sử đó. Rủi ro đã thấy ở khâu soạn nháp là lặp cùng mẫu với nhiều giá trị số, khiến 200 dòng trông nhiều hơn mức đa dạng thật. Sinh viên cần viết lại, kiểm tra tính đúng và ghi lại một prompt thất bại thực tế sau khi thử mô hình.
