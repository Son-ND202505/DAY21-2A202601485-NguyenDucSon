# B2 — Dataset miền riêng: ticket đặt lịch khám phòng khám

## Nguồn

**100% tổng hợp (synthetic), sinh bằng quy tắc, seed cố định `20260821001`.** Không có
dữ liệu bệnh nhân thật, không có PII, không thu thập từ bất kỳ hệ thống nào. Script sinh
data: `scripts_custom/b2_make_data.py` (mirror cấu trúc của `scripts/make_seed_data.py`
dùng cho corpus mặc định của lab).

Cơ chế sinh giống hệt corpus gốc: chọn ngẫu nhiên (có seed) một tổ hợp
`intent × urgency × sentiment × dịch_vụ`, ghép vào một câu ticket theo khuôn mẫu tiếng
Việt tự nhiên, gán `order_id` giả (`LH/PK/BN` + 6 chữ số ngẫu nhiên, không trỏ tới đơn
hàng/bệnh nhân thật nào).

## Vì sao đây là một domain khác, không phải đổi tên corpus cũ

- Tác vụ gốc: ticket CSKH thương mại điện tử (đổi trả, vận chuyển, hoàn tiền...).
- Tác vụ mới: ticket đặt lịch khám tại phòng khám tư nhân — ngữ cảnh, từ vựng, và **toàn
  bộ tập nhãn `intent`** khác hẳn: `dat_lich_moi | doi_lich | huy_lich | hoi_bao_hiem |
  hoi_dich_vu` (so với `doi_tra | van_chuyen | hoan_tien | san_pham_loi | hoi_thong_tin`
  của bản gốc). Trường `product` đổi nghĩa thành "dịch vụ khám" (12 loại dịch vụ y tế)
  thay vì tên sản phẩm thương mại điện tử.
- `urgency`/`sentiment` giữ nguyên 3 lớp (cao/trung_bình/thấp,
  tiêu_cực/trung_tính/tích_cực) vì đây là hai trục cảm xúc/mức độ khẩn cấp tổng quát,
  không đặc thù theo ngành — giữ nguyên để không phải viết lại scorer.

## Kích thước

| Tập | Số mẫu |
|---|---|
| `train_seed` | 250 |
| `eval_target` | 50 |
| `holdout_secret` | 20 (dự phòng, không dùng trong pipeline chấm) |
| **Tổng** | **320** (≥200 theo yêu cầu B2) |

`eval_regression.jsonl` — **dùng lại nguyên bản của lab gốc** (15 câu hỏi kiến thức phổ
thông, không liên quan ngành nào — "Thủ đô Việt Nam", "1km = ? mét"...). Không cần
regenerate vì mục đích của nhóm regression là đo năng lực tổng quát của model, không phụ
thuộc domain của tác vụ target.

## Khử nhiễu / đảm bảo chất lượng

1. **Dedup**: sinh vòng lặp tới khi đủ 320 ticket **duy nhất** (so khớp nguyên văn câu
   ticket), loại trùng lặp ngay tại nguồn.
2. **Decontamination check tự động**: sau khi chia `train_seed` (250) / `eval_target`
   (50) / `holdout_secret` (20), script assert không có câu ticket nào của hai tập eval
   xuất hiện trong tập train — chạy PASSED (xem log chạy `b2_make_data.py`).
3. **Không PII**: mã lịch hẹn là số ngẫu nhiên không tra cứu được, không gắn với người
   thật, không có tên/SĐT/địa chỉ.
4. **Nhãn nhất quán theo cấu trúc**: nhãn `label` sinh ra ngay lúc tạo câu (không suy
   luận ngược từ text), nên 100% chính xác theo định nghĩa — không có nhãn sai/nhiễu.
5. **Giới hạn đã biết**: đây là dữ liệu tổng hợp theo khuôn mẫu cố định (không phải
   ticket thật của phòng khám), nên đa dạng văn phong thấp hơn dữ liệu thực tế — phù hợp
   để kiểm chứng pipeline fine-tune (đúng mục tiêu B2: chứng minh dataset khác domain
   vẫn chạy được qua toàn bộ quy trình), không phù hợp để claim hiệu năng production.

## Prompt riêng cho domain này

`OPTIMIZED_PROMPT`/`NAIVE_PROMPT` gốc (`src/labkit/config.py`) hard-code từ vựng
thương mại điện tử, dùng chung cho cả train lẫn eval (không được sửa vì đó là mốc đã
đóng băng của corpus mặc định — sửa nó sẽ làm hỏng `baselines_frozen.json` đã chấm).
Vì vậy domain mới dùng cặp prompt **riêng**, định nghĩa trong `scripts_custom/
b2_pipeline.py`, không đụng tới `config.py`:

```
OPTIMIZED_PROMPT_CLINIC = """Bạn là hệ thống phân loại ticket đặt lịch khám. Trả về DUY NHẤT
một object JSON, không kèm giải thích, không kèm markdown fence.

Schema bắt buộc — đúng 4 khóa:
{"intent": ..., "urgency": ..., "product": ..., "sentiment": ...}

intent    ∈ dat_lich_moi | doi_lich | huy_lich | hoi_bao_hiem | hoi_dich_vu
urgency   ∈ cao | trung_binh | thap
sentiment ∈ tieu_cuc | trung_tinh | tich_cuc
product   = tên dịch vụ khám xuất hiện nguyên văn trong ticket
...
"""
```

## Kết quả chạy qua pipeline

Xem `results/custom_domain/` và mục B2 trong `submission/REPORT.md` để biết baseline
(a)/(b), kết quả train `correct`-equivalent, và phán quyết bốn-nhóm trên domain này.
