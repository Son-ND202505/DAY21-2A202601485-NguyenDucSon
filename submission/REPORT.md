# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đức Sơn  **MSSV**: 2A202601485  **Ngày**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `NVIDIA GeForce RTX 3080 Ti Laptop 16GB (chạy qua WSL2)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | Corpus mặc định — 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 token *(results/token_stats.json: mean=93.1, p50=93, p95=98, p99=100, max=101)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json: "reasoning preserved — safe to train on traces")*
Không cần xử lý gì thêm — template Qwen3.5 tự đóng khối `<think></think>` rỗng trong generation prompt.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39/94 token) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Đoạn được tính loss (supervised span, `results/mask_proof.json`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần bị mask (KHÔNG tính loss) — cả system prompt lẫn câu hỏi của khách:

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7244 | 0.000 | 2286.8 |
| (b) base + optimized prompt | 0.760 | 0.7244 | 1.000 | 772.2 |
| (c) LoRA fine-tune | 0.975 | 0.5778 | 1.000 | 981.1 |

**(b) có thật sự mạnh hơn (a) không?** Có — target nhảy từ 0.000 lên 0.760, format từ 0.000 lên 1.000 (bản naive không ép được model xuất đúng JSON, `verify.py` cũng xác nhận `(a)=0.000 -> (b)=0.760`).
`OPTIMIZED_PROMPT` **không sửa** — dùng nguyên bản mặc định của lab (`make verify` xác nhận SHA prompt (b) khớp bản gốc, chưa bị chỉnh sau khi thấy kết quả).

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.3512 | **0.975** | 702.1 | 12.01 |
| `attn_only` | q,v (attn-only) | 283 *(matched, 32,456,704 params)* | 32,456,704 | 1e-4 | 0.3515 | **0.980** | 566.8 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.4940 | **0.000** | 668.1 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 (4-bit) | 0.3547 | **0.980** | 700.2 | 6.79 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

`attn_only` (32,456,704 tham số, rank được nâng lên r=283 để khớp ngân sách với `correct` ở
32,464,896 tham số — sai lệch chỉ 0.025%, đạt điều kiện `matched_rank()`) **thắng nhẹ**
`correct` trên tập target: 0.980 so với 0.975. Thứ tự này **không khớp** thứ tự theo train
loss của NB4: `correct` có loss thấp hơn (0.3512 so với 0.3515 của `attn_only`), nghĩa là
nếu chỉ nhìn loss huấn luyện, ta sẽ xếp `correct` đứng đầu — nhưng trên thang đo thật của
tác vụ (field-accuracy) thì `attn_only` mới là run tốt nhất. Với cùng một ngân sách tham
số, gắn adapter chỉ vào `q,v` (attention) và bù bằng rank cao hơn cho ra kết quả ngang
ngửa, thậm chí nhỉnh hơn một chút, so với gắn vào toàn bộ các lớp tuyến tính văn bản với
rank thấp hơn. Điều này gợi ý rằng ở bài toán JSON-triage 4 trường này, **đòn bẩy thật sự
không nằm ở vị trí gắn adapter** — cả hai vị trí đều học được tác vụ tốt như nhau khi rank
được cân bằng đúng ngân sách — mà nằm ở việc có đủ *dung lượng tham số* (rank × số module)
để biểu diễn tác vụ, bất kể phân bổ dung lượng đó vào đâu. Khoảng cách 0.005 giữa hai run
là quá nhỏ để khẳng định vị trí nào "thắng" một cách có ý nghĩa thống kê với chỉ 50 mẫu
eval; kết luận an toàn hơn là: **rank/ngân sách tham số quan trọng hơn vị trí**, ít nhất
trong phạm vi đo được của lab này.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`wrong_lr` dùng LR = 1e-05 (thang full-fine-tune), thấp hơn `correct` đúng 10 lần
(1e-04). Kết quả: train loss cuối cùng của `wrong_lr` là **1.494**, cao gấp hơn 4 lần so
với `correct` (0.3512) — mô hình gần như chưa hội tụ sau 30 step. Hệ quả trên tập target
là thảm khốc: **target = 0.000, format = 0.000** — model không xuất ra được JSON đúng cấu
trúc, và latency lại tăng vọt lên 3983.8ms (so với ~981ms của `correct`), phù hợp với việc
model sinh ra output dài dòng, không đúng định dạng thay vì JSON ngắn gọn. Nếu chỉ nhìn
đường loss huấn luyện (thấy nó giảm dần, không phẳng ngay từ đầu, không có dấu hiệu NaN
hay phát nổ) mà không biết LR đang bị đặt sai thang, người quan sát dễ kết luận nhầm rằng
"model đang học, chỉ là học chậm hơn — train thêm step hoặc thêm epoch là ổn". Kết luận đó
sai: vấn đề không phải là *chưa đủ thời gian*, mà là *tốc độ học sai bậc độ lớn* cho LoRA —
loss vẫn giảm nhưng giảm quá chậm để đạt được vùng tham số hữu ích trong ngân sách step đã
định, và train thêm step ở đúng LR sai này sẽ chỉ trì hoãn chứ không sửa được vấn đề gốc.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` (4-bit, cùng vị trí/rank/LR với `correct`) tiết kiệm **12.01 → 6.79 GB VRAM**, tức
giảm khoảng **43.5%** bộ nhớ đỉnh. Cái giá phải trả không nằm ở chất lượng — target của
`qlora` (0.980) thậm chí nhỉnh hơn `correct` (0.975), ngang với `attn_only` — mà nằm ở
**tốc độ suy luận**: latency 1215.5ms so với 981.1ms của `correct`, tăng khoảng 24%, do chi
phí dequantize trong lúc generate. Train loss của `qlora` (0.3547) cũng gần như tương đương
`correct`. Với số đo cụ thể trong lab này, dữ liệu **không ủng hộ** khuyến nghị chung chung
"không dùng QLoRA cho dòng model này" (deck §12) — ở quy mô 4B tham số và bài toán triage
ngắn này, QLoRA đánh đổi latency lấy VRAM một cách hợp lý, không đánh đổi chất lượng. Khuyến
nghị đó có thể đúng ở các biến thể MoE lớn hơn hoặc tác vụ đòi hỏi suy luận dài (nơi chi phí
dequantize lặp lại nhiều lần trong quá trình sinh token sẽ cộng dồn nặng hơn), nhưng ở đây
tôi đo được ngược lại: nếu VRAM là ràng buộc cứng, QLoRA là lựa chọn hợp lý cho bài toán
này.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.215` · `regression Δ = -0.147` (dung sai cho phép: 0.020) · `valid_trace_rate = 0.00`

Lý do FAIL theo `results/verdict.json`: *"general capability regressed by 0.147 (tolerance 0.020). See deck §14.3 — add 1-5% replay data."*

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

Cổng hồi quy **FAILED**, nhưng không phải vì bản fine-tune học kém — trên tập target nó
vượt xa cả hai baseline (0.975 so với 0.760 của (b) và 0.000 của (a)), tức target Δ =
+0.215 so với mốc phải vượt là (b). Vấn đề nằm ở nhóm thứ hai: **regression giảm từ 0.7244
(cả (a) lẫn (b), vì đây là năng lực gốc của base model, không phụ thuộc prompt) xuống còn
0.5778** ở bản fine-tune — regression Δ = −0.147, vượt xa dung sai cho phép 0.020. Đây là
một ca **quên thảm hoạ (catastrophic forgetting)** kinh điển: dữ liệu train chỉ gồm 250
ticket CSKH hẹp về mặt định dạng và chủ đề, huấn luyện 2 epoch trên toàn bộ text-linear
layer khiến model "quên" một phần khả năng trả lời kiến thức/chỉ dẫn phổ thông vốn có từ
pretraining, để đổi lấy việc bám rất chặt vào định dạng JSON 4 trường. Nói cách khác: lab
này đo được chính xác cái giá phải trả của việc fine-tune hẹp — thắng đậm trên tác vụ mục
tiêu nhưng trả giá bằng năng lực tổng quát, đúng như cảnh báo ở README/rubric về "quên thảm
hoạ". Vì cổng bốn-nhóm bắt cả hai điều kiện (thắng target VÀ không tụt regression) cùng lúc,
một cải thiện target rất lớn vẫn không đủ để PASS nếu cái giá ở nhóm khác vượt ngưỡng — đây
chính là lý do thiết kế cổng có 4 nhóm thay vì chỉ đo target. Hướng khắc phục mà lab gợi ý
(và tôi chưa thực hiện trong lần chạy này) là trộn 1–5% dữ liệu phổ thông vào tập train
(deck §14.3) để giữ lại năng lực gốc trong lúc vẫn học được định dạng JSON.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "...đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện." | san_pham_loi / thap / .../ trung_tinh | intent=**hoan_tien**❌, urgency=**cao**❌ | intent=san_pham_loi✅, urgency=thap✅ | ✅ FT thắng (điểm 0.5→1.0) |
| 2 | "...đặt balo laptop mã đơn DH863123. Đổi size." | doi_tra / thap / .../ tieu_cuc | intent=**hoan_tien**❌, urgency=**cao**❌ | intent=doi_tra✅, urgency=thap✅ | ✅ FT thắng (điểm 0.5→1.0) |
| 3 | "...đặt chuột không dây mã đơn OD538419. Hoàn tiền." | hoan_tien / **trung_binh** / .../ tich_cuc | urgency=trung_binh✅ (đúng) | urgency=**thap**❌ | ❌ **FT thua** (điểm 1.0→0.75) — trường hợp thua rõ ràng duy nhất/50 mẫu |
| 4 | "...đặt máy xay sinh tố mã đơn DH777946. Khi nào có tiền về." | hoan_tien / **trung_binh** / .../ tieu_cuc | intent=hoan_tien✅, urgency=**cao**❌ | intent=hoan_tien✅, urgency=**thap**❌ | ❌ **FT thua ở trường urgency** (điểm hoà 0.75=0.75, nhưng FT sai *urgency* — trường quan trọng cho triage — còn (b) sai *intent*) |
| 5 | "...đặt nồi chiên không dầu mã đơn OD169066. Trả hàng." | doi_tra / **trung_binh** / .../ tich_cuc | intent=**hoan_tien**❌, urgency=trung_binh✅ | intent=doi_tra✅, urgency=**thap**❌ | hoà điểm (0.75=0.75) — FT lại sai đúng *urgency* |

**⚠ Lưu ý minh bạch (đã đo, không suy đoán):** so từng câu trên toàn bộ 50 ticket target
(script phụ, `results/qualitative_with_b.json`), FT thắng (b) ở **34/50**, hoà **15/50**,
và thua theo đúng nghĩa (điểm FT < điểm (b)) chỉ ở **1/50** — ca #3 ở trên. Không đủ 2 ca
"thua tuyệt đối" nên bảng dùng thêm 2 ca **hoà nhưng cùng lộ một lỗi hệ thống**: ở mọi ca
hoà có điểm <1.0 mà tôi kiểm tra, FT luôn đoán sai đúng trường `urgency` thành `thap`
trong khi nhãn thật là `trung_binh` — trong khi (b) thường sai ở trường `intent`. Đây là
mẫu chung thật sự đáng nói trong report, mạnh hơn việc chỉ tìm đủ "2 ca thua" cho có.

Có mẫu chung nào ở các ca FT thua không?

Có, và mẫu này lộ ra rõ ràng hơn khi nhìn toàn bộ 50 mẫu (không chỉ 5 ví dụ trong bảng)
chứ không phải mỗi ca #3: **fine-tune có xu hướng hệ thống đoán trường `urgency` là
`thap` khi nhãn thật là `trung_binh`** — cả ca #3, #4, #5 ở trên đều rơi vào đúng lỗi này,
trong khi (b) hầu như không mắc lỗi này (nó lại hay sai ở trường `intent`). Cách đọc hợp
lý nhất: 250 mẫu train (seed 42) có thể lệch phân bố nhãn `urgency`, khiến model học được
một "prior" thiên về `thap` mỗi khi tín hiệu mức độ khẩn cấp trong câu không thật rõ ràng —
một dạng overfit nhẹ vào phân bố nhãn của tập train hẹp, khác với overfit vào từ vựng. Đây
cũng là một lời giải thích khả dĩ, ở mức trường dữ liệu, cho phần nào của việc regression
tụt: model bị kéo lệch theo thói quen của tập train hẹp thay vì giữ được khả năng suy luận
theo ngữ cảnh chung.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

Với cấu hình đã chạy, tôi **chưa deploy** bản fine-tune `correct` này ở dạng hiện tại,
mặc dù nó thắng đậm trên đúng tác vụ mục tiêu (target 0.975 so với 0.760 của baseline đã
tối ưu prompt, format hoàn hảo 1.0, latency thậm chí nhanh hơn baseline (a) đáng kể). Lý
do là cổng hồi quy bốn-nhóm FAILED một cách rõ ràng và có ý nghĩa: regression tụt 0.147,
vượt xa dung sai — nghĩa là để đổi lấy độ chính xác triage cao hơn, model đã đánh mất một
phần năng lực trả lời kiến thức phổ thông mà một hệ thống CSKH thực tế vẫn cần dùng chung
model cho các việc khác. Deploy nó nguyên trạng là chấp nhận một sự đánh đổi chưa được đo
lường đầy đủ hậu quả trong lab này (ảnh hưởng thực tế đến trải nghiệm người dùng ở các câu
hỏi ngoài phạm vi triage). Nhìn qua bốn run ở NB4, đòn bẩy thật sự **không phải vị trí gắn
adapter** — `attn_only` (chỉ q,v) và `text-linear` (`correct`) cho kết quả target gần như
ngang nhau khi rank được cân bằng đúng ngân sách tham số, chênh lệch 0.005 không đủ để kết
luận vị trí nào ưu việt. Đòn bẩy rõ ràng nhất, gây sập hoàn toàn kết quả, là **learning
rate**: sai một hệ số 10 lần (`wrong_lr`) biến một run khoẻ mạnh thành target=0.000,
format=0.000. Đòn bẩy thứ hai, ít ồn ào hơn nhưng ảnh hưởng đến quyết định deploy nhiều
nhất, là **chất lượng/độ đa dạng của dữ liệu train**: tập 250 mẫu quá hẹp về chủ đề và có
thể lệch phân bố nhãn (thấy rõ qua lỗi hệ thống ở trường `urgency`) là nguyên nhân hợp lý
nhất cho cả hiện tượng quên thảm hoạ lẫn lỗi lặp lại ở một trường cụ thể. `mask` đúng
(assistant-only, đã chứng minh ở NB1) là điều kiện cần nhưng không phải đòn bẩy tạo khác
biệt giữa các run ở đây, vì cả bốn run đều dùng đúng một mask — nó là nền tảng đúng đắn,
không phải biến số đang được so sánh.

**Ba điều tôi học được** (cụ thể, không generic):
1. Chỉ số huấn luyện (train loss) và chỉ số tác vụ thật (field-accuracy trên target) có
   thể xếp hạng các cấu hình theo hai thứ tự khác nhau — `correct` có loss thấp hơn
   `attn_only` nhưng lại thua trên target. Nếu lab này dừng lại ở NB4 và chấm bằng
   `final_loss` thay vì đo lại bằng NB5 §4, kết luận về "vị trí nào tốt hơn" sẽ sai.
2. Một cải thiện rất lớn trên tác vụ mục tiêu (target Δ = +0.215) không tự động là một lý
   do đủ để deploy — phải nhìn đồng thời năng lực bị đánh đổi (ở đây là regression Δ =
   −0.147); cổng đánh giá bốn-nhóm bắt được điều mà một con số target duy nhất che giấu.
3. Sai một bậc độ lớn của learning rate không tạo ra một model "học chậm hơn" mà tạo ra
   một model gần như không học được gì hữu ích (loss vẫn giảm, nhưng quá chậm để tới vùng
   tham số có ích trong ngân sách step cố định) — nhìn đường loss không đủ để phát hiện
   lỗi này nếu không có baseline đối chứng cùng số step để so sánh.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1–5% dữ liệu phổ thông (deck §14.3) vào tập
train để kiểm tra xem có kéo được regression về trong dung sai mà không đánh mất nhiều
target hay không, và chạy thêm một quét nhỏ trên tỉ lệ trộn (ví dụ 1%, 3%, 5%) để tìm điểm
cân bằng — đây là bước tự nhiên tiếp theo sau khi đã xác định đúng nguyên nhân gây FAILED
là quên thảm hoạ chứ không phải lỗi cấu hình LoRA.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap — `results/merge_check.json`: before=0.975, after=0.975, Δ=0.000 (tolerance 0.01); hot-swap thử cả 3 adapter (`correct`, `attn_only`, `qlora`), cùng ticket cho ra 3 kết quả nhất quán
- [x] B2 dataset miền riêng — domain hoàn toàn khác corpus mặc định: **ticket đặt lịch
  khám phòng khám** thay vì CSKH thương mại điện tử (intent khác hẳn: `dat_lich_moi |
  doi_lich | huy_lich | hoi_bao_hiem | hoi_dich_vu`; `product` đổi nghĩa thành dịch vụ
  y tế). 250 train + 50 eval_target + 20 holdout = 320 mẫu tổng hợp theo quy tắc, seed
  cố định, decontamination check PASSED. Mô tả đầy đủ nguồn/khử nhiễu ở
  `data/CUSTOM_DATASET.md`. Chạy full baseline (a)/(b) + train `correct`-tương-đương +
  đánh giá 4 nhóm trên domain này (`results/custom_domain/`), hoàn toàn tách biệt khỏi
  corpus/kết quả đã chấm ở trên:

  | Run | target | regression | format | latency (ms) |
  |---|---|---|---|---|
  | (a) base + naive prompt | 0.000 | 0.7244 | 0.000 | 2255.2 |
  | (b) base + optimized prompt (riêng cho domain này) | 0.830 | 0.7244 | 1.000 | 726.3 |
  | (c) LoRA fine-tune (`correct` config, 32 step) | **1.000** | **0.1333** | 1.000 | 958.9 |

  **Verdict: FAILED** — *"general capability regressed by 0.591 (tolerance 0.020)"*.
  Đáng chú ý: domain mới cho target **hoàn hảo tuyệt đối** (1.000, tốt hơn cả 0.975 của
  corpus gốc) nhưng regression sập nặng hơn nhiều — **−0.591** so với **−0.147** ở
  corpus gốc, dù cùng cấu hình LoRA, cùng epoch, số step gần như y hệt (32 so với 30, vì
  250 mẫu train ở đây không bị NB1 tách 25 mẫu val như corpus gốc). Đây là bằng chứng
  chéo-domain củng cố thêm cho kết luận ở mục 5: mức độ quên thảm hoạ không phải hằng số
  của LoRA nói chung, mà phụ thuộc mạnh vào **độ hẹp/đơn điệu của tập train** — domain
  khám bệnh có vốn từ vựng và cấu trúc câu đơn điệu hơn (ít biến thể mở đầu, ít loại sản
  phẩm) so với corpus e-commerce gốc, nên model "chuyên biệt hoá" nhanh và sâu hơn, trả
  giá đắt hơn ở năng lực tổng quát dù đạt target cao hơn.
- [x] B3 reasoning-trace collapse — corpus mặc định ship câu trả lời JSON trần (không có
  `<think>` thật), nên `response-only`/`masked-think` vốn là no-op trên nó (đúng như cảnh
  báo trong `.env.example`). Để thực sự làm thí nghiệm, tôi tạo một bản sao 225 dòng train
  (`data/split/train_traces.jsonl`, **không đụng file gốc/không ảnh hưởng checksum đã
  khoá**) thêm một khối `<think>` suy luận theo quy tắc, sinh thẳng từ chính nhãn đã biết
  (không bịa dữ kiện mới), rồi train hai lần cùng cấu hình `correct` (text-linear, r=16,
  LR=1e-4, 30 step), chỉ đổi `MASK_MODE`:

  | MASK_MODE | final_loss | target | valid_trace_rate |
  |---|---|---|---|
  | `assistant-only` | 0.3643 | 0.925 | **1.000** |
  | `response-only` | 0.0380 | **0.000** | **0.000** |

  *(Lần đo đầu bị lỗi đo — `valid_trace_rate` của `assistant-only` báo sai 0.0 vì
  `enable_thinking=True` mở thẻ `<think>` trong PROMPT chứ không phải trong phần model
  sinh ra, nên hàm kiểm tra thiếu thẻ mở khi soát completion. Đã sửa và chấm lại — xem
  `results/reasoning_trace_collapse_rescored.json`; số ở bảng trên là số đã sửa đúng.)*

  Đây là ví dụ rõ nhất trong toàn bộ lab về việc train loss có thể **hoàn toàn đánh lừa**:
  `response-only` có loss cuối gần bằng 0 (0.038, gần như học thuộc lòng 225 mẫu dưới
  teacher-forcing), nhưng ở chế độ generate thật (greedy, `enable_thinking=True`) nó sinh
  ra rác — ví dụ thật lấy từ `sample_pred`: `{"intent": "refund", "entities": [...]}` (sai
  hẳn schema đã train), lặp lại token vai trò giả `user`/`assistant`, và một khối
  `<think>\n\n</think>` rỗng. Vì chưa từng được supervise trên nội dung suy luận,
  model không có tín hiệu nào để biết "viết gì" vào khối `<think>` khi tự sinh — nó
  hallucinate tự do rồi trật khỏi đường ray hoàn toàn, kéo cả JSON theo xuống 0.000.
  Ngược lại, `assistant-only` (supervise toàn bộ lượt assistant, gồm cả suy luận) tái
  tạo đúng nguyên văn kiểu suy luận đã train ở mọi mẫu (trace hợp lệ 50/50) và vẫn giữ
  target cao (0.925). Kết luận cho §13.5: nếu dữ liệu train có suy luận thật, **mask
  phải bao trùm suy luận đó** — che nó đi bằng `response-only` không "làm sạch" loss
  (loss thậm chí thấp hơn) mà phá huỷ hoàn toàn khả năng suy luận mạch lạc lúc sinh,
  một cái giá không hề lộ ra nếu chỉ nhìn train loss.
- [x] B4 quét rank có kiểm soát — cố định vị trí `text-linear`, LR=1e-4, cùng 30 step, quét r∈{8,16,64} (`results/rank_sweep.json`):

  | r | trainable | final_loss | target | train_s | VRAM GB |
  |---|---|---|---|---|---|
  | 8 | 16,232,448 | 0.4507 | 0.970 | 617.7 | 11.74 |
  | 16 | 32,464,896 | 0.3512 | 0.975 | 702.1 | 12.01 |
  | 64 | 129,859,584 | 0.2372 | **1.000** | 613.2 | 13.71 |

  **Khi nào rank là đòn bẩy?** Ở bài toán JSON-triage 4 trường này, rank **có** là đòn
  bẩy thật, chỉ là đòn bẩy yếu và có lợi suất giảm dần: đi từ r=8→16 target chỉ nhích
  +0.005 (gần như nhiễu đo lường trên 50 mẫu), nhưng đi từ r=16→64 (tăng ngân sách tham
  số gấp 4 lần) target đạt tuyệt đối 1.000 và train loss giảm gần một nửa. Rank là đòn
  bẩy **khi tác vụ đã gần bão hoà ở ngân sách thấp** — 250 mẫu train, 4 trường nhãn từ
  từ vựng đóng là một tác vụ đơn giản, nên phần lớn "công việc học" đã hoàn thành ở
  r=8–16; phần cải thiện còn lại ở r=64 nhiều khả năng đến từ việc model có đủ dung
  lượng để khớp gần như hoàn hảo 225 mẫu train (final_loss 0.24, thấp nhất trong ba
  run) hơn là học tổng quát hoá tốt hơn — rủi ro đây là overfit vào đúng 250 mẫu chứ
  không phải năng lực triage thật sự tốt hơn, điều mà 50 mẫu eval target không đủ để
  phân biệt. Kết hợp với so sánh vị trí ở mục 4.1 (`attn_only` r=283 thắng `correct`
  r=16 dù cùng ngân sách tham số, không phải cùng rank số), bức tranh nhất quán là: khi
  đã có đủ dung lượng tham số tối thiểu để biểu diễn tác vụ, tăng thêm rank mang lại
  lợi ích biên giảm dần và đến một điểm là đang mua thêm khả năng ghi nhớ tập train hơn
  là năng lực thật.
- [x] B5 HuggingFace Hub — public: https://huggingface.co/Son-ND202505/day21-triage-lora-correct
  (adapter `correct` — text-linear r=16, LR=1e-4, target=0.975 trên corpus mặc định)
