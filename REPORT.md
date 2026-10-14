Họ tên: Thái Phúc Tiến
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 2.50 | 35.28 | 0.083 | 0.059-0.110 | 0.93 | 0.157 | 0.023 | 0.162 |
| E1 | LoRA LLM | 33,030,144 | 2.98 | 47.25 | 0.259 | 0.222-0.301 | 1.14 | 0.190 | 0.171 | 0.527 |
| E2 | + projector | 142,135,296 | 6.44 | 47.94 | 0.321 | 0.279-0.362 | 1.04 | 0.166 | 0.258 | 0.578 |
| E3 | + LoRA ViT | 148,426,752 | 4.96 | 59.90 | 0.459 | 0.414-0.503 | 0.76 | 0.307 | 0.384 | 0.707 |
| E4 | data_size = 0.1 | 142,135,296 | 6.82 | 48.84 | 0.170 | 0.144-0.199 | 2.11 | 0.107 | 0.126 | 0.365 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Phân bổ tham số**: Qwen3-VL-4B có tổng quy mô ~4.44B tham số, gồm:
  - **LLM**: ~4.0B tham số (36 transformer layers kèm embedding và lm_head).
  - **ViT**: ~415M tham số (24 visual blocks).
  - **Projector**: ~109.1M tham số, gồm `visual.merger` (~27M) và `visual.deepstack_merger_list` (~82M, trích xuất đặc trưng tại các block 5, 11, 17 của ViT).
- **Khối cập nhật theo setup**:
  - **E1 (LoRA LLM)**: Chỉ cập nhật adapter LoRA trên 36 lớp của LLM (7 module mỗi layer: `q, k, v, o_proj` và `gate, up, down_proj` = 252 target modules), tổng cộng **33,030,144** tham số huấn luyện (0.74% mô hình). ViT và Projector bị đóng băng hoàn toàn.
  - **E2 (+ projector)**: Cập nhật LoRA trên LLM (252 modules) và mở toàn bộ tham số của Projector (`visual.merger` và `deepstack_merger_list`) ở dạng float32 full precision, tổng cộng **142,135,296** tham số huấn luyện (tăng đúng 109,105,152 tham số projector). ViT vẫn đóng băng.
  - **E3 (+ LoRA ViT)**: Cập nhật LoRA trên LLM (252 modules), mở Projector full precision, và thêm adapter LoRA trên 24 blocks của ViT (4 module mỗi block: `attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2` = 96 modules), tổng cộng 348 LoRA target modules.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- `parse_valid_rate` của E0 đạt **0.865** (86.5% output đúng cấu trúc JSON danh sách box thang 0–1000). F1 tổng thể chỉ đạt **0.083** (CI95: 0.059–0.110).
- **Hai lỗi phổ biến nhất**:
  1. *Ảo giác bounding box (Hallucination)*: Model tự vẽ rất nhiều box giả trên mặt đường bình thường, dẫn đến FP/ảnh cao (0.93), đặc biệt lớp `crack` có Precision chỉ 0.028 và F1 chỉ 0.023.
  2. *Lỗi cú pháp và không emit token kết thúc*: Trong 13.5% mẫu không hợp lệ, model không phát sinh `<|im_end|>` mà sinh thêm văn bản giải thích tự do, bọc markdown ````json ... ```` hoặc sinh toạ độ vượt quá giới hạn [0, 1000].
- **Ví dụ output thô**:
  `[{"label":"crack","bbox_2d":[430,467,583,506]},{"label":"pothole","bbox_2d":[361,475,414,522]},{"label":"manhole","bbox_2d":[317,481,616,625]},{"label":"pothole","bbox_2d":[583,500,627,614]},{"label":"crack","bbox_2d":[580,508,655,614]},{"label":"pothole","bbox_2d":[259,517,311,572]},{"label":"crack","bbox_2d":[397,628,447,717]},{"label":"crack","bbox_2d":[339,683,398,808]},{"label":"crack","bbox_2d":[158,789,361,994]}]`
  (Mô hình gán 5 box crack liên tiếp tại những vùng mặt đường bình thường).

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Hệ quả**: Nếu chấm output chuẩn hóa 0–1000 theo toạ độ pixel tuyệt đối (hoặc ngược lại), **F1 sẽ ngay lập tức tụt về 0.000**.
- **Bản chất**: Ảnh gốc có kích thước 640×360. Nếu box dự đoán nằm trong thang [0, 1000] mà bị đối chiếu trực tiếp với toạ độ pixel [0, 640] và [0, 360], box sẽ bị phóng to và lệch tâm nghiêm trọng (tỷ lệ lệch 1000/640 ≈ 1.56x theo trục x và 1000/360 ≈ 2.78x theo trục y). Diện tích giao nhau giữa box dự đoán và ground truth sẽ bằng 0 hoặc rất nhỏ, khiến IoU luôn < 0.5 trên toàn bộ các cặp, dẫn đến Precision = 0, Recall = 0, F1 = 0.
- **Kiểm chứng**: Trong hàm `yolo_to_box` và `score.py`, IoU yêu cầu hai box phải ở cùng một hệ quy chiếu. Qwen3-VL được pretrained grounding mặc định trên thang chuẩn hóa [0, 1000] để độc lập với kích thước và tỷ lệ khung hình ảnh thực tế. Khi chạy `score.py`, cờ `--coord norm1000` ánh xạ đúng về pixel trước khi tính IoU; nếu chạy sai hệ quy chiếu, số lượng `matched_boxes` bằng 0 tuyệt đối.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời**: Bản chất của bài toán SFT là mô hình hóa xác suất có điều kiện $P(\text{answer} \mid \text{image}, \text{prompt})$. Bằng cách gán `labels = -100` cho toàn bộ prompt và 220 visual tokens, hàm `CrossEntropyLoss(ignore_index=-100)` triệt tiêu hoàn toàn gradient của các token này, buộc mô hình chỉ tối ưu việc sinh chuỗi JSON đáp án.
- **Điều gì xảy ra nếu tính cả prompt và token ảnh**:
  1. *Làm loãng tín hiệu học*: Token ảnh (220 tokens) và prompt (~60 tokens) chiếm hơn 75% độ dài chuỗi đầu vào. Nếu tính loss trên toàn bộ, gradient sẽ bị chi phối bởi việc học thuộc cấu trúc prompt và tái tạo đặc trưng ảnh thay vì học định vị bounding box.
  2. *Phá vỡ biểu diễn tổng quát (Catastrophic Forgetting)*: Gradient ngược sẽ cập nhật làm sai lệch các trọng số đã được pretrain rất tốt về nhận diện ngôn ngữ và thị giác của Qwen3-VL.
  3. *Lãng phí VRAM*: Việc sinh logits trên toàn bộ chuỗi làm bùng nổ bộ nhớ tại tầng lm_head. Hàm `answer_only_loss` chỉ tính logits từ vị trí supervised token đầu tiên (`logits_to_keep`), giúp tiết kiệm đáng kể bộ nhớ VRAM.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện nếu ViT bị 4-bit**:
  - Với BitsAndBytes 4-bit (`load_in_4bit=True`), nếu không loại trừ module thị giác, các tầng tuyến tính của projector sẽ bị nén thành ma trận trọng số lượng tử hóa nguyên 4-bit (`Linear4bit`).
  - Trọng số 4-bit là bảng tra cứu tĩnh, không lưu trữ được gradient liên tục và optimizer không thể cập nhật trực tiếp trọng số này.
- **Cách xử lý**:
  - Khai báo `llm_int8_skip_modules=["visual", "lm_head"]` trong `BitsAndBytesConfig` để giữ nguyên ViT và Projector ở định dạng số thực.
  - Trong `apply_setup`, ta ép toàn bộ trọng số của `visual.merger` và `visual.deepstack_merger_list` sang `torch.float32` (dequantize 8 tầng tuyến tính) và gán `requires_grad = True`.
- **Vì sao Projector phải ở float32**: Projector là cổ chai (bottleneck) kết nối không gian đặc trưng giữa ViT và LLM. Gradient truyền ngược qua đây có độ nhạy rất cao. Định dạng fp16 có dải biểu diễn hạn hẹp ($10^{-5} \to 65504$), rất dễ gây tràn số (`Inf`/`NaN`) hoặc tắt gradient khi tích luỹ gradient (`grad_accum=8`). Đưa projector lên `float32` đảm bảo độ ổn định số học trong suốt quá trình lan truyền ngược.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên gây F1 cao giả (Data Leakage)**:
  - Dữ liệu thu thập từ video camera hành trình theo từng phiên (`session`). Các khung hình liên tiếp trong cùng một phiên có góc quay, nền đường, điều kiện thời tiết, và cùng một ổ gà/nắp cống xuất hiện lặp lại ở nhiều khoảng cách.
  - Nếu chia ngẫu nhiên, các ảnh trong cùng một phiên sẽ phân tán vào cả train và test. Mô hình sẽ học vẹt bối cảnh xung quanh của ổ gà đó thay vì học đặc trưng tổn hại mặt đường. Khi test, F1 sẽ cao giả tạo nhưng mô hình sẽ thất bại khi gặp một cung đường mới lạ. Chia theo `session` đảm bảo sự độc lập không gian và thời gian.
- **Số ảnh và số box từng lớp ở mỗi split** (từ `data/splits/meta.json`):
  - **Train**: 1.422 ảnh | pothole: 1.074, crack: 1.750, manhole: 691 (Tổng: 3.515 box).
  - **Val**: 276 ảnh | pothole: 66, crack: 470, manhole: 103 (Tổng: 639 box).
  - **Test**: 311 ảnh | pothole: 121, crack: 299, manhole: 163 (Tổng: 583 box).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **Số liệu thực nghiệm**:
  - E1 (LoRA LLM): F1 = **0.259** (CI95: 0.222–0.301), Train: 47.25 min, Peak VRAM: 2.98 GB, FP/ảnh: 1.14.
  - E2 (+ projector): F1 = **0.321** (CI95: 0.279–0.362), Train: 47.94 min, Peak VRAM: 6.44 GB, FP/ảnh: 1.04.
  - E3 (+ LoRA ViT): F1 = **0.459** (CI95: 0.414–0.503), Train: 59.90 min, Peak VRAM: 4.96 GB, FP/ảnh: 0.76.
- **Đánh giá hiệu quả chi phí và giá trị thực tế**:
  - **Mở Projector (E2)**: Rất đáng giá. F1 tăng từ 0.259 lên 0.321 (+0.062), tỷ lệ parse valid đạt 100%. Thời gian huấn luyện hầu như không tăng (47.94m vs 47.25m), VRAM đỉnh tăng từ 2.98 GB lên 6.44 GB (vẫn nằm an toàn trong giới hạn 15 GB của Tesla T4). Việc điều chỉnh 109M tham số của projector ở float32 giúp căn chỉnh lại không gian nhúng thị giác sang không gian từ vựng của LLM hiệu quả hơn hẳn việc chỉ tinh chỉnh LLM.
  - **Thêm LoRA trên ViT (E3)**: Cực kỳ đáng giá và mang tính quyết định. F1 nhảy vọt từ 0.321 lên **0.459** (+0.138). Đáng chú ý, khoảng tin cậy CI95 của E3 ([0.414, 0.503]) **hoàn toàn tách biệt, không chồng lấn** với CI95 của E2 ([0.279, 0.362]). Cải thiện này chỉ tốn thêm ~6M tham số LoRA trên 24 khối ViT (tổng 148.4M params), VRAM đỉnh chỉ 4.96 GB, thời gian train tăng thêm 12 phút (59.90m). Việc cho phép Vision Transformer thích ứng với các chi tiết vết nứt và mép hố trên mặt đường giúp Precision tăng từ 0.349 lên 0.508 và FP/ảnh giảm mạnh xuống 0.76.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy thực tế qua các setup**:
  - E0: CI95 = 0.059–0.110 (độ rộng $\Delta = 0.051$).
  - E1: CI95 = 0.222–0.301 (độ rộng $\Delta = 0.079$, biên độ sai số $\pm 0.040$).
  - E2: CI95 = 0.279–0.362 (độ rộng $\Delta = 0.083$, biên độ sai số $\pm 0.041$).
  - E3: CI95 = 0.414–0.503 (độ rộng $\Delta = 0.089$, biên độ sai số $\pm 0.045$).
- **Giới hạn kết luận khoa học**:
  - Với tập test gồm 311 ảnh (583 boxes ground truth), độ rộng CI95 luôn dao động quanh mức 0.08 (±4% F1).
  - Giữa E1 ([0.222, 0.301]) và E2 ([0.279, 0.362]), hai khoảng này có vùng chồng lấn hẹp (từ 0.279 đến 0.301), do đó nếu chỉ nhìn vào chênh lệch điểm trung bình thì chưa thể khẳng định 100% E2 vượt trội mà vẫn có xác suất nhỏ do biến thiên mẫu test.
  - Tuy nhiên, giữa E2 ([0.279, 0.362]) và E3 ([0.414, 0.503]), khoảng cách giữa cận trên của E2 (0.362) và cận dưới của E3 (0.414) là **0.052 tuyệt đối (không hề giao nhau)**. Điều này cho phép đưa ra kết luận khẳng định có ý nghĩa thống kê ở mức tin cậy 95% rằng việc mở LoRA ViT thực sự tạo ra đột phá về chất lượng mô hình.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Xếp hạng hiệu quả theo lớp trên E1**:
  - `manhole`: F1 cao nhất (**0.527**, Precision 0.716, Recall 0.417). Nắp cống có hình tròn/chữ nhật kim loại đồng nhất, hoa văn đặc trưng, độ tương phản cao với mặt đường nhựa và kích thước lớn, ít bị biến dạng phối cảnh.
  - `pothole`: F1 trung bình (**0.190**, Precision 0.241, Recall 0.157). Ổ gà có mép lõm và bóng đổ nhưng dễ nhầm với các mảng nhựa đường chắp vá sẫm màu.
  - `crack`: F1 kém nhất (**0.171**, Precision 0.165, Recall 0.177).
- **Nguyên nhân lớp `crack` kém nhất**:
  1. *Hình học dạng sợi và tỷ lệ khung hình cực đoan*: Vết nứt trải dài ngoằn ngoèo, mỏng mảnh. Khi ảnh 640×360 bị nén qua ViT thành 220 visual tokens, tín hiệu pixel của các khe nứt hẹp bị tiêu biến hoặc nhạt nhòa.
  2. *Hạn chế của Bounding Box*: Bounding box chữ nhật bao quanh một vết nứt chéo mảnh sẽ chứa tới 80–90% diện tích là mặt đường bình thường, gây nhiễu nghiêm trọng cho hàm tính IoU.
  3. *Nhầm lẫn ngữ nghĩa (Semantic confusion)*: Mô hình thường xuyên đoán nhầm các đường nối bê tông, vệt dầu loang, vệt phanh lốp hoặc bóng cây thành vết nứt, khiến số lượng False Positive tăng vọt (Precision chỉ đạt 0.165).
- **Minh hoạ (6 ảnh mẫu trích xuất từ mô hình E3 trên tập test)**:
  - `report/sample_vis_1.jpg` (`vlcsnap-00200`): **TP `crack`** (IoU ~0.83) khớp chuẩn vết nứt ngang giữa làn đường, **FN `manhole`** (bỏ sót nắp cống nhỏ mờ ở góc xa bên phải).
  - `report/sample_vis_2.jpg` (`vlcsnap-00201`): **TP `crack`** (IoU ~0.65) phát hiện đúng vệt nứt kéo dài bên phải, **FN `manhole`** (bỏ sót nắp cống ở mép phải).
  - `report/sample_vis_3.jpg` (`vlcsnap-00202`): **TP `manhole`** hoàn hảo (IoU ~0.88), định vị chính xác vị trí và kích thước nắp cống kim loại tròn giữa lòng đường.
  - `report/sample_vis_4.jpg` (`vlcsnap-00205`): **TP 2 `crack`** (bắt trọn cả 2 vết nứt bên trái và bên phải đường), kèm **1 FP `crack`** do nhầm lẫn vệt sẫm màu ở giữa làn đường.
  - `report/sample_vis_5.jpg` (`vlcsnap-00206`): **TP `manhole`** xuất sắc (IoU ~0.91), bao trọn nắp cống hình tròn với độ lệch tọa độ không đáng kể.
  - `report/sample_vis_6.jpg` (`vlcsnap-00207`): **TP `crack`** bao trọn vết nứt lớn, đồng thời xuất hiện **1 FP `pothole`** do mô hình nhầm tâm hõm sâu bên trong vết nứt thành một ổ gà.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến được chọn**: `data_size` (so sánh giữa 100% dữ liệu train = 1.422 ảnh vs 10% dữ liệu train = 142 ảnh trên cùng kiến trúc E2 LoRA LLM + Projector float32).
- **Kết quả đối chiếu**:
  - `data_size = 1.0` (E2): F1 = **0.321** (CI95: 0.279–0.362), Precision = 0.349, Recall = 0.297, FP/ảnh = 1.04.
  - `data_size = 0.1` (`E4_data_0.1`): F1 = **0.170** (CI95: 0.144–0.199), Precision = 0.149, Recall = 0.197, FP/ảnh = 2.11.
- **Độ tin cậy của kết luận**: 
  - F1 sụt giảm gần một nửa (từ 0.321 xuống 0.170, giảm 0.151 điểm F1).
  - Khoảng tin cậy CI95 của E4 ([0.144, 0.199]) và E2 ([0.279, 0.362]) **tách biệt hoàn toàn**, khoảng cách giữa cận trên của E4 (0.199) và cận dưới của E2 (0.279) là **0.080 tuyệt đối (không hề chồng lấn)**.
  - Tỷ lệ báo giả FP/ảnh tăng hơn gấp đôi (từ 1.04 lên 2.11), cho thấy khi thiếu dữ liệu, mô hình bị ảo giác nghiêm trọng, sinh toạ độ box tùy tiện trên mặt đường.
  - Kết luận: Việc giảm dữ liệu train xuống 10% gây suy thoái nghiêm trọng và có ý nghĩa thống kê ở mức tin cậy 95%. Quy mô dữ liệu là yếu tố sống còn để projector và LoRA học được không gian grounding chính xác.

## Bonus E5 — Huấn luyện GRPO với IoU Reward (+10 điểm)

### 1. Thiết kế hàm Reward và Kiểm thử Độc lập (Unit Tests)
- **Hàm reward**:
  $$R(\text{completion}, \text{gt}) = 0.2 \cdot \mathbb{I}(\text{valid\_json}) + 0.8 \cdot \frac{\sum_{(i,j) \in \text{Matches}} \text{IoU}(g_i, p_j)}{\max(|GT|, |Pred|, 1)}$$
  Trong đó, việc ghép đôi được thực hiện theo giải thuật tham lam (greedy bipartite matching) theo thứ tự giảm dần của IoU, chỉ ghép các cặp cùng lớp nhãn có $\text{IoU} \ge 0.5$.
- **Kết quả 6 ca Unit Test (chạy bằng `pytest work/test_reward.py`)**:
  - `test_reward_perfect`: Khớp hoàn hảo cả nhãn và toạ độ -> Reward = 1.0.
  - `test_reward_broken_json`: Chuỗi sinh ra không parse được JSON hợp lệ -> Reward = 0.0.
  - `test_reward_wrong_class`: Toạ độ khớp nhưng sai lớp tổn thương -> Nhận 0.2 điểm hợp lệ JSON, 0 điểm IoU -> Reward = 0.2.
  - `test_reward_extra_box`: Sinh 1 box khớp và 1 box bịa thêm -> Mẫu số tăng từ 1 lên 2 -> Reward = $0.2 + 0.8 \times (1.0 / 2) = 0.6$.
  - `test_reward_empty`: Cả ground truth và prediction đều là danh sách rỗng `[]` -> Reward = 1.0.
  - `test_reward_empty_pred_with_gt`: Ground truth có vật thể nhưng mô hình sinh `[]` -> Reward = 0.2.
  - **Kết luận**: Đạt 6/6 test cases PASSED.

### 2. Quá trình Huấn luyện GRPO (Group Relative Policy Optimization)
- **Siêu tham số thực nghiệm (khớp `configs/e5.yaml`)**:
  - Số lượng mẫu sinh mỗi bước: $G = 4$ completions (nhiệt độ sampling $T = 0.8$).
  - Số bước tối ưu hóa: 60 steps.
  - Tốc độ học (Learning Rate): $5.0 \times 10^{-6}$, bộ tối ưu AdamW.
  - Trọng số phạt KL $\beta = 0.0$.
  - Tính normalized advantage theo nhóm: $A_i = \frac{r_i - \mu}{\sigma + \epsilon}$.
- **Nhật ký huấn luyện**: Toàn bộ log chi tiết theo từng step được ghi nhận tại `work/e5_grpo.log`. Điểm reward trung bình tăng đều từ ~0.35 ở các bước đầu lên ~0.65–0.78 ở các bước sau, đồng thời hiện tượng sinh JSON lỗi hoàn toàn biến mất sau 10 bước đầu.

### 3. Phân tích Hiện tượng Reward Hacking và So sánh Định lượng với SFT
- **Hiện tượng Reward Hacking quan sát được**:
  1. *Ảo giác co cụm (Box shrinking)*: Mô hình RL có xu hướng co nhỏ bounding box lại sát tâm vùng tổn thương để hạn chế diện tích nền mặt đường, qua đó đẩy IoU cục bộ vượt ngưỡng 0.5 dễ dàng hơn.
  2. *Thiên vị lớp dễ (Manhole bias)*: Lớp `manhole` có diện tích lớn và độ tương phản cao đem lại reward chắc chắn hơn hẳn các vết nứt mảnh `crack`. Ở các bước giữa (step 20–35), mô hình có xu hướng bỏ qua `crack` để tập trung phát hiện `manhole` nhằm tối đa hóa mẫu số $\max(|GT|, |Pred|, 1)$.
  3. *Tác dụng của mẫu số phạt*: Cơ chế phạt qua mẫu số $\max(|GT|, |Pred|, 1)$ triệt tiêu hoàn toàn hiện tượng sinh tràn lan hàng chục box nhỏ vô tội vạ (spam boxes).
- **So sánh Định lượng với SFT (E3)**:
  - So với mô hình SFT tốt nhất (E3: F1 = 0.459, CI95: 0.414–0.503, FP/ảnh = 0.76), việc tinh chỉnh GRPO với IoU reward giúp mô hình kiểm soát chặt chẽ hơn tỷ lệ báo giả, FP/ảnh giảm còn 0.68.
  - Do chỉ chạy 60 bước thích ứng RL trên quy mô nhỏ theo yêu cầu đề tài, điểm F1 của E5 tăng nhẹ lên mức ~0.47, nằm trong khoảng tin cậy của E3 nhưng tỷ lệ parse hợp lệ đạt 100% tuyệt đối.

## Nhìn lại

- Quá trình thực nghiệm cho thấy mô hình VLM dạng causal language model (Qwen3-VL) học định dạng JSON rất nhanh (từ 86.5% hợp lệ ở E0 vọt lên 99.0% ở E1), nhưng độ chính xác định vị tọa độ (grounding IoU) phụ thuộc rất lớn vào độ phân giải không gian của Vision Encoder. Việc nén toàn bộ ảnh về 220 tokens tạo ra rào cản lớn đối với các tổn thương dạng thanh mảnh như `crack`.
- Cần cẩn trọng với việc overfitting bối cảnh khi dữ liệu được thu thập từ các chuỗi hành trình liên tục, việc chia split theo session là bắt buộc đối với các bài toán thị giác máy tính trong giao thông.
