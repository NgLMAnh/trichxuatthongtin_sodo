# Context Summary: Chatbot Trích Xuất Dữ Liệu Giấy Chứng Nhận Quyền Sử Dụng Đất

## 1. Bối cảnh bài toán

Người dùng đang là **thực tập sinh (intern)** và được giao nghiên cứu/xây dựng thử nghiệm cho bài toán doanh nghiệp:

> Trích xuất dữ liệu từ ảnh scan Giấy chứng nhận quyền sử dụng đất / sổ đỏ / sổ hồng, sau đó hướng tới chatbot có thể truy vấn thông tin.

### Đầu vào

- Ảnh scan giấy chứng nhận
- Có thể là JPG / PNG / PDF scan
- Một giấy chứng nhận có thể gồm **nhiều trang**
- Dữ liệu thực tế có **nhiều mẫu cũ và mới**
- Layout giữa các mẫu không hoàn toàn giống nhau

### Mục tiêu dài hạn

```text
Ảnh scan nhiều trang
    ↓
Trích xuất dữ liệu có cấu trúc
    ↓
Chuẩn hóa + kiểm tra
    ↓
Lưu database
    ↓
Chatbot truy vấn dữ liệu
```

### Lưu ý về vai trò người dùng

Người dùng chỉ đang là **intern**, vì vậy khi trao đổi với sếp nên:

- Không overclaim
- Không “chốt kiến trúc” như senior
- Nói theo hướng:
  - “Em đang nghĩ tới...”
  - “Em muốn thử baseline...”
  - “Em muốn benchmark/đánh giá trước...”
  - “Nếu kết quả chưa ổn thì mới cân nhắc...”
- Ưu tiên POC nhỏ, dễ kiểm chứng
- Đề xuất kỹ thuật dưới dạng giả thuyết cần xác nhận với mentor/senior

---

# 2. Phát hiện quan trọng từ dữ liệu mẫu

Người dùng đã kiểm tra dữ liệu và thấy:

- Có mẫu cũ
- Có mẫu mới
- Layout không đồng nhất
- Một giấy chứng nhận có nhiều trang

Điều này dẫn tới kết luận:

> Không nên dùng duy nhất một bộ OCR + spatial rules cố định cho toàn bộ dữ liệu.

Thay vào đó, về dài hạn có thể cần:

```text
Document
    ↓
Template Detection
    ↓
Page Type Detection
    ↓
OCR + Bounding Boxes
    ↓
Rule set phù hợp theo:
template + page type
    ↓
Normalize
    ↓
Validate
    ↓
Merge document-level JSON
```

Tuy nhiên với yêu cầu hiện tại của sếp là **“thử trên 1 mẫu trước”**, chưa cần xây toàn bộ hệ thống multi-template.

---

# 3. Hướng tiếp cận tổng thể đã thống nhất

## 3.1. Không làm chatbot trước

Mục tiêu POC đầu tiên nên là:

> Một bộ giấy chứng nhận nhiều trang → một JSON có cấu trúc đúng.

Chatbot để sau.

## 3.2. Không dùng VLM end-to-end làm đường chính ngay từ đầu

Không nên bắt đầu bằng:

```text
Ảnh scan
    ↓
VLM
    ↓
JSON
```

Hướng an toàn hơn:

```text
OCR
+
Bounding Boxes
+
Anchor Detection
+
Spatial Rules
+
Regex/Parser
+
Validation
```

Sau này mới cân nhắc:

- Layout-aware model
- LayoutLMv3
- PaddleOCR-VL
- Qwen-VL
- VLM fallback

---

# 4. Khái niệm Spatial Rules

## Định nghĩa

**Spatial rules** là các luật dựa trên vị trí tương đối giữa các text box sau OCR.

Ví dụ OCR trả:

```text
"Thửa đất số"   bbox=(100, 200, 250, 230)
"125"           bbox=(270, 200, 320, 230)
```

Rule:

> Tìm anchor `"Thửa đất số"` → lấy candidate gần nhất bên phải, cùng dòng, trong khoảng cách cho phép.

Kết quả:

```json
{
  "parcel_number": "125"
}
```

## Các rule phổ biến

### Nearest right

```text
Thửa đất số: 125
```

### Nearest below

```text
Địa chỉ thửa đất
123 Nguyễn Văn A
```

### Same row

Dựa vào độ chồng lấp theo trục Y.

### Same column

Phù hợp với dữ liệu bảng.

### Distance threshold

Không lấy candidate quá xa anchor.

## Pseudo-code

```python
anchor = find_text("Thửa đất số")

candidates = [
    box for box in ocr_boxes
    if box.x1 > anchor.x2
    and same_row(box, anchor)
    and distance(box, anchor) < 300
]

value = nearest(candidates)
```

---

# 5. Làm rõ thứ tự OCR và Template Classification

Đã có một điểm gây nhầm lẫn trong trao đổi:

Người dùng nghĩ:

```text
Phân loại template
    ↓
OCR + spatial rules theo template
```

Điều này đúng về logic.

Tuy nhiên OCR cũng có thể chạy **trước phân loại template**, vì OCR text và bbox có thể dùng làm feature để nhận diện template.

Có 2 cách:

## Cách 1 — Phân loại template từ ảnh

```text
Ảnh
 ↓
Image Classifier
 ↓
Template A/B/C
 ↓
OCR
 ↓
Spatial Rules tương ứng
```

## Cách 2 — OCR trước rồi phân loại template

```text
Ảnh
 ↓
OCR text + bbox
 ↓
Keyword + layout signature
 ↓
Template A/B/C
 ↓
Spatial Rules tương ứng
```

### Khuyến nghị cho intern/POC

Ưu tiên cách 2 vì:

- Dễ thử
- Chưa cần train classifier
- Có thể dùng keyword đơn giản

Ví dụ:

```python
if "Mã QR" in full_text:
    template = "new"
elif "GIẤY CHỨNG NHẬN QUYỀN SỬ DỤNG ĐẤT" in full_text:
    template = "legacy"
else:
    template = "unknown"
```

---

# 6. Cách nói với sếp đã thống nhất

Nên mở đầu theo hướng:

> Em có check thử một số dữ liệu mẫu thì thấy gồm cả mẫu cũ và mẫu mới, layout cũng không hoàn toàn giống nhau. Hướng em đang nghĩ tới là trước tiên mình nên phân nhóm dữ liệu mẫu, sau đó thử một baseline đơn giản để đánh giá. Với từng nhóm/template phổ biến có thể OCR để lấy text và bounding box, rồi dùng spatial rules hoặc regex phù hợp. Nếu số lượng layout quá nhiều hoặc rule không đủ ổn định thì mới cân nhắc thêm layout-aware model hoặc VLM.

Vì là intern, nên dùng giọng:

- “Em đang nghĩ tới...”
- “Em muốn thử...”
- “Em muốn trao đổi xem hướng này có phù hợp không...”

Không nên nói:

> “Hệ thống phải dùng hybrid OCR + LayoutLM + VLM fallback.”

---

# 7. Benchmark là gì

**Benchmark** trong ngữ cảnh này là:

> Thử một hoặc nhiều phương pháp trên cùng một tập dữ liệu rồi đo kết quả để so sánh.

Ví dụ:

- OCR + Regex
- OCR + Spatial Rules
- VLM

Đánh giá:

- Accuracy số thửa
- Accuracy tờ bản đồ
- Accuracy diện tích
- Tốc độ xử lý
- Tỷ lệ lỗi

Với intern, khi nói với sếp có thể dùng câu dễ hiểu hơn:

> “Sau đó mình thử nghiệm và đánh giá độ chính xác trên từng nhóm mẫu.”

---

# 8. Yêu cầu hiện tại của sếp: Thử trên 1 mẫu trước

Điểm rất quan trọng:

> “1 mẫu” nên hiểu là **1 template**, không nhất thiết chỉ 1 ảnh.

Nếu chỉ có đúng 1 ảnh:

- Có thể demo code chạy
- Không đủ đánh giá độ ổn định

Lý tưởng là có nhiều document cùng template.

---

# 9. POC đề xuất cho 1 template nhiều trang

## Mục tiêu

```text
1 template
+
nhiều trang
+
OCR từng trang
+
trích xuất theo loại trang
+
merge document JSON
```

## Pipeline

```text
Multi-page Document
        ↓
Split / collect pages
        ↓
Preprocess từng page
        ↓
OCR từng page
        ↓
Page Type Detection
        ↓
Rule set theo page type
        ↓
Extract Fields
        ↓
Normalize + Validate
        ↓
Merge Fields
        ↓
Document-level JSON
```

---

# 10. Phân biệt Template Type và Page Type

## Template Type

Là mẫu/phiên bản giấy chứng nhận.

Ví dụ:

```text
Template A = mẫu cũ
Template B = mẫu mới
Template C = mẫu khác
```

## Page Type

Là chức năng của từng trang.

Ví dụ:

```text
holder_info_page
land_info_page
diagram_page
change_history_page
unknown
```

### Kiến trúc dài hạn

```text
Document
   ↓
Template Detection
   ↓
Pages
   ↓
Page Type Detection
   ↓
OCR
   ↓
Rules theo:
template + page type
   ↓
Merge
```

---

# 11. Với POC 1 template, chưa cần template classifier

Vì sếp đã yêu cầu thử 1 mẫu trước, có thể hard-code:

```text
template = template_a
```

Sau đó tập trung vào:

```text
Template A
   ↓
Page 1 → OCR → extract
Page 2 → OCR → extract
Page 3 → OCR → extract
Page 4 → OCR → extract
   ↓
Merge JSON
```

---

# 12. Không nên giả định cứng page 1/page 2/page 3

Không nên mặc định lâu dài:

```text
page_1 = holder info
page_2 = land info
page_3 = diagram
page_4 = changes
```

Vì scan thực tế có thể:

- Thiếu trang
- Đảo thứ tự
- Scan trùng trang
- Trang bị xoay
- Có trang phụ

Do đó nên có `Page Type Detection`.

POC ban đầu có thể dùng keyword từ OCR.

Ví dụ:

```text
"Thửa đất số"
"Tờ bản đồ số"
"Diện tích"
```

→ `land_info_page`

```text
"Nội dung thay đổi"
"Xác nhận thay đổi"
```

→ `change_history_page`

---

# 13. Các field POC nên ưu tiên

Không lấy toàn bộ field ngay.

Đề xuất 5 field:

```text
parcel_number
map_sheet_number
area_m2
holder_name
address
```

Có thể mở rộng:

```text
land_use_purpose
land_use_term
```

---

# 14. Cấu trúc project POC

```text
land_certificate_poc/
│
├── data/
│   └── documents/
│       ├── DOC_001/
│       │   ├── page_01.jpg
│       │   ├── page_02.jpg
│       │   ├── page_03.jpg
│       │   └── page_04.jpg
│       │
│       └── DOC_002/
│           ├── page_01.jpg
│           ├── page_02.jpg
│           └── page_03.jpg
│
├── configs/
│   └── template_a/
│       ├── template.yaml
│       └── pages/
│           ├── holder_info.yaml
│           ├── land_info.yaml
│           └── changes.yaml
│
├── outputs/
│   ├── ocr/
│   ├── visualizations/
│   └── predictions/
│
├── src/
│   ├── preprocess.py
│   ├── ocr_engine.py
│   ├── page_classifier.py
│   ├── anchor_finder.py
│   ├── spatial_rules.py
│   ├── extractors.py
│   ├── normalizers.py
│   ├── validators.py
│   ├── document_merger.py
│   └── pipeline.py
│
├── evaluate.py
├── main.py
└── requirements.txt
```

---

# 15. Config strategy

Config nên tách theo 3 tầng:

```text
configs/
└── template_a/
    ├── template.yaml
    ├── pages/
    │   ├── holder_info.yaml
    │   ├── land_info.yaml
    │   └── changes.yaml
    └── common/
        ├── normalization.yaml
        └── validation.yaml
```

POC tối giản có thể bỏ `common/` trước.

---

# 16. `template.yaml`

Mục đích:

- Định danh template
- Cấu hình cách detect page type

Ví dụ:

```yaml
template_id: template_a
template_name: "Giấy chứng nhận mẫu A"
version: "1.0"

page_types:
  - holder_info
  - land_info
  - change_history

page_detection:
  holder_info:
    keywords:
      - "Người sử dụng đất"
      - "Chủ sở hữu"
      - "Họ và tên"
    min_keyword_matches: 1

  land_info:
    keywords:
      - "Thửa đất số"
      - "Tờ bản đồ số"
      - "Diện tích"
      - "Mục đích sử dụng"
    min_keyword_matches: 2

  change_history:
    keywords:
      - "Nội dung thay đổi"
      - "Xác nhận thay đổi"
      - "Biến động"
    min_keyword_matches: 1
```

---

# 17. `holder_info.yaml`

Ví dụ:

```yaml
page_type: holder_info

fields:
  holder_name:
    anchors:
      - "Người sử dụng đất"
      - "Họ và tên"
      - "Ông"
      - "Bà"

    strategy: nearest

    spatial:
      directions:
        - right
        - below
      max_distance: 500
      same_line_preferred: true
      min_vertical_overlap: 0.4

    value:
      type: text
      min_length: 3
      max_length: 150

    normalization:
      - strip
      - collapse_spaces

  holder_address:
    anchors:
      - "Địa chỉ thường trú"
      - "Địa chỉ"
      - "Nơi thường trú"

    strategy: nearest

    spatial:
      directions:
        - right
        - below
      max_distance: 700
      same_line_preferred: false

    value:
      type: text
      min_length: 5
      max_length: 300

    normalization:
      - strip
      - collapse_spaces
```

---

# 18. `land_info.yaml`

```yaml
page_type: land_info

fields:
  parcel_number:
    anchors:
      - "Thửa đất số"
      - "Số thửa"

    strategy: nearest

    spatial:
      directions:
        - right
      max_distance: 350
      same_line_required: true
      min_vertical_overlap: 0.5

    value:
      type: integer_string

    regex:
      - '\d+'

    normalization:
      - strip
      - remove_punctuation

  map_sheet_number:
    anchors:
      - "Tờ bản đồ số"
      - "Số tờ bản đồ"

    strategy: nearest

    spatial:
      directions:
        - right
      max_distance: 350
      same_line_required: true
      min_vertical_overlap: 0.5

    value:
      type: integer_string

    regex:
      - '\d+'

    normalization:
      - strip
      - remove_punctuation

  area_m2:
    anchors:
      - "Diện tích"
      - "Diện tích thửa đất"

    strategy: nearest

    spatial:
      directions:
        - right
        - below
      max_distance: 450
      same_line_preferred: true

    value:
      type: decimal

    regex:
      - '\d+(?:[.,]\d+)?'

    normalization:
      - strip
      - comma_to_dot

    validation:
      min: 0.01
      max: 100000000

  land_use_purpose:
    anchors:
      - "Mục đích sử dụng"
      - "Mục đích sử dụng đất"

    strategy: nearest

    spatial:
      directions:
        - right
        - below
      max_distance: 600

    value:
      type: text
      min_length: 2
      max_length: 200

    normalization:
      - strip
      - collapse_spaces

  land_use_term:
    anchors:
      - "Thời hạn sử dụng"
      - "Thời hạn"

    strategy: nearest

    spatial:
      directions:
        - right
        - below
      max_distance: 600

    value:
      type: text

    normalization:
      - strip
      - collapse_spaces
```

---

# 19. `changes.yaml`

```yaml
page_type: change_history

fields:
  changes:
    type: repeating_block

    anchors:
      - "Nội dung thay đổi"
      - "Xác nhận thay đổi"
      - "Biến động"

    extraction:
      strategy: below_region
      max_vertical_distance: 1500

    item_schema:
      change_date:
        regex:
          - '\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{4}'

      content:
        type: text
        min_length: 5

    normalization:
      change_date:
        - normalize_date

      content:
        - strip
        - collapse_spaces
```

Ghi chú:

- Phần biến động phức tạp hơn
- POC đầu tiên có thể chưa cần làm sâu

---

# 20. Config tối giản hơn cho POC

Không nên over-engineer YAML ngay.

Ví dụ `land_info.yaml` tối giản:

```yaml
page_type: land_info

fields:
  parcel_number:
    anchors:
      - "Thửa đất số"
    directions:
      - right
    max_distance: 300
    regex: '\d+'

  map_sheet_number:
    anchors:
      - "Tờ bản đồ số"
    directions:
      - right
    max_distance: 300
    regex: '\d+'

  area_m2:
    anchors:
      - "Diện tích"
    directions:
      - right
    max_distance: 400
    regex: '\d+(?:[.,]\d+)?'
```

Chỉ thêm option khi gặp lỗi thật.

---

# 21. Đọc YAML bằng Python

Dùng PyYAML:

```bash
pip install pyyaml
```

Code:

```python
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")

    return data
```

---

# 22. Page Type Detection từ OCR text

```python
def detect_page_type(
    ocr_text: str,
    template_config: dict,
) -> str:
    text = ocr_text.lower()

    best_page_type = "unknown"
    best_score = 0

    page_detection = template_config.get(
        "page_detection",
        {},
    )

    for page_type, config in page_detection.items():
        keywords = config.get("keywords", [])
        min_matches = config.get(
            "min_keyword_matches",
            1,
        )

        matches = sum(
            1
            for keyword in keywords
            if keyword.lower() in text
        )

        if matches >= min_matches and matches > best_score:
            best_page_type = page_type
            best_score = matches

    return best_page_type
```

---

# 23. Load page config tương ứng

```python
from pathlib import Path


def load_page_config(
    template_id: str,
    page_type: str,
) -> dict:
    path = Path(
        f"configs/{template_id}/pages/"
        f"{page_type}.yaml"
    )

    return load_yaml(path)
```

---

# 24. Extraction flow dựa trên config

Pseudo-code:

```python
def extract_fields(
    ocr_boxes: list[dict],
    page_config: dict,
) -> dict:
    results = {}

    fields = page_config.get(
        "fields",
        {},
    )

    for field_name, field_config in fields.items():
        anchors = field_config.get(
            "anchors",
            [],
        )

        anchor_box = find_best_anchor(
            ocr_boxes=ocr_boxes,
            anchors=anchors,
        )

        if anchor_box is None:
            results[field_name] = None
            continue

        candidate = apply_spatial_rule(
            anchor_box=anchor_box,
            ocr_boxes=ocr_boxes,
            spatial_config=field_config.get(
                "spatial",
                field_config,
            ),
        )

        results[field_name] = candidate

    return results
```

---

# 25. OCR output cần giữ những gì

Không chỉ giữ plain text.

Cần giữ:

```json
[
  {
    "text": "Thửa đất số",
    "bbox": [100, 200, 250, 230],
    "confidence": 0.98
  },
  {
    "text": "125",
    "bbox": [270, 200, 320, 230],
    "confidence": 0.96
  }
]
```

Mục đích:

- Spatial rules
- Debug
- Evidence
- Visualization
- Human review sau này

---

# 26. Anchor Matching

Không nên exact match tuyệt đối vì OCR có thể sai nhẹ.

Ví dụ:

```text
"Thửa đất số"
"Thửa đất sô"
"Thửa đât số"
"Thửa đất số:"
```

Nên:

- lowercase
- trim spaces
- remove punctuation
- normalize Unicode
- có thể fuzzy matching nhẹ

Không nên ngay lập tức dùng fuzzy matching quá rộng vì dễ match nhầm.

---

# 27. Normalization

Ví dụ:

```text
"120,5 m²"
```

→

```json
{
  "area_m2": 120.5
}
```

Tên:

```text
"  NGUYỄN   VĂN   A  "
```

→

```text
"NGUYỄN VĂN A"
```

Lưu ý:

- Không tự sửa OCR một cách âm thầm
- Nếu có correction, nên giữ raw value

Ví dụ:

```json
{
  "raw": "I25",
  "normalized": "125",
  "reason": "numeric_field_confusion_I_to_1"
}
```

---

# 28. Validation

Ví dụ:

## Parcel number

```text
125 → hợp lệ
I25 → nghi ngờ
ABC → không hợp lệ
```

## Area

```text
120.5 → hợp lệ
-50   → không hợp lệ
ABC   → không hợp lệ
```

Output:

```json
{
  "parcel_number": {
    "value": "125",
    "valid": true,
    "confidence": 0.96
  },
  "area_m2": {
    "value": 120.5,
    "valid": true,
    "confidence": 0.94
  }
}
```

---

# 29. Merge multi-page results

Ví dụ:

## Page holder info

```json
{
  "holder_name": "Nguyễn Văn A",
  "holder_address": "..."
}
```

## Page land info

```json
{
  "parcel_number": "125",
  "map_sheet_number": "12",
  "area_m2": 120.5
}
```

## Page change history

```json
{
  "changes": [
    {
      "date": "2023-05-12",
      "content": "..."
    }
  ]
}
```

## Document-level JSON

```json
{
  "document_id": "DOC_001",

  "holder": {
    "name": "Nguyễn Văn A",
    "address": "..."
  },

  "land_parcel": {
    "parcel_number": "125",
    "map_sheet_number": "12",
    "area_m2": 120.5
  },

  "changes": [
    {
      "date": "2023-05-12",
      "content": "..."
    }
  ]
}
```

---

# 30. Ground Truth cho POC

Mỗi document nên có JSON chuẩn viết tay để đánh giá.

Ví dụ:

```text
data/
├── documents/
│   └── DOC_001/
│       ├── page_01.jpg
│       ├── page_02.jpg
│       └── page_03.jpg
│
└── ground_truth/
    └── DOC_001.json
```

Ground truth:

```json
{
  "holder_name": "Nguyễn Văn A",
  "parcel_number": "125",
  "map_sheet_number": "12",
  "area_m2": 120.5,
  "land_use_purpose": "Đất ở tại đô thị"
}
```

---

# 31. Evaluation

Nếu có nhiều document cùng template:

| Field | Đúng | Tổng | Accuracy |
|---|---:|---:|---:|
| Số thửa | 19 | 20 | 95% |
| Tờ bản đồ | 20 | 20 | 100% |
| Diện tích | 18 | 20 | 90% |
| Tên chủ | 17 | 20 | 85% |

Phân tích lỗi:

```text
OCR sai: 5 lỗi
Anchor không tìm thấy: 2 lỗi
Spatial rule lấy nhầm: 3 lỗi
Ảnh mờ: 1 lỗi
```

---

# 32. Visualization rất nên làm

Nên vẽ bbox lên ảnh:

```text
- OCR boxes
- Anchor boxes
- Extracted value boxes
```

Mục tiêu:

- Debug
- Demo với sếp
- Chứng minh hệ thống lấy giá trị từ đâu

Ví dụ:

```text
Thửa đất số: [125]
                ↑
         PARCEL_NUMBER
```

---

# 33. Công nghệ đề xuất cho POC

## Preprocessing

```text
OpenCV
```

## OCR

Ưu tiên thử:

```text
PaddleOCR
```

Cần output:

- text
- bbox
- confidence

## Config

```text
YAML + PyYAML
```

## API

Chưa cần ngay.

Nếu cần:

```text
FastAPI
```

## Database

Chưa cần ngay cho POC.

Có thể xuất:

```text
JSON
```

---

# 34. Những thứ chưa nên làm ngay

Với intern và POC đầu tiên, chưa cần:

- Chatbot
- RAG
- Vector DB
- LayoutLMv3
- VLM end-to-end
- Template classifier bằng deep learning
- Microservices
- Kafka
- Kubernetes
- Full production architecture
- 20–30 field
- Diagram reconstruction

---

# 35. Khi nào mới cân nhắc Layout-aware Model hoặc VLM

Sau khi POC OCR + spatial rules được thử trên dữ liệu thật.

Nếu gặp:

- Quá nhiều layout
- Rule khó maintain
- Anchor thay đổi nhiều
- Text bị lệch dòng
- Table phức tạp
- OCR confidence thấp
- Unknown template

Thì mới benchmark:

```text
LayoutLMv3
PaddleOCR-VL
Qwen-VL
```

---

# 36. Decision Log

## Decision 1

Không làm chatbot trước.

## Decision 2

POC đầu tiên là:

```text
1 template
+
multi-page document
+
OCR
+
page type detection
+
spatial rules
+
JSON merge
```

## Decision 3

Không dùng một rule set chung cho tất cả template về dài hạn.

## Decision 4

Phân biệt rõ:

```text
template type
vs
page type
```

## Decision 5

OCR có thể chạy trước template/page classification để cung cấp keyword + bbox.

## Decision 6

Với 1 template POC:

```text
template = hard-coded
```

chưa cần template classifier.

## Decision 7

Page type có thể detect bằng keyword OCR trước.

## Decision 8

Config nên để trong YAML, tránh hard-code toàn bộ rule.

## Decision 9

POC ưu tiên 5 field:

```text
parcel_number
map_sheet_number
area_m2
holder_name
address
```

## Decision 10

Mọi OCR result nên giữ:

```text
text
bbox
confidence
```

---

# 37. Open Questions / Việc cần làm tiếp

1. Chọn chính xác template đầu tiên.
2. Xác định số lượng document cùng template hiện có.
3. Xem mỗi document có bao nhiêu trang.
4. Liệt kê page types thực tế của template đó.
5. Chọn 5 field POC.
6. Chạy OCR thử trên vài document.
7. Kiểm tra chất lượng bbox.
8. Viết `template.yaml`.
9. Viết `holder_info.yaml`.
10. Viết `land_info.yaml`.
11. Implement:
    - anchor finder
    - spatial rules
    - regex parser
    - normalizer
    - validator
    - merger
12. Tạo ground truth.
13. Đánh giá accuracy theo field.
14. Ghi lại lỗi để quyết định bước tiếp theo.

---

# 38. Recommended Immediate Next Step

Bước tiếp theo nên là:

```text
Chọn 1 template thật
    ↓
Lấy 5–20 document cùng template nếu có
    ↓
OCR tất cả page
    ↓
Vẽ bbox kiểm tra
    ↓
Xác định page types
    ↓
Chọn 5 field
    ↓
Viết YAML config
    ↓
Implement spatial extraction
    ↓
Merge document JSON
    ↓
Evaluate
```

---

# 39. Câu ngắn để báo cáo sếp

> Em đang thử theo hướng POC trên một template trước. Vì một giấy chứng nhận có nhiều trang nên em sẽ OCR từng trang, nhận diện loại trang bằng keyword/layout, áp dụng rule trích xuất phù hợp cho từng loại trang, rồi merge kết quả về một JSON chung ở mức document. Trước mắt em chỉ chọn một số field chính để đánh giá độ chính xác, sau đó dựa trên kết quả mới quyết định có cần mở rộng sang template classifier, layout-aware model hay VLM hay không.

---

# 40. Tóm tắt 1 câu cho AI khác

> Đây là một POC Document AI cho Giấy chứng nhận quyền sử dụng đất: hiện tại ưu tiên 1 template nhiều trang, OCR từng trang để lấy text+bbox+confidence, detect page type bằng keyword, áp dụng YAML-configured anchor/spatial rules theo page type, normalize/validate, merge thành document-level JSON, đánh giá accuracy theo field; chưa làm chatbot/VLM/production architecture cho tới khi baseline được kiểm chứng.
