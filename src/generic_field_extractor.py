import re

from src.text_utils import remove_accents

# Bỏ tiền tố đánh mục con kiểu "a. ", "đ. " ở đầu block trước khi tách cặp nhãn:giá trị.
_LABEL_PREFIX_RE = re.compile(r"^\s*[a-zđA-ZĐ]\.\s*")
_KV_RE = re.compile(r"^([^:]{2,50}?)\s*:\s*(.+)$")


def _slugify(label):
    label = remove_accents(label).lower().strip()
    label = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return label or None


def extract_generic_fields(blocks, sections):
    """
    Lớp bổ sung ngoài các field khai báo tay (field_extraction/change_extraction/
    holder_extraction): bắt MỌI dòng dạng "<nhãn>: <giá trị>" (kể cả nhiều cặp
    trong 1 block, phân tách bởi ";", ví dụ "Thửa đất số: 251; tờ bản đồ số: 74")
    thành field key tự sinh từ nhãn - để không thông tin nào trên giấy bị bỏ sót,
    áp dụng được cho cả mẫu chưa từng gặp (không cần khai báo tay).

    Trùng lặp với field đã khai báo tay là CHẤP NHẬN ĐƯỢC (dư thừa nhưng an toàn)
    vì mục đích là mạng lưới an toàn, không thay thế các field đã có.
    """
    block_to_section = {}
    for section_name, block_ids in sections.items():
        for block_id in block_ids:
            block_to_section[block_id] = section_name

    results = []
    seen = set()
    for block in blocks:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        text = _LABEL_PREFIX_RE.sub("", text, count=1)

        for segment in text.split(";"):
            segment = segment.strip().strip(",").strip()
            match = _KV_RE.match(segment)
            if not match:
                continue

            label = match.group(1).strip()
            value = match.group(2).strip().rstrip(",").strip()
            if not label or not value:
                continue

            key = _slugify(label)
            if not key:
                continue

            dedupe_key = (key, value)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            results.append(
                {
                    "label": label,
                    "key": key,
                    "value": value,
                    "section": block_to_section.get(block.get("block_id"), "unknown"),
                }
            )

    return results
