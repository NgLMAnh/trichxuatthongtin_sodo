import re

from src.text_utils import normalize_text, remove_accents

# Bỏ tiền tố đánh mục con ở đầu block trước khi tách cặp nhãn:giá trị, để KEY
# field ổn định giữa các mẫu (không phụ thuộc số thứ tự mục). Bắt các dạng:
#   "a. " "đ. " (1 chữ cái + chấm/ngoặc), "1. " "12. " (số), "IV. " (La Mã),
#   "a) " "1) " (ngoặc đơn).
_LABEL_PREFIX_RE = re.compile(r"^\s*(?:[0-9]{1,2}|[IVX]{1,4}|[a-zđA-ZĐ])[.)]\s*")
_KV_RE = re.compile(r"^([^:]{2,50}?)\s*:\s*(.+)$")

# Số quyết định (VD "QĐ số 138/TTg", "QĐ.Số.6382/QĐ-UB-QLĐT", "Quyết định số...")
# KHÔNG theo dạng "nhãn: giá trị" (toàn bộ block CHÍNH LÀ giá trị), và nội dung đầy
# đủ (ngày ký, nơi ký) thường trải trên NHIỀU block liên tiếp theo cột (do OCR
# tách theo dòng trong 1 ô/cụm văn bản) - _KV_RE ở trên không bắt được. Đây là
# lớp bổ sung riêng để không bỏ sót loại thông tin này.
_DECISION_MARKER_RE = re.compile(r"\bqd\b|\bquyet\s*dinh\b")


def _slugify(label):
    label = remove_accents(label).lower().strip()
    label = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return label or None


def _find_below_same_column(block, blocks, excluded_ids, max_dy=25, min_x_overlap=0.25):
    """Tìm block gần nhất phía dưới, cùng cột (x chồng lấn), bỏ qua các block
    trong excluded_ids (đã dùng, hoặc là mốc quyết định khác - xem
    _extract_decision_refs)."""
    bx1, by1, bx2, by2 = block["bbox"]
    best, best_dy = None, None
    for other in blocks:
        if other.get("block_id") in excluded_ids:
            continue
        ox1, oy1, ox2, oy2 = other["bbox"]
        dy = oy1 - by2
        if dy < -2 or dy > max_dy:
            continue
        overlap = max(0, min(bx2, ox2) - max(bx1, ox1))
        width = min(bx2 - bx1, ox2 - ox1)
        if width <= 0 or overlap / width < min_x_overlap:
            continue
        if best_dy is None or dy < best_dy:
            best_dy, best = dy, other
    return best


def _extract_decision_refs(blocks, block_to_section):
    """Bắt các cụm 'QĐ số ...' / 'Quyết định số ...' bị OCR trải trên nhiều
    block liên tiếp (nhãn+giá trị nằm chung 1 block, không có dấu ':'), ghép
    lại thành 1 thông tin hoàn chỉnh (số QĐ + ngày ký + nơi ký nếu có)."""
    candidates = [
        b for b in blocks if _DECISION_MARKER_RE.search(normalize_text(b.get("text") or ""))
    ]
    if not candidates:
        return []
    candidate_ids = {b["block_id"] for b in candidates}

    results = []
    used = set()
    for marker in candidates:
        if marker["block_id"] in used:
            continue
        used.add(marker["block_id"])
        chain = [marker]
        current = marker
        for _ in range(5):
            # Chỉ loại các block ĐÃ dùng khỏi tìm kiếm (KHÔNG loại candidate_ids ở
            # đây) - nếu không, khi block gần nhất phía dưới là 1 mốc "QĐ" khác,
            # thuật toán sẽ NHẢY QUA nó và nối nhầm sang nội dung của mốc kế tiếp.
            # Thay vào đó: nếu block gần nhất là 1 mốc khác thì DỪNG chuỗi tại đây.
            nxt = _find_below_same_column(current, blocks, used)
            if not nxt:
                break
            if nxt["block_id"] in candidate_ids:
                break
            chain.append(nxt)
            used.add(nxt["block_id"])
            current = nxt

        value = " ".join(b.get("text", "").strip() for b in chain if b.get("text", "").strip())
        results.append(
            {
                "label": "Quyết định",
                "key": None,  # gán số thứ tự ở extract_generic_fields
                "value": value,
                "section": block_to_section.get(marker.get("block_id"), "unknown"),
            }
        )
    return results


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

    decision_refs = _extract_decision_refs(blocks, block_to_section)
    for idx, ref in enumerate(decision_refs, start=1):
        dedupe_key = (f"quyet_dinh_{idx}", ref["value"])
        if dedupe_key in seen or not ref["value"]:
            continue
        seen.add(dedupe_key)
        label = "Quyết định" if len(decision_refs) == 1 else f"Quyết định {idx}"
        results.append(
            {
                "label": label,
                "key": f"quyet_dinh_{idx}",
                "value": ref["value"],
                "section": ref["section"],
            }
        )

    return results
