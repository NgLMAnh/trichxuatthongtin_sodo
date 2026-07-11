import re

def remove_accents(input_str):
    """
    Removes Vietnamese accents from a string using a robust dictionary translation.
    """
    if not input_str:
        return ""
    mapping = {
        'a': 'áàảãạăắằẳẵặâấầẩẫậä',
        'A': 'ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÄ',
        'd': 'đ',
        'D': 'Đ',
        'e': 'éèẻẽẹêếềểễệë',
        'E': 'ÉÈẺẼẸÊẾỀỂỄỆË',
        'i': 'íìỉĩịï',
        'I': 'ÍÌỈĨỊÏ',
        'o': 'óòỏõọôốồổỗộơớờởỡợö',
        'O': 'ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÖ',
        'u': 'úùủũụưứừửữựü',
        'U': 'ÚÙỦŨỤƯỨỪỬỮỰÜ',
        'y': 'ýỳỷỹỵÿ',
        'Y': 'ÝỲỶỸỴŸ'
    }
    trans_dict = {}
    for k, v in mapping.items():
        for char in v:
            trans_dict[ord(char)] = k
    return input_str.translate(trans_dict)

def normalize_text(text):
    """
    Normalizes text for matching (lowercase, remove accents, strip, remove punctuation, replace multiple spaces).
    """
    if not text:
        return ""
    text = remove_accents(text)
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _joined_with_below(block, blocks, max_dy=20, min_x_overlap=0.3):
    """
    Một số mẫu quét (bảng dữ liệu cũ) có nhãn cột bị OCR tách thành NHIỀU block
    xếp theo chiều dọc do wrap dòng trong ô hẹp (VD "Số tờ" / "bản đồ" là 2
    block riêng thay vì 1 block "Số tờ bản đồ"). find_best_anchor so khớp
    từng block ĐƠN LẺ nên bỏ lỡ các nhãn kiểu này. Hàm này ghép thêm text của
    1 block ngay bên dưới, cùng cột (x chồng lấn, khoảng cách y nhỏ), để thử
    so khớp anchor trên cụm đã ghép - không đổi block trả về (vẫn là block
    gốc phía trên) nên không ảnh hưởng logic tìm giá trị lân cận phía sau.
    """
    bx1, by1, bx2, by2 = block["bbox"]
    best = None
    best_dy = None
    for other in blocks:
        if other is block:
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
            best_dy = dy
            best = other
    if best is None:
        return None
    return f"{block.get('text', '')} {best.get('text', '')}".strip()


def find_best_anchor(blocks, anchor_keywords):
    """
    Finds the best block matching one of the anchor keywords.
    """
    flexible_regexes = {
        "thua dat so": r'\bth[uư]a\s+đ[aăâ]t\s+s[oỏóõọôốồổỗộäëïöü]?\b',
        "so thua": r'\bs[oỏóõọôốồổỗộäëïöü]?\s+th[uư]a\b',
        "to ban do so": r'\bt[oơớờởỡợäëïöü]?\s+b[aăâảãạ]n\s+đ[oôốồổỗộäëïöü]?[ \-\.]*s[oôốồổỗộäëïöü]?\b',
        "so to ban do": r'\bs[oôốồổỗộäëïöü]?\s+t[oơớờởỡợäëïöü]?\s+b[aăâảãạ]n\s+đ[oôốồổỗộäëïöü]?\b',
        "dien tich": r'\bd[iíìỉĩịïeë]e[nñ]?\s+t[iíìỉĩịïeë]ch\b',
        "dien tich thua dat": r'\bd[iíìỉĩịïeë]e[nñ]?\s+t[iíìỉĩịïeë]ch\s+th[uư]a\s+đ[aăâ]t\b',
        "ho va ten": r'\bh[oọóỏõôốồổỗộö]\s+v[aàảãạăắằẳẵặâấầẩẫậä]\s+t[eéèẻẽẹêếềểễệë]n\b',
        "dia chi thuong tru": r'\bđ[iịíìỉĩï]\s+ch[iỉíìĩịï]\s+th[uư][oơớờởỡợö]ng\s+tr[uúùủũụưứừửữựü]\b',
        "noi thuong tru": r'\bn[oơớờởỡợö]i\s+th[uư][oơớờởỡợö]ng\s+tr[uúùủũụưứừửữựü]\b',
        "dia chi": r'\bđ[iịíìỉĩï]\s+ch[iỉíìĩịï]\b',
        "ba": r'^\s*b[aàảãạăắằẳẵặâấầẩẫậä]\b',
        "ong": r'^\s*[oôốồổỗộơớờởỡợö]ng\b',
    }
    
    matches = []
    
    for block in blocks:
        text = block.get("text", "")
        if not text:
            continue

        joined_text = _joined_with_below(block, blocks)
        # Thử so khớp trên CẢ text gốc lẫn text đã ghép với block liền dưới
        # cùng cột (xem docstring _joined_with_below) - dùng candidate dài
        # nhất khớp được để không đổi hành vi khi text gốc đã đủ khớp.
        candidate_texts = [text] + ([joined_text] if joined_text and joined_text != text else [])

        for candidate_text in candidate_texts:
            text_lower = candidate_text.lower().strip()
            text_clean = re.sub(r'^[^\w\s]+', '', text_lower).strip()
            norm_text = normalize_text(candidate_text)

            for kw_idx, kw in enumerate(anchor_keywords):
                norm_kw = normalize_text(kw)
                if not norm_kw:
                    continue

                if norm_text == norm_kw:
                    matches.append((block, 100.0, len(text), kw_idx))
                    continue

                pattern = flexible_regexes.get(norm_kw)
                matched = False
                if pattern:
                    deaccented_pattern = remove_accents(pattern)
                    if norm_kw in ["ba", "ong"]:
                        if re.search(pattern, text_clean) or re.search(deaccented_pattern, remove_accents(text_clean)):
                            matches.append((block, 90.0, len(text), kw_idx))
                            matched = True
                    else:
                        if re.search(pattern, text_lower) or re.search(deaccented_pattern, norm_text):
                            similarity = len(norm_kw) / max(1, len(norm_text))
                            matches.append((block, similarity, len(text), kw_idx))
                            matched = True

                if matched:
                    continue

                if norm_kw in ["ba", "ong"]:
                    continue

                pattern_wb = r'\b' + re.escape(norm_kw) + r'\b'
                if re.search(pattern_wb, norm_text):
                    similarity = len(norm_kw) / max(1, len(norm_text))
                    matches.append((block, similarity, len(text), kw_idx))

    if not matches:
        return None
        
    matches.sort(key=lambda x: (x[3], -x[1], x[2]))
    return matches[0][0]
