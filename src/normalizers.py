import re

# Các cặp chữ<->số HAY BỊ VietOCR nhầm (VD "6" đọc thành "G", "0" thành "O").
# CHỈ dùng cho field CHẮC CHẮN là số (CMND/CCCD, diện tích) - KHÔNG bao giờ áp
# dụng cho tên/địa chỉ (sẽ phá chữ hợp lệ). Chỉ các cặp có độ tin cậy cao.
_OCR_LETTER_TO_DIGIT = {
    'G': '6', 'g': '6',
    'O': '0', 'o': '0', 'Q': '0', 'D': '0',
    'I': '1', 'l': '1', 'L': '1',
    'S': '5', 's': '5',
    'B': '8',
    'Z': '2', 'z': '2',
}

def fix_numeric_ocr(s):
    """Sửa lỗi OCR nhầm chữ<->số cho chuỗi ĐÁNG LẼ TOÀN SỐ. Chỉ thay khi chuỗi
    chủ yếu là chữ số (>=50% ký tự là số) để tránh đổi nhầm chuỗi văn bản."""
    if not s:
        return s
    digit_like = sum(1 for c in s if c.isdigit() or c in _OCR_LETTER_TO_DIGIT)
    letters_all = sum(1 for c in s if c.isalnum())
    if letters_all == 0 or digit_like / letters_all < 0.5:
        return s
    return ''.join(_OCR_LETTER_TO_DIGIT.get(c, c) for c in s)

def normalize_text(text):
    """
    Standard text normalization (strip, collapse spaces).
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def correct_ocr_typos(val):
    if not val:
        return ""
    
    # VietOCR is highly accurate for Vietnamese diacritics.
    # We only need to clean up minor artifacts like single dots between letters
    # e.g., "Cu.Xa" -> "Cu Xa", but NOT "18/2/3.5" (number.number)
    val = re.sub(r'(?<=[A-Za-zÀ-ỹ])\.(?=[A-Za-zÀ-ỹ])', ' ', val)
    
    return val

def clean_name_or_address(val):
    if not val:
        return ""
    # Remove leading/trailing dots, spaces, underscores, colons, hyphens, slashes,
    # AND dấu phẩy/chấm phẩy cuối (VD "sở hữu chung," / "...Bình Chiều;" - đuôi
    # dấu câu thừa từ OCR/cách trình bày trên giấy, không phải nội dung).
    val = re.sub(r'^[ \.\-_:\(\)\/,;]+', '', val)
    val = re.sub(r'[ \.\-_:\(\)\/,;]+$', '', val)
    # Remove large clusters of dots in between
    val = re.sub(r'\.{2,}', ' ', val)
    val = re.sub(r'_{2,}', ' ', val)
    # Collapse multiple spaces
    val = re.sub(r'\s+', ' ', val).strip()

    # Correct OCR spelling mistakes
    val = correct_ocr_typos(val)
    return val

def normalize_area(area_str, min_valid=0.1, max_valid=1_000_000):
    """
    Chuẩn hoá diện tích về float m². Xử lý đúng định dạng số Việt Nam:
    - '513,893' hoặc '513.893'  -> 513.893  (1 dấu = phần thập phân)
    - '6.748,4'                 -> 6748.4    (CÓ CẢ '.' và ',' => '.'=ngăn nghìn, ','=thập phân)
    - '6748,4m²' / '509,0'      -> 6748.4 / 509.0
    Trả về None nếu ngoài khoảng hợp lý [min_valid, max_valid] (lọc rác OCR).
    """
    if not area_str:
        return None
    s = str(area_str)
    # Bỏ đơn vị đo TRƯỚC (m², m2, m?, "mét vuông") - nếu không, chữ số '2' trong
    # "m2" sẽ dính vào con số (VD "120,5 m2" -> "120,52" sai).
    s = re.sub(r'(?i)\s*m\s*[2²?]?', '', s)
    s = re.sub(r'(?i)mét\s*vuông', '', s)
    # Sửa lỗi OCR chữ<->số cho phần diện tích (VD "5O9" -> "509", "6G8" -> "668")
    s = fix_numeric_ocr(s)
    # Chỉ giữ chữ số, '.', ',' - bỏ ký tự còn lại
    s = re.sub(r'[^0-9.,]', '', s)
    if not s:
        return None

    if '.' in s and ',' in s:
        # Định dạng VN: '.' ngăn nghìn, ',' thập phân -> bỏ '.', đổi ',' thành '.'
        s = s.replace('.', '').replace(',', '.')
    else:
        # Chỉ 1 loại dấu -> coi là dấu thập phân (giữ nguyên hành vi cũ cho
        # '513.893'/'513,893'); chuẩn hoá về '.'
        s = s.replace(',', '.')
        # Nếu còn nhiều dấu '.' (VD OCR '6.748.4'), chỉ giữ dấu cuối làm thập phân
        if s.count('.') > 1:
            head, _, tail = s.rpartition('.')
            s = head.replace('.', '') + '.' + tail

    match = re.search(r'\d+(?:\.\d+)?', s)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    if value < min_valid or value > max_valid:
        return None
    return value

# Tiền tố vai trò/quan hệ hay dính vào ĐẦU tên do OCR gộp dòng ("VÀ VỢ: BÀ ...").
_NAME_ROLE_PREFIX_RE = re.compile(
    r'^\s*(?:và\s+(?:vợ|chồng)\s*:?\s*)?(?:ông|bà|anh|chị)\s*:?\s*',
    re.IGNORECASE,
)

def clean_name(val):
    if not val:
        return ""
    # Cắt đuôi ", CMND/CCCD: <số>" hoặc "CCCD số ..." dính vào tên (mẫu hợp nhất
    # gộp tên + giấy tờ trong 1 dòng), tránh tên chứa cả số căn cước.
    val = re.sub(r'(?i)[,;]?\s*(?:CMND|CCCD|số\s*định\s*danh)\b.*$', '', val)
    # Strip "sinh năm" or "sn" and anything after it from names
    val = re.sub(r'(?i)\b(?:sinh năm|năm sinh|sn)\b.*$', '', val)
    # Bỏ tiền tố vai trò/quan hệ ở đầu ("VÀ VỢ:", "BÀ", "Ông")
    val = _NAME_ROLE_PREFIX_RE.sub('', val)
    # Use generic clean up
    val = clean_name_or_address(val)
    return val

def normalize_birthday(val, min_year=1900, max_year=2025):
    """Chuẩn hoá năm sinh: chỉ chấp nhận năm 4 chữ số trong khoảng hợp lệ,
    loại rác OCR (mảnh số CCCD, năm cấp giấy, số thửa 4 chữ số...)."""
    if not val:
        return None
    text = normalize_text(str(val))
    m = re.search(r'\b(1\d{3}|20\d{2})\b', text)
    if not m:
        return None
    year = int(m.group(1))
    if year < min_year or year > max_year:
        return None
    return str(year)

def normalize_fields(extracted_data):
    """
    Applies appropriate normalization functions to extracted fields.
    """
    normalized = {}
    
    # 1. holder_name
    if "holder_name" in extracted_data:
        normalized["holder_name"] = clean_name(extracted_data["holder_name"])
        
    # 2. address
    if "address" in extracted_data:
        normalized["address"] = clean_name_or_address(extracted_data["address"])
        
    # 3. id_number (CCCD/CMND)
    if "id_number" in extracted_data:
        val = extracted_data["id_number"]
        if val:
            # ID number should only contain digits and possibly some letters (for passport)
            val = re.sub(r'[^a-zA-Z0-9]', '', val)
            normalized["id_number"] = val if val else None
        else:
            normalized["id_number"] = None
        
    # 4. parcel_number
    if "parcel_number" in extracted_data:
        val = extracted_data["parcel_number"]
        if val:
            # Clean up parcel number to keep alphanumeric and basic hyphens/slashes
            val = re.sub(r'^[ \.\-_:\(\)]+', '', val)
            val = re.sub(r'[ \.\-_:\(\)]+$', '', val)
            normalized["parcel_number"] = val.strip()
        else:
            normalized["parcel_number"] = None
        
    # 5. map_sheet_number
    if "map_sheet_number" in extracted_data:
        val = extracted_data["map_sheet_number"]
        if val:
            val = re.sub(r'^[ \.\-_:\(\)]+', '', val)
            val = re.sub(r'[ \.\-_:\(\)]+$', '', val)
            normalized["map_sheet_number"] = val.strip()
        else:
            normalized["map_sheet_number"] = None
        
    # 6. area_m2
    if "area_m2" in extracted_data:
        normalized["area_m2"] = normalize_area(extracted_data["area_m2"])
        
    # 7. birthday
    if "birthday" in extracted_data:
        normalized["birthday"] = normalize_birthday(extracted_data["birthday"])

    # 8. asset_name (Mục "3. Thông tin tài sản...")
    if "asset_name" in extracted_data:
        normalized["asset_name"] = clean_name_or_address(extracted_data["asset_name"]) or None

    # 9. usable_area_m2 (diện tích SỬ DỤNG của tài sản, khác area_m2 của thửa đất)
    if "usable_area_m2" in extracted_data:
        normalized["usable_area_m2"] = normalize_area(extracted_data["usable_area_m2"])

    # 10. ownership_form / ownership_term
    if "ownership_form" in extracted_data:
        normalized["ownership_form"] = clean_name_or_address(extracted_data["ownership_form"]) or None
    if "ownership_term" in extracted_data:
        normalized["ownership_term"] = clean_name_or_address(extracted_data["ownership_term"]) or None

    return normalized

def normalize_holders(holders):
    """
    Chuẩn hoá danh sách chủ sở hữu/người sử dụng đất (hỗ trợ nhiều người/document).
    """
    normalized = []
    for holder in holders or []:
        id_number = holder.get("id_number")
        if id_number:
            id_number = re.sub(r'[^a-zA-Z0-9]', '', id_number) or None
        normalized.append({
            "role": holder.get("role"),
            "name": clean_name(holder.get("name")) or None,
            "id_number": id_number,
            "birthday": normalize_birthday(holder.get("birthday")),
            "address": clean_name_or_address(holder.get("address")) or None,
        })
    return normalized

def normalize_change_history(records):
    """
    Chuẩn hoá danh sách record biến động (application_number, decision_place).
    decision_date đã được ChangeHistoryExtractor format sẵn dạng dd/mm/yyyy.
    """
    normalized = []
    for record in records or []:
        normalized.append({
            **record,
            "application_number": clean_name_or_address(record.get("application_number")) or None,
            "decision_place": clean_name_or_address(record.get("decision_place")) or None,
            "content": normalize_text(record.get("content")),
        })
    return normalized
