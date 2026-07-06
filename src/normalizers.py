import re

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
    
    # 1. Character level corrections FIRST (European diacritics -> Vietnamese)
    #    These are always safe because ö, ä, ü, ë, ï are not valid Vietnamese
    char_replacements = {
        'ö': 'ô',
        'ä': 'á',
        'ü': 'ủ',
        'ë': 'ê',
        'ï': 'í',
        'Ö': 'Ô',
        'Ä': 'Á',
        'Ü': 'Ủ',
        'Ë': 'Ê',
        'Ï': 'Í',
    }
    for char, replacement in char_replacements.items():
        val = val.replace(char, replacement)
    
    # 2. Safe multi-word corrections only (low risk of false positives)
    safe_replacements = {
        r'\bthanh pho\b': 'Thành phố',
        r'\bthi tran\b': 'Thị trấn',
        r'\bkhu pho\b': 'Khu phố',
        r'\bkhu dat\b': 'Khu đất',
        r'\bdien tich\b': 'Diện tích',
        r'\bban do\b': 'Bản đồ',
        r'\bvi tri\b': 'Vị trí',
        r'\bchung nhan\b': 'Chứng nhận',
    }
    for pattern, replacement in safe_replacements.items():
        val = re.sub(pattern, replacement, val, flags=re.IGNORECASE)
    
    # 3. Clean up single dots between letters (OCR artifact from spaces)
    #    e.g., "Cu.Xa" -> "Cu Xa", but NOT "18/2/3.5" (number.number)
    val = re.sub(r'(?<=[A-Za-zÀ-ỹ])\.(?=[A-Za-zÀ-ỹ])', ' ', val)
    
    return val

def clean_name_or_address(val):
    if not val:
        return ""
    # Remove leading/trailing dots, spaces, underscores, colons, hyphens, slashes
    val = re.sub(r'^[ \.\-_:\(\)\/]+', '', val)
    val = re.sub(r'[ \.\-_:\(\)\/]+$', '', val)
    # Remove large clusters of dots in between
    val = re.sub(r'\.{2,}', ' ', val)
    val = re.sub(r'_{2,}', ' ', val)
    # Collapse multiple spaces
    val = re.sub(r'\s+', ' ', val).strip()
    
    # Correct OCR spelling mistakes
    val = correct_ocr_typos(val)
    return val

def normalize_area(area_str):
    """
    Normalizes area value (e.g., '120,5 m2' -> 120.5).
    """
    if not area_str:
        return None
    # Replace comma with dot
    val = area_str.replace(",", ".")
    # Find all float-like numbers
    match = re.search(r'\d+(?:\.\d+)?', val)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None

def normalize_fields(extracted_data):
    """
    Applies appropriate normalization functions to extracted fields.
    """
    normalized = {}
    
    # 1. holder_name
    if "holder_name" in extracted_data:
        normalized["holder_name"] = clean_name_or_address(extracted_data["holder_name"])
        
    # 2. address
    if "address" in extracted_data:
        normalized["address"] = clean_name_or_address(extracted_data["address"])
        
    # 3. parcel_number
    if "parcel_number" in extracted_data:
        val = extracted_data["parcel_number"]
        if val:
            # Clean up parcel number to keep alphanumeric and basic hyphens/slashes
            val = re.sub(r'^[ \.\-_:\(\)]+', '', val)
            val = re.sub(r'[ \.\-_:\(\)]+$', '', val)
            normalized["parcel_number"] = val.strip()
        else:
            normalized["parcel_number"] = None
        
    # 4. map_sheet_number
    if "map_sheet_number" in extracted_data:
        val = extracted_data["map_sheet_number"]
        if val:
            val = re.sub(r'^[ \.\-_:\(\)]+', '', val)
            val = re.sub(r'[ \.\-_:\(\)]+$', '', val)
            normalized["map_sheet_number"] = val.strip()
        else:
            normalized["map_sheet_number"] = None
        
    # 5. area_m2
    if "area_m2" in extracted_data:
        normalized["area_m2"] = normalize_area(extracted_data["area_m2"])
        
    # 6. birthday
    if "birthday" in extracted_data:
        val = extracted_data["birthday"]
        normalized["birthday"] = normalize_text(val) if val else None
        
    return normalized
