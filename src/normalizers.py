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
    
    # VietOCR is highly accurate for Vietnamese diacritics.
    # We only need to clean up minor artifacts like single dots between letters
    # e.g., "Cu.Xa" -> "Cu Xa", but NOT "18/2/3.5" (number.number)
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

def clean_name(val):
    if not val:
        return ""
    # Strip "sinh năm" or "sn" and anything after it from names
    val = re.sub(r'(?i)\b(?:sinh năm|năm sinh|sn)\b.*$', '', val)
    # Use generic clean up
    val = clean_name_or_address(val)
    return val

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
        val = extracted_data["birthday"]
        normalized["birthday"] = normalize_text(val) if val else None
        
    return normalized
