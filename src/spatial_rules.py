import math
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

def find_best_anchor(ocr_results, anchor_keywords):
    """
    Finds the best OCR box matching one of the anchor keywords.
    """
    # Define custom flexible regexes for known anchors
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
    
    for item in ocr_results:
        text_lower = item["text"].lower().strip()
        # Clean leading punctuation/symbols to find clean starts
        text_clean = re.sub(r'^[^\w\s]+', '', text_lower).strip()
        norm_text = normalize_text(item["text"])
        
        for kw in anchor_keywords:
            norm_kw = normalize_text(kw)
            
            # Check exact match first
            if norm_text == norm_kw:
                matches.append((item, 100.0, len(item["text"])))
                continue
                
            # Check flexible regex
            pattern = flexible_regexes.get(norm_kw)
            if pattern:
                deaccented_pattern = remove_accents(pattern)
                # For ba and ong, require match at the beginning of clean text
                if norm_kw in ["ba", "ong"]:
                    if re.search(pattern, text_clean) or re.search(deaccented_pattern, remove_accents(text_clean)):
                        matches.append((item, 90.0, len(item["text"])))
                        continue
                else:
                    if re.search(pattern, text_lower) or re.search(deaccented_pattern, norm_text):
                        similarity = len(norm_kw) / max(1, len(norm_text))
                        matches.append((item, similarity, len(item["text"])))
                        continue
            
            # For short keywords like ba and ong, do not use standard substring search
            if norm_kw in ["ba", "ong"]:
                continue
                
            # Standard word boundary match
            pattern_wb = r'\b' + re.escape(norm_kw) + r'\b'
            if re.search(pattern_wb, norm_text):
                similarity = len(norm_kw) / max(1, len(norm_text))
                matches.append((item, similarity, len(item["text"])))
                
    if not matches:
        return None
        
    # Sort matches by similarity score descending, and then by text length ascending
    matches.sort(key=lambda x: (-x[1], x[2]))
    return matches[0][0]

def same_row(box_a, box_b, min_overlap=0.4):
    """
    Checks if two boxes are on the same row based on vertical overlap.
    """
    a_y1, a_y2 = box_a[1], box_a[3]
    b_y1, b_y2 = box_b[1], box_b[3]
    
    overlap = max(0, min(a_y2, b_y2) - max(a_y1, b_y1))
    height_a = a_y2 - a_y1
    height_b = b_y2 - b_y1
    
    if height_a == 0 or height_b == 0:
        return False
        
    overlap_ratio = overlap / min(height_a, height_b)
    return overlap_ratio >= min_overlap

def get_distance(box_a, box_b):
    """
    Calculates Euclidean distance between centers of two boxes.
    """
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    
    return math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

def apply_spatial_rule(anchor_box, ocr_results, spatial_config):
    """
    Finds candidate values near the anchor box using spatial rules.
    """
    directions = spatial_config.get("directions", ["right", "below"])
    max_distance = spatial_config.get("max_distance", 500)
    same_line_pref = spatial_config.get("same_line_preferred", True)
    
    a_x1, a_y1, a_x2, a_y2 = anchor_box["bbox"]
    
    candidates = []
    
    for item in ocr_results:
        # Skip the anchor box itself
        if item == anchor_box:
            continue
            
        c_x1, c_y1, c_x2, c_y2 = item["bbox"]
        c_cx = (c_x1 + c_x2) / 2.0
        c_cy = (c_y1 + c_y2) / 2.0
        
        # Calculate distance
        dist = get_distance(anchor_box["bbox"], item["bbox"])
        if dist > max_distance:
            continue
            
        is_candidate = False
        
        for direction in directions:
            if direction == "right":
                if c_cx > a_x2 - 10:  # Allow slight tolerance
                    if same_row(anchor_box["bbox"], item["bbox"]):
                        is_candidate = True
            elif direction == "below":
                if c_cy > a_y2 - 10:
                    is_candidate = True
                    
        if is_candidate:
            candidates.append((item, dist))
            
    if not candidates:
        return None
        
    if same_line_pref:
        line_candidates = [c for c in candidates if same_row(anchor_box["bbox"], c[0]["bbox"])]
        if line_candidates:
            line_candidates.sort(key=lambda x: x[1])
            return line_candidates[0][0]
            
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]
