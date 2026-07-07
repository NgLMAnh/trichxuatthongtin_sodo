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
            
        text_lower = text.lower().strip()
        text_clean = re.sub(r'^[^\w\s]+', '', text_lower).strip()
        norm_text = normalize_text(text)
        
        for kw_idx, kw in enumerate(anchor_keywords):
            norm_kw = normalize_text(kw)
            if not norm_kw: continue
            
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
