from src.spatial_rules import normalize_text

def detect_page_type(ocr_results, template_config):
    """
    Detects the page type of a page based on OCR text keywords.
    ocr_results: list[dict]
    template_config: dict
    """
    # Join all OCR text and normalize (lowercase, de-accented, clean)
    full_text = " ".join([r["text"] for r in ocr_results])
    norm_full_text = normalize_text(full_text)
    
    best_page_type = "unknown"
    best_score = 0
    
    page_detection = template_config.get("page_detection", {})
    
    for page_type, config in page_detection.items():
        keywords = config.get("keywords", [])
        min_matches = config.get("min_keyword_matches", 1)
        
        matches = sum(1 for kw in keywords if normalize_text(kw) in norm_full_text)
        
        if matches >= min_matches and matches > best_score:
            best_page_type = page_type
            best_score = matches
            
    return best_page_type
