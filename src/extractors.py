import re
from src.spatial_rules import find_best_anchor, apply_spatial_rule

def check_multiline_extension(first_box, ocr_results, max_vertical_gap=45, max_lines=1):
    """
    Checks if there are subsequent lines directly below first_box that should be merged
    (e.g., multi-line address). Limited to max_lines continuation lines to prevent
    swallowing entire page content.
    """
    # Stop words: both Vietnamese (accented) AND raw OCR (non-accented) forms
    stop_keywords = [
        # Vietnamese accented
        "thửa đất", "tờ bản đồ", "diện tích", "mục ", "ngày", "năm",
        "tổng", "kết cấu", "số tầng", "hình thức", "chứng nhận",
        "ủy ban", "chủ tịch", "ghi chú", "hồ sơ", "sinh năm",
        "kích thước", "kèm như sau", "như sau", "sơ đồ", "bản đồ",
        # Raw OCR (non-accented) equivalents
        "thua dat", "to ban do", "dien tich", "muc ", "ngay", "nam",
        "tong", "ket cau", "so tang", "hinh thuc", "chung nhan",
        "uy ban", "chu tich", "ghi chu", "ho so", "sinh nam",
        "kich thuoc", "kem nhu sau", "nhu sau", "so do", "ban do",
        # Common section headers
        "muc ii", "muc iii", "muc iv", "kt.", "t.m ",
        "cmnd", "cccd", "giay ch",
    ]
    
    extended_text = first_box["text"]
    current_box = first_box
    lines_added = 0
    visited = {id(first_box)}
    
    while lines_added < max_lines:
        next_line_box = None
        min_dist = float('inf')
        
        a_x1, a_y1, a_x2, a_y2 = current_box["bbox"]
        a_cx = (a_x1 + a_x2) / 2.0
        
        for item in ocr_results:
            if id(item) in visited:
                continue
                
            c_x1, c_y1, c_x2, c_y2 = item["bbox"]
            c_cx = (c_x1 + c_x2) / 2.0
            
            # Check if it is below current_box
            vertical_gap = c_y1 - a_y2
            if -15 <= vertical_gap <= max_vertical_gap:
                # Check horizontal alignment (left edge within 150px)
                if abs(c_x1 - a_x1) < 150:
                    # Check stop words
                    text_lower = item["text"].lower()
                    if any(k in text_lower for k in stop_keywords):
                        continue
                        
                    if vertical_gap < min_dist:
                        min_dist = vertical_gap
                        next_line_box = item
                        
        if next_line_box:
            extended_text += " " + next_line_box["text"]
            visited.add(id(next_line_box))
            current_box = next_line_box
            lines_added += 1
        else:
            break
            
    return extended_text

def extract_value_from_same_box(anchor_box, anchor_keywords, field_config, ocr_results):
    text = anchor_box["text"]
    from src.spatial_rules import normalize_text, remove_accents
    
    # 1. Try splitting by colon
    if ":" in text:
        parts = text.split(":", 1)
        potential_val = parts[1].strip()
        # Clean prefix punctuation
        potential_val = re.sub(r'^[ \.\-_:\(\)\/]+', '', potential_val).strip()
        if potential_val:
            regex_pattern = field_config.get("regex")
            if regex_pattern:
                match = re.search(regex_pattern, potential_val)
                if match:
                    return match.group(0)
            else:
                if len(potential_val) >= 2:
                    # If multiline is enabled, we need a pseudo-box for extension
                    if field_config.get("multiline"):
                        # Create a mock box corresponding to the value part
                        # Coordinates can just be anchor box coordinates for vertical extension
                        mock_box = {"text": potential_val, "bbox": anchor_box["bbox"]}
                        return check_multiline_extension(mock_box, ocr_results)
                    return potential_val
                    
    # 2. Try matching after the matched anchor keyword
    text_deaccent = remove_accents(text.lower())
    for kw in anchor_keywords:
        norm_kw = normalize_text(kw)
        words = norm_kw.split()
        if not words:
            continue
        last_word = words[-1]
        idx = text_deaccent.rfind(last_word)
        if idx != -1:
            suffix = text[idx + len(last_word):].strip()
            suffix = re.sub(r'^[ \.\-_:\(\)\/]+', '', suffix).strip()
            if suffix:
                regex_pattern = field_config.get("regex")
                if regex_pattern:
                    match = re.search(regex_pattern, suffix)
                    if match:
                        return match.group(0)
                else:
                    if len(suffix) >= 2:
                        if field_config.get("multiline"):
                            mock_box = {"text": suffix, "bbox": anchor_box["bbox"]}
                            return check_multiline_extension(mock_box, ocr_results)
                        return suffix
    return None

def extract_fields(ocr_results, page_config):
    """
    Extracts fields from OCR results based on the page YAML config.
    """
    results = {}
    fields = page_config.get("fields", {})
    
    for field_name, field_config in fields.items():
        anchors = field_config.get("anchors", [])
        
        # Filter OCR results by y_max or y_min coordinates if specified
        filtered_results = ocr_results
        y_max = field_config.get("y_max")
        y_min = field_config.get("y_min")
        if y_max is not None or y_min is not None:
            filtered_results = []
            for item in ocr_results:
                y1 = item["bbox"][1]
                if y_max is not None and y1 > y_max:
                    continue
                if y_min is not None and y1 < y_min:
                    continue
                filtered_results.append(item)
        
        # Find anchor box
        anchor_box = find_best_anchor(filtered_results, anchors)
        if anchor_box is None:
            results[field_name] = None
            continue
            
        # Try to extract value from the same box first
        same_box_val = extract_value_from_same_box(anchor_box, anchors, field_config, ocr_results)
        if same_box_val is not None:
            results[field_name] = same_box_val
            continue
            
        # Fallback to spatial rule (find candidate value box)
        candidate_box = apply_spatial_rule(ocr_box_result_format(ocr_results, anchor_box), ocr_results, field_config)
        if candidate_box is None:
            results[field_name] = None
            continue
            
        raw_val = candidate_box["text"]
        if field_config.get("multiline"):
            raw_val = check_multiline_extension(candidate_box, ocr_results)
        
        # Apply regex if specified
        regex_pattern = field_config.get("regex")
        if regex_pattern:
            match = re.search(regex_pattern, raw_val)
            if match:
                results[field_name] = match.group(0)
            else:
                results[field_name] = None
        else:
            results[field_name] = raw_val
            
    return results

def ocr_box_result_format(ocr_results, box):
    # Simply helper to ensure anchor_box is in correct format
    return box
