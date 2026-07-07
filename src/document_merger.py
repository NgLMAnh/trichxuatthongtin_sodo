def get_value_score(val, field_name):
    if val is None or val == "":
        return -100
        
    val_str = str(val).lower()
    score = len(val_str)
    
    # Noise indicators
    noise_keywords = ["sinh nam", "cmnd", "ngay", "qd", "quyet dinh", "so do nen", "kich thuoc", "quy hoach", "chuyen nhugng"]
    for kw in noise_keywords:
        if kw in val_str:
            score -= 50
            
    # For names, penalize digits
    if field_name == "name":
        if any(c.isdigit() for c in val_str):
            score -= 40
            
    return score

def is_better_value(new_val, old_val, field_name):
    old_score = get_value_score(old_val, field_name)
    new_score = get_value_score(new_val, field_name)
    
    # In land certificates, the later pages contain the most recent updates
    # (Thay đổi về chủ). We want the FINAL owner.
    # So if the new value is reasonably valid (score >= 5), we ALWAYS prefer it
    # over the old value from an earlier page.
    if new_score >= 5:
        return True
        
    return new_score > old_score

def merge_pages(document_id, page_results):
    """
    Merges page-level extracted data into a unified document-level dict.
    page_results: list[dict] where each dict contains:
                  - 'page_type': str
                  - 'fields': dict (normalized fields)
    """
    doc_json = {
        "document_id": document_id,
        "holder": {
            "name": None,
            "id_number": None,
            "address": None,
            "birthday": None
        },
        "land_parcel": {
            "parcel_number": None,
            "map_sheet_number": None,
            "area_m2": None
        }
    }
    
    for page in page_results:
        ptype = page.get("page_type")
        fields = page.get("fields", {})
        
        if "holder_name" in fields:
            new_val = fields["holder_name"]
            old_val = doc_json["holder"]["name"]
            if is_better_value(new_val, old_val, "name"):
                doc_json["holder"]["name"] = new_val
        if "id_number" in fields:
            new_val = fields["id_number"]
            old_val = doc_json["holder"]["id_number"]
            if is_better_value(new_val, old_val, "id_number"):
                doc_json["holder"]["id_number"] = new_val
        if "address" in fields:
            new_val = fields["address"]
            old_val = doc_json["holder"]["address"]
            if is_better_value(new_val, old_val, "address"):
                doc_json["holder"]["address"] = new_val
        if "birthday" in fields:
            new_val = fields["birthday"]
            old_val = doc_json["holder"]["birthday"]
            if is_better_value(new_val, old_val, "birthday"):
                doc_json["holder"]["birthday"] = new_val

        if "parcel_number" in fields:
            new_val = fields["parcel_number"]
            old_val = doc_json["land_parcel"]["parcel_number"]
            if is_better_value(new_val, old_val, "parcel_number"):
                doc_json["land_parcel"]["parcel_number"] = new_val
        if "map_sheet_number" in fields:
            new_val = fields["map_sheet_number"]
            old_val = doc_json["land_parcel"]["map_sheet_number"]
            if is_better_value(new_val, old_val, "map_sheet_number"):
                doc_json["land_parcel"]["map_sheet_number"] = new_val
        if "area_m2" in fields:
            new_val = fields["area_m2"]
            old_val = doc_json["land_parcel"]["area_m2"]
            if is_better_value(new_val, old_val, "area_m2"):
                doc_json["land_parcel"]["area_m2"] = new_val
                    
    return doc_json
