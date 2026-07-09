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
        },
        "change_history": [],
        "holders": [],
        "asset": {
            "asset_name": None,
            "usable_area_m2": None,
            "ownership_form": None,
            "ownership_term": None
        },
        "extra_fields": []
    }
    seen_extra = set()
    
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

        for asset_field in ("asset_name", "usable_area_m2", "ownership_form", "ownership_term"):
            if asset_field in fields:
                new_val = fields[asset_field]
                old_val = doc_json["asset"][asset_field]
                if is_better_value(new_val, old_val, asset_field):
                    doc_json["asset"][asset_field] = new_val

        # change_history: nối theo thứ tự trang (nhiều record/document, không overwrite)
        doc_json["change_history"].extend(page.get("change_history", []))

        # holders: lấy trang có nhiều chủ sở hữu nhất (mẫu GCN hợp nhất gộp tên+CCCD
        # thường nằm trọn trong 1 trang; mẫu cũ trả về [] và sẽ fallback dưới đây).
        page_holders = page.get("holders", [])
        if len(page_holders) > len(doc_json["holders"]):
            doc_json["holders"] = page_holders

        # extra_fields: nối qua các trang, dedupe theo (key, value) toàn document
        # (mỗi trang đã tự dedupe nội bộ, nhưng label giống nhau có thể lặp giữa trang).
        for item in page.get("extra_fields", []):
            dedupe_key = (item.get("key"), item.get("value"))
            if dedupe_key in seen_extra:
                continue
            seen_extra.add(dedupe_key)
            doc_json["extra_fields"].append(item)

    if doc_json["holders"]:
        # HolderExtractor khớp được (mẫu GCN hợp nhất) -> đồng bộ lại holder scalar
        # (tương thích ngược cho các nơi vẫn chỉ đọc doc_json["holder"]) từ người đầu tiên.
        primary = doc_json["holders"][0]
        doc_json["holder"]["name"] = primary.get("name") or doc_json["holder"]["name"]
        doc_json["holder"]["id_number"] = primary.get("id_number") or doc_json["holder"]["id_number"]
    elif doc_json["holder"].get("name"):
        # Mẫu cũ: tên/CMND ở 2 block riêng, HolderExtractor không khớp được gì.
        # Dùng lại holder scalar đã merge làm 1 chủ sở hữu duy nhất, để "holders"
        # luôn có sẵn cho mọi mẫu (không chỉ mẫu GCN hợp nhất).
        doc_json["holders"] = [dict(doc_json["holder"], role=None)]

    return doc_json
