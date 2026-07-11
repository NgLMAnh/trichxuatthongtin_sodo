"""
Sinh báo cáo Markdown DỄ ĐỌC (cho người, không phải cho RAG/LLM) từ JSON trích
xuất đã có trong outputs/predictions/*.json. Khác với text_formatter.py (sinh
Markdown phục vụ chunking/RAG, chứa block OCR thô để LLM tra cứu), file này chỉ
trình bày lại toàn bộ thông tin đã trích xuất theo cấu trúc rõ ràng, gọn, không
lẫn dữ liệu thô của pipeline OCR.
"""
import json
import os


def _fmt(value):
    if value is None or value == "":
        return "_(không có dữ liệu)_"
    return str(value)


def _format_holders(doc):
    holders = doc.get("holders") or []
    if not holders:
        holder = doc.get("holder", {})
        holders = [dict(holder, role=None)]

    lines = []
    for idx, h in enumerate(holders, start=1):
        prefix = f"{idx}. " if len(holders) > 1 else ""
        role = f"{h.get('role')} " if h.get("role") else ""
        lines.append(f"- {prefix}**{role}{_fmt(h.get('name'))}**")
        lines.append(f"  - CMND/CCCD: {_fmt(h.get('id_number'))}")
        lines.append(f"  - Năm sinh: {_fmt(h.get('birthday'))}")
        address = h.get("address") or doc.get("holder", {}).get("address")
        lines.append(f"  - Địa chỉ: {_fmt(address)}")
    return "\n".join(lines)


def _format_change_history(doc):
    records = doc.get("change_history") or []
    if not records:
        return "_(không có lịch sử biến động được ghi nhận)_"

    lines = []
    for idx, r in enumerate(records, start=1):
        lines.append(f"**Lần {idx}:**")
        lines.append(f"- Ngày ký/quyết định: {_fmt(r.get('decision_date'))}")
        lines.append(f"- Số hồ sơ: {_fmt(r.get('application_number'))}")
        lines.append(f"- Nơi ký/ra quyết định: {_fmt(r.get('decision_place'))}")
        lines.append(f"- Nội dung: {_fmt(r.get('content'))}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_extra_fields(doc):
    items = doc.get("extra_fields") or []
    if not items:
        return "_(không có thông tin bổ sung)_"
    lines = []
    for item in items:
        lines.append(f"- **{item.get('label')}**: {_fmt(item.get('value'))}")
    return "\n".join(lines)


# Nhãn tiếng Việt cho tên section (đồng bộ với section_detector).
_SECTION_TITLES = {
    "holder_info": "Chủ sở hữu / Người sử dụng đất",
    "land_info": "Thông tin thửa đất",
    "asset_info": "Thông tin tài sản",
    "land_diagram": "Sơ đồ thửa đất",
    "owner_changes": "Thay đổi chủ sở hữu",
    "property_changes": "Thay đổi tài sản",
    "post_issue_changes": "Thay đổi sau khi cấp GCN",
    "unknown": "Khác",
}


def _format_full_text(doc):
    """Liệt kê MỌI dòng chữ OCR đọc được, gom theo section - bằng chứng 'đã lưu
    tất cả thông tin trên giấy', kể cả dòng chưa map vào field cấu trúc nào."""
    lines = doc.get("full_text") or []
    if not lines:
        return "_(không có dữ liệu)_"

    # Gom theo thứ tự section xuất hiện, giữ nguyên thứ tự đọc trong mỗi section
    order = []
    grouped = {}
    for ln in lines:
        sec = ln.get("section", "unknown")
        if sec not in grouped:
            grouped[sec] = []
            order.append(sec)
        grouped[sec].append(ln.get("text", ""))

    out = []
    for sec in order:
        out.append(f"**{_SECTION_TITLES.get(sec, sec)}** ({len(grouped[sec])} dòng):")
        for text in grouped[sec]:
            out.append(f"- {text}")
        out.append("")
    return "\n".join(out).strip()


def generate_readable_report(doc):
    """Sinh nội dung Markdown dễ đọc cho 1 document (dict JSON đã load)."""
    doc_id = doc.get("document_id", "UNKNOWN")
    land = doc.get("land_parcel", {})
    asset = doc.get("asset", {})

    parts = [
        f"# Sổ đỏ {doc_id}",
        "",
        "## 1. Chủ sở hữu / Người sử dụng đất",
        "",
        _format_holders(doc),
        "",
        "## 2. Thông tin thửa đất",
        "",
        f"- Số thửa đất: {_fmt(land.get('parcel_number'))}",
        f"- Tờ bản đồ số: {_fmt(land.get('map_sheet_number'))}",
        f"- Diện tích: {_fmt(land.get('area_m2'))} m²",
        "",
        "## 3. Thông tin tài sản gắn liền với đất",
        "",
        f"- Tên / mô tả tài sản: {_fmt(asset.get('asset_name'))}",
        f"- Diện tích sử dụng: {_fmt(asset.get('usable_area_m2'))} m²",
        f"- Hình thức sở hữu: {_fmt(asset.get('ownership_form'))}",
        f"- Thời hạn sở hữu: {_fmt(asset.get('ownership_term'))}",
        "",
        "## 4. Lịch sử biến động",
        "",
        _format_change_history(doc),
        "",
        "## 5. Thông tin bổ sung (trích xuất tự động từ mọi dòng trên giấy)",
        "",
        _format_extra_fields(doc),
        "",
        "## 6. Toàn văn OCR (tất cả chữ đọc được, theo mục)",
        "",
        _format_full_text(doc),
        "",
    ]
    if doc.get("failed_pages"):
        parts.append("## ⚠️ Trang xử lý lỗi (thông tin có thể bị thiếu)")
        parts.append("")
        for fp in doc["failed_pages"]:
            parts.append(f"- {fp.get('page_name')}: {fp.get('error')}")
        parts.append("")
    return "\n".join(parts)


def generate_all_reports(predictions_dir="outputs/predictions", reports_dir="outputs/reports"):
    """Đọc mọi JSON trong predictions_dir, sinh file .md dễ đọc trong reports_dir."""
    os.makedirs(reports_dir, exist_ok=True)
    written = []
    for fname in sorted(os.listdir(predictions_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(predictions_dir, fname), "r", encoding="utf-8") as f:
            doc = json.load(f)
        doc_id = doc.get("document_id", fname.replace(".json", ""))
        content = generate_readable_report(doc)
        out_path = os.path.join(reports_dir, f"{doc_id}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(out_path)
    return written


if __name__ == "__main__":
    for path in generate_all_reports():
        print("Wrote", path)
