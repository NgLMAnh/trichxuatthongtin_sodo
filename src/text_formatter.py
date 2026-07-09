from collections import OrderedDict


PIPELINE_STAGES = ["PP-StructureV3", "PaddleOCR", "VietOCR"]
SECTION_TITLES = {
    "holder_info": "Muc I - Chu so huu / nguoi su dung dat",
    "land_info": "Muc II - Thong tin nha dat / thua dat",
    "asset_info": "Thong tin tai san gan lien voi dat",
    "land_diagram": "Muc IIc - So do",
    "owner_changes": "Muc III - Thay doi ve chu",
    "property_changes": "Muc IV - Thay doi nha dat / the chap",
    "post_issue_changes": "VI - Thay doi sau khi cap giay",
    "unknown": "Noi dung khac",
}


def group_boxes_into_lines(ocr_results, y_tolerance_ratio=0.5):
    """
    Groups OCR bounding boxes into lines based on their vertical overlap.
    ocr_results: list of dicts with 'text' and 'bbox' ([x1, y1, x2, y2]).
    """
    boxes = sorted(ocr_results, key=lambda b: b["bbox"][1])

    lines = []
    current_line = []

    for box in boxes:
        if not current_line:
            current_line.append(box)
            continue

        line_y1 = sum(b["bbox"][1] for b in current_line) / len(current_line)
        line_y2 = sum(b["bbox"][3] for b in current_line) / len(current_line)
        line_height = line_y2 - line_y1

        box_y1, box_y2 = box["bbox"][1], box["bbox"][3]
        box_height = box_y2 - box_y1

        overlap_y1 = max(line_y1, box_y1)
        overlap_y2 = min(line_y2, box_y2)
        overlap_height = max(0, overlap_y2 - overlap_y1)

        min_height = min(line_height, box_height)
        if min_height > 0 and overlap_height / min_height >= y_tolerance_ratio:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]

    if current_line:
        lines.append(current_line)

    return lines


def _escape_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("\n", " ").strip()


def _format_bbox(bbox):
    if not bbox:
        return ""
    return ",".join(str(int(round(v))) for v in bbox)


def _page_payload_to_parts(page_payload):
    if isinstance(page_payload, dict) and "blocks" in page_payload:
        return (
            page_payload.get("blocks", []),
            page_payload.get("sections", {}),
            page_payload.get("fields", {}),
            page_payload.get("extra_fields", []),
        )
    return page_payload or [], {}, {}, []


def _group_blocks_by_section(blocks, sections):
    ordered_sections = OrderedDict()
    for section_name in [
        "holder_info",
        "land_info",
        "asset_info",
        "land_diagram",
        "owner_changes",
        "property_changes",
        "post_issue_changes",
        "unknown",
    ]:
        ordered_sections[section_name] = []

    block_by_id = {block.get("block_id"): block for block in blocks}
    for section_name, block_ids in sections.items():
        ordered_sections.setdefault(section_name, [])
        for block_id in block_ids:
            block = block_by_id.get(block_id)
            if block is not None:
                ordered_sections[section_name].append(block)

    assigned_ids = {
        block.get("block_id")
        for section_blocks in ordered_sections.values()
        for block in section_blocks
    }
    for block in blocks:
        if block.get("block_id") not in assigned_ids:
            section_name = block.get("section", "unknown") or "unknown"
            ordered_sections.setdefault(section_name, []).append(block)

    return OrderedDict(
        (name, sorted(section_blocks, key=lambda b: b.get("reading_order", 0)))
        for name, section_blocks in ordered_sections.items()
        if section_blocks
    )


def _format_field_summary(fields):
    if not fields:
        return []

    lines = ["### extracted_fields", ""]
    for key in sorted(fields):
        value = _escape_value(fields.get(key))
        if value:
            lines.append(f"- {key}: {value}")
    if len(lines) == 2:
        return []
    lines.append("")
    return lines


def _format_change_history(change_history):
    if not change_history:
        return []

    lines = []
    for idx, record in enumerate(change_history, start=1):
        prefix = f"bien_dong_{idx}"
        date_val = _escape_value(record.get("decision_date"))
        app_val = _escape_value(record.get("application_number"))
        place_val = _escape_value(record.get("decision_place"))
        content_val = _escape_value(record.get("content"))

        if date_val:
            lines.append(f"- {prefix}_ngay: {date_val}")
        if app_val:
            lines.append(f"- {prefix}_so_ho_so: {app_val}")
        if place_val:
            lines.append(f"- {prefix}_noi_ky: {place_val}")
        if content_val:
            lines.append(f"- {prefix}_noi_dung: {content_val}")

    return lines


def _format_holders(holders):
    """Chỉ xuất thêm khi document có >1 chủ sở hữu (mẫu GCN hợp nhất vợ/chồng);
    trường hợp 1 chủ đã có sẵn chu_so_huu/cmnd_cccd trong document_summary."""
    if not holders or len(holders) < 2:
        return []

    lines = []
    for idx, holder in enumerate(holders, start=1):
        role = _escape_value(holder.get("role"))
        name = _escape_value(holder.get("name"))
        id_number = _escape_value(holder.get("id_number"))
        if name:
            lines.append(f"- chu_so_huu_{idx}: {name}")
        if id_number:
            lines.append(f"- cmnd_cccd_{idx}: {id_number}")
        if role:
            lines.append(f"- vai_tro_{idx}: {role}")

    return lines


def _format_extra_fields(extra_fields):
    """Lớp bổ sung: mọi cặp '<nhãn>: <giá trị>' chưa có field khai báo tay nào bắt được."""
    if not extra_fields:
        return []

    lines = ["### extra_fields", ""]
    for item in extra_fields:
        key = _escape_value(item.get("key"))
        value = _escape_value(item.get("value"))
        label = _escape_value(item.get("label"))
        if key and value:
            lines.append(f"- {key}: {value}  # {label}")

    if len(lines) == 2:
        return []
    lines.append("")
    return lines


def _format_block(block, page_name, section_name):
    text = _escape_value(block.get("text", ""))
    if not text:
        return []

    block_id = _escape_value(block.get("block_id", ""))
    label = _escape_value(block.get("label", "text"))
    confidence = _escape_value(block.get("confidence", ""))
    bbox = _format_bbox(block.get("bbox"))

    lines = [
        f"#### block {block_id}",
        f"- document_page: {page_name}",
        f"- section: {section_name}",
        f"- layout_label: {label}",
        f"- bbox_xyxy: {bbox}",
        f"- confidence: {confidence}",
        f"- text: {text}",
    ]

    text_lines = block.get("text_lines") or []
    if text_lines:
        lines.append("- lines:")
        for idx, line in enumerate(text_lines, start=1):
            line_text = _escape_value(line.get("text", ""))
            if not line_text:
                continue
            line_bbox = _format_bbox(line.get("bbox"))
            line_conf = _escape_value(line.get("confidence", ""))
            lines.append(
                f"  - line_{idx}: text={line_text} | bbox={line_bbox} | confidence={line_conf}"
            )
    lines.append("")
    return lines


def format_as_markdown(page_blocks_dict, document_id=None, doc_json=None):
    """
    Formats structured OCR output as RAG-friendly Markdown.

    The output intentionally keeps Markdown headings for the existing chunker,
    while adding stable key-value metadata for retrieval, filtering, and audits.
    """
    md_lines = [
        "---",
        f"document_id: {document_id or ''}",
        "source_pipeline: PP-StructureV3 -> PaddleOCR -> VietOCR",
        "runtime: cpu",
        "format_version: rag_markdown_v1",
        "---",
        "",
    ]

    if doc_json:
        md_lines.extend(["# document_summary", ""])
        holder = doc_json.get("holder", {})
        land = doc_json.get("land_parcel", {})
        asset = doc_json.get("asset", {})
        summary_fields = {
            "chu_so_huu": holder.get("name"),
            "cmnd_cccd": holder.get("id_number"),
            "dia_chi": holder.get("address"),
            "nam_sinh": holder.get("birthday"),
            "thua_dat_so": land.get("parcel_number"),
            "to_ban_do_so": land.get("map_sheet_number"),
            "dien_tich_m2": land.get("area_m2"),
            "ten_tai_san": asset.get("asset_name"),
            "dien_tich_su_dung_m2": asset.get("usable_area_m2"),
            "hinh_thuc_so_huu": asset.get("ownership_form"),
            "thoi_han_so_huu": asset.get("ownership_term"),
        }
        for key, value in summary_fields.items():
            value = _escape_value(value)
            if value:
                md_lines.append(f"- {key}: {value}")

        md_lines.extend(_format_holders(doc_json.get("holders")))
        md_lines.extend(_format_change_history(doc_json.get("change_history")))
        md_lines.append("")

    for page_index, (page_name, page_payload) in enumerate(page_blocks_dict.items(), start=1):
        blocks, sections, fields, extra_fields = _page_payload_to_parts(page_payload)
        sorted_blocks = sorted(blocks, key=lambda b: b.get("reading_order", 0))
        section_groups = _group_blocks_by_section(sorted_blocks, sections)

        md_lines.extend(
            [
                f"## {page_name}",
                "",
                "### page_metadata",
                "",
                f"- page_index: {page_index}",
                f"- page_name: {page_name}",
                f"- source_pipeline: {' -> '.join(PIPELINE_STAGES)}",
                "- runtime: cpu",
                f"- block_count: {len(sorted_blocks)}",
                "",
            ]
        )

        md_lines.extend(_format_field_summary(fields))
        md_lines.extend(_format_extra_fields(extra_fields))

        for section_name, section_blocks in section_groups.items():
            section_title = SECTION_TITLES.get(section_name, section_name)
            md_lines.extend(
                [
                    f"### {section_name}: {section_title}",
                    "",
                    f"- section_id: {section_name}",
                    f"- block_count: {len(section_blocks)}",
                    "",
                ]
            )
            for block in section_blocks:
                md_lines.extend(_format_block(block, page_name, section_name))

        md_lines.extend(["---", ""])

    return "\n".join(md_lines).rstrip() + "\n"
