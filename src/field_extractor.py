import re

from src.text_utils import find_best_anchor, remove_accents
from src.normalizers import fix_numeric_ocr


class FieldExtractor:
    def __init__(self, config):
        self.config = config.get("field_extraction", {})

    def extract(self, blocks, sections, graph):
        results = {}

        for field_name, field_cfg in self.config.items():
            anchors = field_cfg.get("anchors", [])
            target_sections = field_cfg.get("target_sections", [])

            candidate_blocks = []
            if target_sections:
                for section_name in target_sections:
                    block_ids = sections.get(section_name, [])
                    candidate_blocks.extend(
                        [block for block in blocks if block["block_id"] in block_ids]
                    )
            else:
                candidate_blocks = blocks

            if not candidate_blocks:
                # strict_section: field này CHỈ có ý nghĩa trong 1 mẫu cụ thể (VD: các
                # field asset_info chỉ tồn tại ở mẫu GCN hợp nhất). Nếu section đó không
                # tồn tại trong document (mẫu khác), coi như field không áp dụng - KHÔNG
                # tìm kiếm toàn document (tránh khớp nhầm sang nội dung mục khác).
                if field_cfg.get("strict_section"):
                    results[field_name] = None
                    continue
                candidate_blocks = blocks

            anchor_block = find_best_anchor(candidate_blocks, anchors)
            if not anchor_block and candidate_blocks is not blocks:
                anchor_block = find_best_anchor(blocks, anchors)
            if not anchor_block:
                results[field_name] = None
                continue

            results[field_name] = self._extract_from_anchor(anchor_block, {**field_cfg, "_field_name": field_name}, graph)

        return results

    @staticmethod
    def _validate_id_number(value):
        """Validate CMND (9 số) hoặc CCCD (12 số). Loại bỏ false positive.
        Thử sửa lỗi OCR chữ->số (VD 'S'->'5', 'O'->'0') trước khi đếm, để không
        mất số CMND/CCCD chỉ vì 1 ký tự bị nhận nhầm thành chữ."""
        if not value:
            return None
        # Sửa lỗi OCR chữ<->số TRƯỚC khi lọc (VD "O66..." phải thành "066...",
        # không được xoá 'O' như ký tự lạ rồi ra số sai độ dài).
        digits = re.sub(r"\D", "", fix_numeric_ocr(str(value)))
        if len(digits) in (9, 12):
            return digits
        # Fallback: thử lọc trực tiếp (phòng khi fix_numeric_ocr không kích hoạt)
        raw = re.sub(r"\D", "", str(value))
        if len(raw) in (9, 12):
            return raw
        return None

    @staticmethod
    def _extract_after_anchor_keyword(text, anchors, regex_pattern, require_mid_line=False):
        """
        Trích giá trị NGAY SAU đúng anchor keyword trong block, thay vì sau dấu
        ':' đầu tiên. Cần cho mẫu hợp nhất gộp NHIỀU cặp nhãn:giá trị trong 1
        dòng, VD "a. Thửa đất số: 251; tờ bản đồ số: 74," - map_sheet phải lấy
        74 (sau "tờ bản đồ số"), không phải 251 (sau dấu ':' đầu).

        remove_accents dịch 1:1 từng ký tự nên vị trí trong chuỗi bỏ dấu khớp
        đúng vị trí trong chuỗi gốc - dùng để định vị anchor rồi cắt trên text gốc.

        require_mid_line=True: CHỈ trích khi anchor keyword nằm SAU đầu dòng
        (có nội dung/cặp khác đứng trước, pos>0 sau khi bỏ tiền tố "a."/"1.") -
        đảm bảo mẫu 1-cặp (anchor ở đầu dòng) giữ nguyên hành vi Bước 1 cũ.
        Trả về None nếu không khớp (caller dùng lại logic cũ).
        """
        if not anchors:
            return None
        deacc = remove_accents(text).lower()
        # Độ dài tiền tố đánh mục ("a. ", "1. ", "đ. ") ở đầu dòng - anchor nằm
        # ngay sau tiền tố vẫn coi là "đầu dòng" (không phải cặp thứ 2).
        prefix_m = re.match(r"^\s*(?:[0-9]{1,2}|[a-zđ])[.)]\s*", deacc)
        prefix_len = prefix_m.end() if prefix_m else 0
        for anchor in anchors:
            akw = remove_accents(anchor).lower().strip()
            if not akw:
                continue
            pos = deacc.find(akw)
            if pos == -1:
                continue
            if require_mid_line and pos <= prefix_len:
                # anchor ở đầu dòng -> đây là cặp đầu tiên, để Bước 1 xử lý
                continue
            after = text[pos + len(akw):]
            # bỏ dấu phân cách/nhiễu ngay sau nhãn
            after = re.sub(r"^[\s:.\-_)\(]+", "", after)
            # giới hạn tới cặp kế tiếp (ngăn cách bởi ';') để không nuốt sang nhãn sau
            after = re.split(r";", after, 1)[0].strip()
            if not after:
                continue
            if regex_pattern:
                m = re.search(regex_pattern, after)
                if m:
                    return m.group(0)
            else:
                return after
        return None

    def _extract_from_anchor(self, anchor_block, field_cfg, graph):
        text = anchor_block["text"]
        regex_pattern = field_cfg.get("regex")
        field_name = field_cfg.get("_field_name", "")

        # Bước 0: CHỈ cho các field GIÁ TRỊ SỐ INLINE (số thửa, tờ bản đồ, diện
        # tích) - khi anchor keyword nằm GIỮA dòng gộp nhiều cặp "nhãn: giá trị"
        # (VD "a. Thửa đất số: 251; tờ bản đồ số: 74,"), trích giá trị ngay sau
        # ĐÚNG anchor đó, tránh lấy nhầm giá trị của nhãn đứng trước (74 chứ không
        # phải 251). KHÔNG áp dụng cho tên/địa chỉ (holder_name, address...) vì
        # các field đó cần logic join_same_row/nối dòng riêng - Bước 0 sẽ cắt sai.
        # Và CHỈ khi anchor nằm sau vị trí đầu dòng (pos>0 = có cặp đứng trước),
        # để mẫu 1-cặp giữ nguyên hành vi Bước 1.
        _INLINE_NUMERIC_FIELDS = {"parcel_number", "map_sheet_number", "area_m2", "usable_area_m2"}
        if field_name in _INLINE_NUMERIC_FIELDS:
            anchors = field_cfg.get("anchors", [])
            anchored_val = self._extract_after_anchor_keyword(
                text, anchors, regex_pattern, require_mid_line=True
            )
            if anchored_val is not None:
                return anchored_val

        # Bước 1: Thử tách giá trị sau dấu phân cách (: hoặc bất kỳ)
        # label_only: block chỉ chứa NHÃN (VD "4. Diện tích:"), giá trị thật
        # nằm ở block KHÁC (do OCR tách nhãn/giá trị làm 2 block riêng - gặp
        # thật ở mẫu mới). Khi đó KHÔNG được quét regex trên chính text anchor
        # ở Bước 2 nữa, vì dễ bắt nhầm số thứ tự của nhãn (VD "4." trong "4.
        # Diện tích:" bị hiểu nhầm thành giá trị diện tích) - phải đi thẳng
        # sang tìm ở block lân cận (_ordered_neighbor_candidates).
        label_only = False
        for sep in [":", "-", "="]:
            if sep in text:
                potential_val = text.split(sep, 1)[1].strip()
                potential_val = re.sub(r"^[ \.\-_:\(\)\/]+", "", potential_val).strip()
                if potential_val:
                    if regex_pattern:
                        match = re.search(regex_pattern, potential_val)
                        if match:
                            extracted = match.group(0)
                            if field_name == "id_number":
                                # Chỉ nhận nếu đúng 9 (CMND) hoặc 12 (CCCD) số;
                                # nếu không, KHÔNG trả về giá trị rác - để Bước 2 /
                                # neighbor tìm ứng viên hợp lệ hơn.
                                validated = self._validate_id_number(extracted)
                                if validated:
                                    return validated
                            else:
                                return extracted
                    elif len(potential_val) >= 2:
                        if field_cfg.get("multiline"):
                            return self._extend_multiline(anchor_block, potential_val, graph, field_cfg)
                        return potential_val
                else:
                    label_only = True
                break  # Chỉ thử separator đầu tiên tìm thấy

        # Bước 2: Fallback — quét toàn bộ text bằng regex (bất kể separator),
        # CHỈ khi anchor không phải block "chỉ có nhãn" (xem label_only ở trên).
        if regex_pattern and not label_only:
            matches = list(re.finditer(regex_pattern, text))
            if matches:
                extracted = matches[-1].group(0)
                if field_name == "id_number":
                    validated = self._validate_id_number(extracted)
                    if validated:
                        return validated
                else:
                    return extracted

        ordered_candidates = self._ordered_neighbor_candidates(anchor_block, field_cfg, graph)
        if not ordered_candidates:
            return None

        first_raw_val = None
        if field_cfg.get("join_same_row") and ordered_candidates:
            primary = ordered_candidates[0]
            same_row_cands = [
                c for c in ordered_candidates
                if graph._same_row(primary["bbox"], c["bbox"]) and c["bbox"][2] > anchor_block["bbox"][0] - 50
            ]
            same_row_cands.sort(key=lambda x: x["bbox"][0])
            joined_text = " ".join([c["text"] for c in same_row_cands])
            
            if field_cfg.get("multiline"):
                joined_text = self._extend_multiline(primary, joined_text, graph, field_cfg)
                
            if regex_pattern:
                match = re.search(regex_pattern, joined_text)
                return match.group(0) if match else None
            return joined_text

        for candidate in ordered_candidates:
            raw_val = candidate["text"]
            if field_cfg.get("multiline"):
                raw_val = self._extend_multiline(candidate, raw_val, graph, field_cfg)

            if first_raw_val is None:
                first_raw_val = raw_val

            if regex_pattern:
                match = re.search(regex_pattern, raw_val)
                if match:
                    return match.group(0)
            else:
                return raw_val

        return None if regex_pattern else first_raw_val

    def _ordered_neighbor_candidates(self, anchor_block, field_cfg, graph):
        directions = field_cfg.get("direction", ["right", "below"])
        same_line_pref = field_cfg.get("same_line_preferred", False)
        candidates = []

        for direction in directions:
            neighbors = graph.get_neighbors(anchor_block["block_id"], direction, max_distance=600)
            if not neighbors:
                continue

            if direction == "right" and same_line_pref:
                same_row = [
                    neighbor
                    for neighbor in neighbors
                    if graph._same_row(anchor_block["bbox"], neighbor["bbox"])
                ]
                candidates.extend(same_row)
                candidates.extend([neighbor for neighbor in neighbors if neighbor not in same_row])
            else:
                candidates.extend(neighbors)

        return candidates

    def _extend_multiline(self, start_block, current_text, graph, field_cfg):
        extended_text = current_text
        current_block_id = start_block["block_id"]
        max_dist = field_cfg.get("max_distance_y", 150)

        for _ in range(3):
            below_neighbors = graph.get_neighbors(current_block_id, "below", max_distance=max_dist)
            if not below_neighbors:
                break

            next_block = below_neighbors[0]
            next_text = next_block.get("text", "").lower()
            stop_keywords = [
                "thửa đất",
                "thá»­a Ä‘áº¥t",
                "tờ bản đồ",
                "tá» báº£n Ä‘á»“",
                "diện tích",
                "diá»‡n tÃ­ch",
                "hình thức",
                "hÃ¬nh thá»©c",
                "kết cấu",
                "káº¿t cáº¥u",
                "mục ",
                "má»¥c ",
                "thông tin tài sản",
                "sơ đồ thửa đất",
                "kích thước",
                "bản đồ vị trí",
            ]
            if any(keyword in next_text for keyword in stop_keywords):
                break

            if abs(start_block["bbox"][0] - next_block["bbox"][0]) < 250:
                extended_text += " " + next_block["text"]
                current_block_id = next_block["block_id"]
            else:
                break

        return extended_text
