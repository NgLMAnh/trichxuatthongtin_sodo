import re

from src.text_utils import find_best_anchor


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
        """Validate CMND (9 số) hoặc CCCD (12 số). Loại bỏ false positive."""
        if not value:
            return None
        digits = re.sub(r"\D", "", value)
        if len(digits) in (9, 12):
            return digits
        return None

    def _extract_from_anchor(self, anchor_block, field_cfg, graph):
        text = anchor_block["text"]
        regex_pattern = field_cfg.get("regex")
        field_name = field_cfg.get("_field_name", "")

        # Bước 1: Thử tách giá trị sau dấu phân cách (: hoặc bất kỳ)
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
                                return self._validate_id_number(extracted) or extracted
                            return extracted
                    elif len(potential_val) >= 2:
                        if field_cfg.get("multiline"):
                            return self._extend_multiline(anchor_block, potential_val, graph, field_cfg)
                        return potential_val
                break  # Chỉ thử separator đầu tiên tìm thấy

        # Bước 2: Fallback — quét toàn bộ text bằng regex (bất kể separator)
        if regex_pattern:
            matches = list(re.finditer(regex_pattern, text))
            if matches:
                extracted = matches[-1].group(0)
                if field_name == "id_number":
                    return self._validate_id_number(extracted) or extracted
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
