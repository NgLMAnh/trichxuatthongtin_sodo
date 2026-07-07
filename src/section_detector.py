import re

from src.text_utils import normalize_text


class SectionDetector:
    SECTION_ORDER = [
        "unknown",
        "holder_info",
        "land_info",
        "land_diagram",
        "owner_changes",
        "property_changes",
        "post_issue_changes",
    ]

    HEADING_RULES = [
        (r"\bmuc\s+i\b", "holder_info"),
        (r"\bmuc\s+ii\s*c\b", "land_diagram"),
        (r"\bmuc\s+ii\b", "land_info"),
        (r"\bmuc\s+iii\b", "owner_changes"),
        (r"\bmuc\s+iv\b", "property_changes"),
        (r"\bvi\b.*\bthay\s+doi\b", "post_issue_changes"),
    ]

    def __init__(self, config):
        self.config = config.get("section_detection", {})

    def detect(self, blocks):
        """
        Classifies blocks into stable document sections.

        The field extractor still relies on holder_info and land_info. Extra
        sections are kept for cleaner Markdown/RAG structure.
        """
        sections = {name: [] for name in self.SECTION_ORDER}
        section_boundaries = []

        for block in sorted(blocks, key=lambda b: b.get("reading_order", 0)):
            text = normalize_text(block.get("text", ""))
            section_type = self._detect_heading_type(text)
            if section_type:
                section_boundaries.append(
                    {
                        "type": section_type,
                        "y": block["bbox"][1],
                        "order": block.get("reading_order", 0),
                        "block_id": block["block_id"],
                    }
                )

        section_boundaries.sort(key=lambda x: (x["y"], x["order"]))

        for block in blocks:
            assigned_section = self._assign_section(block, section_boundaries)
            sections.setdefault(assigned_section, []).append(block["block_id"])

        return {name: ids for name, ids in sections.items() if ids}

    def _detect_heading_type(self, normalized_text):
        if not normalized_text:
            return None

        for pattern, section_type in self.HEADING_RULES:
            if re.search(pattern, normalized_text):
                return section_type

        return self._detect_from_config(normalized_text)

    def _detect_from_config(self, normalized_text):
        holder_kws = [
            normalize_text(kw)
            for kw in self.config.get("holder_info", {}).get("keywords", [])
        ]
        if any(kw and kw in normalized_text for kw in holder_kws):
            return "holder_info"

        land_kws = [
            normalize_text(kw)
            for kw in self.config.get("land_info", {}).get("keywords", [])
        ]
        if any(kw and kw in normalized_text for kw in land_kws):
            return "land_info"

        return None

    def _assign_section(self, block, boundaries):
        if not boundaries:
            return "unknown"

        y = block["bbox"][1]
        order = block.get("reading_order", 0)
        assigned_section = "unknown"

        for boundary in boundaries:
            if (y, order) >= (boundary["y"] - 20, boundary["order"]):
                assigned_section = boundary["type"]
            else:
                break

        return assigned_section
