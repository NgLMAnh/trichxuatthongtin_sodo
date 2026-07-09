import re

from src.text_utils import normalize_text, remove_accents

# Mục III/IV dùng "Ngày 30 tháng 12 năm 2002"; Mục VI dùng "ngày 19/01/2009"
# (sau normalize_text, "/" và "." bị gộp thành khoảng trắng nên 2 format tách biệt).
DATE_VERBAL_RE = re.compile(r"ngay\s+(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{4})")
DATE_SLASH_RE = re.compile(r"ngay\s+(\d{1,2})\s+(\d{1,2})\s+(\d{4})\b")
APPLICATION_NUMBER_RE = re.compile(
    r"[Hh]ồ\s*sơ(?:\s*gốc)?\s*số[\s\.:]*([A-Za-z0-9][\w\/\.\-]*)"
)
APPLICATION_NUMBER_RE_NOACCENT = re.compile(
    r"[Hh]o\s*so(?:\s*goc)?\s*so[\s\.:]*([A-Za-z0-9][\w\/\.\-]*)"
)
DECISION_PLACE_RE = re.compile(
    r"(?:UBND|Ủy\s*ban\s*[Nn]hân\s*[Dd]ân)[^\n,\.]{0,40}"
)


class ChangeHistoryExtractor:
    """
    Trích xuất lịch sử biến động (Mục III/IV/VI) thành danh sách record
    {section, page, decision_date, application_number, decision_place, content}.

    Khác với FieldExtractor (1 giá trị/field), 1 document có thể có nhiều
    lần biến động nên mỗi section được cắt thành nhiều record, mỗi record
    bắt đầu tại 1 block chứa mốc ngày ("Ngày ... tháng ... năm ...").
    """

    def __init__(self, config):
        self.config = config.get("change_extraction", {})

    def extract(self, blocks, sections):
        target_sections = self.config.get(
            "target_sections",
            ["owner_changes", "property_changes", "post_issue_changes"],
        )

        records = []
        for section_name in target_sections:
            block_ids = sections.get(section_name, [])
            if not block_ids:
                continue
            section_blocks = [b for b in blocks if b["block_id"] in block_ids]
            section_blocks.sort(key=lambda b: b.get("reading_order", 0))
            records.extend(self._extract_section_records(section_name, section_blocks))

        return records

    def _extract_section_records(self, section_name, section_blocks):
        marker_indexes = [
            idx
            for idx, block in enumerate(section_blocks)
            if self._find_date_match(block.get("text", ""))
        ]

        if marker_indexes:
            groups = [
                section_blocks[start:end]
                for start, end in zip(
                    marker_indexes, marker_indexes[1:] + [len(section_blocks)]
                )
            ]
        else:
            groups = [section_blocks] if section_blocks else []

        records = []
        for group in groups:
            content = " ".join(
                block.get("text", "").strip()
                for block in group
                if block.get("text", "").strip()
            )
            if not content:
                continue

            records.append(
                {
                    "section": section_name,
                    "page": None,
                    "decision_date": self._extract_date(content),
                    "application_number": self._extract_application_number(content),
                    "decision_place": self._extract_decision_place(content),
                    "content": content,
                }
            )

        return records

    @staticmethod
    def _find_date_match(text):
        normalized = normalize_text(text)
        return DATE_VERBAL_RE.search(normalized) or DATE_SLASH_RE.search(normalized)

    @classmethod
    def _extract_date(cls, text):
        match = cls._find_date_match(text)
        if not match:
            return None
        day, month, year = match.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}"

    @staticmethod
    def _extract_application_number(text):
        match = APPLICATION_NUMBER_RE.search(text)
        if not match:
            match = APPLICATION_NUMBER_RE_NOACCENT.search(remove_accents(text))
        if not match:
            return None
        value = match.group(1).strip().strip(".-/")
        return value or None

    @staticmethod
    def _extract_decision_place(text):
        match = DECISION_PLACE_RE.search(text)
        if not match:
            return None
        return match.group(0).strip().rstrip(".,")
