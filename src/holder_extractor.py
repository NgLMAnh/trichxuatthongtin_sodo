import re

# Mẫu GCN hợp nhất (2009+) thường gộp tên + CCCD/CMND trong 1 dòng, ví dụ:
#   "Ông Nguyễn Ngọc Ngà, CCCD: 066095010633"
#   "VÀ VỢ: BÀ LÊ THỊ XUÂN THAO, CCCD: 066194020228"
# Khác mẫu cũ (1 chủ, tên và CMND ở 2 block riêng, do FieldExtractor xử lý).
HOLDER_RE = re.compile(
    r"(?:và\s+(?:vợ|chồng)\s*:?\s*)?"
    r"(ông|bà)\s+([^,:]+?)\s*,?\s*"
    r"(?:CMND|CCCD)(?:\s*số)?\s*:?\s*(\d{9,12})",
    re.IGNORECASE,
)


class HolderExtractor:
    """
    Trích xuất TẤT CẢ chủ sở hữu/người sử dụng đất trong 1 document (hỗ trợ
    mẫu GCN có 2 người - vợ và chồng - cùng đứng tên, mỗi người có CCCD riêng).

    Chỉ khớp khi tên + CMND/CCCD nằm gộp trong CÙNG 1 block (đặc trưng mẫu GCN
    hợp nhất 2009+). Mẫu cũ (tên/CMND ở 2 block riêng) sẽ không khớp gì -
    pipeline.py sẽ fallback dùng lại kết quả FieldExtractor cho trường hợp đó
    (xem src/pipeline.py), nên không ảnh hưởng tài liệu mẫu cũ.
    """

    def __init__(self, config):
        self.config = config.get("holder_extraction", {})

    def extract(self, blocks, sections):
        target_sections = self.config.get("target_sections", ["holder_info"])

        candidate_blocks = []
        for section_name in target_sections:
            block_ids = sections.get(section_name, [])
            candidate_blocks.extend(b for b in blocks if b["block_id"] in block_ids)
        if not candidate_blocks:
            candidate_blocks = blocks

        candidate_blocks = sorted(candidate_blocks, key=lambda b: b.get("reading_order", 0))

        holders = []
        for block in candidate_blocks:
            text = block.get("text", "")
            match = HOLDER_RE.search(text)
            if not match:
                continue

            role_raw, name, id_number = match.groups()
            holders.append(
                {
                    "role": role_raw.strip().title(),
                    "name": name.strip(" ,"),
                    "id_number": id_number,
                    "birthday": None,
                    "address": None,
                }
            )

        return holders
