"""
Synonym Expander: Mở rộng query bằng từ đồng nghĩa trước khi search.
Giải quyết vấn đề bất đồng ngôn ngữ (CMND ≠ holder_id_number).
"""
import re


SYNONYMS = {
    "cmnd": ["cccd", "chứng minh nhân dân", "căn cước công dân", "cmnd_cccd", "số cmnd", "số cccd", "giấy tờ tùy thân"],
    "cccd": ["cmnd", "chứng minh nhân dân", "căn cước công dân", "cmnd_cccd"],
    "chủ sở hữu": ["chu_so_huu", "tên chủ", "người sở hữu", "holder", "chủ đất"],
    "diện tích": ["dien_tich_m2", "m2", "mét vuông", "area"],
    "địa chỉ": ["dia_chi", "nơi ở", "thường trú", "address"],
    "năm sinh": ["nam_sinh", "sinh năm", "tuổi", "birthday"],
    "thửa đất": ["thua_dat_so", "số thửa", "parcel"],
    "tờ bản đồ": ["to_ban_do_so", "số tờ bản đồ", "map sheet"],
    "thay đổi": ["biến động", "chuyển nhượng", "thế chấp", "tặng cho", "thừa kế"],
    "nhà ở": ["nhà", "đất ở", "bất động sản", "nhà đất"],
    "hồ sơ": ["số hồ sơ", "mã hồ sơ", "hồ sơ gốc"],
    "nơi ký": ["cơ quan ký", "ubnd", "nơi ra quyết định", "nơi cấp"],
    "đổi chủ": ["biến động", "chuyển nhượng", "mua bán", "thay đổi chủ"],
}


def expand_query(question):
    """
    Mở rộng query bằng cách thêm các từ đồng nghĩa.
    Không thay đổi câu gốc, chỉ nối thêm ở cuối để tăng recall.
    
    VD: "CMND của Hồ Lệ Hồng"
    → "CMND của Hồ Lệ Hồng [cccd chứng minh nhân dân căn cước cmnd_cccd]"
    """
    q_lower = question.lower()
    expansions = set()

    for keyword, syns in SYNONYMS.items():
        if keyword in q_lower:
            for syn in syns:
                if syn.lower() not in q_lower:
                    expansions.add(syn)

    if expansions:
        # Nối từ đồng nghĩa vào cuối query (không thay đổi câu gốc)
        expansion_text = " ".join(sorted(expansions)[:5])  # Giới hạn 5 từ
        return f"{question} [{expansion_text}]"

    return question
