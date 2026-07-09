"""
Query Router: Phân loại câu hỏi thành field-based hoặc open-ended.
- Field-based: tra trực tiếp JSON, không cần RAG/LLM.
- Open-ended: đi qua pipeline RAG như bình thường.
"""
import os
import re
import json
from difflib import SequenceMatcher

from src.text_utils import remove_accents

# Bảng chữ cái tiếng Việt tường minh (không dùng range Unicode "À-Ỹ" vì range đó
# vô tình chứa cả chữ thường có dấu như "đ"/"ư", khiến regex nhận nhầm cụm như
# "Thửa đất" là tên người).
_VI_UPPER = "AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÄBCDĐEÉÈẺẼẸÊẾỀỂỄỆËFGHIÍÌỈĨỊÏJKLMNOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÖPQRSTUÚÙỦŨỤƯỨỪỬỮỰÜVWXYÝỲỶỸỴŸZ"
_VI_LOWER = "aáàảãạăắằẳẵặâấầẩẫậäbcdđeéèẻẽẹêếềểễệëfghiíìỉĩịïjklmnoóòỏõọôốồổỗộơớờởỡợöpqrstuúùủũụưứừửữựüvwxyýỳỷỹỵÿz"

# Từ khóa ánh xạ câu hỏi -> trường JSON
FIELD_PATTERNS = {
    "holder_name": [
        r"(?:tên|họ\s*tên|ai\s*là)\s*(?:chủ\s*sở\s*hữu|người\s*sở\s*hữu|chủ\s*đất)",
        r"chủ\s*sở\s*hữu[^.?!]{0,25}\s*(?:là\s*ai|tên)",
        r"(?:sổ\s*đỏ|thửa\s*đất|mảnh\s*đất).*(?:của\s*ai|thuộc\s*về\s*ai)",
    ],
    "id_number": [
        r"(?:số\s*)?(?:cmnd|cccd|chứng\s*minh\s*nhân\s*dân|căn\s*cước)",
        r"giấy\s*tờ\s*tùy\s*thân",
    ],
    "address": [
        r"(?:địa\s*chỉ|nơi\s*ở|thường\s*trú)",
    ],
    "birthday": [
        r"(?:năm\s*sinh|sinh\s*năm|tuổi|bao\s*nhiêu\s*tuổi)",
    ],
    "parcel_number": [
        r"(?:số\s*thửa|thửa\s*đất\s*số|thửa\s*(?:đất\s*)?(?:bao\s*nhiêu|mấy|nào))",
    ],
    "map_sheet_number": [
        r"(?:tờ\s*bản\s*đồ|số\s*tờ\s*bản\s*đồ)",
    ],
    # Đặt TRƯỚC area_m2: "diện tích sử dụng" (của tài sản/căn hộ) khác "diện tích"
    # (của thửa đất) - nếu không, area_m2 sẽ bắt nhầm vì cùng chứa "diện tích".
    "usable_area_m2": [
        r"diện\s*tích\s*sử\s*dụng",
    ],
    "asset_name": [
        r"tên\s*tài\s*sản|căn\s*hộ\s*(?:số|tên|là\s*gì)",
    ],
    "ownership_form": [
        r"hình\s*thức\s*sở\s*hữu",
    ],
    "ownership_term": [
        r"thời\s*hạn\s*sở\s*hữu",
    ],
    # Đặt TRƯỚC area_m2: "diện tích còn lại" (sau biến động, nằm trong nội dung
    # change_history) khác "diện tích" thửa đất hiện tại - và các câu hỏi về nội
    # dung/thời điểm biến động nên tra change_history đã có sẵn, tránh đẩy sang
    # RAG rồi bị LLM tự bịa thêm chi tiết (đã gặp thật khi test).
    "change_history": [
        r"(?:biến\s*động|đổi\s*chủ|chuyển\s*nhượng|mua\s*bán|thế\s*chấp|tặng\s*cho|thừa\s*kế)",
        r"(?:lịch\s*sử|quá\s*trình)\s*(?:thay\s*đổi|biến\s*động)?",
        r"(?:số\s*hồ\s*sơ|mã\s*hồ\s*sơ|hồ\s*sơ\s*gốc)",
        r"(?:ngày\s*ký|ngày\s*quyết\s*định|ngày\s*ra\s*quyết\s*định|vào\s*sổ\s*(?:ngày|lúc))",
        r"(?:nơi\s*ký|cơ\s*quan\s*ký|nơi\s*ra\s*quyết\s*định|nơi\s*cấp\s*quyết\s*định)",
        r"(?:ký\s*(?:ở|tại)\s*đâu|quyết\s*định.*ký|ký\s*quyết\s*định)",
        r"(?:thông\s*tin\s*thay\s*đổi|những\s*thay\s*đổi)\s*(?:sau\s*khi\s*cấp)?",
        r"có\s*gì\s*thay\s*đổi",
        r"xác\s*nhận\s*(?:điều\s*gì|những\s*gì|gì)",
        r"diện\s*tích\s*còn\s*lại",
    ],
    "area_m2": [
        r"(?:diện\s*tích|bao\s*nhiêu\s*m2|mét\s*vuông|bao\s*nhiêu\s*mét)",
    ],
}

# Ánh xạ field name -> đường dẫn trong JSON (chỉ còn dùng cho field cấp-document,
# không phải cấp-người; field cấp-người xem PERSON_FIELDS + _iter_persons)
FIELD_TO_JSON_PATH = {
    "parcel_number": ("land_parcel", "parcel_number"),
    "map_sheet_number": ("land_parcel", "map_sheet_number"),
    "area_m2": ("land_parcel", "area_m2"),
    "asset_name": ("asset", "asset_name"),
    "usable_area_m2": ("asset", "usable_area_m2"),
    "ownership_form": ("asset", "ownership_form"),
    "ownership_term": ("asset", "ownership_term"),
}

# Field cấp-người (mỗi document có thể có NHIỀU người - vợ/chồng cùng đứng tên) ->
# key tương ứng trong 1 phần tử của doc_data["holders"]
PERSON_FIELDS = {
    "holder_name": "name",
    "id_number": "id_number",
    "address": "address",
    "birthday": "birthday",
}

# Nhãn tiếng Việt cho các trường
FIELD_LABELS = {
    "holder_name": "Chủ sở hữu",
    "id_number": "Số CMND/CCCD",
    "address": "Địa chỉ",
    "birthday": "Năm sinh",
    "parcel_number": "Số thửa đất",
    "map_sheet_number": "Tờ bản đồ số",
    "area_m2": "Diện tích (m²)",
    "asset_name": "Tên tài sản",
    "usable_area_m2": "Diện tích sử dụng (m²)",
    "ownership_form": "Hình thức sở hữu",
    "ownership_term": "Thời hạn sở hữu",
    "change_history": "Lịch sử biến động",
}

# Câu hỏi TỔNG HỢP qua NHIỀU tài liệu (đếm/so sánh/liệt kê) - phải tính trực tiếp
# trên toàn bộ JSON đã load bằng Python, KHÔNG qua LLM, vì RAG (top-k chunk) không
# đủ đại diện để đếm/so sánh chính xác và dễ ảo giác (đã gặp thật khi test).
AGGREGATE_PATTERNS = {
    "count_documents": [
        r"bao\s*nhiêu\s*(?:tài\s*liệu|sổ\s*đỏ|sổ\s*hồng|hồ\s*sơ|giấy\s*chứng\s*nhận)",
        r"(?:có\s*)?mấy\s*(?:tài\s*liệu|sổ\s*đỏ|sổ\s*hồng)",
    ],
    # area_extreme_or_compare PHẢI đứng TRƯỚC list_holders: câu hỏi so sánh diện
    # tích nhiều chủ (VD: "So sánh diện tích đất của tất cả chủ sở hữu, ai nhiều
    # nhất, ai ít nhất?") vẫn chứa cụm "tất cả chủ sở hữu" nên nếu list_holders
    # được check trước sẽ bắt nhầm thành liệt kê tên - bug thật gặp khi test.
    "area_extreme_or_compare": [
        r"diện\s*tích.*(?:lớn\s*nhất|nhỏ\s*nhất|nhiều\s*nhất|ít\s*nhất)",
        r"(?:lớn\s*nhất|nhỏ\s*nhất).*diện\s*tích",
        r"so\s*sánh\s*diện\s*tích",
    ],
    "list_holders": [
        r"liệt\s*kê.*chủ\s*sở\s*hữu",
        r"danh\s*sách.*chủ\s*sở\s*hữu",
        r"tất\s*cả\s*chủ\s*sở\s*hữu",
    ],
    "co_owned_docs": [
        r"đồng\s*sở\s*hữu",
        r"(?:2|hai)\s*(?:người|chủ)[^.?!]{0,15}trở\s*lên",
        r"nhiều\s*hơn\s*(?:1|một)\s*chủ",
    ],
    "no_change_history": [
        r"không\s*có[^.?!]{0,20}(?:lịch\s*sử\s*)?biến\s*động",
        r"chưa\s*(?:từng\s*)?(?:có\s*)?biến\s*động",
    ],
}


class QueryRouter:
    def __init__(self, predictions_dir="outputs/predictions"):
        self.predictions_dir = predictions_dir
        self.documents = self._load_all_predictions()
        # Conversation memory: tài liệu/người được nói tới gần nhất, để câu hỏi
        # tiếp theo không nêu rõ tên/thửa đất vẫn hiểu đúng ngữ cảnh (VD: hỏi
        # "diện tích" rồi hỏi tiếp "diện tích sử dụng thì sao?").
        self.last_doc_id = None
        self.last_person_name = None

    def _remember(self, doc_id, person_name=None):
        self.last_doc_id = doc_id
        if person_name:
            self.last_person_name = person_name

    def _load_all_predictions(self):
        """Load tất cả JSON predictions vào bộ nhớ."""
        docs = {}
        if not os.path.exists(self.predictions_dir):
            return docs
        for fname in os.listdir(self.predictions_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.predictions_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        doc_id = data.get("document_id", fname.replace(".json", ""))
                        docs[doc_id] = data
                except Exception:
                    pass
        return docs

    def classify(self, question):
        """
        Phân loại câu hỏi.
        Returns: ("field", field_name) hoặc ("rag", None)
        """
        q_lower = question.lower().strip()

        for field_name, patterns in FIELD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q_lower):
                    return "field", field_name

        return "rag", None

    def lookup_aggregate(self, question):
        """
        Câu hỏi tổng hợp/so sánh qua NHIỀU tài liệu (đếm, liệt kê, lớn/nhỏ nhất,
        đồng sở hữu, không có biến động). Tính trực tiếp trên JSON đã load bằng
        Python - không qua LLM - để không thể ảo giác và luôn chính xác 100%.
        Gọi TRƯỚC classify()/lookup_json(), vì các mẫu câu này dễ bị field-pattern
        (VD "thửa đất nào") bắt nhầm thành field-lookup đơn lẻ.
        Returns: (answer_text, "ALL") hoặc (None, None) nếu không phải câu hỏi tổng hợp.
        """
        q_lower = question.lower().strip()
        intent = None
        for key, patterns in AGGREGATE_PATTERNS.items():
            if any(re.search(p, q_lower) for p in patterns):
                intent = key
                break
        if not intent:
            return None, None

        docs = self.documents
        if not docs:
            return None, None

        if intent == "count_documents":
            return (
                f"Hệ thống hiện có {len(docs)} tài liệu Sổ đỏ: {', '.join(docs.keys())}.",
                "ALL",
            )

        if intent == "list_holders":
            lines = ["Danh sách chủ sở hữu trong hệ thống:"]
            for doc_id, doc_data in docs.items():
                names = [p.get("name") for p in self._iter_persons(doc_id, doc_data) if p.get("name")]
                lines.append(f"  - {doc_id}: {', '.join(names) if names else '(không rõ)'}")
            return "\n".join(lines), "ALL"

        if intent == "area_extreme_or_compare":
            rows = [
                (doc_id, doc_data.get("land_parcel", {}).get("area_m2"), doc_data.get("holder", {}).get("name"))
                for doc_id, doc_data in docs.items()
            ]
            rows = [row for row in rows if row[1] is not None]
            if not rows:
                return None, None
            rows.sort(key=lambda row: row[1], reverse=True)
            lines = ["So sánh diện tích thửa đất giữa các tài liệu (từ lớn đến nhỏ):"]
            for doc_id, area, name in rows:
                lines.append(f"  - {doc_id} ({name}): {area} m²")
            max_row, min_row = rows[0], rows[-1]
            lines.append(f"→ Lớn nhất: {max_row[2]} ({max_row[0]}) với {max_row[1]} m².")
            if min_row is not max_row:
                lines.append(f"→ Nhỏ nhất: {min_row[2]} ({min_row[0]}) với {min_row[1]} m².")
            return "\n".join(lines), "ALL"

        if intent == "co_owned_docs":
            matches = [doc_id for doc_id, doc_data in docs.items() if len(doc_data.get("holders") or []) >= 2]
            if not matches:
                return "Không có tài liệu nào ghi nhận từ 2 chủ sở hữu trở lên.", "ALL"
            lines = ["Tài liệu có từ 2 chủ sở hữu trở lên:"]
            for doc_id in matches:
                names = [h.get("name") for h in docs[doc_id].get("holders", [])]
                lines.append(f"  - {doc_id}: {', '.join(n for n in names if n)}")
            return "\n".join(lines), "ALL"

        if intent == "no_change_history":
            matches = [doc_id for doc_id, doc_data in docs.items() if not doc_data.get("change_history")]
            if not matches:
                return "Tất cả tài liệu đều có ghi nhận lịch sử biến động.", "ALL"
            return "Tài liệu không có lịch sử biến động: " + ", ".join(matches) + ".", "ALL"

        return None, None

    def _fuzzy_match_name(self, query_name, candidate_name, threshold=0.5):
        """So khớp mờ (fuzzy match) tên người."""
        if not query_name or not candidate_name:
            return 0.0
        q = query_name.upper().strip()
        c = candidate_name.upper().strip()
        # Exact substring
        if q in c or c in q:
            return 1.0
        return SequenceMatcher(None, q, c).ratio()

    # Từ hỏi (không phải tên người) hay bị regex bắt nhầm trong các câu như
    # "...là của ai?" -> "ai" không phải tên, tránh coi đó là person_name.
    _INTERROGATIVE_WORDS = {"ai", "gì", "nào", "đâu", "sao", "bao nhiêu"}

    def _extract_doc_id_from_query(self, question):
        """
        Nhận diện MÃ TÀI LIỆU (VD "DOC_004") được nêu trực tiếp trong câu hỏi.
        Phải kiểm tra TRƯỚC khi suy luận tên người: nếu không, "của DOC_004" bị
        _extract_person_name_from_query bắt nhầm thành tên người "DOC_004", rồi
        fuzzy-match với tên chủ sở hữu thất bại (score thấp) -> trả None dù
        document đó tồn tại và có đủ dữ liệu - bug thật đã gặp khi test (câu
        "Số thửa đất của DOC_004 là bao nhiêu?" bị đẩy nhầm sang RAG rồi báo
        không tìm thấy, dù land_parcel.parcel_number của DOC_004 = "251").
        """
        q_norm = question.upper()
        for doc_id in self.documents:
            if doc_id.upper() in q_norm:
                return doc_id
        return None

    def _extract_person_name_from_query(self, question):
        """Trích xuất tên người từ câu hỏi (heuristic)."""
        q = question.strip()
        # Tìm pattern: "của <TÊN>"
        match = re.search(r"(?:của|cho)\s+(?:ông|bà|anh|chị)?\s*(.+?)(?:\?|$|là|ở|tại)", q, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().rstrip("?").strip()
            if candidate.lower() not in self._INTERROGATIVE_WORDS:
                return candidate
        # Tìm pattern: "<TÊN> là chủ sở hữu"
        match = re.search(r"(?:ông|bà)\s+(.+?)(?:\s+là|\s+sở\s*hữu|\s+có|\?|$)", q, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().rstrip("?").strip()
            if candidate.lower() not in self._INTERROGATIVE_WORDS:
                return candidate
        # Tìm pattern: tên viết hoa liên tiếp
        match = re.search(
            rf"([{_VI_UPPER}][{_VI_LOWER}]+(?:\s+[{_VI_UPPER}][{_VI_LOWER}]+){{1,5}})", q
        )
        if match:
            return match.group(1).strip()
        return None

    # Câu hỏi về change_history có thể chỉ hỏi ĐÚNG 1 khía cạnh cụ thể (mã hồ
    # sơ / ngày ký / nơi ký) - nếu luôn trả về toàn bộ content thô, người dùng
    # phải tự đọc để tìm, không "trả lời đúng trọng tâm". Nhận diện khía cạnh
    # hỏi để trả lời gọn, chỉ fallback về nội dung đầy đủ khi câu hỏi mở/chung.
    _CHANGE_ASPECT_PATTERNS = {
        "application_number": [r"số\s*hồ\s*sơ", r"mã\s*hồ\s*sơ", r"hồ\s*sơ\s*gốc"],
        "decision_date": [r"ngày\s*ký", r"ngày\s*quyết\s*định", r"ngày\s*ra\s*quyết\s*định", r"vào\s*sổ"],
        "decision_place": [r"nơi\s*ký", r"cơ\s*quan\s*ký", r"nơi\s*ra\s*quyết\s*định", r"ký\s*(?:ở|tại)\s*đâu"],
    }
    _CHANGE_ASPECT_LABELS = {
        "application_number": "Số hồ sơ",
        "decision_date": "Ngày ký/quyết định",
        "decision_place": "Nơi ký/ra quyết định",
    }

    @classmethod
    def _detect_change_aspect(cls, question):
        q_lower = question.lower()
        for aspect, patterns in cls._CHANGE_ASPECT_PATTERNS.items():
            if any(re.search(p, q_lower) for p in patterns):
                return aspect
        return None

    def _format_change_history_answer(self, doc_id, doc_data, aspect=None):
        """Định dạng change_history (list) thành câu trả lời. Nếu `aspect` được
        chỉ định (mã hồ sơ/ngày ký/nơi ký), chỉ trả lời đúng khía cạnh đó."""
        change_history = doc_data.get("change_history") or []
        if not change_history:
            return None

        holder = doc_data.get("holder", {}).get("name", "")

        if aspect:
            label = self._CHANGE_ASPECT_LABELS[aspect]
            values = [(idx, r.get(aspect)) for idx, r in enumerate(change_history, start=1) if r.get(aspect)]
            if not values:
                return (
                    f"Không tìm thấy {label.lower()} trong lịch sử biến động của {holder} ({doc_id}) "
                    f"trong dữ liệu đã trích xuất."
                )
            lines = [f"{label} trong lịch sử biến động của {holder} ({doc_id}):"]
            for idx, val in values:
                lines.append(f"  - Lần biến động {idx}: {val}")
            return "\n".join(lines)

        lines = [f"Lịch sử biến động của {holder} ({doc_id}):"]
        for idx, record in enumerate(change_history, start=1):
            parts = []
            if record.get("decision_date"):
                parts.append(record["decision_date"])
            if record.get("application_number"):
                parts.append(f"hồ sơ {record['application_number']}")
            if record.get("decision_place"):
                parts.append(f"ký tại {record['decision_place']}")
            meta = ", ".join(parts)
            content = record.get("content", "")
            if meta:
                lines.append(f"  {idx}. ({meta}): {content}")
            else:
                lines.append(f"  {idx}. {content}")

        return "\n".join(lines)

    def _iter_persons(self, doc_id, doc_data):
        """
        Duyệt qua TỪNG NGƯỜI trong 1 document. Hỗ trợ mẫu GCN có nhiều chủ sở
        hữu (vợ/chồng cùng đứng tên, mỗi người 1 CCCD riêng) qua doc_data["holders"];
        mẫu cũ (1 chủ) không có "holders" khớp được thì fallback dùng doc_data["holder"].
        """
        holders = doc_data.get("holders")
        if holders:
            for holder in holders:
                yield {
                    "doc_id": doc_id,
                    "name": holder.get("name"),
                    "id_number": holder.get("id_number"),
                    "birthday": holder.get("birthday"),
                    "address": holder.get("address") or doc_data.get("holder", {}).get("address"),
                    "role": holder.get("role"),
                }
            return

        holder = doc_data.get("holder", {})
        yield {
            "doc_id": doc_id,
            "name": holder.get("name"),
            "id_number": holder.get("id_number"),
            "birthday": holder.get("birthday"),
            "address": holder.get("address"),
            "role": None,
        }

    @staticmethod
    def _value_in_question(question, value):
        """Kiểm tra GIÁ TRỊ (số CCCD, tên...) có xuất hiện trực tiếp trong câu hỏi không."""
        if value is None or value == "":
            return False
        val_str = str(value)
        if len(val_str) < 4:
            return False
        val_digits = re.sub(r"\D", "", val_str)
        if val_digits and len(val_digits) >= 4 and val_digits in re.sub(r"\D", "", question):
            return True
        return val_str.lower() in question.lower()

    def _lookup_person_field(self, question, field_name):
        """Tra cứu field cấp-người (tên/CCCD/địa chỉ/năm sinh), hỗ trợ document nhiều người."""
        person_key = PERSON_FIELDS[field_name]
        label = FIELD_LABELS.get(field_name, field_name)
        person_name = self._extract_person_name_from_query(question)

        all_persons = [
            person
            for doc_id, doc_data in self.documents.items()
            for person in self._iter_persons(doc_id, doc_data)
        ]

        # Câu hỏi nêu rõ mã tài liệu (VD "của DOC_004") -> tra theo doc_id trực
        # tiếp, ưu tiên hơn suy luận tên người (tránh "DOC_004" bị hiểu nhầm là
        # tên và fuzzy-match thất bại).
        explicit_doc_id = self._extract_doc_id_from_query(question)
        if explicit_doc_id:
            doc_persons = [p for p in all_persons if p["doc_id"] == explicit_doc_id]
            if doc_persons:
                self._remember(explicit_doc_id)
                if len(doc_persons) == 1 or person_name:
                    p = (
                        max(doc_persons, key=lambda p: self._fuzzy_match_name(person_name, p.get("name") or ""))
                        if person_name
                        else doc_persons[0]
                    )
                    if p.get(person_key) is None:
                        return (
                            f"Không tìm thấy {label.lower()} của {p['name']} ({p['doc_id']}) "
                            f"trong dữ liệu đã trích xuất.",
                            p["doc_id"],
                        )
                    return f"{label} của {p['name']} ({p['doc_id']}): {p.get(person_key)}", p["doc_id"]
                results = [
                    f"  - {p['doc_id']} ({p['name']}): {label} = {p.get(person_key)}" for p in doc_persons
                ]
                return "\n".join(results), explicit_doc_id

        if not person_name:
            # Thử tra theo GIÁ TRỊ có sẵn trong câu hỏi trước (VD: hỏi theo số
            # CCCD cụ thể) để tránh trả lời thừa/nhầm người khi có nhiều người
            # cùng field đó (nhiều document, hoặc nhiều chủ trong 1 document).
            value_matches = [
                p for p in all_persons if self._value_in_question(question, p.get(person_key))
            ]
            if len(value_matches) == 1:
                p = value_matches[0]
                self._remember(p["doc_id"], p.get("name"))
                return f"{label} của {p['name']} ({p['doc_id']}): {p.get(person_key)}", p["doc_id"]

            # Không nêu tên/giá trị cụ thể -> ưu tiên người/tài liệu đang nói tới (memory)
            if len(value_matches) == 0 and self.last_doc_id:
                mem_person = next(
                    (
                        p for p in all_persons
                        if p["doc_id"] == self.last_doc_id
                        and (not self.last_person_name or p.get("name") == self.last_person_name or len(
                            [q for q in all_persons if q["doc_id"] == self.last_doc_id]
                        ) == 1)
                        and p.get(person_key) is not None
                    ),
                    None,
                )
                if mem_person:
                    self._remember(mem_person["doc_id"], mem_person.get("name"))
                    return (
                        f"{label} của {mem_person['name']} ({mem_person['doc_id']}): {mem_person.get(person_key)}",
                        mem_person["doc_id"],
                    )

            # Không đủ thông tin để xác định đúng 1 người -> liệt kê tất cả
            results = [
                f"  - {p['doc_id']} ({p['name']}): {label} = {p.get(person_key)}"
                for p in all_persons
                if p.get(person_key) is not None
            ]
            return ("\n".join(results), "ALL") if results else (None, None)

        # QUAN TRỌNG: chọn người khớp TÊN tốt nhất trước, KHÔNG lọc theo "có giá
        # trị field" ngay từ đầu — nếu lọc trước, người đúng nhưng field null sẽ
        # bị loại và hệ thống rơi qua nhầm 1 người khác chỉ vì họ có giá trị (dù
        # khớp tên rất thấp). Bug thật đã gặp: hỏi CMND của "Tang Quang Hưng"
        # (CMND=null) lại trả về CMND của người khác do lọc sai thứ tự.
        best_person, best_score = None, 0.0
        for p in all_persons:
            score = self._fuzzy_match_name(person_name, p.get("name") or "")
            if score > best_score:
                best_score, best_person = score, p

        if not best_person or best_score < 0.4:
            return None, None

        self._remember(best_person["doc_id"], best_person.get("name"))

        if best_person.get(person_key) is None:
            return (
                f"Không tìm thấy {label.lower()} của {best_person['name']} ({best_person['doc_id']}) "
                f"trong dữ liệu đã trích xuất.",
                best_person["doc_id"],
            )

        return (
            f"{label} của {best_person['name']} ({best_person['doc_id']}): {best_person.get(person_key)}",
            best_person["doc_id"],
        )

    def _find_doc_by_value_in_question(self, question, json_path):
        """
        Tìm document mà GIÁ TRỊ field đó xuất hiện trực tiếp trong câu hỏi
        (VD: hỏi "...số 020168965" thì tra theo giá trị, không cần tên người).
        Chỉ dùng khi câu hỏi không nêu tên người, để tránh trả lời thừa/nhầm
        tài liệu khi có nhiều document cùng loại field.
        Returns: [(doc_id, value), ...] các document khớp.
        """
        q_lower = question.lower()
        q_digits = re.sub(r"\D", "", question)
        matches = []

        for doc_id, doc_data in self.documents.items():
            obj = doc_data
            for key in json_path:
                obj = obj.get(key, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break
            if obj is None or obj == "":
                continue

            val_str = str(obj)
            val_digits = re.sub(r"\D", "", val_str)
            if val_digits and len(val_digits) >= 4 and val_digits in q_digits:
                matches.append((doc_id, obj))
            elif len(val_str) >= 4 and val_str.lower() in q_lower:
                matches.append((doc_id, obj))

        return matches

    def lookup_json(self, question, field_name):
        """
        Tra cứu trực tiếp từ JSON predictions.
        Returns: (answer_text, doc_id) hoặc (None, None)
        """
        if not self.documents:
            return None, None

        if field_name == "change_history":
            person_name = self._extract_person_name_from_query(question)
            aspect = self._detect_change_aspect(question)

            explicit_doc_id = self._extract_doc_id_from_query(question)
            if explicit_doc_id and explicit_doc_id in self.documents:
                self._remember(explicit_doc_id)
                doc_data = self.documents[explicit_doc_id]
                if not doc_data.get("change_history"):
                    holder = doc_data.get("holder", {}).get("name", "")
                    return f"Không tìm thấy lịch sử biến động nào của {holder} ({explicit_doc_id}).", explicit_doc_id
                return self._format_change_history_answer(explicit_doc_id, doc_data, aspect), explicit_doc_id

            if not person_name:
                # Không nêu rõ chủ/thửa đất -> ưu tiên tài liệu đang nói tới (memory)
                if self.last_doc_id and self.documents.get(self.last_doc_id, {}).get("change_history"):
                    answer = self._format_change_history_answer(
                        self.last_doc_id, self.documents[self.last_doc_id], aspect
                    )
                    if answer:
                        self._remember(self.last_doc_id)
                        return answer, self.last_doc_id

                # Không có ngữ cảnh nào -> liệt kê biến động của mọi tài liệu
                answers = [
                    self._format_change_history_answer(doc_id, doc_data, aspect)
                    for doc_id, doc_data in self.documents.items()
                    if doc_data.get("change_history")
                ]
                if answers:
                    return "\n\n".join(answers), "ALL"
                return None, None

            # Chọn tài liệu khớp TÊN tốt nhất trước - không lọc theo "có change_history"
            # ngay từ đầu (tránh rơi nhầm sang người khác nếu người đúng không có biến động).
            best_doc_id, best_score = None, 0.0
            for doc_id, doc_data in self.documents.items():
                holder_name = doc_data.get("holder", {}).get("name", "")
                score = self._fuzzy_match_name(person_name, holder_name)
                if score > best_score:
                    best_score, best_doc_id = score, doc_id

            if not best_doc_id or best_score < 0.4:
                return None, None

            self._remember(best_doc_id)
            if not self.documents[best_doc_id].get("change_history"):
                holder = self.documents[best_doc_id].get("holder", {}).get("name", "")
                return f"Không tìm thấy lịch sử biến động nào của {holder} ({best_doc_id}).", best_doc_id

            answer = self._format_change_history_answer(best_doc_id, self.documents[best_doc_id], aspect)
            return answer, best_doc_id

        if field_name in PERSON_FIELDS:
            return self._lookup_person_field(question, field_name)

        person_name = self._extract_person_name_from_query(question)
        json_path = FIELD_TO_JSON_PATH.get(field_name)
        if not json_path:
            return None, None

        # Câu hỏi nêu rõ mã tài liệu (VD "của DOC_004") -> tra trực tiếp theo
        # doc_id, không suy luận tên người (xem docstring _extract_doc_id_from_query).
        explicit_doc_id = self._extract_doc_id_from_query(question)
        if explicit_doc_id and explicit_doc_id in self.documents:
            doc_data = self.documents[explicit_doc_id]
            obj = doc_data
            for key in json_path:
                obj = obj.get(key, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break
            label = FIELD_LABELS.get(field_name, field_name)
            holder = doc_data.get("holder", {}).get("name", explicit_doc_id)
            self._remember(explicit_doc_id)
            if obj is None:
                return (
                    f"Không tìm thấy {label.lower()} của {holder} ({explicit_doc_id}) "
                    f"trong dữ liệu đã trích xuất.",
                    explicit_doc_id,
                )
            return f"{label} của {holder} ({explicit_doc_id}): {obj}", explicit_doc_id

        # Tìm document phù hợp nhất
        best_doc_id = None
        best_score = 0.0
        best_value = None

        for doc_id, doc_data in self.documents.items():
            holder_name = doc_data.get("holder", {}).get("name", "")

            # Nếu câu hỏi có tên người, dùng fuzzy match
            if person_name:
                score = self._fuzzy_match_name(person_name, holder_name)
            else:
                score = 1.0  # Không chỉ định tên → lấy tất cả

            # Lấy giá trị field
            obj = doc_data
            for key in json_path:
                obj = obj.get(key, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break

            if person_name:
                # Chọn tài liệu khớp TÊN tốt nhất trước - không lọc theo "có giá trị"
                # (tránh rơi nhầm sang tài liệu khác nếu người đúng có field null).
                if score > best_score:
                    best_score, best_doc_id, best_value = score, doc_id, obj
            elif obj is not None and score > best_score:
                best_score = score
                best_doc_id = doc_id
                best_value = obj

        # Nếu không chỉ định tên người cụ thể, thử tra theo GIÁ TRỊ field có
        # sẵn trong câu hỏi trước (VD: hỏi theo số CMND cụ thể) để tránh trả
        # lời thừa toàn bộ document khi câu hỏi đã đủ để xác định đúng 1 hồ sơ.
        if not person_name:
            value_matches = self._find_doc_by_value_in_question(question, json_path)
            if len(value_matches) == 1:
                doc_id, value = value_matches[0]
                holder = self.documents[doc_id].get("holder", {}).get("name", doc_id)
                label = FIELD_LABELS.get(field_name, field_name)
                self._remember(doc_id)
                return f"{label} của {holder} ({doc_id}): {value}", doc_id

            # Không nêu tên/giá trị cụ thể -> ưu tiên tài liệu đang nói tới (memory)
            if len(value_matches) == 0 and self.last_doc_id:
                mem_doc = self.documents.get(self.last_doc_id)
                if mem_doc:
                    obj = mem_doc
                    for key in json_path:
                        obj = obj.get(key, {}) if isinstance(obj, dict) else None
                        if obj is None:
                            break
                    if obj is not None:
                        holder = mem_doc.get("holder", {}).get("name", self.last_doc_id)
                        label = FIELD_LABELS.get(field_name, field_name)
                        self._remember(self.last_doc_id)
                        return f"{label} của {holder} ({self.last_doc_id}): {obj}", self.last_doc_id

        # Nếu không chỉ định tên người cụ thể, trả về tất cả documents
        if not person_name:
            results = []
            for doc_id, doc_data in self.documents.items():
                obj = doc_data
                for key in json_path:
                    obj = obj.get(key, {}) if isinstance(obj, dict) else None
                    if obj is None:
                        break
                if obj is not None:
                    holder = doc_data.get("holder", {}).get("name", doc_id)
                    label = FIELD_LABELS.get(field_name, field_name)
                    results.append(f"  - {doc_id} ({holder}): {label} = {obj}")
            if results:
                return "\n".join(results), "ALL"
            return None, None

        if best_doc_id and best_score >= 0.4:
            label = FIELD_LABELS.get(field_name, field_name)
            holder = self.documents[best_doc_id].get("holder", {}).get("name", "")
            self._remember(best_doc_id)
            if best_value is None:
                return (
                    f"Không tìm thấy {label.lower()} của {holder} ({best_doc_id}) trong dữ liệu đã trích xuất.",
                    best_doc_id,
                )
            return f"{label} của {holder} ({best_doc_id}): {best_value}", best_doc_id

        return None, None

    def lookup_extra_field(self, question):
        """
        Tra field bổ sung (tự sinh từ mọi dòng "<nhãn>: <giá trị>" trên giấy -
        xem src/generic_field_extractor.py), dùng khi câu hỏi không khớp field
        nào đã khai báo tay trong FIELD_PATTERNS. Khớp bằng cách tìm NHÃN xuất
        hiện trong câu hỏi (đã bỏ dấu, không phân biệt hoa/thường).
        Returns: (answer_text, doc_id) hoặc (None, None)
        """
        q_norm = remove_accents(question).lower()

        candidates = []
        for doc_id, doc_data in self.documents.items():
            for item in doc_data.get("extra_fields", []):
                label_norm = remove_accents(item.get("label", "")).lower().strip()
                if label_norm and len(label_norm) >= 4 and label_norm in q_norm:
                    candidates.append((doc_id, item, label_norm))

        if not candidates:
            return None, None

        # Câu hỏi có thể có điều kiện phụ ngoài nhãn (VD: "...ở Huyện Thủ Đức").
        # Ưu tiên tài liệu mà các từ CÒN LẠI (sau khi bỏ nhãn) trong câu hỏi cũng
        # xuất hiện trong dữ liệu của tài liệu đó (tên/địa chỉ/field khác) - tránh
        # chọn nhầm tài liệu đầu tiên khớp nhãn nhưng không khớp ngữ cảnh còn lại.
        def doc_text(doc_id):
            doc_data = self.documents[doc_id]
            parts = [doc_data.get("holder", {}).get("name", ""), doc_data.get("holder", {}).get("address", "")]
            parts += [h.get("name", "") for h in doc_data.get("holders", [])]
            parts += [it.get("value", "") for it in doc_data.get("extra_fields", [])]
            return remove_accents(" ".join(p for p in parts if p)).lower()

        def overlap_score(doc_id, label_norm):
            residual = q_norm.replace(label_norm, " ")
            words = [w for w in re.findall(r"[a-z0-9]+", residual) if len(w) >= 4]
            text = doc_text(doc_id)
            return sum(1 for w in words if w in text)

        # Câu hỏi nêu rõ mã tài liệu (VD "của DOC_004") -> ưu tiên tuyệt đối,
        # trước cả overlap-score và memory (xem docstring _extract_doc_id_from_query).
        explicit_doc_id = self._extract_doc_id_from_query(question)
        if explicit_doc_id:
            for doc_id, item, _ in candidates:
                if doc_id == explicit_doc_id:
                    return self._format_extra_field_answer(doc_id, item)

        # Nhãn dài hơn thường cụ thể hơn (ít nhầm); trong các nhãn ngang nhau,
        # ưu tiên tài liệu khớp nhiều từ khoá còn lại của câu hỏi hơn.
        candidates.sort(key=lambda c: (overlap_score(c[0], c[2]), len(c[1]["label"])), reverse=True)

        # Trong các nhãn khớp, ưu tiên tài liệu đang nói tới (memory) nếu có
        if self.last_doc_id:
            for doc_id, item, _ in candidates:
                if doc_id == self.last_doc_id:
                    return self._format_extra_field_answer(doc_id, item)

        doc_id, item, _ = candidates[0]
        return self._format_extra_field_answer(doc_id, item)

    def _format_extra_field_answer(self, doc_id, item):
        holder = self.documents[doc_id].get("holder", {}).get("name", doc_id)
        self._remember(doc_id)
        return f"{item['label']} của {holder} ({doc_id}): {item['value']}", doc_id

    def grounding_check(self, llm_response):
        """
        Kiểm tra câu trả lời LLM có khớp với dữ liệu JSON gốc không.
        Nếu phát hiện entity bị sai → sửa lại và flag cảnh báo.
        """
        corrected = llm_response
        corrections = []

        for doc_id, doc_data in self.documents.items():
            # Duyệt qua TỪNG NGƯỜI (hỗ trợ document có nhiều chủ sở hữu - vợ/chồng)
            for person in self._iter_persons(doc_id, doc_data):
                true_name = person.get("name") or ""
                true_id = person.get("id_number") or ""

                if not true_name:
                    continue

                # Kiểm tra tên bị hallucinate: tìm tên gần giống nhưng không khớp chính xác
                for word_boundary in re.finditer(
                    rf"[{_VI_UPPER}][{_VI_LOWER}]+(?:\s+[{_VI_UPPER}][{_VI_LOWER}]+){{1,5}}", corrected
                ):
                    found_name = word_boundary.group(0)
                    similarity = SequenceMatcher(None, found_name.upper(), true_name.upper()).ratio()
                    if 0.6 <= similarity < 1.0 and found_name.upper() != true_name.upper():
                        # Tên gần giống nhưng không khớp → có thể hallucinate
                        corrected = corrected.replace(found_name, true_name)
                        corrections.append(f"'{found_name}' → '{true_name}'")

                # Kiểm tra số CMND/CCCD bị sai
                if true_id:
                    for num_match in re.finditer(r"\d{9,12}", corrected):
                        found_num = num_match.group(0)
                        if len(found_num) == len(true_id) and found_num != true_id:
                            sim = SequenceMatcher(None, found_num, true_id).ratio()
                            if sim >= 0.6:
                                corrected = corrected.replace(found_num, true_id)
                                corrections.append(f"CMND '{found_num}' → '{true_id}'")

            # Kiểm tra ngày biến động (decision_date) bị sai lệch so với change_history
            known_dates = [
                record.get("decision_date")
                for record in doc_data.get("change_history", [])
                if record.get("decision_date")
            ]
            for date_match in re.finditer(r"\d{1,2}/\d{1,2}/\d{4}", corrected):
                found_date = date_match.group(0)
                if found_date in known_dates:
                    continue
                for true_date in known_dates:
                    sim = SequenceMatcher(None, found_date, true_date).ratio()
                    if sim >= 0.6:
                        corrected = corrected.replace(found_date, true_date)
                        corrections.append(f"Ngày '{found_date}' → '{true_date}'")
                        break

        if corrections:
            warning = "\n⚠️ [Grounding Check] Đã tự động sửa lỗi ảo giác: " + ", ".join(corrections)
            return corrected + warning

        return corrected
