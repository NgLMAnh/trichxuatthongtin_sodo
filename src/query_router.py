"""
Query Router: Phân loại câu hỏi thành field-based hoặc open-ended.
- Field-based: tra trực tiếp JSON, không cần RAG/LLM.
- Open-ended: đi qua pipeline RAG như bình thường.
"""
import os
import re
import json
from difflib import SequenceMatcher

# Bảng chữ cái tiếng Việt tường minh (không dùng range Unicode "À-Ỹ" vì range đó
# vô tình chứa cả chữ thường có dấu như "đ"/"ư", khiến regex nhận nhầm cụm như
# "Thửa đất" là tên người).
_VI_UPPER = "AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÄBCDĐEÉÈẺẼẸÊẾỀỂỄỆËFGHIÍÌỈĨỊÏJKLMNOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÖPQRSTUÚÙỦŨỤƯỨỪỬỮỰÜVWXYÝỲỶỸỴŸZ"
_VI_LOWER = "aáàảãạăắằẳẵặâấầẩẫậäbcdđeéèẻẽẹêếềểễệëfghiíìỉĩịïjklmnoóòỏõọôốồổỗộơớờởỡợöpqrstuúùủũụưứừửữựüvwxyýỳỷỹỵÿz"

# Từ khóa ánh xạ câu hỏi -> trường JSON
FIELD_PATTERNS = {
    "holder_name": [
        r"(?:tên|họ\s*tên|ai\s*là)\s*(?:chủ\s*sở\s*hữu|người\s*sở\s*hữu|chủ\s*đất)",
        r"chủ\s*sở\s*hữu\s*(?:là\s*ai|tên)",
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
    "area_m2": [
        r"(?:diện\s*tích|bao\s*nhiêu\s*m2|mét\s*vuông|bao\s*nhiêu\s*mét)",
    ],
    "change_history": [
        r"(?:biến\s*động|đổi\s*chủ|chuyển\s*nhượng|mua\s*bán|thế\s*chấp|tặng\s*cho|thừa\s*kế)",
        r"(?:lịch\s*sử|quá\s*trình)\s*(?:thay\s*đổi|biến\s*động)?",
        r"(?:số\s*hồ\s*sơ|mã\s*hồ\s*sơ|hồ\s*sơ\s*gốc)",
        r"(?:ngày\s*ký|ngày\s*quyết\s*định|ngày\s*ra\s*quyết\s*định|vào\s*sổ\s*(?:ngày|lúc))",
        r"(?:nơi\s*ký|cơ\s*quan\s*ký|nơi\s*ra\s*quyết\s*định|nơi\s*cấp\s*quyết\s*định)",
        r"(?:ký\s*(?:ở|tại)\s*đâu|quyết\s*định.*ký|ký\s*quyết\s*định)",
    ],
}

# Ánh xạ field name -> đường dẫn trong JSON
FIELD_TO_JSON_PATH = {
    "holder_name": ("holder", "name"),
    "id_number": ("holder", "id_number"),
    "address": ("holder", "address"),
    "birthday": ("holder", "birthday"),
    "parcel_number": ("land_parcel", "parcel_number"),
    "map_sheet_number": ("land_parcel", "map_sheet_number"),
    "area_m2": ("land_parcel", "area_m2"),
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
    "change_history": "Lịch sử biến động",
}


class QueryRouter:
    def __init__(self, predictions_dir="outputs/predictions"):
        self.predictions_dir = predictions_dir
        self.documents = self._load_all_predictions()

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

    def _extract_person_name_from_query(self, question):
        """Trích xuất tên người từ câu hỏi (heuristic)."""
        q = question.strip()
        # Tìm pattern: "của <TÊN>"
        match = re.search(r"(?:của|cho)\s+(?:ông|bà|anh|chị)?\s*(.+?)(?:\?|$|là|ở|tại)", q, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip("?").strip()
        # Tìm pattern: "<TÊN> là chủ sở hữu"
        match = re.search(r"(?:ông|bà)\s+(.+?)(?:\s+là|\s+sở\s*hữu|\s+có|\?|$)", q, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip("?").strip()
        # Tìm pattern: tên viết hoa liên tiếp
        match = re.search(
            rf"([{_VI_UPPER}][{_VI_LOWER}]+(?:\s+[{_VI_UPPER}][{_VI_LOWER}]+){{1,5}})", q
        )
        if match:
            return match.group(1).strip()
        return None

    def _format_change_history_answer(self, doc_id, doc_data):
        """Định dạng change_history (list) thành câu trả lời dạng liệt kê theo thời gian."""
        change_history = doc_data.get("change_history") or []
        if not change_history:
            return None

        holder = doc_data.get("holder", {}).get("name", "")
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

            if not person_name:
                # Không chỉ định chủ/thửa đất cụ thể -> liệt kê biến động của mọi tài liệu
                answers = [
                    self._format_change_history_answer(doc_id, doc_data)
                    for doc_id, doc_data in self.documents.items()
                    if doc_data.get("change_history")
                ]
                if answers:
                    return "\n\n".join(answers), "ALL"
                return None, None

            best_doc_id, best_score = None, 0.0
            for doc_id, doc_data in self.documents.items():
                holder_name = doc_data.get("holder", {}).get("name", "")
                score = self._fuzzy_match_name(person_name, holder_name)
                if doc_data.get("change_history") and score > best_score:
                    best_score, best_doc_id = score, doc_id

            if not best_doc_id or best_score < 0.4:
                return None, None

            answer = self._format_change_history_answer(best_doc_id, self.documents[best_doc_id])
            return answer, best_doc_id

        person_name = self._extract_person_name_from_query(question)
        json_path = FIELD_TO_JSON_PATH.get(field_name)
        if not json_path:
            return None, None

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

            if obj is not None and score > best_score:
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
                return f"{label} của {holder} ({doc_id}): {value}", doc_id

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

        if best_value is not None and best_score >= 0.4:
            label = FIELD_LABELS.get(field_name, field_name)
            holder = self.documents[best_doc_id].get("holder", {}).get("name", "")
            return f"{label} của {holder} ({best_doc_id}): {best_value}", best_doc_id

        return None, None

    def grounding_check(self, llm_response):
        """
        Kiểm tra câu trả lời LLM có khớp với dữ liệu JSON gốc không.
        Nếu phát hiện entity bị sai → sửa lại và flag cảnh báo.
        """
        corrected = llm_response
        corrections = []

        for doc_id, doc_data in self.documents.items():
            holder = doc_data.get("holder", {})
            true_name = holder.get("name", "")
            true_id = holder.get("id_number", "")
            parcel = doc_data.get("land_parcel", {}).get("parcel_number", "")

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

            # Kiểm tra số CMND bị sai
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
