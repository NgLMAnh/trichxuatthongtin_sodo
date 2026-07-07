import os
import json
import re

def extract_json_from_llm_response(response_text):
    """
    Extracts JSON from LLM response which might contain markdown code blocks.
    """
    # Remove markdown code block markers
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = response_text.strip()
        
    try:
        return json.loads(json_str)
    except Exception as e:
        print(f"Error parsing JSON from LLM: {e}")
        return None

def extract_information(document_id, markdown_content):
    """
    Uses Local LLM (Ollama) to extract structured JSON from the markdown text of a document.
    """
    try:
        import requests
        # Kiểm tra xem Ollama có đang chạy không
        response = requests.get("http://localhost:11434/")
        if response.status_code != 200:
            raise Exception("Ollama server is not responding.")
    except Exception as e:
        print(f"Lỗi kết nối Ollama: {e}. Đảm bảo bạn đã khởi động Ollama.")
        return {
            "document_id": document_id,
            "error": "Ollama not running",
            "holder": {"name": None, "id_number": None, "address": None, "birthday": None},
            "land_parcel": {"parcel_number": None, "map_sheet_number": None, "area_m2": None}
        }
        
    try:
        from langchain_community.llms import Ollama
        from langchain_core.prompts import PromptTemplate
    except ImportError:
        print("WARNING: 'langchain_community' package not installed.")
        return None

    # Khởi tạo mô hình Ollama (qwen2.5:1.5b đã được tải)
    llm = Ollama(model="qwen2.5:1.5b", temperature=0.0)
    
    prompt_template = """Bạn là một chuyên gia trích xuất thông tin từ Giấy chứng nhận quyền sử dụng đất (Sổ đỏ/Sổ hồng).
Dưới đây là nội dung đã được OCR (chuyển thành văn bản) của giấy tờ {document_id}.
Văn bản có thể bị lỗi chính tả hoặc OCR sai đôi chút, hãy cố gắng sửa lỗi logic.

Lưu ý quan trọng:
1. "Chủ sở hữu" hiện tại là người cuối cùng có tên trong phần "Thay đổi về chủ" (nếu có). Nếu không có, đó là người ở Mục I (Chủ sở hữu).
2. Người ký giấy chứng nhận (Chủ tịch, Phó Chủ tịch, Ủy viên, ví dụ có các chữ "KT. CHỦ TỊCH", "PHÓ CHỦ TỊCH") KHÔNG PHẢI là chủ sở hữu. Nếu một tên nằm gần chức vụ, hãy bỏ qua. Chủ sở hữu thực sự thường đi kèm với "Ông:", "Bà:", Năm sinh, hoặc CMND/CCCD.
3. CMND/CCCD thường có 9 hoặc 12 số.
4. Diện tích thửa đất (area_m2) là một con số, ví dụ 120.5

Hãy trích xuất thông tin và trả về DUY NHẤT một chuỗi JSON hợp lệ, không kèm bất kỳ giải thích nào. Định dạng như sau:
{{
  "document_id": "{document_id}",
  "holder": {{
    "name": "Tên chủ sở hữu hiện tại",
    "id_number": "Số CMND/CCCD",
    "address": "Địa chỉ thường trú",
    "birthday": "Năm sinh (hoặc Ngày tháng năm sinh)"
  }},
  "land_parcel": {{
    "parcel_number": "Số thửa đất",
    "map_sheet_number": "Tờ bản đồ số",
    "area_m2": <số float>
  }}
}}

NỘI DUNG SỔ ĐỎ:
{markdown_content}

KẾT QUẢ JSON:"""

    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    try:
        print(f"  -> Đang gọi LLM (Ollama - qwen2.5:1.5b) trích xuất cho {document_id}...")
        response_text = chain.invoke({
            "document_id": document_id,
            "markdown_content": markdown_content
        })
        return extract_json_from_llm_response(response_text)
    except Exception as e:
        print(f"LLM Local Error: {e}")
        return None
