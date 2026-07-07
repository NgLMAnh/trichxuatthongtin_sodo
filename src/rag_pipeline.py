import json
import requests
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from src.vector_store import load_vector_store

# Prompt để LLM trích xuất JSON dựa trên ngữ cảnh cung cấp
PROMPT_TEMPLATE = """Bạn là một chuyên gia trích xuất dữ liệu từ Giấy chứng nhận quyền sử dụng đất (Sổ đỏ) của Việt Nam.
Dựa vào NGỮ CẢNH (Context) dưới đây, hãy trích xuất các thông tin sau và trả về DUY NHẤT một chuỗi JSON hợp lệ. KHÔNG giải thích, KHÔNG thêm bất kỳ chữ nào ngoài JSON.

NGỮ CẢNH:
{context}

YÊU CẦU TRÍCH XUẤT (định dạng JSON):
{{
  "holder": {{
    "name": "Tên chủ sở hữu (nếu nhiều người thì cách nhau bằng dấu phẩy, null nếu không có)",
    "id_number": "Số CMND/CCCD (null nếu không có)",
    "address": "Địa chỉ thường trú (null nếu không có)",
    "birthday": "Năm sinh hoặc ngày sinh (null nếu không có)"
  }},
  "land_parcel": {{
    "parcel_number": "Số thửa đất (ví dụ: 169, g-169, 360, null nếu không có)",
    "map_sheet_number": "Tờ bản đồ số (null nếu không có)",
    "area_m2": "Diện tích đất bằng số (ví dụ: 41.9, 699, null nếu không có)"
  }}
}}

KẾT QUẢ JSON CỦA BẠN:"""

def is_ollama_running():
    """Kiểm tra xem Ollama có đang chạy trên máy không."""
    try:
        response = requests.get("http://localhost:11434/")
        return response.status_code == 200
    except:
        return False

def extract_information_rag(document_id, query_hints=None, model_name="qwen2.5:1.5b"):
    """
    Sử dụng Local LLM (Ollama) và ChromaDB (RAG) để trích xuất thông tin JSON.
    """
    if not is_ollama_running():
        return {
            "document_id": document_id,
            "error": "Ollama is not running. Please start Ollama first."
        }

    # 1. Load Vector DB
    vector_store = load_vector_store()
    
    # 2. Sinh các query để lấy context tốt nhất
    queries = [
        "thông tin người sở hữu, chủ hộ, tên, CMND, năm sinh, địa chỉ",
        "thực trạng nhà ở, đất ở, thửa đất số, tờ bản đồ số, diện tích m2"
    ]
    if query_hints:
        queries.extend(query_hints)
        
    context_chunks = []
    # Chỉ lấy các chunk thuộc về document_id này
    # Tuy nhiên ChromaDB mặc định của chúng ta chứa tất cả doc, ta nên filter theo metadata
    # Lấy k=3 cho mỗi query để đảm bảo bao phủ đủ context
    for q in queries:
        # ChromaDB Hỗ trợ filter metadata
        results = vector_store.similarity_search(q, k=3, filter={"page": {"$regex": ".*"}})
        
        # Chỉ giữ lại các kết quả thuộc về document_id hiện tại
        for doc in results:
            # Metadata có dạng {'page': 'page_001.png', ...}
            # Trong code chunking hiện tại không lưu doc_id vào metadata.
            # RÚT KINH NGHIỆM: Nếu có doc_id thì tốt. Tạm thời gom hết kết quả tìm được.
            context_chunks.append(doc.page_content)
            
    # Xóa trùng lặp chunk
    unique_context = list(set(context_chunks))
    combined_context = "\n\n---\n\n".join(unique_context)
    
    # 3. Khởi tạo Local LLM qua Ollama
    llm = Ollama(model=model_name, temperature=0.0)
    
    # 4. Gửi Prompt cho LLM
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    
    print(f"\n[RAG] Đang trích xuất thông tin cho {document_id} bằng {model_name}...")
    try:
        response_text = chain.invoke({"context": combined_context})
        
        # Tiền xử lý để loại bỏ các Markdown tag nếu LLM sinh ra
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()
            
        result = json.loads(json_str)
        result["document_id"] = document_id
        return result
    except Exception as e:
        print(f"Lỗi khi parse JSON từ LLM: {e}")
        return {
            "document_id": document_id,
            "error": str(e),
            "raw_response": response_text if 'response_text' in locals() else ""
        }
