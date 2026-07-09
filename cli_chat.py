import sys
# Tự động chuyển console output sang UTF-8 để không bị lỗi font tiếng Việt trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from src.vector_store import load_vector_store
from src.query_router import QueryRouter
from src.synonym_expander import expand_query

def main():
    print("==================================================")
    print("🤖 CHATBOT HỎI ĐÁP SỔ ĐỎ (CLI MODE) v2.0")
    print("   [Query Router + Grounding Check + Synonym]")
    print("==================================================")
    
    # --- Khởi tạo Query Router ---
    print("⏳ Đang tải JSON predictions cho Query Router...")
    router = QueryRouter()
    print(f"   → Đã load {len(router.documents)} tài liệu JSON")

    # --- Khởi tạo Vector Store ---
    print("⏳ Đang tải cơ sở dữ liệu Vector (ChromaDB)...")
    try:
        vector_store = load_vector_store()
    except Exception as e:
        print(f"❌ Lỗi tải Vector DB: {e}. Bạn đã chạy bước Embedding (test_embedding.py) chưa?")
        return
        
    # --- Khởi tạo LLM ---
    print("⏳ Đang khởi tạo mô hình AI (Ollama qwen2.5:7b)...")
    try:
        llm = Ollama(model="qwen2.5:7b", temperature=0.0)
    except Exception as e:
        print(f"Lỗi tải Ollama: {e}")
        return

    prompt_template = """Bạn là một trợ lý thông minh chuyên giải đáp các câu hỏi về thông tin Giấy chứng nhận quyền sử dụng đất (Sổ đỏ/Sổ hồng).
Dựa vào các NỘI DUNG TRÍCH XUẤT từ Sổ đỏ dưới đây, hãy trả lời câu hỏi của người dùng bằng tiếng Việt một cách ngắn gọn, chính xác.
Nếu thông tin không có trong NỘI DUNG TRÍCH XUẤT, hãy trả lời trung thực là "Tôi không tìm thấy thông tin này trong tài liệu hiện tại."

ĐẶC BIỆT LƯU Ý:
1. Nếu có thông tin từ nhiều nguồn tài liệu khác nhau, hãy phân loại câu trả lời rõ ràng theo từng tài liệu.
2. ƯU TIÊN SỐ 1: Nếu trong NỘI DUNG TRÍCH XUẤT có phần "# document_summary" hoặc "extracted_fields" (chứa các trường như chu_so_huu, cmnd_cccd, dia_chi, dien_tich_m2...), BẮT BUỘC phải sử dụng thông tin ở phần này vì đây là dữ liệu chính xác nhất.
3. Người ký giấy tờ (Chủ tịch, Phó Chủ tịch, Ủy viên, Chủ hộ, Giám đốc) KHÔNG PHẢI là chủ sở hữu.
4. Nếu người dùng hỏi về Chủ sở hữu và không có phần summary, hãy chú ý tìm người có các từ "Ông:", "Bà:" kèm theo Năm sinh và CMND.
5. LUÔN trả lời ĐÚNG TRỌNG TÂM câu hỏi. Ví dụ: Hỏi địa chỉ thì CHỈ trả lời địa chỉ, không trả lời tên chủ sở hữu.
6. LƯU Ý TỪ VỰNG: "cmnd_cccd" chính là số Chứng minh nhân dân (CMND) hoặc Căn cước công dân (CCCD). "chu_so_huu" là Chủ sở hữu.
7. SAO CHÉP CHÍNH XÁC: Tên người, số CMND, số thửa đất phải được sao chép NGUYÊN VĂN từ dữ liệu, KHÔNG ĐƯỢC tự ý thay đổi hay suy luận.

NỘI DUNG TRÍCH XUẤT:
{context}

CÂU HỎI CỦA NGƯỜI DÙNG: {question}

TRẢ LỜI:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm

    print("\n✅ Khởi tạo thành công! (Gõ 'quit' hoặc 'exit' để thoát)")
    
    while True:
        try:
            print("\n--------------------------------------------------")
            user_input = input("🗣️ Bạn: ")
            if user_input.lower().strip() in ['quit', 'exit', 'q']:
                print("Tạm biệt!")
                break
            if not user_input.strip():
                continue
            
            # ====== BƯỚC 1: Query Router — phân loại câu hỏi ======
            route_type, field_name = router.classify(user_input)
            
            if route_type == "field":
                print(f"\n🔀 [Router] Câu hỏi field-based → Tra cứu trực tiếp JSON (field: {field_name})")
                answer, doc_id = router.lookup_json(user_input, field_name)
                if answer:
                    print(f"\n🤖 Chatbot (JSON Lookup):\n{answer}")
                else:
                    print(f"\n⚠️ Không tìm thấy thông tin '{field_name}' trong JSON. Chuyển sang RAG...")
                    route_type = "rag"  # Fallback sang RAG
            
            if route_type == "rag":
                print(f"\n🔀 [Router] Câu hỏi mở → Đi qua RAG Pipeline")
                
                # ====== BƯỚC 2: Synonym Expansion ======
                expanded_query = expand_query(user_input)
                if expanded_query != user_input:
                    print(f"🔍 [Synonym] Query mở rộng: {expanded_query}")
                
                # ====== BƯỚC 3: Vector Search ======
                print("⏳ Đang tìm kiếm thông tin...")
                results = vector_store.similarity_search(expanded_query, k=4)
                
                context_chunks = []
                for doc in results:
                    doc_id = doc.metadata.get("document_id", "Không rõ")
                    page = doc.metadata.get("page", "Không rõ")
                    context_chunks.append(f"[Nguồn: {doc_id} - Trang: {page}]:\n{doc.page_content}")
                    
                unique_context = list(set(context_chunks))
                combined_context = "\n\n---\n\n".join(unique_context)
                
                print("\n[DEBUG] RAG đã tìm thấy các đoạn văn bản sau:")
                print(combined_context[:500] + "..." if len(combined_context) > 500 else combined_context)
                print("--------------------------------------------------\n")
                
                # ====== BƯỚC 4: LLM Generate ======
                print("⏳ Đang suy nghĩ trả lời...")
                response = chain.invoke({
                    "context": combined_context,
                    "question": user_input
                })
                
                # ====== BƯỚC 5: Grounding Check ======
                checked_response = router.grounding_check(response.strip())
                
                print(f"\n🤖 Chatbot:\n{checked_response}")
            
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()
