"""Test script để nhúng các chunks vào ChromaDB."""
import os, sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vector_store import create_vector_store

def main():
    all_chunks = []
    
    # Đọc các file JSON chứa chunk
    chunk_dir = os.path.join("outputs", "chunks")
    if not os.path.exists(chunk_dir):
        print("Thư mục outputs/chunks không tồn tại. Vui lòng chạy test_chunking.py trước.")
        return
        
    for doc_id in ["DOC_001", "DOC_002", "DOC_003"]:
        chunk_file = os.path.join(chunk_dir, f"{doc_id}_chunks.json")
        if os.path.exists(chunk_file):
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
                
    if not all_chunks:
        print("Không tìm thấy chunk nào.")
        return
        
    print(f"Tổng số chunks đã thu thập: {len(all_chunks)}")
    
    # Tạo vector store
    db_dir = os.path.join("outputs", "chroma_db")
    vector_store = create_vector_store(all_chunks, persist_directory=db_dir)
    
    if vector_store:
        # Thử một truy vấn nhỏ để test
        query = "Ai là chủ sở hữu của thửa đất 169?"
        print(f"\nTruy vấn thử nghiệm: '{query}'")
        results = vector_store.similarity_search_with_score(query, k=2)
        
        from src.vector_store import get_parent_content
        for i, (doc, score) in enumerate(results):
            print(f"\n--- Kết quả {i+1} (Score: {score:.4f}) ---")
            print(f"Metadata: {doc.metadata}")
            print(f"Trích đoạn: {doc.page_content[:200]}...")
            
            # Kiểm tra xem chunk này có parent content không
            # ChromaDB lưu nội dung parent trong _parent_store theo index nội bộ, 
            # tuy nhiên để test đơn giản ta chỉ check metadata
            if doc.metadata.get("has_parent"):
                print("Có Parent Content (Section đầy đủ)")

if __name__ == "__main__":
    main()
