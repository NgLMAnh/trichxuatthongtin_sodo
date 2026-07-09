import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Lưu trữ parent chunks (không embed, chỉ dùng khi trả context cho LLM)
_parent_store = {}


def create_vector_store(chunks, persist_directory="outputs/chroma_db"):
    """
    Nhận danh sách các chunks và đưa vào Vector Database (Chroma).
    Hỗ trợ parent-child chunking: child embed vào Chroma, parent lưu riêng.
    """
    global _parent_store
    _parent_store = {}

    if not chunks:
        print("Không có chunks nào để đưa vào DB.")
        return None
        
    import shutil
    if os.path.exists(persist_directory):
        print(f"Đang xóa database cũ tại {persist_directory}...")
        shutil.rmtree(persist_directory)
        
    print(f"Đang load embedding model...")
    # Sử dụng BGE-M3: Mô hình đa ngôn ngữ siêu việt, tối ưu RAG
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"batch_size": 2, "normalize_embeddings": True}
    )
    
    # Chuyển đổi định dạng list[dict] sang list[str] và list[dict] metadatas cho Chroma
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    # Lưu parent content (nếu có) vào bộ nhớ
    for i, chunk in enumerate(chunks):
        if "parent_content" in chunk:
            # Key = index trong Chroma
            _parent_store[i] = chunk["parent_content"]
    
    print(f"Đang embedding {len(texts)} chunks và lưu vào {persist_directory}...")
    parent_count = len(_parent_store)
    if parent_count > 0:
        print(f"   → {parent_count} child chunks có parent content")
    
    # Khởi tạo và lưu vào Chroma DB
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )

    # Lưu parent store ra file để load_vector_store có thể dùng
    import json
    parent_file = os.path.join(persist_directory, "parent_store.json")
    serializable = {str(k): v for k, v in _parent_store.items()}
    with open(parent_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)
    
    print("Hoàn tất!")
    return vector_store

def load_vector_store(persist_directory="outputs/chroma_db"):
    """Load Chroma DB đã lưu + parent store."""
    global _parent_store
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"batch_size": 2, "normalize_embeddings": True}
    )
    
    # Load parent store
    import json
    parent_file = os.path.join(persist_directory, "parent_store.json")
    if os.path.exists(parent_file):
        with open(parent_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            _parent_store = {int(k): v for k, v in data.items()}
    
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def get_parent_content(chunk_index):
    """Lấy parent content cho một child chunk (nếu có)."""
    return _parent_store.get(chunk_index)
