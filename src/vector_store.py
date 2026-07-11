import json
import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Lưu trữ parent chunks (không embed, chỉ dùng khi trả context cho LLM).
# Key = "<document_id>_<index>" (string), khớp với id dùng để add_texts vào Chroma -
# cho phép xoá/ghi đè đúng phần của 1 tài liệu mà không đụng tài liệu khác.
_parent_store = {}

# Cache model embedding (BGE-M3, ~2.2GB) - KHÔNG tạo mới mỗi lần gọi. Trước đây
# mỗi lần create_vector_store/upsert_document_chunks/load_vector_store chạy
# đều tự load lại model từ đầu (dù đã load rồi), gây chậm rõ rệt mỗi khi có
# tài liệu mới trích xuất (auto-embedding) hoặc mỗi lần hỏi-đáp đầu tiên.
_embeddings_cache = None


def _get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        # Sử dụng BGE-M3: Mô hình đa ngôn ngữ siêu việt, tối ưu RAG
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={"batch_size": 2, "normalize_embeddings": True},
        )
    return _embeddings_cache


def _parent_file(persist_directory):
    return os.path.join(persist_directory, "parent_store.json")


def _load_parent_map(persist_directory):
    path = _parent_file(persist_directory)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_parent_map(persist_directory, parent_map):
    os.makedirs(persist_directory, exist_ok=True)
    with open(_parent_file(persist_directory), "w", encoding="utf-8") as f:
        json.dump(parent_map, f, ensure_ascii=False)


def create_vector_store(chunks, persist_directory="outputs/chroma_db"):
    """
    XÂY LẠI TOÀN BỘ ChromaDB từ đầu (xoá DB cũ). Dùng cho lần khởi tạo đầu
    tiên hoặc khi cần rebuild thủ công toàn bộ (VD sau khi sửa logic chunking).
    Cho việc thêm 1 tài liệu MỚI vào DB đã có, dùng upsert_document_chunks()
    thay vì hàm này - hàm này luôn xoá sạch dữ liệu cũ.
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
    embeddings = _get_embeddings()

    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    # id ổn định theo document_id để về sau có thể xoá/ghi đè đúng phần của
    # từng tài liệu (xem upsert_document_chunks) thay vì phải rebuild toàn bộ.
    ids = [f"{m.get('document_id', 'unknown')}_{i}" for i, m in enumerate(metadatas)]

    for chunk_id, chunk in zip(ids, chunks):
        if "parent_content" in chunk:
            _parent_store[chunk_id] = chunk["parent_content"]

    print(f"Đang embedding {len(texts)} chunks và lưu vào {persist_directory}...")
    if _parent_store:
        print(f"   → {len(_parent_store)} child chunks có parent content")

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_directory,
    )

    _save_parent_map(persist_directory, _parent_store)
    print("Hoàn tất!")
    return vector_store


def upsert_document_chunks(document_id, chunks, persist_directory="outputs/chroma_db"):
    """
    Thêm/CẬP NHẬT các chunk của 1 tài liệu vào ChromaDB đã có, KHÔNG đụng tới
    dữ liệu của các tài liệu khác - dùng khi trích xuất thêm 1 tài liệu mới
    (hoặc trích xuất lại 1 tài liệu cũ) thay vì phải chunk lại + nhúng lại
    toàn bộ hệ thống mỗi lần (create_vector_store).

    Nếu DB chưa tồn tại, tự tạo mới (bootstrap) với đúng các chunk này.
    Nếu tài liệu này đã từng được nhúng trước đó, xoá các chunk CŨ của riêng
    nó trước khi thêm chunk MỚI - tránh chunk cũ/mới lẫn lộn khi tài liệu
    được trích xuất lại (VD OCR lại sau khi sửa bug).
    """
    global _parent_store

    embeddings = _get_embeddings()

    if not os.path.exists(persist_directory):
        return create_vector_store(chunks, persist_directory=persist_directory)

    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Xoá chunk CŨ của đúng tài liệu này (nếu có) - Chroma cho phép xoá theo
    # điều kiện metadata, không cần biết trước id cụ thể.
    try:
        vector_store.delete(where={"document_id": document_id})
    except Exception:
        pass  # DB rỗng hoặc chưa từng có tài liệu này - không sao.

    parent_map = _load_parent_map(persist_directory)
    # Xoá parent content cũ của tài liệu này (id có dạng "<document_id>_<i>")
    prefix = f"{document_id}_"
    parent_map = {k: v for k, v in parent_map.items() if not k.startswith(prefix)}

    if chunks:
        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{prefix}{i}" for i in range(len(chunks))]
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        for chunk_id, c in zip(ids, chunks):
            if "parent_content" in c:
                parent_map[chunk_id] = c["parent_content"]

    _save_parent_map(persist_directory, parent_map)
    _parent_store = parent_map
    return vector_store


def load_vector_store(persist_directory="outputs/chroma_db"):
    """Load Chroma DB đã lưu + parent store."""
    global _parent_store
    embeddings = _get_embeddings()
    _parent_store = _load_parent_map(persist_directory)
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def get_parent_content(chunk_id):
    """Lấy parent content cho một child chunk theo id (VD 'DOC_001_3')."""
    return _parent_store.get(chunk_id)
