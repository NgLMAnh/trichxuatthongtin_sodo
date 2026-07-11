"""
Đóng gói bước "chunk tài liệu -> nhúng vào ChromaDB" thành các hàm dùng chung,
gọi TỰ ĐỘNG mỗi khi có tài liệu mới được trích xuất (qua main.py,
extract_image.py, hoặc API /api/extract của webapp) - không cần chạy tay
test_chunking.py/test_embedding.py như trước.

Cơ chế THÊM TỪNG PHẦN (add_document_embedding): chỉ chunk + nhúng lại ĐÚNG 1
tài liệu vừa trích xuất, dùng upsert_document_chunks() để xoá riêng chunk cũ
của tài liệu đó (nếu trích xuất lại) rồi thêm chunk mới - không đụng tới các
tài liệu khác đã có trong DB. Đây là cách dùng MẶC ĐỊNH, phù hợp khi số tài
liệu tăng dần theo thời gian (không phải rebuild lại toàn bộ mỗi lần).

rebuild_embeddings() (chunk lại + nhúng lại TOÀN BỘ) vẫn được giữ lại như một
tiện ích thủ công, dùng khi cần đồng bộ lại từ đầu (VD sau khi sửa logic
chunking, hoặc phát hiện DB bị lệch dữ liệu) - KHÔNG còn được gọi tự động.
"""
import json
import os

from src.chunking import chunk_document
from src.vector_store import create_vector_store, upsert_document_chunks


def add_document_embedding(
    document_id,
    markdowns_dir="outputs/markdowns",
    chunks_dir="outputs/chunks",
    chroma_dir="outputs/chroma_db",
):
    """Chunk + nhúng THÊM/CẬP NHẬT đúng 1 tài liệu vào ChromaDB đã có (thêm
    từng phần - không đụng tới các tài liệu khác). Trả về (vector_store,
    số_chunks) của riêng tài liệu này; (None, 0) nếu không tìm thấy Markdown."""
    md_path = os.path.join(markdowns_dir, f"{document_id}.md")
    if not os.path.exists(md_path):
        return None, 0

    os.makedirs(chunks_dir, exist_ok=True)
    chunks = chunk_document(md_path)

    chunk_path = os.path.join(chunks_dir, f"{document_id}_chunks.json")
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    vector_store = upsert_document_chunks(document_id, chunks, persist_directory=chroma_dir)
    return vector_store, len(chunks)


def rebuild_embeddings(
    markdowns_dir="outputs/markdowns",
    chunks_dir="outputs/chunks",
    chroma_dir="outputs/chroma_db",
):
    """[Tiện ích thủ công] Chunk lại toàn bộ Markdown hiện có và XÂY LẠI TOÀN
    BỘ ChromaDB từ đầu. Không còn được gọi tự động sau mỗi lần trích xuất
    (xem add_document_embedding) - dùng khi cần đồng bộ lại toàn bộ, VD sau
    khi sửa logic chunking hoặc nghi ngờ DB bị lệch dữ liệu.
    Trả về (vector_store, số_tài_liệu, tổng_số_chunks); (None, 0, 0) nếu
    không có tài liệu nào."""
    if not os.path.isdir(markdowns_dir):
        return None, 0, 0

    os.makedirs(chunks_dir, exist_ok=True)

    all_chunks = []
    doc_count = 0
    for fname in sorted(os.listdir(markdowns_dir)):
        if not fname.endswith(".md"):
            continue
        doc_id = fname[:-3]
        md_path = os.path.join(markdowns_dir, fname)
        chunks = chunk_document(md_path)

        chunk_path = os.path.join(chunks_dir, f"{doc_id}_chunks.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        all_chunks.extend(chunks)
        doc_count += 1

    if not all_chunks:
        return None, 0, 0

    vector_store = create_vector_store(all_chunks, persist_directory=chroma_dir)
    return vector_store, doc_count, len(all_chunks)


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    _, docs, chunks = rebuild_embeddings()
    print(f"Đã nhúng lại {docs} tài liệu ({chunks} chunks) vào ChromaDB.")
