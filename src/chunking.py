"""
Chunking module sử dụng LangChain để tách nội dung Markdown có cấu trúc
thành các chunks tối ưu cho RAG pipeline.

Chiến lược chunking:
1. MarkdownHeaderTextSplitter: Tách theo tiêu đề (## page_xxx.png) để mỗi trang
   là một chunk riêng, giữ nguyên metadata về trang nguồn.
2. RecursiveCharacterTextSplitter: Nếu một trang quá dài, tiếp tục tách nhỏ hơn
   nhưng vẫn giữ overlap để không mất ngữ cảnh.
"""

import os
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def chunk_markdown(markdown_text, chunk_size=1000, chunk_overlap=200, document_id="unknown", doc_summary=""):
    """
    Tách nội dung Markdown có cấu trúc thành các chunks tối ưu cho RAG.
    
    Bước 1: Tách theo tiêu đề Markdown (## = trang) để giữ ngữ cảnh trang.
    Bước 2: Nếu một chunk quá dài (> chunk_size), tiếp tục tách nhỏ hơn
            bằng RecursiveCharacterTextSplitter với overlap.
    
    Args:
        markdown_text: Nội dung Markdown đầy đủ của một document.
        chunk_size: Kích thước tối đa mỗi chunk (ký tự).
        chunk_overlap: Số ký tự overlap giữa các chunk liên tiếp.
        
    Returns:
        list[dict]: Mỗi dict chứa 'content' (nội dung chunk) và 'metadata' 
                    (thông tin về trang nguồn).
    """
    
    
    # "##" tương ứng với mỗi trang (## page_001.png, ## page_002.png, ...)
    # "###" tương ứng với mỗi mục (### Mục I - ..., ### Mục II - ...)
    headers_to_split_on = [
        ("##", "page"),
        ("###", "section"),
    ]
    
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # Giữ lại tiêu đề trong nội dung
    )
    
    md_chunks = md_splitter.split_text(markdown_text)
    
    # Bước 2: Tách nhỏ hơn nếu chunk quá dài
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n---\n",    # Phân cách trang (horizontal rule)
            "\nMục I",    # Các mục chính
            "\nMục II",
            "\nMục III",
            "\nMục IV",
            "\nVI-",
            "\n\n",       # Đoạn văn
            "\n",         # Dòng
            " | ",        # Cột trong cùng một dòng
            " ",          # Từ
        ],
        length_function=len,
    )
    
    final_chunks = []
    
    for doc in md_chunks:
        content = doc.page_content
        metadata = dict(doc.metadata)
        
        # Thêm thông tin cơ bản
        metadata["document_id"] = document_id
        
        # Nếu chunk nhỏ hơn chunk_size, giữ nguyên
        if len(content) <= chunk_size:
            # Tiêm metadata vào text để hỗ trợ Embedding
            page_name = metadata.get("page", "unknown")
            section_name = metadata.get("section", "")
            prefix = f"[Tài liệu: {document_id}{doc_summary} - Trang: {page_name}]"
            if section_name:
                prefix = f"[Tài liệu: {document_id}{doc_summary} - Trang: {page_name} - {section_name}]"
            injected_content = f"{prefix}\n{content}"
            
            final_chunks.append({
                "content": injected_content,
                "metadata": metadata,
            })
        else:
            # Tách nhỏ hơn bằng RecursiveCharacterTextSplitter
            sub_docs = text_splitter.create_documents(
                texts=[content],
                metadatas=[metadata],
            )
            for i, sub_doc in enumerate(sub_docs):
                chunk_meta = dict(sub_doc.metadata)
                chunk_meta["sub_chunk"] = i + 1
                
                page_name = chunk_meta.get("page", "unknown")
                section_name = chunk_meta.get("section", "")
                
                prefix = f"[Tài liệu: {document_id}{doc_summary} - Trang: {page_name}]"
                if section_name:
                    prefix = f"[Tài liệu: {document_id}{doc_summary} - Trang: {page_name} - {section_name}]"
                    
                injected_content = f"{prefix} (Phần {i+1})\n{sub_doc.page_content}"
                
                final_chunks.append({
                    "content": injected_content,
                    "metadata": chunk_meta,
                })
    
    return final_chunks

def chunk_document(md_file_path, chunk_size=1000, chunk_overlap=200):
    """
    Đọc file Markdown và trả về danh sách chunks.
    
    Args:
        md_file_path: Đường dẫn tới file .md
        chunk_size: Kích thước tối đa mỗi chunk
        chunk_overlap: Số ký tự overlap
        
    Returns:
        list[dict]: Danh sách chunks với content và metadata.
    """
    with open(md_file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()
    
    document_id = os.path.basename(md_file_path).replace(".md", "")
    
    # Load JSON prediction to enrich context
    import json
    doc_summary = ""
    json_path = os.path.join(os.path.dirname(md_file_path), "..", "predictions", f"{document_id}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                holder_name = _compact_summary_value(data.get("holder", {}).get("name", ""))
                parcel = _compact_summary_value(data.get("land_parcel", {}).get("parcel_number", ""))
                
                summary_parts = []
                if holder_name:
                    summary_parts.append(f"Chủ sở hữu: {holder_name}")
                if parcel:
                    summary_parts.append(f"Thửa đất: {parcel}")
                    
                if summary_parts:
                    doc_summary = " | " + " | ".join(summary_parts)
        except Exception:
            pass

    return chunk_markdown(markdown_text, chunk_size, chunk_overlap, document_id, doc_summary)


def _compact_summary_value(value, max_len=80):
    if not value:
        return ""
    value = " ".join(str(value).split())
    if len(value) > max_len:
        return ""
    return value


def print_chunks(chunks):
    """In ra danh sách chunks để debug."""
    print(f"\n{'='*60}")
    print(f" TOTAL CHUNKS: {len(chunks)}")
    print(f"{'='*60}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"  Metadata: {chunk['metadata']}")
        print(f"  Length: {len(chunk['content'])} chars")
        print(f"  Content preview: {chunk['content'][:150]}...")
        print()
