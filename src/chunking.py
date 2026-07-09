"""
Chunking module sử dụng LangChain để tách nội dung Markdown có cấu trúc
thành các chunks tối ưu cho RAG pipeline.

Chiến lược chunking v2 (Structure-aware + Parent-Child):
1. MarkdownHeaderTextSplitter: Tách theo tiêu đề (## page, ### section)
   để mỗi mục (Mục I, II, III, IV) là một chunk riêng.
2. Parent-Child: Nếu section quá dài (> max_child_size), tạo child chunks
   nhỏ để embed chính xác, nhưng giữ parent chunk nguyên vẹn.
3. Child chunks dùng cho Vector Search, parent chunks dùng đưa context cho LLM.
"""

import os
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Giới hạn context window của BGE-M3 (8192 tokens ≈ ~6000 ký tự tiếng Việt)
MAX_EMBED_CHARS = 6000
# Kích thước child chunk để embed chính xác
CHILD_CHUNK_SIZE = 1500
CHILD_CHUNK_OVERLAP = 200


def chunk_markdown(markdown_text, document_id="unknown", doc_summary=""):
    """
    Tách nội dung Markdown có cấu trúc thành các chunks tối ưu cho RAG.
    
    Chiến lược Structure-aware:
    - Bước 1: Tách theo ranh giới section (### = mục).
    - Bước 2: Nếu section nhỏ hơn MAX_EMBED_CHARS → giữ nguyên (1 chunk = 1 section).
    - Bước 3: Nếu section quá dài → tạo parent-child chunks.
    
    Returns:
        list[dict]: Mỗi dict chứa 'content', 'metadata', và tùy chọn 'parent_content'.
    """
    
    headers_to_split_on = [
        ("##", "page"),
        ("###", "section"),
    ]
    
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    
    md_chunks = md_splitter.split_text(markdown_text)
    
    # RecursiveCharacterTextSplitter cho child chunks
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=[
            "\n---\n",
            "\n#### ",      # Ranh giới block
            "\n\n",
            "\n",
            " | ",
            " ",
        ],
        length_function=len,
    )
    
    final_chunks = []
    
    for doc in md_chunks:
        content = doc.page_content
        metadata = dict(doc.metadata)
        metadata["document_id"] = document_id
        
        page_name = metadata.get("page", "unknown")
        section_name = metadata.get("section", "")
        
        prefix = f"[Tài liệu: {document_id}{doc_summary} - Trang: {page_name}]"
        if section_name:
            prefix = f"[Tài liệu: {document_id}{doc_summary} - Trang: {page_name} - {section_name}]"
        
        injected_content = f"{prefix}\n{content}"
        
        if len(injected_content) <= MAX_EMBED_CHARS:
            # Section nhỏ → giữ nguyên, 1 chunk = 1 section
            final_chunks.append({
                "content": injected_content,
                "metadata": metadata,
            })
        else:
            # Section quá dài → Parent-Child chunking
            # Parent: toàn bộ section (lưu riêng, không embed)
            parent_content = injected_content
            
            # Child: cắt nhỏ để embed chính xác
            sub_docs = child_splitter.create_documents(
                texts=[content],
                metadatas=[metadata],
            )
            
            for i, sub_doc in enumerate(sub_docs):
                chunk_meta = dict(sub_doc.metadata)
                chunk_meta["sub_chunk"] = i + 1
                chunk_meta["has_parent"] = True
                
                child_prefix = f"{prefix} (Phần {i+1})"
                child_content = f"{child_prefix}\n{sub_doc.page_content}"
                
                final_chunks.append({
                    "content": child_content,
                    "metadata": chunk_meta,
                    "parent_content": parent_content,  # Lưu parent để đưa cho LLM
                })
    
    return final_chunks


def chunk_document(md_file_path):
    """
    Đọc file Markdown và trả về danh sách chunks.
    Sử dụng structure-aware chunking (không cần chunk_size cứng).
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

    return chunk_markdown(markdown_text, document_id, doc_summary)


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
        has_parent = "parent_content" in chunk
        print(f"  Has Parent: {has_parent}")
        if has_parent:
            print(f"  Parent Length: {len(chunk['parent_content'])} chars")
        print(f"  Content preview: {chunk['content'][:150]}...")
        print()
