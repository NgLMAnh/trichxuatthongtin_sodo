"""Test script để xem kết quả chunking từ file Markdown."""
import os, sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.chunking import chunk_document, print_chunks

import json

# Test với cả 3 documents
for doc_id in ["DOC_001", "DOC_002", "DOC_003"]:
    md_path = os.path.join("outputs", "markdowns", f"{doc_id}.md")
    if not os.path.exists(md_path):
        print(f"SKIP: {md_path} not found")
        continue
    
    print(f"\n{'#'*60}")
    print(f" CHUNKING: {doc_id}")
    print(f"{'#'*60}")
    
    chunks = chunk_document(md_path, chunk_size=3000, chunk_overlap=300)
    print_chunks(chunks)
    
    # Lưu output chunking
    chunk_dir = os.path.join("outputs", "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_file = os.path.join(chunk_dir, f"{doc_id}_chunks.json")
    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu các chunk vào: {chunk_file}")
