import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys

# Tự động chuyển console output sang UTF-8 để không bị lỗi font tiếng Việt trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import yaml
from src.pipeline import DocumentPipeline

# Disable oneDNN to avoid incompatibility issues on Windows CPU
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(base_dir, "configs")
    documents_dir = os.path.join(base_dir, "data", "documents")
    outputs_dir = os.path.join(base_dir, "outputs", "predictions")
    md_dir = os.path.join(base_dir, "outputs", "markdowns")
    reports_dir = os.path.join(base_dir, "outputs", "reports")

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load pipeline configuration
    try:
        pipeline_config = load_yaml(os.path.join(configs_dir, "pipeline.yaml"))
    except Exception as e:
        print(f"Error loading pipeline config: {e}")
        sys.exit(1)
        
    print(f"Loaded pipeline configuration: {pipeline_config.get('pipeline', {}).get('name')}")
    
    # 2. Find document folders
    if not os.path.exists(documents_dir):
        print(f"Error: Documents directory '{documents_dir}' does not exist.")
        return
        
    doc_folders = sorted([d for d in os.listdir(documents_dir) if os.path.isdir(os.path.join(documents_dir, d))])
    
    if not doc_folders:
        print("No document folders found in data/documents.")
        return
        
    print(f"Found {len(doc_folders)} documents to process: {', '.join(doc_folders)}")
    
    # 3. Initialize Pipeline
    print("\nInitializing Document Pipeline (PPStructure + OCR + Extraction)...")
    try:
        pipeline = DocumentPipeline(pipeline_config)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error initializing Pipeline: {e}")
        sys.exit(1)
        
    print("Pipeline ready.\n")
    print("=" * 80)
    print(" PROCESSING DOCUMENTS")
    print("=" * 80)
    
    # 4. Process each document folder
    from src.file_converter import convert_to_images, SUPPORTED_EXTENSIONS

    for doc_id in doc_folders:
        doc_path = os.path.join(documents_dir, doc_id)
        raw_files = sorted(
            os.path.join(doc_path, f) for f in os.listdir(doc_path) if f.lower().endswith(SUPPORTED_EXTENSIONS)
        )

        if not raw_files:
            print(f"Skipping {doc_id}: No supported files found (.png/.jpg/.jpeg/.pdf/.docx).")
            continue

        image_exts = (".png", ".jpg", ".jpeg")
        if all(f.lower().endswith(image_exts) for f in raw_files):
            # Trường hợp phổ biến: toàn ảnh sẵn có - dùng thẳng, không cần convert.
            image_files = raw_files
        else:
            # Có PDF/Word lẫn trong thư mục -> render thành ảnh. KHÔNG tự động
            # xoay trang ở đây (auto_rotate=False): đây là corpus/dữ liệu đã có
            # sẵn, một số mẫu scan gốc nằm ngang HỢP LỆ (khổ giấy scan ngang) -
            # tự ý xoay sẽ phá hỏng dữ liệu đã hiệu chỉnh sẵn theo đúng hướng
            # gốc (bug thật đã gặp). Tính năng tự xoay chỉ áp dụng cho tài liệu
            # MỚI người dùng đưa vào qua webapp/extract_image.py.
            convert_dir = os.path.join(doc_path, "_converted")
            if os.path.isdir(convert_dir):
                import shutil as _shutil
                _shutil.rmtree(convert_dir)
            image_files = []
            for raw in raw_files:
                pages, _ = convert_to_images(raw, convert_dir, auto_rotate=False)
                image_files.extend(pages)
            image_files.sort()

        if not image_files:
            print(f"Skipping {doc_id}: No image files found.")
            continue

        print(f"\nProcessing document: {doc_id} ({len(image_files)} pages)")

        try:
            # Chạy toàn bộ pipeline cho document
            doc_json, page_blocks_dict = pipeline.process_document(doc_id, image_files)
            
            # Format as structured Markdown for RAG
            from src.text_formatter import format_as_markdown
            md_text = format_as_markdown(page_blocks_dict, document_id=doc_id, doc_json=doc_json)
            
            # Save Markdown
            md_file = os.path.join(md_dir, f"{doc_id}.md")
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"    -> Saved markdown to: {md_file}")
            
            # (Optional) Extract using LLM - Vẫn giữ để so sánh hoặc làm fallback
            # print("  - Extracting information using LLM...")
            # from src.llm_extractor import extract_information
            # llm_json = extract_information(doc_id, md_text)
            
            # Save output JSON (từ rule-based pipeline)
            output_file = os.path.join(outputs_dir, f"{doc_id}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(doc_json, f, ensure_ascii=False, indent=2)

            # Sinh báo cáo Markdown DỄ ĐỌC (khác với md_text ở trên - file đó
            # phục vụ chunking/RAG, chứa block OCR thô; file này chỉ trình bày
            # lại JSON đã trích xuất theo mục rõ ràng cho người đọc trực tiếp).
            from src.report_generator import generate_readable_report
            report_text = generate_readable_report(doc_json)
            report_file = os.path.join(reports_dir, f"{doc_id}.md")
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"    -> Saved readable report to: {report_file}")

            print(f"\nSUCCESS: Document {doc_id} result saved to: {output_file}")
            print(json.dumps(doc_json, ensure_ascii=False, indent=2))

            # Tự động nhúng THÊM TỪNG PHẦN đúng tài liệu này vào ChromaDB
            # (không đụng tới các tài liệu khác đã có) - để có thể hỏi-đáp
            # ngay mà không cần chạy tay test_embedding.py.
            from src.embedding_pipeline import add_document_embedding
            _, n_chunks = add_document_embedding(doc_id, markdowns_dir=md_dir)
            print(f"    -> Đã nhúng {n_chunks} chunks của {doc_id} vào ChromaDB.")

        except Exception as e:
            import traceback
            print(f"Error processing document {doc_id}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
