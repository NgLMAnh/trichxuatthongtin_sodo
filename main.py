import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import yaml

import re

# Disable oneDNN to avoid incompatibility issues on Windows CPU
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"


from src.ocr_engine import OCREngine
from src.spatial_rules import normalize_text
from src.extractors import extract_fields
from src.normalizers import normalize_fields
from src.document_merger import merge_pages

def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(base_dir, "configs", "template_a")
    documents_dir = os.path.join(base_dir, "data", "documents")
    outputs_dir = os.path.join(base_dir, "outputs", "predictions")
    
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Load configuration
    try:
        template_config = load_yaml(os.path.join(configs_dir, "template.yaml"))
    except Exception as e:
        print(f"Error loading template config: {e}")
        sys.exit(1)
        
    print(f"Loaded template: {template_config.get('template_name')} (ID: {template_config.get('template_id')})")
    
    # 2. Find document folders
    if not os.path.exists(documents_dir):
        print(f"Error: Documents directory '{documents_dir}' does not exist.")
        return
        
    doc_folders = sorted([d for d in os.listdir(documents_dir) if os.path.isdir(os.path.join(documents_dir, d))])
    
    if not doc_folders:
        print("No document folders found in data/documents.")
        return
        
    print(f"Found {len(doc_folders)} documents to process: {', '.join(doc_folders)}")
    
    # 3. Initialize OCREngine
    print("\nInitializing OCR Engine (PaddleOCR)... This might take a few seconds.")
    try:
        # Using CPU (gpu=False) for compatibility
        ocr_engine = OCREngine(languages=['vi'], gpu=False)
    except Exception as e:
        print(f"Error initializing OCR Engine: {e}")
        sys.exit(1)
        
    print("OCR Engine ready.\n")
    print("=" * 80)
    print(" PROCESSING DOCUMENTS")
    print("=" * 80)
    
    # 4. Process each document folder
    for doc_id in doc_folders:
        doc_path = os.path.join(documents_dir, doc_id)
        image_files = sorted([f for f in os.listdir(doc_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print(f"Skipping {doc_id}: No image files found.")
            continue
            
        print(f"\nProcessing document: {doc_id} ({len(image_files)} pages)")
        page_results = []
        
        for img_name in image_files:
            img_path = os.path.join(doc_path, img_name)
            print(f"  - Running OCR on: {img_name}...")
            
            try:
                # Perform OCR
                ocr_results = ocr_engine.run_ocr(img_path)
                
                # Join all OCR text and normalize
                full_text = " ".join([r["text"] for r in ocr_results])
                norm_full_text = normalize_text(full_text)
                
                # Check keyword matches for each page type
                matched_types = []
                page_detection = template_config.get("page_detection", {})
                
                for ptype, config in page_detection.items():
                    keywords = config.get("keywords", [])
                    min_matches = config.get("min_keyword_matches", 1)
                    
                    matches = sum(1 for kw in keywords if re.search(r'\b' + re.escape(normalize_text(kw)) + r'\b', norm_full_text))
                    if matches >= min_matches:
                        matched_types.append(ptype)
                        
                print(f"    -> Matched page types: {matched_types}")
                
                for page_type in matched_types:
                    # Load page config
                    page_config_path = os.path.join(configs_dir, "pages", f"{page_type}.yaml")
                    page_config = load_yaml(page_config_path)
                    
                    # Extract raw fields
                    raw_fields = extract_fields(ocr_results, page_config)
                    
                    # Normalize fields
                    normalized_fields = normalize_fields(raw_fields)
                    
                    print(f"    -> Extracted {page_type} fields: {normalized_fields}")
                    page_results.append({
                        "page_type": page_type,
                        "fields": normalized_fields
                    })
            except Exception as e:
                import traceback
                print(f"    -> Error processing page {img_name}: {e}")
                traceback.print_exc()
                
        # 5. Merge results for this document
        doc_json = merge_pages(doc_id, page_results)
        
        # Save output JSON
        output_file = os.path.join(outputs_dir, f"{doc_id}.json")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(doc_json, f, ensure_ascii=False, indent=2)
            print(f"\nSUCCESS: Document {doc_id} result saved to: {output_file}")
            print(json.dumps(doc_json, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Error saving result for {doc_id}: {e}")
            
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
