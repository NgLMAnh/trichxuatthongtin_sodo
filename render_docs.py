import os
import sys

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF (fitz) is not installed in the active environment.")
    sys.exit(1)

def main():
    # Base directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    doc1_dir = os.path.join(base_dir, "data", "documents")
    output_base_dir = os.path.join(base_dir, "data", "documents", "DOC_004")

    if not os.path.exists(doc1_dir):
        print(f"Error: Directory '{doc1_dir}' does not exist.")
        return

    # List and sort PDF files in DOC1
    files = sorted([f for f in os.listdir(doc1_dir) if f.lower().endswith(".pdf")])
    
    if not files:
        print(f"No PDF files found in '{doc1_dir}'.")
        return

    print(f"Found {len(files)} PDF files in DOC1. Rendering to images...")

    for i, filename in enumerate(files, 1):
        pdf_path = os.path.join(doc1_dir, filename)
        
        # Create folder name like DOC_001, DOC_002, etc.
        folder_name = f"DOC_{i:03d}"
        target_dir = os.path.join(output_base_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\n[{i}/{len(files)}] Processing '{filename}' -> '{folder_name}'...")
        
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                # Render page to an image (zoom=2.0 for clear quality)
                zoom = 2.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Image filename e.g. page_001.png
                img_name = f"page_{page_num + 1:03d}.png"
                img_path = os.path.join(target_dir, img_name)
                
                pix.save(img_path)
                print(f"  - Saved page {page_num + 1}/{num_pages} as '{img_name}'")
                
            doc.close()
            print(f"Finished processing '{filename}'. Images saved in: {target_dir}")
            
        except Exception as e:
            print(f"Error processing '{filename}': {e}")

    print("\nAll files rendered successfully.")

if __name__ == "__main__":
    main()
