import os
import csv
import sys
import numpy as np

try:
    import fitz  # PyMuPDF
    import cv2
    from PIL import Image
except ImportError as e:
    print(f"Error: Missing required library. {e}")
    print("Please make sure you have installed: pymupdf, opencv-python-headless, pillow, numpy")
    print("Run: .\\.venv\\Scripts\\pip install pymupdf opencv-python-headless pillow numpy")
    sys.exit(1)

# Configuration threshold values
BLUR_THRESHOLD = 100.0       # Laplacian variance below this is considered blurry
LOW_BRIGHTNESS_LIMIT = 40.0   # Average pixel intensity below this is too dark
HIGH_BRIGHTNESS_LIMIT = 240.0 # Average pixel intensity above this is too bright
LOW_CONTRAST_LIMIT = 15.0     # Standard deviation below this is low contrast

def analyze_image_array(img_array):
    """
    Analyzes image quality using numpy and opencv.
    img_array: grayscale or BGR numpy array
    """
    # Convert to grayscale if it's color (3 channels)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array

    # Get dimensions
    height, width = gray.shape[:2]

    # Calculate blur using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = laplacian_var < BLUR_THRESHOLD

    # Calculate brightness (mean of grayscale values)
    brightness = np.mean(gray)
    is_too_dark = brightness < LOW_BRIGHTNESS_LIMIT
    is_too_bright = brightness > HIGH_BRIGHTNESS_LIMIT

    # Calculate contrast (std dev of grayscale values)
    contrast = np.std(gray)
    is_low_contrast = contrast < LOW_CONTRAST_LIMIT

    # Determine overall status
    issues = []
    if is_blurry:
        issues.append(f"Mờ (Độ sắc nét: {laplacian_var:.1f} < {BLUR_THRESHOLD})")
    if is_too_dark:
        issues.append(f"Quá tối (Độ sáng: {brightness:.1f} < {LOW_BRIGHTNESS_LIMIT})")
    if is_too_bright:
        issues.append(f"Quá sáng (Độ sáng: {brightness:.1f} > {HIGH_BRIGHTNESS_LIMIT})")
    if is_low_contrast:
        issues.append(f"Tương phản thấp (Tương phản: {contrast:.1f} < {LOW_CONTRAST_LIMIT})")

    status = "Đạt" if not issues else "Cần kiểm tra"
    
    return {
        "width": width,
        "height": height,
        "sharpness": round(laplacian_var, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "status": status,
        "issues": "; ".join(issues) if issues else "Không có"
    }

def analyze_pdf(pdf_path):
    """
    Opens PDF, renders each page as an image, and analyzes quality.
    """
    results = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return [{
            "filename": os.path.basename(pdf_path),
            "page": "N/A",
            "width": "N/A",
            "height": "N/A",
            "sharpness": "N/A",
            "brightness": "N/A",
            "contrast": "N/A",
            "status": "Lỗi file",
            "issues": f"Không thể mở file PDF: {str(e)}"
        }]

    file_basename = os.path.basename(pdf_path)
    
    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            # Render page to a high-quality image (2.0 zoom factor = 144 DPI)
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert pixmap to numpy array
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            
            metrics = analyze_image_array(img_bgr)
            metrics["filename"] = file_basename
            metrics["page"] = page_num + 1
            results.append(metrics)
        except Exception as e:
            results.append({
                "filename": file_basename,
                "page": page_num + 1,
                "width": "N/A",
                "height": "N/A",
                "sharpness": "N/A",
                "brightness": "N/A",
                "contrast": "N/A",
                "status": "Lỗi trang",
                "issues": f"Lỗi render trang: {str(e)}"
            })
            
    doc.close()
    return results

def analyze_image_file(image_path):
    """
    Loads a static image file and analyzes quality.
    """
    file_basename = os.path.basename(image_path)
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Không thể đọc ảnh (có thể file bị lỗi/hỏng)")
            
        metrics = analyze_image_array(img)
        metrics["filename"] = file_basename
        metrics["page"] = 1
        return [metrics]
    except Exception as e:
        return [{
            "filename": file_basename,
            "page": 1,
            "width": "N/A",
            "height": "N/A",
            "sharpness": "N/A",
            "brightness": "N/A",
            "contrast": "N/A",
            "status": "Lỗi file",
            "issues": f"Không thể mở file ảnh: {str(e)}"
        }]

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir):
        print(f"Lỗi: Thư mục '{data_dir}' không tồn tại.")
        return

    print("=" * 80)
    print(" BẮT ĐẦU KIỂM TRA CHẤT LƯỢNG FILE VÀ ẢNH TRONG THƯ MỤC DATA")
    print("=" * 80)

    # Supported extensions
    pdf_exts = {".pdf"}
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    all_results = []
    
    # Scan the directory
    files = sorted(os.listdir(data_dir))
    files_to_process = [f for f in files if os.path.splitext(f)[1].lower() in (pdf_exts | img_exts)]
    
    if not files_to_process:
        print("Không tìm thấy file ảnh hoặc PDF hợp lệ nào trong thư mục data.")
        return

    print(f"Tìm thấy {len(files_to_process)} file cần xử lý.\n")

    for idx, filename in enumerate(files_to_process, 1):
        file_path = os.path.join(data_dir, filename)
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        print(f"[{idx}/{len(files_to_process)}] Đang xử lý: {filename}...")
        
        if ext in pdf_exts:
            results = analyze_pdf(file_path)
        else:
            results = analyze_image_file(file_path)
            
        all_results.extend(results)

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Tên File':<35} | {'Trang':<5} | {'Kích Thước':<10} | {'Độ Sắc Nét':<10} | {'Trạng Thái':<12} | {'Vấn Đề'}")
    print("-" * 100)
    
    passed_count = 0
    warning_count = 0
    error_count = 0

    for r in all_results:
        size_str = f"{r['width']}x{r['height']}" if isinstance(r['width'], int) else "N/A"
        sharpness_str = f"{r['sharpness']}" if isinstance(r['sharpness'], float) else "N/A"
        
        status = r['status']
        if status == "Đạt":
            passed_count += 1
            status_display = "🟢 Đạt"
        elif status == "Cần kiểm tra":
            warning_count += 1
            status_display = "🟡 Cần KTra"
        else:
            error_count += 1
            status_display = "🔴 Lỗi"

        # Format display filename
        fn = r['filename']
        if len(fn) > 35:
            fn = fn[:32] + "..."
            
        print(f"{fn:<35} | {r['page']:<5} | {size_str:<10} | {sharpness_str:<10} | {status_display:<12} | {r['issues']}")

    print("=" * 100)
    print(f"KẾT LUẬN: Đạt: {passed_count} trang | Cần kiểm tra: {warning_count} trang | Lỗi: {error_count} trang/file.")
    print("=" * 100)

    # Export to CSV
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_report.csv")
    fields = ["filename", "page", "width", "height", "sharpness", "brightness", "contrast", "status", "issues"]
    
    try:
        with open(csv_file, mode="w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nBáo cáo chi tiết đã được xuất ra file: {csv_file}")
    except Exception as e:
        print(f"Lỗi khi lưu file CSV báo cáo: {e}")

if __name__ == "__main__":
    main()
