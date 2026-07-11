"""
Chuyển đổi file đầu vào (ảnh / PDF / Word) thành danh sách ảnh PNG theo đúng
thứ tự trang, để DocumentPipeline xử lý như bình thường (pipeline vốn chỉ
nhận ảnh). Kèm tự động xoay các trang nằm ngang về khổ dọc chuẩn.
"""
import os
import shutil

from PIL import Image

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".pdf", ".docx")


def auto_rotate_landscape(image_path):
    """
    Nếu ảnh đang nằm NGANG (width > height), xoay 90 độ để đưa về khổ DỌC
    chuẩn của Sổ đỏ - ghi đè luôn vào file gốc. Trả về True nếu đã xoay.

    Lưu ý: không thể biết chắc 100% chiều xoay đúng (cùng/ngược chiều kim
    đồng hồ) chỉ dựa vào kích thước ảnh - đây là giả định hợp lý nhất khi
    không có thêm thông tin (VD EXIF orientation của ảnh chụp). Nếu xoay sai
    chiều, layout/OCR vẫn có thể trích xuất kém - hãy báo lại nếu gặp trường
    hợp này để cải tiến thêm (VD thử OCR cả 2 chiều rồi chọn chiều tốt hơn).
    """
    with Image.open(image_path) as img:
        if img.width <= img.height:
            return False
        rotated = img.rotate(-90, expand=True)  # -90 = xoay theo chiều kim đồng hồ
        rotated.save(image_path)
    return True


def _extract_pdf_pages(file_path, output_dir):
    import fitz  # PyMuPDF

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    paths = []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=200)
            page_path = os.path.join(output_dir, f"{base_name}_page{i:02d}.png")
            pix.save(page_path)
            paths.append(page_path)
    return paths


def _extract_docx_images(file_path, output_dir):
    """
    Trích các ảnh nhúng trong file Word (trường hợp phổ biến: ảnh Sổ đỏ được
    scan rồi dán/chèn vào Word) theo ĐÚNG THỨ TỰ xuất hiện trong tài liệu -
    quan trọng để giữ đúng thứ tự trang khi có nhiều ảnh.
    """
    import docx
    from docx.oxml.ns import qn

    document = docx.Document(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    image_parts = {
        rel_id: rel.target_part
        for rel_id, rel in document.part.rels.items()
        if "image" in rel.reltype
    }
    if not image_parts:
        raise ValueError("Không tìm thấy ảnh nào được nhúng trong file Word.")

    # Duyệt theo thứ tự xuất hiện thật trong nội dung (không phải thứ tự rels,
    # thứ tự đó không đảm bảo khớp thứ tự trang).
    ordered_rel_ids = []
    for blip in document.element.body.iter(qn("a:blip")):
        rel_id = blip.get(qn("r:embed"))
        if rel_id and rel_id in image_parts and rel_id not in ordered_rel_ids:
            ordered_rel_ids.append(rel_id)
    for rel_id in image_parts:  # ảnh hiếm khi không bắt được qua XML - thêm vào cuối
        if rel_id not in ordered_rel_ids:
            ordered_rel_ids.append(rel_id)

    paths = []
    for i, rel_id in enumerate(ordered_rel_ids, start=1):
        part = image_parts[rel_id]
        _, orig_ext = os.path.splitext(part.partname)
        ext = orig_ext.lower() if orig_ext else ".png"
        page_path = os.path.join(output_dir, f"{base_name}_page{i:02d}{ext}")
        with open(page_path, "wb") as f:
            f.write(part.blob)

        if ext not in (".png", ".jpg", ".jpeg"):
            with Image.open(page_path) as im:
                new_path = os.path.splitext(page_path)[0] + ".png"
                im.convert("RGB").save(new_path)
            os.remove(page_path)
            page_path = new_path

        paths.append(page_path)

    return paths


def convert_to_images(file_path, output_dir, auto_rotate=False):
    """
    Chuyển 1 file đầu vào (.png/.jpg/.jpeg/.pdf/.docx) thành danh sách đường
    dẫn ảnh PNG/JPG theo đúng thứ tự trang trong output_dir.

    auto_rotate=False (mặc định): KHÔNG tự xoay. Thực tế nhiều ảnh/scan Sổ đỏ
    có tỷ lệ khung hình rộng hơn cao (landscape) một cách HỢP LỆ - nội dung
    vẫn đọc được bình thường, không phải ảnh bị chụp/scan lệch 90°. Quy tắc
    "rộng hơn cao thì xoay" từng gây lỗi trích xuất TRỐNG HOÀN TOÀN cho những
    ảnh này (bug thật đã gặp 2 lần: corpus DOC_001 và 1 ảnh người dùng upload)
    vì xoay nhầm biến ảnh vốn đọc được thành ảnh sai chiều thật sự. Chỉ bật
    auto_rotate=True khi bạn CHẮC CHẮN ảnh bị chụp/scan lệch ngang thật.

    Returns: (image_paths, rotated_filenames) - rotated_filenames là tên các
    trang đã bị xoay, để báo lại cho người dùng biết.
    """
    os.makedirs(output_dir, exist_ok=True)
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".png", ".jpg", ".jpeg"):
        dest = os.path.join(output_dir, os.path.basename(file_path))
        if os.path.abspath(dest) != os.path.abspath(file_path):
            shutil.copyfile(file_path, dest)
        paths = [dest]
    elif ext == ".pdf":
        paths = _extract_pdf_pages(file_path, output_dir)
    elif ext == ".docx":
        paths = _extract_docx_images(file_path, output_dir)
    else:
        raise ValueError(
            f"Định dạng file không được hỗ trợ: '{ext}'. Chỉ hỗ trợ .png/.jpg/.jpeg/.pdf/.docx"
        )

    rotated = []
    if auto_rotate:
        rotated = [os.path.basename(p) for p in paths if auto_rotate_landscape(p)]
    return paths, rotated
