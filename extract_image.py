"""
CLI trích xuất NHANH cho 1 ảnh/PDF/Word Sổ đỏ ĐƯA TỪ NGOÀI VÀO - không cần đặt
sẵn vào data/documents/ như main.py. Dùng khi có 1 tài liệu mới muốn trích
xuất ngay, không cần hỏi-đáp: chỉ đưa đường dẫn, hệ thống tự render PDF/Word
thành ảnh (nếu cần), tự xoay trang nằm ngang về khổ dọc, OCR + trích xuất toàn
bộ thông tin, in ra JSON + Markdown dễ đọc.

Cách dùng:
    python extract_image.py <đường/dẫn/ảnh.jpg> [<ảnh_trang_2.jpg> ...]
    python extract_image.py <đường/dẫn/file.pdf>
    python extract_image.py <đường/dẫn/file.docx>
    python extract_image.py <thư_mục_chứa_nhiều_trang>/

Kết quả được lưu vào outputs/predictions/<DOC_ID>.json và
outputs/reports/<DOC_ID>.md (DOC_ID tự sinh từ tên ảnh/thư mục đầu vào, hoặc
truyền qua --doc-id).
"""
import argparse
import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import yaml

from src.pipeline import DocumentPipeline
from src.report_generator import generate_readable_report
from src.text_formatter import format_as_markdown


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_image_files(inputs, convert_dir):
    """Thu thập đường dẫn ảnh từ inputs (ảnh/PDF/Word/thư mục), tự động
    render PDF/Word thành ảnh (lưu vào convert_dir) và xoay trang nằm ngang.
    Trả về (danh_sách_đường_dẫn_ảnh, danh_sách_trang_đã_xoay)."""
    from src.file_converter import convert_to_images, SUPPORTED_EXTENSIONS

    raw_inputs = []
    for path in inputs:
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                    raw_inputs.append(os.path.join(path, fname))
        elif os.path.isfile(path):
            if not path.lower().endswith(SUPPORTED_EXTENSIONS):
                raise ValueError(
                    f"Định dạng không được hỗ trợ: {path} (chỉ .png/.jpg/.jpeg/.pdf/.docx)"
                )
            raw_inputs.append(path)
        else:
            raise FileNotFoundError(f"Không tìm thấy: {path}")

    files, rotated_pages = [], []
    for path in raw_inputs:
        pages, rotated = convert_to_images(path, convert_dir)
        files.extend(pages)
        rotated_pages.extend(rotated)

    if not files:
        raise ValueError("Không tìm thấy ảnh nào để xử lý.")
    return files, rotated_pages


def default_doc_id(inputs):
    first = inputs[0]
    base = os.path.basename(os.path.normpath(first))
    name = os.path.splitext(base)[0]
    return name.upper().replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="Trích xuất thông tin Sổ đỏ từ 1 ảnh bên ngoài (không cần hỏi-đáp).")
    parser.add_argument("images", nargs="+", help="Đường dẫn ảnh (1 hoặc nhiều trang) hoặc thư mục chứa ảnh.")
    parser.add_argument("--doc-id", default=None, help="Mã tài liệu tuỳ chỉnh (mặc định: tự sinh từ tên file/thư mục).")
    parser.add_argument("--no-save", action="store_true", help="Chỉ in kết quả ra màn hình, không lưu file.")
    parser.add_argument(
        "--embed", action="store_true",
        help="Sinh thêm Markdown RAG + nhúng vào ChromaDB để có thể hỏi-đáp về tài liệu này qua chatbot "
             "(chậm hơn - tải model embedding BGE-M3 lần đầu ~10-15s). Mặc định TẮT, chỉ trích xuất JSON/report.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_config = load_yaml(os.path.join(base_dir, "configs", "pipeline.yaml"))

    import tempfile
    convert_dir = tempfile.mkdtemp(prefix="extract_image_")
    image_files, rotated_pages = collect_image_files(args.images, convert_dir)
    doc_id = args.doc_id or default_doc_id(args.images)

    if rotated_pages:
        print(f"Đã tự động xoay {len(rotated_pages)} trang nằm ngang: {', '.join(rotated_pages)}")
    print(f"Đang xử lý {len(image_files)} ảnh cho tài liệu '{doc_id}'...")
    pipeline = DocumentPipeline(pipeline_config)
    doc_json, page_blocks_dict = pipeline.process_document(doc_id, image_files)

    report_text = generate_readable_report(doc_json)

    print("\n" + "=" * 80)
    print(" JSON TRÍCH XUẤT")
    print("=" * 80)
    print(json.dumps(doc_json, ensure_ascii=False, indent=2))

    print("\n" + "=" * 80)
    print(" BÁO CÁO DỄ ĐỌC")
    print("=" * 80)
    print(report_text)

    if not args.no_save:
        outputs_dir = os.path.join(base_dir, "outputs", "predictions")
        reports_dir = os.path.join(base_dir, "outputs", "reports")
        md_dir = os.path.join(base_dir, "outputs", "markdowns")
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(md_dir, exist_ok=True)

        json_path = os.path.join(outputs_dir, f"{doc_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc_json, f, ensure_ascii=False, indent=2)

        report_path = os.path.join(reports_dir, f"{doc_id}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print("\n" + "=" * 80)
        print(f" Đã lưu: {json_path}")
        print(f" Đã lưu: {report_path}")

        if args.embed:
            md_text = format_as_markdown(page_blocks_dict, document_id=doc_id, doc_json=doc_json)
            md_path = os.path.join(md_dir, f"{doc_id}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f" Đã lưu (RAG markdown): {md_path}")

            print(f" Đang tự động nhúng {doc_id} vào ChromaDB (thêm từng phần)...")
            from src.embedding_pipeline import add_document_embedding

            try:
                _, n_chunks = add_document_embedding(doc_id, markdowns_dir=md_dir)
                print(f" -> Đã nhúng {n_chunks} chunks. Có thể hỏi-đáp ngay về tài liệu này.")
            except Exception as e:
                print(f" -> Lỗi khi nhúng embedding: {e}")
        else:
            print(" (Chưa nhúng vào ChromaDB - dùng --embed nếu cần hỏi-đáp về tài liệu này.)")


if __name__ == "__main__":
    main()
