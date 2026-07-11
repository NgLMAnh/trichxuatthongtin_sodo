import os
import re
from src.text_utils import normalize_text
from src.layout_analyzer import LayoutAnalyzer
from src.ocr_engine import OCREngine
from src.block_builder import BlockBuilder
from src.section_detector import SectionDetector
from src.spatial_graph import SpatialGraph
from src.field_extractor import FieldExtractor
from src.change_extractor import ChangeHistoryExtractor
from src.holder_extractor import HolderExtractor
from src.generic_field_extractor import extract_generic_fields
from src.normalizers import normalize_fields, normalize_change_history, normalize_holders
from src.document_merger import merge_pages

class DocumentPipeline:
    def __init__(self, config):
        self.config = config
        self.layout_analyzer = LayoutAnalyzer(config)
        self.ocr_engine = OCREngine(languages=['vi'], gpu=config.get("ocr_detection", {}).get("use_gpu", False))
        self.block_builder = BlockBuilder()
        self.section_detector = SectionDetector(config)
        self.field_extractor = FieldExtractor(config)
        self.change_extractor = ChangeHistoryExtractor(config)
        self.holder_extractor = HolderExtractor(config)

    def process_image(self, image_path):
        """
        Xử lý một trang ảnh.
        """
        # Decode ảnh 1 LẦN (BGR, an toàn unicode) rồi tái sử dụng cho cả layout
        # analyzer lẫn OCR engine - tránh đọc/decode lại file 2-3 lần mỗi trang.
        image_bgr = self.ocr_engine._read_image_bgr(image_path)

        print(f"    - Phân tích Layout: {os.path.basename(image_path)}")
        layout = self.layout_analyzer.analyze(image_path, image_bgr=image_bgr)

        print(f"    - Chạy OCR...")
        ocr_results = self.ocr_engine.run_ocr(image_path, layout_blocks=layout["blocks"], image_bgr=image_bgr)
        
        print(f"    - Xây dựng Structured Blocks...")
        blocks = self.block_builder.build(layout, ocr_results)
        
        print(f"    - Phân tích Sections...")
        sections = self.section_detector.detect(blocks)
        self._attach_sections_to_blocks(blocks, sections)
        
        print(f"    - Xây dựng Spatial Graph & Trích xuất Fields...")
        graph = SpatialGraph(blocks)
        
        raw_fields = self.field_extractor.extract(blocks, sections, graph)
        norm_fields = normalize_fields(raw_fields)

        # Mẫu cũ: mục "Tài sản gắn liền với đất" là 1 ĐOẠN MÔ TẢ TỰ DO (VD "Nhà
        # hai tầng, tường gạch, sàn BTCT, mái ngói, diện tích xây dựng 136,2m²...")
        # không có nhãn "Tên tài sản:" -> anchor không khớp, asset_name = None dù
        # thông tin có trên giấy. Fallback: ghi NGUYÊN VĂN đoạn mô tả đó làm
        # asset_name; các field con (diện tích sử dụng, hình thức/thời hạn sở hữu)
        # vẫn chỉ điền khi có nhãn rõ ràng - không suy diễn từ mô tả.
        if not norm_fields.get("asset_name"):
            desc = self._asset_description_fallback(blocks, sections)
            if desc:
                norm_fields["asset_name"] = desc

        raw_change_history = self.change_extractor.extract(blocks, sections)
        change_history = normalize_change_history(raw_change_history)

        # Chỉ khớp khi mẫu GCN gộp tên+CMND/CCCD trong 1 block (xem holder_extractor.py).
        # Mẫu cũ (tên/CMND ở 2 block riêng) sẽ trả về [] - document_merger.py sẽ
        # fallback dùng lại holder scalar đã merge (từ FieldExtractor) cho trường hợp đó.
        raw_holders = self.holder_extractor.extract(blocks, sections)
        holders = normalize_holders(raw_holders)

        # Lớp bổ sung: bắt mọi dòng "<nhãn>: <giá trị>" chưa được field nào ở trên
        # khai báo tay xử lý, để không mất thông tin khi gặp mẫu/mục con mới.
        extra_fields = extract_generic_fields(blocks, sections)

        return {
            "blocks": blocks,
            "sections": sections,
            "fields": norm_fields,
            "change_history": change_history,
            "holders": holders,
            "extra_fields": extra_fields,
        }

    @staticmethod
    def _asset_description_fallback(blocks, sections):
        """Ghép nguyên văn các dòng trong mục 'Tài sản gắn liền với đất' (trừ
        chính dòng tiêu đề mục) làm mô tả tài sản, khi anchor 'Tên tài sản'
        không khớp (mẫu cũ viết mô tả tự do thay vì nhãn:giá trị)."""
        block_ids = set(sections.get("asset_info", []))
        if not block_ids:
            return None

        texts = []
        for block in sorted(blocks, key=lambda b: b.get("reading_order", 0)):
            if block.get("block_id") not in block_ids:
                continue
            text = (block.get("text") or "").strip()
            if not text:
                continue
            norm = normalize_text(text)
            # Bỏ dòng tiêu đề của chính mục (cả kiểu cũ lẫn kiểu đánh số mới)
            if re.search(r"tai\s+san\s+gan\s+lien|thong\s+tin\s+tai\s+san", norm):
                continue
            # Bỏ block chữ ký/chức danh/con dấu (cùng dải toạ độ y với mục tài
            # sản nên bị gán nhầm vào section) và block confidence thấp - chữ
            # trong con dấu/chữ ký OCR ra rác (VD "TRONG THỊ CHỐNG NHÂN MINH")
            # với confidence đặc trưng < 0.75, còn dòng mô tả thật ~0.9+.
            if re.search(r"chu\s+tich|ky\s+ten|dong\s+dau|thay\s+mat|giam\s+doc", norm):
                continue
            if float(block.get("confidence", 1.0)) < 0.75:
                continue
            texts.append(text)

        joined = " ".join(texts).strip()
        return joined or None

    def _attach_sections_to_blocks(self, blocks, sections):
        block_lookup = {block.get("block_id"): block for block in blocks}
        for section_name, block_ids in sections.items():
            for block_id in block_ids:
                block = block_lookup.get(block_id)
                if block is not None:
                    block["section"] = section_name

    def process_document(self, doc_id, image_files):
        """
        Xử lý tất cả các trang của một tài liệu.
        """
        page_results = []
        page_blocks_dict = {}
        failed_pages = []

        for img_path in image_files:
            img_name = os.path.basename(img_path)
            print(f"  - Đang xử lý: {img_name}")
            try:
                res = self.process_image(img_path)

                change_history = res["change_history"]
                for record in change_history:
                    record["page"] = img_name

                # MỌI dòng chữ OCR đọc được trên trang (kể cả dòng không map vào
                # field nào) - để không bỏ sót thông tin nào khỏi JSON kết quả,
                # phục vụ yêu cầu "OCR tất cả các chữ và lưu thành field rõ ràng".
                text_lines = [
                    {
                        "page": img_name,
                        "section": b.get("section", "unknown"),
                        "text": b.get("text", "").strip(),
                        "confidence": round(float(b.get("confidence", 0.0)), 4),
                    }
                    for b in sorted(res["blocks"], key=lambda x: x.get("reading_order", 0))
                    if b.get("text", "").strip()
                ]

                page_results.append({
                    "page_name": img_name,
                    # Phân loại page_type cho document_merger.
                    # Nếu có field nào được extract, ta coi nó là loại trang tương ứng.
                    "page_type": "holder_info" if res["fields"].get("holder_name") else "land_info",
                    "fields": res["fields"],
                    "change_history": change_history,
                    "holders": res["holders"],
                    "extra_fields": res["extra_fields"],
                    "text_lines": text_lines,
                })
                page_blocks_dict[img_name] = {
                    "blocks": res["blocks"],
                    "sections": res["sections"],
                    "fields": res["fields"],
                    "extra_fields": res["extra_fields"],
                    "source_image": img_path,
                }
            except Exception as e:
                import traceback
                print(f"    -> Error processing page {img_name}: {e}")
                traceback.print_exc()
                # KHÔNG nuốt lỗi im lặng: ghi lại trang thất bại để merger phản ánh
                # trong kết quả (người dùng biết trang nào thiếu, không tưởng đủ).
                failed_pages.append({"page_name": img_name, "error": str(e)})

        # Merge các trang lại thành 1 JSON
        doc_json = merge_pages(doc_id, page_results)
        if failed_pages:
            doc_json["failed_pages"] = failed_pages

        return doc_json, page_blocks_dict
