import os
from src.layout_analyzer import LayoutAnalyzer
from src.ocr_engine import OCREngine
from src.block_builder import BlockBuilder
from src.section_detector import SectionDetector
from src.spatial_graph import SpatialGraph
from src.field_extractor import FieldExtractor
from src.normalizers import normalize_fields
from src.document_merger import merge_pages

class DocumentPipeline:
    def __init__(self, config):
        self.config = config
        self.layout_analyzer = LayoutAnalyzer(config)
        self.ocr_engine = OCREngine(languages=['vi'], gpu=config.get("ocr_detection", {}).get("use_gpu", False))
        self.block_builder = BlockBuilder()
        self.section_detector = SectionDetector(config)
        self.field_extractor = FieldExtractor(config)

    def process_image(self, image_path):
        """
        Xử lý một trang ảnh.
        """
        print(f"    - Phân tích Layout: {os.path.basename(image_path)}")
        layout = self.layout_analyzer.analyze(image_path)
        
        print(f"    - Chạy OCR...")
        ocr_results = self.ocr_engine.run_ocr(image_path, layout_blocks=layout["blocks"])
        
        print(f"    - Xây dựng Structured Blocks...")
        blocks = self.block_builder.build(layout, ocr_results)
        
        print(f"    - Phân tích Sections...")
        sections = self.section_detector.detect(blocks)
        self._attach_sections_to_blocks(blocks, sections)
        
        print(f"    - Xây dựng Spatial Graph & Trích xuất Fields...")
        graph = SpatialGraph(blocks)
        
        raw_fields = self.field_extractor.extract(blocks, sections, graph)
        norm_fields = normalize_fields(raw_fields)
        
        return {
            "blocks": blocks,
            "sections": sections,
            "fields": norm_fields
        }

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
        
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            print(f"  - Đang xử lý: {img_name}")
            try:
                res = self.process_image(img_path)
                
                page_results.append({
                    "page_name": img_name,
                    # Phân loại page_type cho document_merger. 
                    # Nếu có field nào được extract, ta coi nó là loại trang tương ứng.
                    "page_type": "holder_info" if res["fields"].get("holder_name") else "land_info",
                    "fields": res["fields"]
                })
                page_blocks_dict[img_name] = {
                    "blocks": res["blocks"],
                    "sections": res["sections"],
                    "fields": res["fields"],
                    "source_image": img_path,
                }
            except Exception as e:
                import traceback
                print(f"    -> Error processing page {img_name}: {e}")
                traceback.print_exc()
                
        # Merge các trang lại thành 1 JSON
        doc_json = merge_pages(doc_id, page_results)
        
        return doc_json, page_blocks_dict
