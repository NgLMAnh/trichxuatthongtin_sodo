import os
import cv2
import numpy as np

try:
    from paddleocr import PPStructure
except ImportError:
    PPStructure = None

try:
    from paddleocr import PPStructureV3
except ImportError:
    PPStructureV3 = None

class LayoutAnalyzer:
    def __init__(self, config):
        """
        Khởi tạo Layout Analyzer sử dụng PPStructure
        """
        layout_cfg = config.get("layout", {})
        self.use_gpu = layout_cfg.get("use_gpu", False)
        self.show_log = layout_cfg.get("show_log", False)
        
        self.engine_name = "PPStructureV3"
        self.engine = self._init_ppstructure_v3()
        if self.engine is None:
            self.engine_name = "PPStructure"
            self.engine = self._init_ppstructure()

        if self.engine is None:
            raise ImportError("Could not initialize PPStructureV3 or PPStructure from paddleocr.")

        print(f"LayoutAnalyzer initialized with {self.engine_name} on CPU={not self.use_gpu}.")

    def _init_ppstructure_v3(self):
        if PPStructureV3 is None:
            return None

        try:
            return PPStructureV3(
                use_gpu=self.use_gpu,
                show_log=self.show_log,
                layout=True,
                table=False,
                ocr=False,
                det=False,
                rec=False,
            )
        except TypeError:
            try:
                return PPStructureV3(use_gpu=self.use_gpu, show_log=self.show_log)
            except Exception:
                return None
        except Exception:
            return None

    def _init_ppstructure(self):
        if PPStructure is None:
            return None

        return PPStructure(
            use_gpu=self.use_gpu,
            show_log=self.show_log,
            layout=True,
            table=False,
            ocr=False,
            det=False,
            rec=False,
        )

    def analyze(self, image_path):
        """
        Phân tích layout của ảnh.
        Args:
            image_path (str): Đường dẫn tới ảnh
        Returns:
            dict: Chứa 'blocks' (danh sách các vùng layout) và 'image_size' (width, height)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at: {image_path}")
            
        height, width = img.shape[:2]
        
        # Thực hiện phân tích layout
        result = self.engine(img)
        
        blocks = []
        for i, region in enumerate(self._iter_regions(result)):
            block_type = region.get('type') or region.get('label') or 'unknown'
            bbox = region.get('bbox') or region.get('coordinate') or [0, 0, 0, 0]
            
            blocks.append({
                "block_id": f"block_{i}",
                "label": block_type.lower(), # text, title, figure, table, header, footer...
                "bbox": bbox,
                "reading_order": i # Giả định PPStructure trả về thứ tự đọc hợp lý
            })
            
        # Optional: Cải thiện reading order sorting nếu cần (thay vì dùng thứ tự của PPStructure)
        # Sắp xếp các block từ trên xuống dưới, trái qua phải
        blocks = self._sort_reading_order(blocks)
            
        return {
            "blocks": blocks,
            "image_size": (width, height)
        }

    def _iter_regions(self, result):
        if result is None:
            return []
        if isinstance(result, dict):
            for key in ("blocks", "layout", "res", "result"):
                value = result.get(key)
                if isinstance(value, list):
                    return value
            return []
        if isinstance(result, list):
            if len(result) == 1 and isinstance(result[0], dict):
                for key in ("blocks", "layout", "res", "result"):
                    value = result[0].get(key)
                    if isinstance(value, list):
                        return value
            return result
        return []
        
    def _sort_reading_order(self, blocks):
        """
        Sắp xếp các blocks theo thứ tự đọc (từ trên xuống, từ trái qua).
        """
        # Nhóm các block trên cùng một hàng
        # Sử dụng logic tương tự group_boxes_into_lines
        if not blocks:
            return []
            
        # Sắp xếp theo y1 trước
        blocks.sort(key=lambda b: b['bbox'][1])
        
        lines = []
        current_line = []
        
        y_tolerance_ratio = 0.5
        
        for block in blocks:
            if not current_line:
                current_line.append(block)
                continue
                
            line_y1 = sum(b['bbox'][1] for b in current_line) / len(current_line)
            line_y2 = sum(b['bbox'][3] for b in current_line) / len(current_line)
            line_height = line_y2 - line_y1
            
            box_y1, box_y2 = block['bbox'][1], block['bbox'][3]
            box_height = box_y2 - box_y1
            
            overlap_y1 = max(line_y1, box_y1)
            overlap_y2 = min(line_y2, box_y2)
            overlap_height = max(0, overlap_y2 - overlap_y1)
            
            min_height = min(line_height, box_height)
            if min_height > 0 and overlap_height / min_height >= y_tolerance_ratio:
                current_line.append(block)
            else:
                lines.append(current_line)
                current_line = [block]
                
        if current_line:
            lines.append(current_line)
            
        # Sắp xếp các block trong cùng một hàng từ trái sang phải
        sorted_blocks = []
        order_idx = 0
        for line in lines:
            line.sort(key=lambda b: b['bbox'][0])
            for block in line:
                block['reading_order'] = order_idx
                sorted_blocks.append(block)
                order_idx += 1
                
        return sorted_blocks
