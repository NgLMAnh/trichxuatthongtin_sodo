import os
import sys
import tempfile
import numpy as np
import cv2
from PIL import Image

try:
    from paddleocr import PaddleOCR
except ImportError:
    pass

try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
except ImportError:
    pass

class OCREngine:
    def __init__(self, languages=['vi'], gpu=False):
        """
        Initializes the Hybrid OCR engine:
        - PaddleOCR for Text Detection (Bounding Boxes)
        - VietOCR for Text Recognition (Reading Text)
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            print("Error: paddleocr is not installed.")
            sys.exit(1)
            
        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg
        except ImportError as e:
            print(f"Error: vietocr is not installed or missing dependencies. Details: {e}")
            sys.exit(1)
            
        # 1. Initialize PaddleOCR ONLY for detection
        self.det_model = PaddleOCR(
            use_angle_cls=True, 
            det=True, 
            rec=False, 
            lang='vi', 
            show_log=False, 
            use_gpu=gpu,
            det_limit_side_len=2048,     # Ngăn model resize ảnh xuống quá nhỏ, giúp giữ chi tiết chữ
            det_db_thresh=0.3,           # Ngưỡng nhị phân hóa, giảm nhẹ để bắt được chữ mờ
            det_db_box_thresh=0.5,       # Ngưỡng box, giảm nhẹ để không sót box
            det_db_unclip_ratio=1.6      # Mở rộng bounding box một chút để không bị cắt lẹm dấu tiếng Việt
        )
        
        # 2. Initialize VietOCR for recognition
        config = self._load_vietocr_config()
        config['cnn']['pretrained'] = False
        config['device'] = 'cuda:0' if gpu else 'cpu'
        self._prefer_local_vietocr_weights(config)
        self.rec_model = Predictor(config)

    def _load_vietocr_config(self):
        local_config = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "vietocr",
            "vgg_transformer.yml",
        )
        if os.path.exists(local_config):
            return Cfg.load_config_from_file(local_config)
        return Cfg.load_config_from_name('vgg_transformer')

    def _prefer_local_vietocr_weights(self, config):
        project_weight = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "vietocr",
            "weights",
            "vgg_transformer.pth",
        )
        if os.path.exists(project_weight):
            config["weights"] = project_weight
            return

        cached_weight = os.path.join(tempfile.gettempdir(), "vgg_transformer.pth")
        if os.path.exists(cached_weight):
            config["weights"] = cached_weight

    @staticmethod
    def _read_image_bgr(image_path):
        """Đọc ảnh về ndarray BGR, an toàn với đường dẫn chứa ký tự tiếng Việt
        (cv2.imread dùng codepage ANSI nên fail âm thầm với path có dấu -
        dùng np.fromfile + cv2.imdecode để tránh)."""
        buf = np.fromfile(image_path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def run_ocr(self, image_path, layout_blocks=None, image_bgr=None):
        """
        Performs OCR on an image.

        Tối ưu hiệu năng (CPU): (1) chỉ chạy DETECTION thuần của PaddleOCR
        (det=True, rec=False, cls=False) - trước đây gọi .ocr(path, cls=True)
        khiến Paddle chạy CẢ recognition riêng của nó (~10x chậm hơn) rồi VỨT
        kết quả để VietOCR đọc lại, đồng thời âm thầm loại các box confidence
        thấp (mất chữ). (2) Gom TẤT CẢ crop rồi nhận dạng 1 lượt bằng VietOCR
        predict_batch thay vì predict từng crop tuần tự.

        image_bgr: ndarray BGR đã decode sẵn (tránh decode lại) - nếu None sẽ tự đọc.

        Returns:
            list[dict]: keys 'text', 'bbox' (x1,y1,x2,y2), 'confidence', 'block_id'.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        if image_bgr is None:
            image_bgr = self._read_image_bgr(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not read image at: {image_path}")

        # PIL RGB cho VietOCR (crop) - từ chính ndarray BGR đã có, không đọc lại file
        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        parsed_results = []

        # DETECTION thuần trên toàn bộ ảnh (không rec, không cls) - nhanh & không mất box
        results = self.det_model.ocr(image_bgr, det=True, rec=False, cls=False)
        if not results or not results[0]:
            print(
                f"    [CẢNH BÁO] OCR không tìm thấy chữ nào trong '{os.path.basename(image_path)}' "
                f"(kích thước ảnh: {img.width}x{img.height}). Kiểm tra lại ảnh có bị trắng/mờ/lỗi/xoay sai "
                f"chiều không - kết quả trích xuất cho trang này sẽ TRỐNG HOÀN TOÀN."
            )
            return parsed_results

        # Bước 1: cắt toàn bộ crop + lưu bbox tương ứng (chưa nhận dạng)
        crops = []
        boxes = []
        for line in results[0]:
            if isinstance(line[0][0], (int, float)):
                box = line
            else:
                box = line[0]

            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            lx1, ly1, lx2, ly2 = min(xs), min(ys), max(xs), max(ys)

            # Padding để VietOCR đọc không bị lẹm nét (dấu huyền/ngã/nặng tiếng Việt)
            pad_x = 3
            pad_y = 5
            crop_img = img.crop((
                max(0, lx1 - pad_x),
                max(0, ly1 - pad_y),
                min(img.width, lx2 + pad_x),
                min(img.height, ly2 + pad_y),
            ))
            crops.append(crop_img)
            boxes.append((lx1, ly1, lx2, ly2))

        # Bước 2: nhận dạng TẤT CẢ crop 1 lượt (batch) - nhanh hơn nhiều so với tuần tự
        texts, probs = self.rec_model.predict_batch(crops, return_prob=True)

        # Bước 3: ghép text + bbox + block_id
        for (lx1, ly1, lx2, ly2), text, prob in zip(boxes, texts, probs):
            if not text or not text.strip():
                continue

            assigned_block_id = None
            if layout_blocks:
                lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
                for b in layout_blocks:
                    bx1, by1, bx2, by2 = b['bbox']
                    if bx1 <= lcx <= bx2 and by1 <= lcy <= by2:
                        assigned_block_id = b['block_id']
                        break

            parsed_results.append({
                "text": text,
                "bbox": [lx1, ly1, lx2, ly2],
                "confidence": float(prob),
                "block_id": assigned_block_id,
            })

        n_boxes = len(results[0])
        if n_boxes > 0 and not parsed_results:
            print(
                f"    [CẢNH BÁO] OCR định vị được {n_boxes} vùng chữ trong '{os.path.basename(image_path)}' "
                f"nhưng bước nhận dạng (VietOCR) không đọc ra được ký tự nào cho bất kỳ vùng nào - kết quả "
                f"trích xuất cho trang này sẽ TRỐNG HOÀN TOÀN. Kiểm tra chất lượng ảnh (độ phân giải, độ nét, "
                f"độ nghiêng) hoặc thử ảnh khác."
            )

        return parsed_results
