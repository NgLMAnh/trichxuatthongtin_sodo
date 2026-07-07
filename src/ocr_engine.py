import os
import sys
import tempfile
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
        self.det_model = PaddleOCR(use_angle_cls=True, det=True, rec=False, lang='vi', show_log=False, use_gpu=gpu)
        
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

    def run_ocr(self, image_path, layout_blocks=None):
        """
        Performs OCR on an image.
        Returns:
            list[dict]: A list of dicts with keys: 'text', 'bbox' (x1, y1, x2, y2), 'confidence', 'block_id'.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        img = Image.open(image_path).convert('RGB')
        parsed_results = []
        
        # Luôn chạy det trên toàn bộ ảnh để không bị sót chữ do layout model nhận diện thiếu
        results = self.det_model.ocr(image_path, cls=True)
        if not results or not results[0]:
            return parsed_results
            
        for line in results[0]:
            if isinstance(line[0][0], (int, float)):
                box = line
            else:
                box = line[0]
                
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            lx1, ly1, lx2, ly2 = min(xs), min(ys), max(xs), max(ys)
            
            pad = 2
            crop_img = img.crop((max(0, lx1 - pad), max(0, ly1 - pad), min(img.width, lx2 + pad), min(img.height, ly2 + pad)))
            text, prob = self.rec_model.predict(crop_img, return_prob=True)
            
            if text.strip():
                # Tìm block_id tương ứng
                assigned_block_id = None
                if layout_blocks:
                    best_iou = 0
                    lcx, lcy = (lx1 + lx2)/2, (ly1 + ly2)/2
                    for b in layout_blocks:
                        bx1, by1, bx2, by2 = b['bbox']
                        # Check if center is inside block
                        if bx1 <= lcx <= bx2 and by1 <= lcy <= by2:
                            assigned_block_id = b['block_id']
                            break
                            
                parsed_results.append({
                    "text": text,
                    "bbox": [lx1, ly1, lx2, ly2],
                    "confidence": float(prob),
                    "block_id": assigned_block_id
                })
                    
        return parsed_results
