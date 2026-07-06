import os
import sys
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
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = False
        config['device'] = 'cuda:0' if gpu else 'cpu'
        self.rec_model = Predictor(config)

    def run_ocr(self, image_path):
        """
        Performs OCR on an image.
        Returns:
            list[dict]: A list of dicts with keys: 'text', 'bbox' (x1, y1, x2, y2), 'confidence'.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # Step 1: Detect Text Bounding Boxes using PaddleOCR
        results = self.det_model.ocr(image_path, cls=True)
        
        parsed_results = []
        if not results or not results[0]:
            return parsed_results
            
        # Open original image to crop
        img = Image.open(image_path).convert('RGB')
        
        # PaddleOCR returns a list of boxes for results[0] when rec=False
        # Each box is a list of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        boxes = results[0]
        
        for line in results[0]:
            # Depending on cls=True/False and rec=True/False, line could be:
            # - A bounding box: [[x1, y1], [x2, y2], ...]
            # - A list: [box, (label, score)]
            if isinstance(line[0][0], (int, float)):
                box = line
            else:
                box = line[0]
                
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            
            # Step 2: Crop image
            # Add a small padding to prevent cutting off text edges
            pad = 2
            crop_img = img.crop((max(0, x1 - pad), max(0, y1 - pad), min(img.width, x2 + pad), min(img.height, y2 + pad)))
            
            # Step 3: Recognize text using VietOCR
            # VietOCR returns (text, confidence) if return_prob=True
            text, prob = self.rec_model.predict(crop_img, return_prob=True)
            
            # Only keep results with some text
            if text.strip():
                parsed_results.append({
                    "text": text,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(prob)
                })
                
        return parsed_results
