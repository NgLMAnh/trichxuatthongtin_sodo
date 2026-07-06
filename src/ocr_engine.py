import os
import sys

try:
    from paddleocr import PaddleOCR
except ImportError:
    pass

class OCREngine:
    def __init__(self, languages=['vi'], gpu=False):
        """
        Initializes the PaddleOCR engine (v2.x).
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            print("Error: paddleocr is not installed.")
            sys.exit(1)
            
        # Standard PaddleOCR 2.x initialization parameters
        self.ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False, use_gpu=gpu)

    def run_ocr(self, image_path):
        """
        Performs OCR on an image.
        Returns:
            list[dict]: A list of dicts with keys: 'text', 'bbox' (x1, y1, x2, y2), 'confidence'.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # Run PaddleOCR with classification model (angle classification)
        results = self.ocr.ocr(image_path, cls=True)
        
        parsed_results = []
        if results and results[0]:
            for line in results[0]:
                bbox_points = line[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                text, confidence = line[1]
                
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                
                parsed_results.append({
                    "text": text,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(confidence)
                })
                
        return parsed_results
