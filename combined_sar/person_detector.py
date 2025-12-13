"""Person detection via YOLO"""

from typing import List, Tuple
import numpy as np


class PersonDetector:
    """YOLO-based person detection"""
    
    def __init__(self, yolo_model="yolov8n.pt", confidence=0.5, nms_threshold=0.45):
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.model = None
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(yolo_model)
        except ImportError:
            print("[PersonDetector] ultralytics not available")
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, List]:
        """Detect persons in frame"""
        if self.model is None:
            return False, []
        
        try:
            # Convert RGBA to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                # RGBA to RGB conversion
                frame = frame[:, :, :3]
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                # Grayscale to RGB
                frame = np.repeat(frame, 3, axis=2)
            elif len(frame.shape) == 2:
                # Pure grayscale to RGB
                frame = np.stack([frame] * 3, axis=2)
            
            results = self.model(frame, conf=self.confidence, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:
                        detections.append({
                            "bbox": box.xyxy[0].cpu().numpy(),
                            "confidence": float(box.conf[0]),
                        })
            return len(detections) > 0, detections
        except Exception as e:
            print(f"[PersonDetector] Error: {e}")
            return False, []
