"""
detector.py
Wraps a pretrained YOLOv8 model so we can detect every COCO object class
(person, ball, backpack, bottle, phone, chair, dog, ... 80 classes total),
including small objects like a ball, as long as they're visible enough
in the frame. YOLOv8n/s/m trade speed for accuracy - 'n' is fastest and
good enough for a live desktop demo.
"""

from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.30):
        # ultralytics auto-downloads the weights the first time this runs
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names  # dict: id -> label

    def detect(self, frame):
        """
        Returns a list of dicts:
        {"bbox": [x1, y1, x2, y2], "confidence": float, "label": str, "class_id": int}
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            imgsz=max(frame.shape[:2]),
            verbose=False,
        )[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.class_names[cls_id]
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "label": label,
            })
        return detections
