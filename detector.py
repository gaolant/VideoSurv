from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
    
    def detect(self, frame, class_id, conf_threshold):
        """
        Детектирует объекты на кадре.
        Возвращает: список bounding boxes [x1, y1, x2, y2]
        """
        # device=0 для GPU, device='cpu' для CPU
        results = self.model(frame, verbose=False, device='cpu', classes=[class_id], conf=conf_threshold)
        
        detections = []
        for box in results[0].boxes:
            # Конвертируем в [x1, y1, x2, y2] и в int
            detections.append(box.xyxy[0].cpu().numpy().astype(int))
            
        return detections