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
        
        Возвращает: 
            Список кортежей [(bbox, conf, cls_id)], 
            где bbox = [x1, y1, x2, y2]
        """
        # device='cpu' - не забываем про исправление из прошлого шага
        results = self.model(frame, verbose=False, device='cpu', classes=[class_id], conf=conf_threshold)
        
        detections = []
        
        # results[0] содержит все найденные "коробки"
        for res in results[0]:
            # res.boxes содержит всю информацию
            box = res.boxes.xyxy[0].cpu().numpy().astype(int)
            conf = res.boxes.conf[0].cpu().numpy()
            cls = res.boxes.cls[0].cpu().numpy()
            
            # Собираем все в один кортеж
            detections.append((box, float(conf), int(cls)))
            
        return detections