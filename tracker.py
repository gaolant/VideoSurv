from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class ObjectTracker:
    def __init__(self, max_age=30):
        """
        Инициализация DeepSORT (deep-sort-realtime).
        max_age: Сколько кадров ждать трек, прежде чем удалить его.
        """
        try:
            self.deepsort = DeepSort(
                max_age=max_age,
                n_init=3,
                nms_max_overlap=1.0,
                max_iou_distance=0.7,
                max_cosine_distance=0.2, # Насколько "похожими" должны быть объекты
                nn_budget=None,
                override_track_class=False, # Используем стандартный класс Track
                embedder="mobilenet", # Используем встроенный MobileNet
                half=True, # Включить FP16 (быстрее, если есть GPU, но работает и на CPU)
                bgr=True, # Ожидаем BGR кадры от OpenCV
                polygon=False,
                today=None
            )
        except Exception as e:
            print(f"Error initializing DeepSort: {e}")
            raise

    def update(self, frame, detections):
        """
        Обновляет трекер и возвращает активные треки.
        
        detections: список от YOLO в формате [(bbox, conf, cls_id)]
        frame: BGR-кадр (важно для извлечения признаков!)
        
        Возвращает: список [[x1, y1, x2, y2, track_id], ...]
        """
        
        # 1. Преобразуем детекции в формат, который DeepSORT ожидает:
        # ([x1, y1, w, h], confidence, class_name)
        
        formatted_detections = []
        for (bbox, conf, cls_id) in detections:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            if w <= 0 or h <= 0: # Игнорируем вырожденные "коробки"
                continue
                
            # Важно: deep-sort-realtime ожидает имя класса (строку), 
            # а не ID. Мы знаем, что наш cls_id это "person".
            class_name = "person" 
            
            # Формат: ([x1, y1, w, h], confidence, class_name)
            formatted_detections.append(
                ([x1, y1, w, h], conf, class_name)
            )
        
        # 2. Обновляем DeepSort
        # Он ожидает список Detections и BGR-кадр
        tracks = self.deepsort.update_tracks(formatted_detections, frame=frame)
        
        # 3. Извлекаем нужные данные из объектов Track
        output_tracks = []
        for track in tracks:
            # Пропускаем треки, которые еще не "подтверждены" или "потеряны"
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlbr() # Конвертация [x1, y1, w, h] -> [x1, y1, x2, y2]
            track_id = track.track_id
            
            output_tracks.append([
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), track_id
            ])
        
        return output_tracks