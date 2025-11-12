import cv2
import time
import config
from detector_ds import ObjectDetector
from zone import RestrictedZone
import visualizer
from tracker import ObjectTracker  

def main():
    # 1. Инициализация
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {config.VIDEO_PATH}")
        return

    detector = ObjectDetector(config.YOLO_MODEL_PATH)
    zone = RestrictedZone(config.ZONE_CONFIG_PATH)
    tracker = ObjectTracker() 
    
    # 3. НОВАЯ ЛОГИКА ТРЕВОГИ
    # Вместо alarm_active и last_intrusion_time, используем словарь
    # {track_id: last_seen_time}
    intruders = {} 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()

        # 2. Детекция (YOLO)
        # detections теперь = [(bbox, conf, cls_id), ...]
        detections = detector.detect(frame, config.PERSON_CLASS_ID, config.CONFIDENCE_THRESHOLD)

        # 3. Трекинг (DeepSORT)
        # Передаем кадр (для анализа признаков) и детекции
        # tracked_objects = [[x1, y1, x2, y2, track_id], ...]
        tracked_objects = tracker.update(frame, detections)

        # 4. Логика проникновения (на основе ТРЕКОВ)
        visualizer.draw_zone(frame, zone.get_polygon())
        
        # Сет ID нарушителей, которые видны в ЭТОМ кадре
        current_frame_intruder_ids = set()

        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            bbox = [x1, y1, x2, y2]
            
            # Точка для проверки - центр нижней границы BBox
            ref_point = ((x1 + x2) // 2, y2)
            
            text_to_draw = f"ID: {track_id}"
            
            if zone.is_point_inside(ref_point):
                # ЧЕЛОВЕК В ЗОНЕ
                current_frame_intruder_ids.add(track_id)
                
                # Добавляем или обновляем время его последнего визита
                intruders[track_id] = current_time 
                
                visualizer.draw_bbox(frame, bbox, config.COLOR_RED, text_to_draw)
            else:
                # ЧЕЛОВЕК НЕ В ЗОНЕ
                visualizer.draw_bbox(frame, bbox, config.COLOR_GREEN, text_to_draw)

        # 5. Управление состоянием тревоги (Новая логика)
        
        # "Сборщик мусора" для словаря intruders
        # Проверяем все ID, которые были в зоне, но не видны в этом кадре
        ids_to_remove = []
        for track_id, last_seen in intruders.items():
            if track_id not in current_frame_intruder_ids:
                # ID не в зоне. Проверяем, прошло ли 3 секунды.
                if current_time - last_seen > config.ALARM_COOLDOWN_SEC:
                    ids_to_remove.append(track_id) # Запоминаем, что его надо удалить
        
        # Удаляем "устаревшие" ID из словаря нарушителей
        for track_id in ids_to_remove:
            del intruders[track_id]

        # 6. Включение тревоги
        # Тревога активна, если в словаре есть ХОТЯ БЫ ОДИН нарушитель
        if len(intruders) > 0:
            alarm_active = True
        else:
            alarm_active = False
        
        if alarm_active:
            # Получаем ID активных нарушителей для отображения (просто для информации)
            active_ids = ", ".join(str(id) for id in intruders.keys())
            visualizer.draw_alarm(frame, f"ALARM! (IDs: {active_ids})")

        # 7. Отображение
        cv2.imshow("Intrusion Detection (with DeepSORT)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. Очистка
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()