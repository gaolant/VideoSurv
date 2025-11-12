import cv2
import time
import config
from detector import ObjectDetector
from zone import RestrictedZone
import visualizer

def main():
    # 1. Инициализация
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {config.VIDEO_PATH}")
        return

    detector = ObjectDetector(config.YOLO_MODEL_PATH)
    zone = RestrictedZone(config.ZONE_CONFIG_PATH)
    
    # Переменные для управления состоянием тревоги
    alarm_active = False
    last_intrusion_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Детекция людей
        detections = detector.detect(frame, config.PERSON_CLASS_ID, config.CONFIDENCE_THRESHOLD)

        # 3. Логика проникновения
        person_in_zone = False
        visualizer.draw_zone(frame, zone.get_polygon())

        for bbox in detections:
            # Точка для проверки - центр нижней границы BBox
            # Это лучше, чем центр бокса, т.к. показывает "ноги"
            x1, y1, x2, y2 = bbox
            ref_point = ((x1 + x2) // 2, y2)
            
            if zone.is_point_inside(ref_point):
                person_in_zone = True
                visualizer.draw_bbox(frame, bbox, config.COLOR_RED)
            else:
                visualizer.draw_bbox(frame, bbox, config.COLOR_GREEN)

        current_time = time.time()
        
        if person_in_zone:
            alarm_active = True
            last_intrusion_time = current_time  # Сбрасываем таймер при каждом обнаружении
        
        if alarm_active:
            if not person_in_zone:
                # Человек пропал, запускаем 3-секундный обратный отсчет
                if current_time - last_intrusion_time > config.ALARM_COOLDOWN_SEC:
                    alarm_active = False
            
            if alarm_active:
                visualizer.draw_alarm(frame)

        # 5. Отображение
        cv2.imshow("Intrusion Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Очистка
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()