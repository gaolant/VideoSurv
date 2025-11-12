import cv2
import json
import numpy as np
import config

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    frame = param['frame']
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(frame, (x, y), 5, config.COLOR_RED, -1)
        
        if len(points) > 1:
            cv2.line(frame, tuple(points[-2]), tuple(points[-1]), config.COLOR_GREEN, 2)
        cv2.imshow("Zone Selector", frame)

def main():
    global points
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {config.VIDEO_PATH}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return
    
    cap.release()
    
    frame_copy = frame.copy()
    cv2.namedWindow("Zone Selector")
    cv2.setMouseCallback("Zone Selector", mouse_callback, {"frame": frame_copy})

    print("Нажмите 's' для сохранения, 'r' для сброса, 'q' для выхода.")
    
    while True:
        cv2.imshow("Zone Selector", frame_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            if len(points) < 3:
                print("Нужно как минимум 3 точки для полигона.")
            else:
                with open(config.ZONE_CONFIG_PATH, 'w') as f:
                    json.dump(points, f, indent=4)
                print(f"Зона сохранена в {config.ZONE_CONFIG_PATH}")
                # Нарисуем замкнутый полигон
                cv2.polylines(frame_copy, [np.array(points, dtype=np.int32)], isClosed=True, color=config.COLOR_GREEN, thickness=2)
                cv2.imshow("Zone Selector", frame_copy)
        elif key == ord('r'):
            points = []
            frame_copy = frame.copy()
            print("Сброс. Рисуйте заново.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()