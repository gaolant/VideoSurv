import cv2
import config

def draw_zone(frame, polygon):
    if polygon is not None:
        cv2.polylines(frame, [polygon], isClosed=True, color=config.COLOR_GREEN, thickness=2)

def draw_bbox(frame, bbox, color=config.COLOR_GREEN, text=None):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def draw_alarm(frame, text="ALARM!"):
    h, w, _ = frame.shape
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)
    text_x = (w - text_size[0]) // 2
    text_y = text_size[1] + 50  # 50px от верха
    
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, config.COLOR_RED, 4)