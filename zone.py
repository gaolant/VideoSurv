import cv2
import json
import numpy as np
import os

class RestrictedZone:
    def __init__(self, config_path):
        self.polygon = self._load_zone(config_path)

    def _load_zone(self, config_path):
        if not os.path.exists(config_path):
            print(f"Warning: Zone config '{config_path}' not found. No zone will be active.")
            return None
        try:
            with open(config_path, 'r') as f:
                points = json.load(f)
            return np.array(points, dtype=np.int32)
        except Exception as e:
            print(f"Error loading zone config: {e}")
            return None

    def get_polygon(self):
        return self.polygon

    def is_point_inside(self, point):
        """
        Проверяет, находится ли точка (x, y) внутри полигона.
        """
        if self.polygon is None:
            return False
        
        # Явно преобразуем к кортежу из стандартных Python int
        # Это защищает от ошибок типов данных NumPy
        safe_point = (int(point[0]), int(point[1]))

        # cv2.pointPolygonTest - быстрая C++ реализация
        # Возвращает +1 (внутри), 0 (на границе), -1 (снаружи)
        return cv2.pointPolygonTest(self.polygon, safe_point, False) >= 0