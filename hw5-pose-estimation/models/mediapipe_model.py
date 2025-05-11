import cv2
import mediapipe as mp
import numpy as np


class MediaPipeHandDetector:
    """
    Класс для обнаружения ключевых точек рук с использованием MediaPipe Hands
    """
    
    def __init__(self, static_image_mode=False, max_num_hands=1, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Инициализация детектора рук MediaPipe
        
        Args:
            static_image_mode (bool): Режим статичного изображения (или видео)
            max_num_hands (int): Максимальное количество рук для обнаружения
            min_detection_confidence (float): Минимальная достоверность для обнаружения рук
            min_tracking_confidence (float): Минимальная достоверность для отслеживания рук
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def detect_hands(self, image):
        """
        Обнаружение ключевых точек руки на изображении
        
        Args:
            image (numpy.ndarray): Входное изображение в формате BGR
            
        Returns:
            dict: Словарь с координатами ключевых точек {index: (x, y)} 
                  или None, если руки не обнаружены
        """
        # Конвертация изображения из BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Получение результатов обнаружения
        results = self.hands.process(image_rgb)
        
        # Проверка наличия результатов
        if not results.multi_hand_landmarks:
            return None
        
        # Берем первую обнаруженную руку
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Получаем размеры изображения
        h, w, _ = image.shape
        
        # Создаем словарь с координатами ключевых точек
        keypoints = {}
        for idx, landmark in enumerate(hand_landmarks.landmark):
            # Преобразуем нормализованные координаты в пиксельные
            x, y = int(landmark.x * w), int(landmark.y * h)
            keypoints[idx] = (x, y)
        
        return keypoints
    
    def draw_landmarks(self, image, keypoints):
        """
        Отрисовка ключевых точек и соединений на изображении
        
        Args:
            image (numpy.ndarray): Входное изображение
            keypoints (dict): Словарь с координатами ключевых точек {index: (x, y)}
            
        Returns:
            numpy.ndarray: Изображение с отрисованными ключевыми точками
        """
        if keypoints is None:
            return image
        
        # Создаем копию изображения
        image_copy = image.copy()
        
        # Рисуем ключевые точки
        for idx, (x, y) in keypoints.items():
            cv2.circle(image_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # Рисуем соединения между ключевыми точками
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # большой палец
            (0, 5), (5, 6), (6, 7), (7, 8),  # указательный палец
            (0, 9), (9, 10), (10, 11), (11, 12),  # средний палец
            (0, 13), (13, 14), (14, 15), (15, 16),  # безымянный палец
            (0, 17), (17, 18), (18, 19), (19, 20),  # мизинец
            (5, 9), (9, 13), (13, 17),  # поперечные соединения ладони
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx in keypoints and end_idx in keypoints:
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(image_copy, start_point, end_point, (0, 255, 0), 2)
        
        return image_copy


def load_mediapipe_model():
    """
    Загружает и возвращает модель MediaPipe для обнаружения рук
    
    Returns:
        MediaPipeHandDetector: Экземпляр детектора рук MediaPipe
    """
    return MediaPipeHandDetector()