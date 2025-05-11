import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def draw_keypoints(frame, keypoints, radius=3, color=(0, 255, 0), thickness=-1):
    """
    Рисует ключевые точки на кадре
    
    Args:
        frame (numpy.ndarray): Входной кадр
        keypoints (dict): Словарь с ключевыми точками {index: (x, y)}
        radius (int): Радиус точек
        color (tuple): Цвет точек (BGR)
        thickness (int): Толщина контура точек
        
    Returns:
        numpy.ndarray: Кадр с нарисованными точками
    """
    output = frame.copy()
    
    if keypoints is None:
        return output
    
    # Рисуем каждую точку
    for idx, (x, y) in keypoints.items():
        cv2.circle(output, (int(x), int(y)), radius, color, thickness)
    
    return output


def draw_hand_connections(frame, keypoints, connections=None, color=(0, 255, 0), thickness=2):
    """
    Рисует соединения между ключевыми точками руки
    
    Args:
        frame (numpy.ndarray): Входной кадр
        keypoints (dict): Словарь с ключевыми точками {index: (x, y)}
        connections (list): Список пар индексов точек для соединения
        color (tuple): Цвет линий (BGR)
        thickness (int): Толщина линий
        
    Returns:
        numpy.ndarray: Кадр с нарисованными соединениями
    """
    output = frame.copy()
    
    if keypoints is None:
        return output
    
    # Стандартная схема соединений для руки (MediaPipe)
    if connections is None:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # большой палец
            (0, 5), (5, 6), (6, 7), (7, 8),  # указательный палец
            (0, 9), (9, 10), (10, 11), (11, 12),  # средний палец
            (0, 13), (13, 14), (14, 15), (15, 16),  # безымянный палец
            (0, 17), (17, 18), (18, 19), (19, 20),  # мизинец
            (5, 9), (9, 13), (13, 17),  # соединения между основаниями пальцев
            (0, 5), (0, 17)  # соединения к центру ладони
        ]
    
    # Рисуем соединения
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx in keypoints and end_idx in keypoints:
            start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            cv2.line(output, start_point, end_point, color, thickness)
    
    return output


def draw_trajectory(frame, trajectory, color=(0, 0, 255), thickness=2, fade=True, current_point_color=(255, 0, 0)):
    """
    Рисует траекторию движения указательного пальца на кадре
    
    Args:
        frame (numpy.ndarray): Входной кадр
        trajectory (list): Список точек траектории [(x1, y1), (x2, y2), ...]
        color (tuple): Цвет траектории (BGR)
        thickness (int): Толщина линии
        fade (bool): Использовать ли градиентное затухание цвета
        current_point_color (tuple): Цвет текущей точки (BGR)
        
    Returns:
        numpy.ndarray: Кадр с нарисованной траекторией
    """
    output = frame.copy()
    
    # Рисуем линии траектории
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
            pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
            
            if fade:
                # Градиентное затухание цвета: чем старее точка, тем прозрачнее линия
                alpha = 0.3 + 0.7 * (i / len(trajectory))  # от 0.3 до 1.0
                if isinstance(color, tuple):
                    line_color = tuple([int(c * alpha) for c in color])
                else:
                    line_color = color
            else:
                line_color = color
                
            cv2.line(output, pt1, pt2, line_color, thickness)
    
    # Рисуем текущую точку (последнюю в траектории)
    if trajectory:
        current_point = (int(trajectory[-1][0]), int(trajectory[-1][1]))
        cv2.circle(output, current_point, radius=5, color=current_point_color, thickness=-1)
    
    return output


def create_trajectory_plot(trajectory, output_path, title="Траектория движения пальца", invert_y=True):
    """
    Создает и сохраняет график траектории движения
    
    Args:
        trajectory (list): Список точек траектории [(x1, y1), (