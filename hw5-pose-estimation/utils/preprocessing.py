import os
import cv2
import numpy as np


def extract_frames_from_video(video_path, output_dir=None, return_frames=False):
    """
    Извлекает кадры из видео
    
    Args:
        video_path (str): Путь к видеофайлу
        output_dir (str, optional): Директория для сохранения кадров. Если None, кадры не сохраняются
        return_frames (bool): Возвращать ли кадры как список numpy массивов
        
    Returns:
        list or int: Список кадров, если return_frames=True, иначе количество извлеченных кадров
    """
    # Создаем папку для кадров, если указана и её нет
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    frame_idx = 0
    frames = [] if return_frames else None
    
    # Получаем информацию о видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Обработка видео: {video_path}")
    print(f"Размер: {width}x{height}, FPS: {fps}, Всего кадров: {total_frames}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        if output_dir:
            frame_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        if return_frames:
            frames.append(frame)
        
        frame_idx += 1

    cap.release()
    
    if output_dir:
        print(f"Сохранено {frame_idx} кадров в папку: {output_dir}")
    
    return frames if return_frames else frame_idx


def preprocess_frame(frame):
    """
    Предобработка кадра для модели
    
    Args:
        frame (numpy.ndarray): Входной кадр в BGR формате
        
    Returns:
        numpy.ndarray: Предобработанный кадр в RGB формате
    """
    # Конвертация из BGR в RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Здесь можно добавить дополнительную предобработку при необходимости
    # Например, улучшение контрастности или фильтрацию шума
    
    return rgb_frame


def get_video_info(video_path):
    """
    Получает информацию о видео
    
    Args:
        video_path (str): Путь к видеофайлу
        
    Returns:
        dict: Словарь с информацией о видео (ширина, высота, fps, всего кадров)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    
    cap.release()
    
    return info