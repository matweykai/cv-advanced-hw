import os
import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.custom_model import load_model, heatmaps_to_keypoints, preprocess_image
from models.mediapipe_model import load_mediapipe_model


def process_video(video_path, output_path, model_type="mediapipe", model_path=None, 
                 finger_index=8, show_result=False):
    """
    Обрабатывает видео, отслеживает движение пальца и создает график траектории движения.
    
    Args:
        video_path (str): Путь к входному видео
        output_path (str): Путь для сохранения графика траектории
        model_type (str): Тип используемой модели ('mediapipe' или 'custom')
        model_path (str): Путь к весам модели (нужен только для model_type='custom')
        finger_index (int): Индекс отслеживаемого пальца (8 для указательного по схеме)
        show_result (bool): Показывать ли обработку в реальном времени
    
    Returns:
        dict: Результаты обработки
    """
    # Проверяем наличие входного видео
    if not os.path.exists(video_path):
        raise ValueError(f"Видеофайл не найден: {video_path}")
    
    # Проверяем расширение выходного файла
    _, ext = os.path.splitext(output_path)
    if not ext or ext.lower() not in ['.jpg', '.jpeg', '.png', '.pdf']:
        # Если расширение не указано или неподдерживаемое, добавляем .png
        output_path = output_path.rstrip('.') + ".png"
        print(f"Расширение изменено, новый путь: {output_path}")
    
    # Создаем директорию для выходного файла, если её нет
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Загружаем соответствующую модель
    if model_type.lower() == "mediapipe":
        print("Используется модель MediaPipe Hands")
        model = load_mediapipe_model()
    elif model_type.lower() == "custom":
        if not model_path:
            raise ValueError("Для использования custom модели необходимо указать путь к весам (model_path)")
        if not os.path.exists(model_path):
            raise ValueError(f"Файл модели не найден: {model_path}")
        
        print(f"Используется пользовательская модель: {model_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Устройство: {device}")
        model = load_model(model_path, device)
    else:
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}. Используйте 'mediapipe' или 'custom'")
    
    # Открываем входное видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео {video_path}")
    
    # Получаем параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Обработка видео: {video_path}")
    print(f"Размер: {width}x{height}, FPS: {fps}, Всего кадров: {total_frames}")
    
    # Список для хранения полной траектории
    full_trajectory = []
    
    # Счетчик кадров и прогресс-бар
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Обработка видео")
    
    # Засекаем время выполнения
    start_time = time.time()
    
    # Обработка видео
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Определяем ключевые точки в зависимости от типа модели
                if model_type.lower() == "mediapipe":
                    # Используем MediaPipe
                    keypoints = model.detect_hands(frame)
                else:
                    # Используем пользовательскую модель
                    # Сохраняем оригинальные размеры
                    orig_h, orig_w = frame.shape[:2]
                    
                    # Предобработка изображения и получение тензора
                    image_tensor = preprocess_image(frame).to(device)
                    
                    # Получение предсказаний модели
                    heatmaps = model(image_tensor)
                    
                    # Получение координат ключевых точек
                    keypoints_array = heatmaps_to_keypoints(heatmaps.cpu())[0]
                    
                    # Масштабируем координаты к оригинальному размеру
                    keypoints_scaled = keypoints_array * np.array([orig_w / 256, orig_h / 256])
                    
                    # Преобразуем в формат словаря {index: (x, y)}
                    keypoints = {i: (float(x), float(y)) for i, (x, y) in enumerate(keypoints_scaled)}
                
                # Добавляем координату пальца в траекторию, если она валидна
                if keypoints and finger_index in keypoints:
                    finger_point = keypoints[finger_index]
                    full_trajectory.append((finger_point[0], finger_point[1]))
                
                # Отображаем обработку в реальном времени, если нужно
                if show_result:
                    frame_vis = frame.copy()
                    
                    # Рисуем ключевые точки в зависимости от типа модели
                    if model_type.lower() == "mediapipe":
                        frame_vis = model.draw_landmarks(frame_vis, keypoints)
                    else:
                        # Рисуем ключевые точки вручную
                        if keypoints:
                            for idx, (x, y) in keypoints.items():
                                cv2.circle(frame_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
                    
                    # Рисуем текущую траекторию
                    if len(full_trajectory) > 1:
                        for j in range(1, len(full_trajectory)):
                            pt1 = (int(full_trajectory[j-1][0]), int(full_trajectory[j-1][1]))
                            pt2 = (int(full_trajectory[j][0]), int(full_trajectory[j][1]))
                            cv2.line(frame_vis, pt1, pt2, (0, 0, 255), 2)
                    
                    cv2.imshow('Finger Tracking', frame_vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
            except Exception as e:
                print(f"Ошибка при обработке кадра {frame_idx}: {e}")
            
            frame_idx += 1
            pbar.update(1)
    
    # Вычисляем общее время выполнения
    elapsed_time = time.time() - start_time
    
    # Освобождаем ресурсы
    cap.release()
    if show_result:
        cv2.destroyAllWindows()
    
    pbar.close()
    
    print(f"Обработка завершена за {elapsed_time:.2f} секунд")
    print(f"Общее количество точек в траектории: {len(full_trajectory)}")
    
    # Создаем график траектории
    success = False
    if full_trajectory:
        try:
            create_trajectory_plot(full_trajectory, output_path)
            if os.path.exists(output_path):
                print(f"График траектории успешно сохранен: {output_path}")
                success = True
            else:
                print(f"Предупреждение: Файл {output_path} не был создан.")
        except Exception as e:
            print(f"Ошибка при создании графика: {e}")
    else:
        print("Предупреждение: Не удалось обнаружить траекторию движения пальца")
    
    return {
        'output_image': output_path if success else None,
        'full_trajectory': full_trajectory,
        'frames_processed': frame_idx,
        'elapsed_time': elapsed_time,
        'success': success
    }


def create_trajectory_plot(trajectory, output_path, title="Траектория движения пальца"):
    """
    Создает и сохраняет график траектории движения (как в исходном файле vid.py)
    
    Args:
        trajectory (list): Список точек траектории [(x1, y1), (x2, y2), ...]
        output_path (str): Путь для сохранения графика
        title (str): Заголовок графика
        
    Returns:
        bool: True если график успешно сохранен, иначе False
    """
    if not trajectory:
        print("Пустая траектория, график не создан")
        return False
    
    # Создаем директорию для сохранения, если она не существует
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Создана директория: {output_dir}")
        except Exception as e:
            print(f"Ошибка при создании директории {output_dir}: {e}")
            # Попробуем сохранить в текущую директорию
            output_path = os.path.basename(output_path)
            print(f"Изменен путь сохранения на: {output_path}")
    
    try:
        plt.figure(figsize=(10, 8))
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        
        # Рисуем линию траектории
        plt.plot(xs, ys, 'b-')
        
        # Градиентная окраска точек по времени (как в исходном коде)
        plt.scatter(xs, ys, c=range(len(xs)), cmap='cool', s=50)
        
        plt.title(title)
        plt.gca().invert_yaxis()  # Инвертируем ось Y для соответствия координатам изображения
        
        # Сохраняем график
        plt.savefig(output_path)
        plt.close()
        
        # Проверяем, что файл действительно создан
        if os.path.exists(output_path):
            print(f"График траектории успешно сохранен: {output_path}")
            return True
        else:
            print(f"Файл не был создан, несмотря на отсутствие ошибок: {output_path}")
            return False
            
    except Exception as e:
        print(f"Ошибка при создании графика: {e}")
        # Попробуем сохранить в текущую директорию как запасной вариант
        try:
            backup_path = "trajectory_plot_backup.png"
            plt.savefig(backup_path)
            plt.close()
            if os.path.exists(backup_path):
                print(f"Резервный график сохранен как: {backup_path}")
                return True
            else:
                print(f"Резервный файл не был создан, несмотря на отсутствие ошибок")
                return False
        except Exception as e2:
            print(f"Не удалось сохранить резервный график: {e2}")
            return False


def extract_frames_from_video(video_path, output_dir):
    """
    Извлекает кадры из видео и сохраняет их в указанную директорию
    
    Args:
        video_path (str): Путь к видеофайлу
        output_dir (str): Директория для сохранения кадров
        
    Returns:
        int: Количество извлеченных кадров
    """
    # Создаем папку, если её нет
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
        
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_idx += 1

    cap.release()
    print(f"Сохранено {frame_idx} кадров в папку: {output_dir}")
    
    return frame_idx