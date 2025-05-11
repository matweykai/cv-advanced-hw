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
        trajectory (list): Список точек траектории [(x1, y1), (x2, y2), ...]
        output_path (str): Путь для сохранения графика
        title (str): Заголовок графика
        invert_y (bool): Инвертировать ли ось Y для соответствия координатам изображения
    """
    if not trajectory:
        print("Пустая траектория, график не создан")
        return
    
    plt.figure(figsize=(10, 8))
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    
    # Создаем цветовую карту для отображения временной последовательности
    colors = np.linspace(0, 1, len(trajectory))
    
    # Рисуем линию траектории
    plt.plot(xs, ys, 'b-', linewidth=1, alpha=0.7)
    
    # Рисуем точки с цветовой индикацией времени
    scatter = plt.scatter(xs, ys, c=colors, cmap='cool', s=30, zorder=2)
    
    # Добавляем цветовую шкалу
    cbar = plt.colorbar(scatter)
    cbar.set_label('Время (порядок точек)')
    
    # Добавляем стрелку направления в конце траектории
    if len(trajectory) > 1:
        last_x, last_y = trajectory[-1]
        prev_x, prev_y = trajectory[-2]
        dx = last_x - prev_x
        dy = last_y - prev_y
        
        # Нормализация вектора направления
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
            
            # Рисуем стрелку
            plt.arrow(last_x, last_y, dx*20, dy*20, head_width=15, head_length=15, 
                     fc='red', ec='red', zorder=3)
    
    plt.title(title)
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.grid(True, alpha=0.3)
    
    if invert_y:
        plt.gca().invert_yaxis()  # Инвертируем ось Y для соответствия координатам изображения
    
    # Сохраняем график
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"График траектории сохранен: {output_path}")


def create_trajectory_video(frames, trajectory, output_path, fps=30, include_plot=True):
    """
    Создает видео с отображением траектории на кадрах
    
    Args:
        frames (list): Список кадров
        trajectory (list): Список точек траектории [(x1, y1), (x2, y2), ...]
        output_path (str): Путь для сохранения видео
        fps (int): Частота кадров в секунду
        include_plot (bool): Включать ли график траектории рядом с видео
    """
    if not frames or not trajectory:
        print("Нет кадров или траектории, видео не создано")
        return
    
    # Получаем размеры кадра
    height, width = frames[0].shape[:2]
    
    # Создаем видеописатель
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if include_plot:
        # Создаем временный файл для графика
        plot_path = os.path.join(os.path.dirname(output_path), "temp_trajectory_plot.png")
        create_trajectory_plot(trajectory, plot_path)
        
        # Загружаем график
        plot_img = cv2.imread(plot_path)
        if plot_img is None:
            print(f"Ошибка при загрузке графика: {plot_path}")
            include_plot = False
        else:
            # Изменяем размер графика под высоту кадра
            plot_h, plot_w = plot_img.shape[:2]
            new_plot_w = int(plot_w * (height / plot_h))
            plot_img = cv2.resize(plot_img, (new_plot_w, height))
            
            # Общая ширина выходного кадра (видео + график)
            combined_width = width + new_plot_w
            
            # Создаем видеописатель для комбинированного видео
            out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, height))
    else:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Накопительная траектория
    current_trajectory = []
    
    # Обрабатываем каждый кадр
    for i, frame in enumerate(frames):
        # Добавляем точку к текущей траектории, если доступна
        if i < len(trajectory):
            current_trajectory.append(trajectory[i])
        
        # Рисуем траекторию на кадре
        frame_with_trajectory = draw_trajectory(frame, current_trajectory)
        
        if include_plot and 'plot_img' in locals():
            # Объединяем кадр и график
            combined_frame = np.zeros((height, combined_width, 3), dtype=np.uint8)
            combined_frame[:, :width] = frame_with_trajectory
            combined_frame[:, width:] = plot_img
            out.write(combined_frame)
        else:
            out.write(frame_with_trajectory)
    
    # Освобождаем ресурсы
    out.release()
    
    # Удаляем временный файл графика
    if include_plot and os.path.exists(plot_path):
        os.remove(plot_path)
    
    print(f"Видео с траекторией сохранено: {output_path}")


def add_text_info(frame, info_dict, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, 
                 font_scale=0.7, color=(255, 255, 255), thickness=2, line_spacing=25):
    """
    Добавляет текстовую информацию на кадр
    
    Args:
        frame (numpy.ndarray): Входной кадр
        info_dict (dict): Словарь с информацией для отображения {label: value}
        position (tuple): Начальная позиция текста (x, y)
        font, font_scale, color, thickness: Параметры шрифта
        line_spacing (int): Расстояние между строками
        
    Returns:
        numpy.ndarray: Кадр с добавленной информацией
    """
    output = frame.copy()
    x, y = position
    
    for label, value in info_dict.items():
        text = f"{label}: {value}"
        cv2.putText(output, text, (x, y), font, font_scale, color, thickness)
        y += line_spacing
    
    return output