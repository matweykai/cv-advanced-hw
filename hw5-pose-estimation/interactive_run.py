#!/usr/bin/env python3
import os
import subprocess
import sys
import time
import glob


def clear_screen():
    """Очищает экран терминала."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Печатает заголовок программы."""
    print("\n" + "=" * 70)
    print(" ТРЕКИНГ ДВИЖЕНИЯ ПАЛЬЦА ".center(70, "="))
    print("=" * 70)


def find_video_files():
    """Находит все видеофайлы в папке test_video."""
    # Проверяем существование папки test_video
    if not os.path.exists("test_video"):
        print("\nОшибка: Папка 'test_video' не найдена.")
        print("Пожалуйста, убедитесь, что папка 'test_video' существует и содержит видеофайлы.")
        sys.exit(1)
    
    # Ищем все видеофайлы в папке test_video
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join("test_video", ext)))
    
    return video_files


def select_video_file(video_files):
    """Выбор видеофайла из списка найденных."""
    if not video_files:
        print("\nВ папке 'test_video' не найдено видеофайлов.")
        print("Поддерживаемые форматы: .mp4, .avi, .mov, .mkv")
        print("Пожалуйста, добавьте видеофайлы в папку 'test_video' и запустите программу снова.")
        sys.exit(1)
    
    if len(video_files) == 1:
        print(f"\nНайден видеофайл: {os.path.basename(video_files[0])}")
        print("Он будет использован автоматически.")
        return video_files[0]
    
    print("\nНайдены следующие видеофайлы:")
    for i, file in enumerate(video_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    while True:
        choice = input("\nВыберите номер файла для обработки: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(video_files):
                return video_files[idx]
            else:
                print(f"Пожалуйста, введите число от 1 до {len(video_files)}.")
        except ValueError:
            print("Пожалуйста, введите корректный номер.")


def select_model():
    """Запрашивает у пользователя тип модели."""
    print("\nВыберите модель для обнаружения руки:")
    print("1. MediaPipe (быстрее, работает везде)")
    print("2. Собственная модель (точнее для конкретных задач)")
    
    while True:
        choice = input("\nВаш выбор [1/2] (по умолчанию: 1): ").strip()
        if choice == "":
            return "mediapipe", None
        
        if choice == "1":
            return "mediapipe", None
        elif choice == "2":
            # Проверяем наличие модели в стандартных местах
            default_path = "pose_estimation_model.pth"
            if os.path.exists(default_path):
                print(f"Найдена модель: {default_path}")
                return "custom", default_path
            else:
                print(f"Предупреждение: Файл модели не найден: {default_path}")
                print("Будет использована модель MediaPipe.")
                return "mediapipe", None
        else:
            print("Пожалуйста, выберите 1 или 2.")


def select_finger():
    """Запрашивает у пользователя палец для отслеживания."""
    print("\nВыберите палец для отслеживания:")
    print("1. Указательный палец (кончик)")
    print("2. Большой палец (кончик)")
    print("3. Средний палец (кончик)")
    print("4. Безымянный палец (кончик)")
    print("5. Мизинец (кончик)")
    
    finger_indices = {
        "1": 8,   # Указательный
        "2": 4,   # Большой
        "3": 12,  # Средний
        "4": 16,  # Безымянный
        "5": 20,  # Мизинец
    }
    
    while True:
        choice = input("\nВаш выбор [1-5] (по умолчанию: 1): ").strip()
        if choice == "":
            return 8  # Указательный по умолчанию
        
        if choice in finger_indices:
            return finger_indices[choice]
        else:
            print("Пожалуйста, выберите число от 1 до 5.")


def generate_output_path(video_path):
    """Автоматически генерирует путь для сохранения результата."""
    # Создаем папку result, если её нет
    if not os.path.exists("result"):
        os.makedirs("result")
    
    # Получаем имя входного файла без расширения
    video_name = os.path.basename(video_path)
    name, _ = os.path.splitext(video_name)
    
    # Создаем путь для выходного файла
    output_path = os.path.join("result", f"{name}_trajectory.png")
    
    return output_path


def run_main_script(video_path, output_path, model_type, model_path, finger_index):
    """Запускает основной скрипт с указанными параметрами."""
    cmd = [sys.executable, "run.py", 
          f"--video={video_path}", 
          f"--output={output_path}", 
          f"--model-type={model_type}", 
          f"--finger-index={finger_index}"]
    
    if model_path:
        cmd.append(f"--model-path={model_path}")
    
    # Добавляем флаг для отображения процесса
    cmd.append("--show")
    
    print("\nЗапуск обработки со следующими параметрами:")
    print(f"Видео: {video_path}")
    print(f"Выходной файл: {output_path}")
    print(f"Тип модели: {model_type}")
    if model_path:
        print(f"Путь к модели: {model_path}")
    print(f"Индекс пальца: {finger_index}")
    
    print("\n" + "=" * 70)
    print(" НАЧАЛО ОБРАБОТКИ ".center(70, "="))
    print("=" * 70 + "\n")
    
    # Запуск команды
    try:
        process = subprocess.run(cmd, check=True)
        print("\n" + "=" * 70)
        print(" ОБРАБОТКА ЗАВЕРШЕНА ".center(70, "="))
        print("=" * 70 + "\n")
    except subprocess.CalledProcessError as e:
        print(f"\nОшибка при выполнении команды: {e}")
    
    # Открываем результат автоматически
    if os.path.exists(output_path):
        print(f"Результат сохранен в: {output_path}")
        try:
            if os.name == 'nt':  # Windows
                os.system(f'start {output_path}')
            elif os.name == 'posix':  # macOS или Linux
                if sys.platform == 'darwin':  # macOS
                    os.system(f'open {output_path}')
                else:  # Linux
                    os.system(f'xdg-open {output_path}')
            print("Результат открыт автоматически.")
        except Exception as e:
            print(f"Не удалось открыть файл: {e}")
    else:
        print(f"Файл не найден: {output_path}")


def main():
    try:
        clear_screen()
        print_header()
        
        # Находим и выбираем видеофайл из папки test_video
        video_files = find_video_files()
        video_path = select_video_file(video_files)
        
        # Выбираем модель и палец
        model_type, model_path = select_model()
        finger_index = select_finger()
        
        # Генерируем путь для сохранения результата
        output_path = generate_output_path(video_path)
        
        # Запускаем основной скрипт
        run_main_script(video_path, output_path, model_type, model_path, finger_index)
        
    except KeyboardInterrupt:
        print("\n\nПрограмма была прервана пользователем.")
        sys.exit(1)


if __name__ == "__main__":
    main()