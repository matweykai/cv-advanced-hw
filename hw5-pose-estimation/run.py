#!/usr/bin/env python3
import os
import argparse
import sys
from app.main import process_video, extract_frames_from_video


def main():
    # Создаем основной парсер
    parser = argparse.ArgumentParser(
        description="Отслеживание движения указательного пальца и создание графика траектории",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Показывать значения по умолчанию
    )
    
    # Входные и выходные данные
    parser.add_argument("--video", type=str, required=True, 
                       help="Путь к входному видео")
    parser.add_argument("--output", type=str, default="trajectory_plot.png", 
                       help="Путь для сохранения графика траектории (.png, .jpg, .pdf)")
    
    # Создаем группу для параметров модели
    model_group = parser.add_argument_group('Параметры модели')
    
    # Добавляем выбор типа модели
    model_group.add_argument("--model-type", type=str, default="mediapipe", choices=["mediapipe", "custom"],
                       help="Тип модели: mediapipe (по умолчанию) или custom (ваша обученная модель)")
    model_group.add_argument("--model-path", type=str, default="pose_estimation_model.pth", 
                       help="Путь к файлу модели (используется только с --model-type=custom)")
    model_group.add_argument("--finger-index", type=int, default=8, 
                       help="Индекс отслеживаемого пальца: 0=запястье, 4=большой, 8=указательный, 12=средний, 16=безымянный, 20=мизинец")
    
    # Дополнительные параметры
    extra_group = parser.add_argument_group('Дополнительные параметры')
    extra_group.add_argument("--extract-frames", action="store_true", 
                           help="Извлечь кадры из видео")
    extra_group.add_argument("--frames-dir", type=str, default="frames", 
                           help="Директория для сохранения извлеченных кадров")
    extra_group.add_argument("--show", action="store_true", 
                           help="Показывать обработку в реальном времени")
    
    # Если аргументы не указаны, показываем помощь
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Выводим баннер и выбранные параметры
    print("\n" + "=" * 70)
    print(" Отслеживание движения пальца на видео ".center(70, "="))
    print("=" * 70)
    print(f"Входное видео: {args.video}")
    print(f"Выходной файл: {args.output}")
    print(f"Тип модели: {args.model_type}")
    if args.model_type == "custom":
        print(f"Путь к модели: {args.model_path}")
    print(f"Отслеживаемый палец: {get_finger_name(args.finger_index)} (индекс {args.finger_index})")
    print("=" * 70 + "\n")
    
    # Проверяем наличие входного видео
    if not os.path.exists(args.video):
        print(f"Ошибка: Видеофайл не найден: {args.video}")
        return
    
    # Проверяем наличие модели для custom типа
    if args.model_type.lower() == "custom" and not os.path.exists(args.model_path):
        print(f"Ошибка: Файл модели не найден: {args.model_path}")
        return
    
    # Извлекаем кадры, если требуется
    if args.extract_frames:
        extract_frames_from_video(args.video, args.frames_dir)
    
    # Проверяем расширение выходного файла и при необходимости исправляем
    _, ext = os.path.splitext(args.output)
    if not ext or ext.lower() not in ['.jpg', '.jpeg', '.png', '.pdf']:
        args.output = args.output.rstrip('.') + ".png"
        print(f"Добавлено расширение .png к выходному файлу: {args.output}")
    
    # Обрабатываем видео и создаем график траектории
    result = process_video(
        video_path=args.video, 
        output_path=args.output,
        model_type=args.model_type,
        model_path=args.model_path if args.model_type.lower() == "custom" else None,
        finger_index=args.finger_index,
        show_result=args.show
    )
    
    # Проверяем существование выходного файла
    if result and 'output_image' in result and os.path.exists(result['output_image']):
        print(f"Готово! График траектории сохранен в: {result['output_image']}")
    else:
        print("Внимание: Файл графика не был создан. Проверьте наличие ошибок выше.")
        # Пробуем найти файл по стандартному паттерну
        possible_output = args.output
        if os.path.exists(possible_output):
            print(f"Однако, файл {possible_output} существует. Возможно, сохранение прошло успешно.")


def get_finger_name(finger_index):
    """
    Возвращает название пальца по его индексу
    """
    finger_names = {
        0: "Запястье",
        1: "Основание большого",
        2: "Средняя фаланга большого",
        3: "Дистальная фаланга большого",
        4: "Кончик большого",
        5: "Основание указательного",
        6: "Средняя фаланга указательного",
        7: "Дистальная фаланга указательного",
        8: "Кончик указательного",
        9: "Основание среднего",
        10: "Средняя фаланга среднего",
        11: "Дистальная фаланга среднего",
        12: "Кончик среднего",
        13: "Основание безымянного",
        14: "Средняя фаланга безымянного",
        15: "Дистальная фаланга безымянного",
        16: "Кончик безымянного",
        17: "Основание мизинца",
        18: "Средняя фаланга мизинца",
        19: "Дистальная фаланга мизинца",
        20: "Кончик мизинца"
    }
    
    return finger_names.get(finger_index, f"Неизвестный палец ({finger_index})")


if __name__ == "__main__":
    main()