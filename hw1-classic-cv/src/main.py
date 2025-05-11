import argparse
import yaml
import os
import pandas as pd
import torch

# Проверка доступности GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {DEVICE}")

def load_config(path="configs/config.yaml"):
    """Загрузка конфигурации из YAML файла"""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def ensure_result_dir():
    """Создает директорию для результатов, если она не существует"""
    if not os.path.exists("result"):
        os.makedirs("result")
    return "result"

def save_duplicates_list(detected, output_dir="result"):
    """Сохраняет список дубликатов в CSV файл и выводит статистику"""
    ensure_result_dir()
    
    # Создаем простой список файлов-дубликатов
    if isinstance(detected, pd.DataFrame):
        # Если результат - DataFrame (от ResNet matcher)
        duplicate_files = detected[detected['IsLeaked'] == 1]['Image'].tolist()
        num_duplicates = len(duplicate_files)
        
        # Создаем формат словаря, похожий на классический подход
        detected_dict = {}
        for filename in duplicate_files:
            detected_dict[filename] = {"match": "Найдено в обучающем наборе", "score": 1.0}
        
        # Создаем датафрейм с одной колонкой
        df = pd.DataFrame(duplicate_files, columns=['duplicate_filename'])
    else:
        # Если результат - словарь (от классического matcher)
        duplicate_files = list(detected.keys())
        num_duplicates = len(duplicate_files)
        
        # Создаем датафрейм с одной колонкой
        df = pd.DataFrame(duplicate_files, columns=['duplicate_filename'])
    
    # Сохраняем в CSV
    csv_path = os.path.join(output_dir, "duplicate_files.csv")
    df.to_csv(csv_path, index=False)
    
    # Выводим информацию о результатах
    if num_duplicates > 0:
        print(f"\n Найдено {num_duplicates} дубликатов.")
        print(f"📂 Список сохранен в файл: {csv_path}")
    else:
        print("\n Дубликаты не найдены.")
    
    return df

def save_groups_list(groups, output_dir="result"):
    """Сохраняет список групп в CSV файл и выводит статистику"""
    ensure_result_dir()
    
    # Обрабатываем группы в зависимости от их типа
    if isinstance(groups, pd.DataFrame):
        # Если уже датафрейм
        df = groups
        num_groups = df['cluster'].nunique() if 'cluster' in df.columns else 0
        total_files = len(df)
    else:
        # Подсчет общего количества файлов во всех группах
        total_files = sum(len(g) for g in groups)
        num_groups = len(groups)
        
        # Создаем датафрейм для групп
        group_data = []
        for group_id, files in enumerate(groups):
            for file in files:
                group_data.append({
                    'group_id': group_id + 1,
                    'filename': file
                })
        
        df = pd.DataFrame(group_data)
    
    # Сохраняем в CSV
    csv_path = os.path.join(output_dir, "grouped_files.csv")
    df.to_csv(csv_path, index=False)
    
    # Выводим информацию о результатах
    if num_groups > 0:
        print(f"\n Сформировано {num_groups} групп, содержащих всего {total_files} файлов.")
        print(f" Распределение файлов по группам:")
        
        if isinstance(groups, pd.DataFrame):
            group_counts = df.groupby('cluster').size()
            for group_id, count in group_counts.items():
                print(f"  Группа {group_id + 1}: {count} файлов")
        else:
            for i, group in enumerate(groups):
                print(f"  Группа {i+1}: {len(group)} файлов")
                
        print(f" Результаты сохранены в файл: {csv_path}")
    else:
        print("\n Не удалось сформировать группы.")
    
    return df

def run_classical(task):
    if task == "match":
        from src.classical.matcher import find_duplicates
        config = load_config()
        train_path = config["paths"]["train_dir"]
        test_path = config["paths"]["test_dir"]
        print("🔍 Запуск поиска дубликатов методом classical...")
        detected = find_duplicates(train_path, test_path)
        
        # Сохраняем список дубликатов и выводим информацию о количестве
        save_duplicates_list(detected)
        return detected
    
    elif task == "group":
        from src.classical.grouper import group_images
        config = load_config()
        train_path = config["paths"].get("train_dir")
        test_path = config["paths"].get("test_dir")
        print(" Запуск группировки изображений методом classical...")
        result = group_images(train_path, test_path, eps=30, min_samples=2)
        
        # Сохраняем группы и выводим информацию о количестве
        save_groups_list(result)
        return result

def run_resnet(task):
    # Устанавливаем PyTorch для использования подходящего устройства
    if task == "match":
        from src.resnet.matcher import match_resnet
        config = load_config()
        train_path = config["paths"]["train_dir"]
        test_path = config["paths"]["test_dir"]
        print(" Запуск поиска дубликатов методом ResNet...")
        result = match_resnet(train_path, test_path)
        
        # Сохраняем список дубликатов и выводим информацию о количестве
        save_duplicates_list(result)
        return result
    
    elif task == "group":
        from src.resnet.grouper import group_images
        config = load_config()
        train_path = config["paths"]["train_dir"]
        test_path = config["paths"]["test_dir"]
        print(" Запуск группировки изображений методом ResNet...")
        result = group_images(train_path, test_path)
        
        # Сохраняем группы и выводим информацию о количестве
        save_groups_list(result)
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["classical", "resnet"], required=True, help="Метод анализа: classical или resnet")
    parser.add_argument("--task", choices=["match", "group"], required=True, help="Задача: match (поиск дубликатов) или group (группировка)")
    args = parser.parse_args()

    # Проверяем существование папки result
    ensure_result_dir()
    
    print(f"Запуск задачи: {args.task} с методом: {args.method}")
    
    try:
        if args.method == "classical":
            run_classical(args.task)
        else:
            run_resnet(args.task)
        
        print("\n Задача успешно завершена. Результаты сохранены в папку 'result'.")
    except Exception as e:
        print(f"\n Ошибка при выполнении задачи: {str(e)}")
        import traceback
        traceback.print_exc()