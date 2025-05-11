# группировка изображений в датафрем с номером группы
import pandas as pd
from src.utils.metrics import load_images_from_folder, extract_features, compute_similarity_matrix, cluster_images


def group_images(train_path, test_path, eps=30, min_samples=2):

    # Загружаем все изображения
    all_images = {}
    all_images.update(load_images_from_folder(train_path))
    all_images.update(load_images_from_folder(test_path))
    
    # Извлекаем признаки
    features = extract_features(all_images)
    
    # Вычисляем матрицу сходства
    sim_matrix, names = compute_similarity_matrix(features)
    
    # Группируем изображения
    clusters = cluster_images(sim_matrix, names, eps, min_samples)
    
    # Построение датафрейма из результата кластеризации
    rows = []
    for cluster_id, filenames in enumerate(clusters):
        for fname in filenames:
            rows.append({'filename': fname, 'cluster': cluster_id})

    df = pd.DataFrame(rows)
    print(df)
    return df

