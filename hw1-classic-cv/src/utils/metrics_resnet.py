import cv2
import os
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import models, transforms
from torchvision.models import resnet50
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import gc
from collections import deque

# Получаем устройство - используем GPU, если доступен
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Кэширование модели ResNet для эффективности
_model_cache = {}
def get_resnet_model():
    """Получение модели ResNet с кэшированием для эффективности"""
    global _model_cache
    if 'resnet50' not in _model_cache:
        # Загружаем предобученную модель ResNet
        model = resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # удаляем классификатор
        model.eval().to(DEVICE)
        _model_cache['resnet50'] = model
    return _model_cache['resnet50']

# Вычисление эмбеддинга
def extract_embedding(image):
    """Извлечение эмбеддинга из изображения с помощью ResNet"""
    # Преобразование для входа в модель
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    # Получаем модель из кэша
    model = get_resnet_model()
    
    # Обработка изображения
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(image).squeeze().cpu().numpy()
    return embedding

def load_images_from_folder(folder):
    """Загрузка изображений из папки"""
    filenames = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = []
    for filename in filenames:
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        else:
            filenames.remove(filename)
    return images, filenames

# Поиск похожих изображений
def find_similar_images_resnet(test_images, test_filenames, train_images, train_filenames, threshold=0.90):
    """Поиск похожих изображений с использованием ResNet"""
    results = []

    print("Извлечение эмбеддингов для обучающих изображений...")
    train_embeddings = [extract_embedding(img) for img in tqdm(train_images)]

    print("Извлечение эмбеддингов для тестовых изображений...")
    test_embeddings = [extract_embedding(img) for img in tqdm(test_images)]

    print("Сравнение эмбеддингов...")
    for test_emb, test_name in zip(test_embeddings, test_filenames):
        similarities = cosine_similarity([test_emb], train_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        best_match_name = train_filenames[best_match_idx]

        is_leaked = int(best_score > threshold)  # 1 — найдено в train, 0 — не найдено
        
        # Сохраняем более подробные результаты
        results.append({
            "Image": test_name, 
            "IsLeaked": is_leaked,
            "Score": float(best_score),
            "BestMatch": best_match_name if is_leaked else ""
        })

    df = pd.DataFrame(results)
    return df 

# ГРУППИРОВКА

# Сбор признаков
def list_image_paths(folder):
    """Получение списка путей к изображениям"""
    return {fname: os.path.join(folder, fname) for fname in os.listdir(folder)
            if fname.lower().endswith(('.jpg', '.png', '.jpeg'))}

def extract_features_resnet(image_paths, batch_size=16):
    """
    Извлечение признаков с использованием ResNet50 с пакетной обработкой для эффективности
    """
    # Получаем кэшированную модель или загружаем её
    model = get_resnet_model()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    features = {}
    names = list(image_paths.keys())
    
    # Обработка пакетами для эффективности
    for i in tqdm(range(0, len(names), batch_size), desc="Извлечение признаков (ResNet50)"):
        batch_names = names[i:i+batch_size]
        batch_tensors = []
        
        # Загрузка и предобработка пакета
        for name in batch_names:
            try:
                path = image_paths[name]
                img = cv2.imread(path)
                if img is None:
                    print(f"Не удалось загрузить изображение: {path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = preprocess(img).unsqueeze(0)
                batch_tensors.append((name, tensor))
            except Exception as e:
                print(f"Ошибка обработки {name}: {e}")
                
        if not batch_tensors:
            continue
            
        # Извлечение признаков для пакета
        batch_names = [item[0] for item in batch_tensors]
        batch_input = torch.cat([item[1] for item in batch_tensors]).to(DEVICE)
        
        with torch.no_grad():
            batch_output = model(batch_input).squeeze().cpu().numpy()
            
            # Обработка случая с одним изображением
            if len(batch_input) == 1:
                batch_output = batch_output.reshape(1, -1)
                
            # Нормализация и сохранение признаков
            for idx, name in enumerate(batch_names):
                output = batch_output[idx]
                output /= np.linalg.norm(output)  # нормализация для косинусной меры сходства
                features[name] = output

        # Очистка памяти
        del batch_input, batch_output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    return features

# Построение групп
def build_groups(features, threshold=0.95):
    """Формирование групп похожих изображений"""
    names = list(features.keys())
    vectors = np.array([features[name] for name in names])

    similarity_matrix = cosine_similarity(vectors)
    n = len(names)
    visited = set()
    groups = []

    for idx in range(n):
        if names[idx] in visited:
            continue

        group = []
        queue = deque()
        queue.append(idx)
        visited.add(names[idx])

        while queue:
            current_idx = queue.popleft()
            group.append(names[current_idx])

            for neighbor_idx in range(n):
                if neighbor_idx != current_idx and similarity_matrix[current_idx, neighbor_idx] >= threshold:
                    if names[neighbor_idx] not in visited:
                        visited.add(names[neighbor_idx])
                        queue.append(neighbor_idx)

        if len(group) >= 2:  # Включаем только группы с как минимум 2 изображениями
            groups.append(group)

    return groups