import os
import imagehash
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from collections import defaultdict


# Функция для вычисления перцептивного хеша
def compute_phash(image):
    # Преобразуем OpenCV изображение в формат PIL
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Вычисляем хеш
    hash_value = imagehash.phash(img_pil)
    return hash_value

# CLAHE для улучшения контраста
def apply_clahe(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed_images = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe_img = clahe.apply(gray)
        processed_images.append(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR))
    return processed_images

# Изменение размера изображений
def resize_images(images, size=(512, 512)):
    return [cv2.resize(img, size) for img in images]

# Вычисление цветовых гистограмм
def compute_color_histograms(images):
    histograms = []
    for img in images:
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
        histograms.append(hist)
    return histograms

# Вычисление SIFT-дескрипторов
def compute_sift_descriptors(images):
    sift = cv2.SIFT_create(nfeatures=1500, nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=10)
    descriptor_data = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        descriptor_data.append((keypoints, descriptors))
    return descriptor_data

# Аугментации изображений
def generate_augmented_versions(image):
    # Основные аугментации
    augmented = [image]
    
    # Добавляем повороты только для изображений достаточного размера
    if image.shape[0] > 100 and image.shape[1] > 100:
        augmented.extend([
            cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(image, cv2.ROTATE_180),
            cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ])
    
    # Добавляем отражения
    augmented.extend([
        cv2.flip(image, 1),  # Горизонтальное отражение
        cv2.flip(image, 0)   # Вертикальное отражение
    ])
    
    return augmented

# Определяем тип изображения (мультипликационное, фотография и т.д.)
def classify_image_type(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Анализ цветовой палитры
    unique_colors = np.unique(image.reshape(-1, 3), axis=0).shape[0]
    color_richness = unique_colors / (image.shape[0] * image.shape[1])
    
    if edge_ratio < 0.05 and color_richness < 0.1:
        return "simple"  # Простые изображения с небольшим количеством цветов
    elif edge_ratio > 0.15:
        return "detailed"  # Детализированные изображения
    else:
        return "cartoon"  # Мультипликационные изображения
    
#GROUPER
def load_images_from_folder(folder):
    images = {}
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is not None:
            images[fname] = img
    return images

def extract_features(images):
    orb = cv2.ORB_create()
    features = {}
    for name, img in images.items():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        features[name] = descriptors
    return features

def compute_similarity_matrix(features):
    names = list(features.keys())
    sim_matrix = np.zeros((len(names), len(names)))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            des1 = features[names[i]]
            des2 = features[names[j]]
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                score = -np.mean([m.distance for m in matches]) if matches else 0
                sim_matrix[i, j] = sim_matrix[j, i] = score
    return sim_matrix, names

def cluster_images(sim_matrix, names, eps=30, min_samples=2):
    db = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples)
    dist_matrix = -sim_matrix  
    labels = db.fit_predict(dist_matrix)
    clusters = defaultdict(list)

    for name, label in zip(names, labels):
        if label != -1:
            clusters[label].append(name)

    # Фильтруем группы, где меньше 3 изображений
    valid_clusters = [group for group in clusters.values() if len(group) > 2]
    return valid_clusters
