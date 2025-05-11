import os
import cv2
import numpy as np
from src.utils.metrics import apply_clahe, compute_phash, compute_sift_descriptors, compute_color_histograms, resize_images, generate_augmented_versions, classify_image_type


# Поиск похожих изображений с адаптивными порогами
def find_similar_images(test_images, test_filenames, train_images, train_filenames):
    similar_images = {}
    
    # Задаем разные пороги для разных типов изображений
    thresholds = {
        "simple": {"hist": 0.7, "ratio": 0.6, "matches": 40, "inliers": 10, "phash": 8},
        "cartoon": {"hist": 0.6, "ratio": 0.6, "matches": 60, "inliers": 15, "phash": 12},
        "detailed": {"hist": 0.5, "ratio": 0.6, "matches": 80, "inliers": 20, "phash": 15}
    }
    
    # Предварительная обработка изображений
    train_images_resized = resize_images(train_images)
    test_images_resized = resize_images(test_images)
    
    # Вычисление хешей
    train_hashes = [compute_phash(img) for img in train_images_resized]
    
    # Вычисление гистограмм
    train_hists = compute_color_histograms(train_images_resized)
    
    # Вычисление SIFT-дескрипторов
    train_data = compute_sift_descriptors(train_images_resized)
    
    # Инициализация FLANN-матчера
    index_params = dict(algorithm=1, trees=10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Для каждого тестового изображения
    for i, test_img in enumerate(test_images_resized):
        
        # Определяем тип изображения для адаптивных порогов
        img_type = classify_image_type(test_img)
        current_thresholds = thresholds[img_type]
        
        # Вычисляем хеш тестового изображения
        test_hash = compute_phash(test_img)
        
        # Вычисляем гистограмму тестового изображения
        test_hist = compute_color_histograms([test_img])[0]
        
        # Лучшее совпадение
        best_match = None
        best_score = 0
        best_details = {}
        
        # Сначала быстрая фильтрация по хешам
        potential_matches = []
        for j, train_hash in enumerate(train_hashes):
            hash_diffs = [(j, test_hash - h) for j, h in enumerate(train_hashes)]
            hash_diffs.sort(key=lambda x: x[1])
            potential_matches = [j for j, d in hash_diffs[:10] if d <= current_thresholds["phash"] + 5]
                    
        if not potential_matches:
            continue
        
        # Проверяем только потенциальные совпадения
        for j in potential_matches:
            # Сравниваем гистограммы
            hist_sim = cv2.compareHist(test_hist, train_hists[j], cv2.HISTCMP_CORREL)
            if hist_sim < current_thresholds["hist"]:
                continue
            
            # Если гистограммы достаточно похожи, переходим к SIFT
            augmented_images = generate_augmented_versions(test_img)
            
            for aug_img in augmented_images:
                test_kp, test_desc = compute_sift_descriptors([aug_img])[0]
                train_kp, train_desc = train_data[j]
                
                if test_desc is None or train_desc is None:
                    continue
                    
                if test_desc.shape[0] < 10 or train_desc.shape[0] < 10:
                    continue
                
                # Находим соответствия между дескрипторами
                matches = flann.knnMatch(test_desc, train_desc, k=2)
                
                # Применяем ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < current_thresholds["ratio"] * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) < current_thresholds["matches"]:
                    continue
                    
                # RANSAC для нахождения геометрического преобразования
                try:
                    src_pts = np.float32([test_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([train_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    inliers = int(mask.sum()) if mask is not None else 0
                    
                    if inliers < current_thresholds["inliers"]:
                        continue
                        
                    # Вычисляем общий балл сходства
                    inlier_ratio = inliers / len(good_matches) if good_matches else 0
                    match_quality = len(good_matches) / min(len(test_desc), len(train_desc))
                    
                    # Обновленная формула без hash_distance
                    similarity_score = 0.5 * hist_sim + 0.5 * inlier_ratio
                    
                    if similarity_score > best_score:
                        best_score = similarity_score
                        best_match = train_filenames[j]
                        best_details = {
                            "similarity_score": similarity_score,
                            "hist_similarity": hist_sim,
                            "inliers": inliers,
                            "good_matches": len(good_matches),
                            "match_quality": match_quality
                        }
                except:
                    continue
        
        # Если найдено хорошее совпадение
        if best_match and best_score > 0.6:  # Минимальный порог общего сходства
            similar_images[test_filenames[i]] = {
                "match": best_match,
                "score": best_score,
                "details": best_details
            }
    
    return similar_images

def find_duplicates(train_path, test_path, output_dir="matches"):
    # Загрузка изображений из папок
    train_filenames = [f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    test_filenames = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    train_images = []
    for filename in train_filenames:
        img = cv2.imread(os.path.join(train_path, filename))
        if img is not None:
            train_images.append(img)
        else:
            print(f"Ошибка при загрузке {filename}")
            train_filenames.remove(filename)

    test_images = []
    for filename in test_filenames:
        img = cv2.imread(os.path.join(test_path, filename))
        if img is not None:
            test_images.append(img)
        else:
            print(f"Ошибка при загрузке {filename}")
            test_filenames.remove(filename)

    # Находим похожие изображения
    similar_images = find_similar_images(test_images, test_filenames, train_images, train_filenames)

    # Получаем упорядоченный список файлов с дубликатами
    duplicate_filenames = sorted(similar_images.keys())

    print("\nНайденные дубликаты:")
    for filename in duplicate_filenames:
        print(filename)
    
    return similar_images