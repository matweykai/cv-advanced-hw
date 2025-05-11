import torch
import torchvision
from torchvision.models import resnet50
from src.utils.metrics_resnet import load_images_from_folder, find_similar_images_resnet


def match_resnet(train_path, test_path):
    # Основной вызов
    train_images, train_filenames = load_images_from_folder(train_path)
    test_images, test_filenames = load_images_from_folder(test_path)

    df_results = find_similar_images_resnet(test_images, test_filenames, train_images, train_filenames)
    return df_results