import os
import torch
import timm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import fiftyone as fo
import fiftyone.zoo as foz
import argparse

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class EmbeddingNet(nn.Module):
    def __init__(self, backbone_name='resnet18', embedding_dim=128, pretrained=True):
        """
        Параметры:
            backbone_name (str): Имя модели из timm (например, 'resnet18').
            embedding_dim (int): Размерность выходного эмбеддинга.
            pretrained (bool): Использовать ли предобученные веса.
        """
        super(EmbeddingNet, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        backbone_features = self.backbone.num_features
        self.fc = nn.Linear(backbone_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def prepare_knn_model(model, dataloader, device, k_val) -> KNeighborsClassifier:
    model.eval()
    
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model(images)
            labels = labels.to(device)
            for emb, label in zip(embeddings, labels):
                label = label.item()

                embeddings_list.append(emb.cpu().numpy())
                labels_list.append(label)

    knn_classifier = KNeighborsClassifier(k_val)

    embeddings_matrix = np.vstack(embeddings_list)

    assert embeddings_matrix.shape == (len(labels_list), 128)

    return knn_classifier.fit(embeddings_matrix, labels_list)


def validate_classification(model: EmbeddingNet, knn_model: KNeighborsClassifier, dataloader: DataLoader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)

            # Compute similarity to prototypes
            pred_classes = knn_model.predict(embeddings.cpu().numpy())

            total += labels.size(0)
            correct += (torch.from_numpy(pred_classes).to(device) == labels).sum().item()

    accuracy = correct / total
    return accuracy


class Caltech256ClassificationDataset(Dataset):
    def __init__(self, samples, transform=None, label_to_idx=None):
        """
        Параметры:
            samples (list): Список кортежей (filepath, label) – путь к изображению и строковая метка.
            transform: Трансформации для изображения.
            label_to_idx (dict): Словарь отображения строковой метки в числовой индекс.
        """
        self.transform = transform
        if label_to_idx is None:
            labels = sorted({label for _, label in samples})
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx
        self.samples = [(filepath, self.label_to_idx[label]) for filepath, label in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        img = Image.open(filepath).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)


def main(args):
    # Путь к сохранённой модели и настройки
    model_path = args.model_path
    backbone_name = 'levit_128'
    embedding_dim = 128
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Загружаем датасет Caltech256 через FiftyOne
    dataset = foz.load_zoo_dataset("caltech256")
    print(f"Загружен Caltech256: {len(dataset)} образцов")

    # Читаем CSV с валидационными сэмплами (столбец filename)
    val_df = pd.read_csv("val.csv")
    val_filenames = set(val_df["filename"].tolist())

    train_samples = []
    val_samples = []

    for sample in dataset:
        filename = os.path.basename(sample.filepath)
        if "ground_truth" in sample and sample["ground_truth"] is not None:
            label = sample["ground_truth"]["label"]
        else:
            label = sample.get("label", None)
        if label is None:
            continue
        if filename in val_filenames:
            val_samples.append((sample.filepath, label))
        else:
            train_samples.append((sample.filepath, label))

    print(f"Обучающих сэмплов: {len(train_samples)}")
    print(f"Валидационных сэмплов: {len(val_samples)}")

    # Создаём отображение меток: label -> числовой индекс
    all_labels = {label for _, label in (train_samples + val_samples)}
    labels_sorted = sorted(all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels_sorted)}

    # Создаём датасеты для обучения и валидации
    train_dataset = Caltech256ClassificationDataset(train_samples, transform=transform, label_to_idx=label_to_idx)
    val_dataset = Caltech256ClassificationDataset(val_samples, transform=transform, label_to_idx=label_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Загружаем модель и веса
    model = EmbeddingNet(backbone_name=backbone_name, embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print("Подготовка KNN классификатора..")
    knn_classifier = prepare_knn_model(model, train_loader, device, k_val=args.k_val)

    print("Валидация модели по задаче классификации...")
    accuracy = validate_classification(model, knn_classifier, val_loader, device)
    print(f"Accuracy: {accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--k_val", type=int, default=3)

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
