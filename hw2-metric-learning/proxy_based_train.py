import os
import random
import argparse
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import numpy as np

import fiftyone.zoo as foz
from torch.utils.tensorboard.writer import SummaryWriter


class EmbeddingNet(nn.Module):
    def __init__(self, backbone_name="resnet18", embedding_dim=128, pretrained=True):
        """
        Модель-эмбеддер, использующая бэкбон из timm и дополнительный FC слой.
        Параметры:
            backbone_name (str): Имя модели-бэкбона (например, "resnet18").
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
    

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, embeddings, targets):
        cos_sim = embeddings @ F.normalize(self.proxies.T)  # Calcluate cosine similarity
        P_one_hot = F.one_hot(targets, num_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos_sim - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos_sim + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss
    

class ImageDataset(Dataset):
    def __init__(self, samples, transform=None, label_to_idx=None):
        """
        Параметры:
            samples (list): Список кортежей (filepath, label) – путь к изображению и его строковая метка.
            transform: Трансформации для изображения.
            label_to_idx (dict): Словарь для отображения строковой метки в числовой индекс.
                            Если None, он будет вычислен по списку samples.
        """
        self.transform = transform
        # Если не передан mapping, вычисляем его из всех меток
        if label_to_idx is None:
            labels = sorted({label for _, label in samples})
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        # Преобразуем метки в числовые индексы
        self.samples = [(filepath, self.label_to_idx[label]) for filepath, label in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Возвращает кортеж:
        (anchor_img, positive_img, negative_img, anchor_label, negative_label)
        """
        filepath, anchor_label = self.samples[index]

        anchor_img = Image.open(filepath).convert("RGB")
        if self.transform:
            anchor_img = self.transform(anchor_img)

        # Приводим метки к тензорам
        return (anchor_img, torch.tensor(anchor_label))


def train_one_epoch(model, proxy_based_loss, dataloader, optimizer, device, sum_writer: SummaryWriter):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        model_pred = model(images)

        loss = proxy_based_loss(model_pred, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            sum_writer.add_scalar("loss/train", running_loss / (batch_idx + 1), sum_writer.global_step + batch_idx)
            print(f"Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.4f}")

    sum_writer.global_step += len(dataloader)
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            model_pred = model(images)

            loss = criterion(model_pred, labels)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate_recall_at_k(model, proxies: torch.Tensor, dataloader, k, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)

            emb = model(images)

            embeddings_list.append(emb)
            labels_list.append(labels.to(device))

    embeddings_all = torch.cat(embeddings_list, dim=0)  # all x dim
    labels_all = torch.cat(labels_list, dim=0)

    distances = 1 - torch.mm(embeddings_all, F.normalize(proxies).T)  # batch x batch
    sorted_indices = torch.argsort(distances, dim=1)

    hits = 0
    N = embeddings_all.size(0)
    for i in range(N):
        proxies_indices = sorted_indices[i, :k]
        if (proxies_indices == labels_all[i]).any():
            hits += 1

    recall_at_k = hits / N
    return recall_at_k


def main(args):
    # Hyperparams
    run_name = args.run_name
    lr = args.lr
    margin = args.margin
    alpha = args.alpha

    project_path = os.path.dirname(os.path.abspath(__file__))
    cur_models_dir = os.path.join(project_path, "models", run_name)

    if not os.path.exists(cur_models_dir):
        os.makedirs(cur_models_dir)

    # Config tensorboard
    writer = SummaryWriter(os.path.join(project_path, "runs", run_name))
    writer.global_step = 0

    layout = {
        "Train Process": {
            "loss": ["Multiline", ["loss/train", "loss/val"]],
        },
    }
    writer.add_custom_scalars(layout)

    # Загружаем датасет Caltech256 через FiftyOne
    dataset = foz.load_zoo_dataset("caltech256")
    print(f"Загружен Caltech256: {len(dataset)} образцов")

    # Читаем CSV с валидационными сэмплами (в столбце filename)
    val_df = pd.read_csv("val.csv")
    val_filenames = set(val_df["filename"].tolist())

    train_samples = []
    val_samples = []

    for sample in dataset:
        filename = os.path.basename(sample.filepath)
        # Предполагается, что метка хранится в поле ground_truth с ключом "label"
        if "ground_truth" in sample and sample["ground_truth"] is not None:
            label = sample["ground_truth"]["label"]
        else:
            # Если поле отсутствует, можно попробовать sample["label"]
            label = sample.get("label", None)
        if label is None:
            continue
        if filename in val_filenames:
            val_samples.append((sample.filepath, label))
        else:
            train_samples.append((sample.filepath, label))

    print(f"Обучающих сэмплов: {len(train_samples)}")
    print(f"Валидационных сэмплов: {len(val_samples)}")

    # Вычисляем общее отображение меток (label -> числовой индекс)
    all_labels = {label for _, label in (train_samples + val_samples)}
    labels_sorted = sorted(all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels_sorted)}

    # Определяем трансформации для изображений
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop((224, 224), pad_if_needed=True),
        # transforms.RandomRotation(30),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Создаем PyTorch-датасеты
    train_dataset = ImageDataset(train_samples, transform=train_transform, label_to_idx=label_to_idx)
    val_dataset = ImageDataset(val_samples, transform=valid_transform, label_to_idx=label_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    model = EmbeddingNet(backbone_name="levit_128", embedding_dim=128, pretrained=True)
    model.to(device)

    train_samples_class_count = len({label for _, label in train_samples})
    criterion = Proxy_Anchor(train_samples_class_count, 128, margin, alpha).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=lr)

    num_epochs = 2
    k = 1

    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device, writer)
        val_loss = validate(model, val_loader, criterion, device)
        recall_at_k = validate_recall_at_k(model, criterion.proxies, val_loader, k, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Recall@{k}: {recall_at_k:.4f}")

        writer.add_scalar("loss/train", train_loss, writer.global_step)
        writer.add_scalar("loss/val", val_loss, writer.global_step)
        writer.add_scalar("Recall@k", recall_at_k, writer.global_step)

        torch.save(model.state_dict(), f"{cur_models_dir}/model_epoch_{epoch + 1}.pth")
        torch.save(criterion.state_dict(), f"{cur_models_dir}/criterion_epoch_{epoch + 1}.pth")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=32.0)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
