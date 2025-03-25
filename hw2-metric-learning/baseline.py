import os
import random
import pandas as pd
from PIL import Image

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
import timm

import fiftyone.zoo as foz
from pytorch_metric_learning.samplers import MPerClassSampler


class TripletFODataset(Dataset):
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

        # Построим словарь: для каждого класса список индексов образцов данного класса
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.samples):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

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

        # Выбираем позитив: другое изображение того же класса
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(self.class_to_indices[anchor_label])
        positive_filepath, _ = self.samples[positive_index]
        positive_img = Image.open(positive_filepath).convert("RGB")
        if self.transform:
            positive_img = self.transform(positive_img)

        # Выбираем негатив: изображение из другого класса
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.class_to_indices.keys()))
        negative_index = random.choice(self.class_to_indices[negative_label])
        negative_filepath, negative_label = self.samples[negative_index]
        negative_img = Image.open(negative_filepath).convert("RGB")
        if self.transform:
            negative_img = self.transform(negative_img)

        # Приводим метки к тензорам
        return (anchor_img, positive_img, negative_img,
                torch.tensor(anchor_label), torch.tensor(negative_label))


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


def train_one_epoch(model, dataloader, optimizer, device, sum_writer: SummaryWriter, pow_val=1.0, margin=1.0, cutoff=0.5, semi_hard=True, distance_weighted=False):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # Распаковка батча: anchor, positive, negative, anchor_label, negative_label
        anchor, positive, negative, anchor_label, negative_label = batch

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        anchor_label = anchor_label.to(device)
        negative_label = negative_label.to(device)

        optimizer.zero_grad()

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        if semi_hard:
            candidate_embeddings = torch.cat([anchor_out, negative_out], dim=0)
            candidate_labels = torch.cat([anchor_label, negative_label], dim=0)
            batch_loss = 0.0
            batch_size = anchor_out.size(0)

            for i in range(batch_size):
                d_ap = torch.norm(anchor_out[i] - positive_out[i], p=pow_val)
                mask = (candidate_labels != anchor_label[i])
                if mask.sum() == 0:
                    chosen_negative = negative_out[i]
                else:
                    candidate_emb = candidate_embeddings[mask]
                    d_an = torch.norm(anchor_out[i].unsqueeze(0) - candidate_emb, p=pow_val, dim=1)
                    semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + margin)
                    if semi_hard_mask.sum() > 0:
                        candidate_d_an = d_an[semi_hard_mask]
                        chosen_idx = torch.argmin(candidate_d_an)
                        chosen_negative = candidate_emb[semi_hard_mask][chosen_idx]
                    else:
                        chosen_negative = negative_out[i]
                d_an_final = torch.norm(anchor_out[i] - chosen_negative, p=pow_val)
                loss_i = torch.relu(d_ap - d_an_final + margin)
                batch_loss += loss_i
            loss = batch_loss / batch_size
        elif distance_weighted:
            p_val = 2

            d_ap = torch.norm(anchor_out - positive_out, p=p_val, dim=1)
            d_an = torch.norm(anchor_out - negative_out, p=p_val, dim=1)

            batch_size, emb_dim = anchor_out.shape

            with torch.no_grad():
                neg_distance_matrix = (2 - 2 * (anchor_out @ negative_out.T)).sqrt()
                neg_distance_matrix = neg_distance_matrix.clamp(min=cutoff)
                
                log_weights = (2.0 - float(emb_dim)) * neg_distance_matrix.log() - \
                    float(emb_dim - 3) / 2 * torch.log(torch.clamp(1 - 0.25 * neg_distance_matrix * neg_distance_matrix, min=1e-8))

                weights = torch.exp(log_weights - torch.max(log_weights))

                if weights.device != d_ap.device:
                    weights = weights.to(d_ap.device)

                weights_sum = torch.sum(weights, dim=1, keepdim=True)
                weights = weights / weights_sum

            d_an_final = []

            for temp_ind in range(batch_size):
                try:
                    neg_sample_index = np.random.choice(batch_size, 1, p=np.nan_to_num(weights[temp_ind].detach().cpu().numpy(), nan=0))
                except ValueError:
                    neg_sample_index = temp_ind

                d_an_final.append(d_an[neg_sample_index])
            
            d_an_final = torch.vstack(d_an_final)

            loss = torch.relu(d_ap - d_an_final + margin).mean()
        else:
            loss = nn.TripletMarginLoss(margin=margin, p=pow_val)(anchor_out, positive_out, negative_out)

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
            anchor, positive, negative, _, _ = batch

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate_recall_at_k(model, dataloader, k, device, pow_val=2):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Из батча берём только anchor и его метку
            anchor, _, _, labels, _ = batch
            anchor = anchor.to(device)
            emb = model(anchor)
            embeddings_list.append(emb)
            labels_list.append(labels.to(device))

    embeddings_all = torch.cat(embeddings_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    distances = torch.cdist(embeddings_all, embeddings_all, p=pow_val)
    sorted_indices = torch.argsort(distances, dim=1)

    hits = 0
    N = embeddings_all.size(0)
    for i in range(N):
        neighbors = sorted_indices[i, 1:k+1]
        if (labels_all[neighbors] == labels_all[i]).any():
            hits += 1

    recall_at_k = hits / N
    return recall_at_k


def main(args):
    run_name = args.run_name

    project_path = os.path.dirname(os.path.abspath(__file__))
    cur_models_dir = os.path.join(project_path, "models", run_name)

    if not os.path.exists(cur_models_dir):
        os.makedirs(cur_models_dir)

    # Init tensorboard
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
        transforms.RandomHorizontalFlip(),
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
    train_dataset = TripletFODataset(train_samples, transform=train_transform, label_to_idx=label_to_idx)
    val_dataset = TripletFODataset(val_samples, transform=valid_transform, label_to_idx=label_to_idx)

    # balanced_sampler = MPerClassSampler([item[1] for item in train_samples], m=args.m_per_class, batch_size=args.batch_size, length_before_new_iter=len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    model = EmbeddingNet(backbone_name="levit_128", embedding_dim=128, pretrained=True)
    model.to(device)

    # Hyperparams
    lr = args.learning_rate
    margin = args.margin
    pow_val = args.pow_val

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin, p=pow_val)

    num_epochs = 2
    k = 1

    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, writer, cutoff=args.cutoff, margin=margin, semi_hard=True if not args.distance_weighted else False, distance_weighted=args.distance_weighted)
        val_loss = validate(model, val_loader, criterion, device)
        recall_at_k = validate_recall_at_k(model, val_loader, k, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Recall@{k}: {recall_at_k:.4f}")

        writer.add_scalar("loss/train", train_loss, writer.global_step)
        writer.add_scalar("loss/val", val_loss, writer.global_step)
        writer.add_scalar("Recall@k", recall_at_k, writer.global_step)

        torch.save(model.state_dict(), f"{cur_models_dir}/model_epoch_{epoch + 1}.pth")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--pow_val", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--m_per_class", type=int, default=4)
    parser.add_argument("--distance_weighted", action="store_true")
    parser.add_argument("--cutoff", type=float, default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())