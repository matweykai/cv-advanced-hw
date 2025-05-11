import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetResNet34(nn.Module):
    def __init__(self, num_keypoints=21):
        super().__init__()
        resnet = models.resnet34(pretrained=True)

        self.input_block = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels

        # Декодеры с проверкой размерностей
        self.up4 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.dec4 = ConvBlock(256+256, 256)

        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.dec3 = ConvBlock(128+128, 128)

        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.dec2 = ConvBlock(64+64, 64)

        self.up1 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.dec1 = ConvBlock(64+64, 64)

        self.final_conv = nn.Conv2d(64, num_keypoints, 1)

    def forward(self, x):
        # Запоминаем исходный размер для финального ресайза
        input_size = x.size()[2:]
        
        # Encoder path
        x0 = self.input_block(x)  # 1/4 размера входа
        x1 = self.encoder1(x0)    # 1/4 размера входа
        x2 = self.encoder2(x1)    # 1/8 размера входа
        x3 = self.encoder3(x2)    # 1/16 размера входа
        x4 = self.encoder4(x3)    # 1/32 размера входа
        
        # Decoder path с resize для гарантии совпадения размерностей
        d4 = self.up4(x4)  # 1/16 размера входа
        # Обеспечиваем одинаковый размер с x3
        if d4.size()[2:] != x3.size()[2:]:
            d4 = F.interpolate(d4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, x3], dim=1))
        
        d3 = self.up3(d4)  # 1/8 размера входа
        # Обеспечиваем одинаковый размер с x2
        if d3.size()[2:] != x2.size()[2:]:
            d3 = F.interpolate(d3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))
        
        d2 = self.up2(d3)  # 1/4 размера входа
        # Обеспечиваем одинаковый размер с x1
        if d2.size()[2:] != x1.size()[2:]:
            d2 = F.interpolate(d2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))
        
        d1 = self.up1(d2)  # 1/2 размера входа
        # Обеспечиваем одинаковый размер с x0
        if d1.size()[2:] != x0.size()[2:]:
            d1 = F.interpolate(d1, size=x0.size()[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))
        
        # Финальная свертка
        out = self.final_conv(d1)
        
        # Финальный ресайз до размера входа
        if out.size()[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        return out


def load_model(model_path, device=None):
    """
    Загружает предобученную модель из файла
    
    Args:
        model_path (str): Путь к файлу модели
        device (torch.device, optional): Устройство для загрузки модели
        
    Returns:
        torch.nn.Module: Загруженная модель
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    print(f"Загрузка модели на устройство: {device}")
    
    try:
        # Создаем модель с правильной архитектурой
        model = UNetResNet34(num_keypoints=21).to(device)
        
        # Загружаем веса из файла
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Переводим в режим оценки
        model.eval()
        
        print(f"Модель успешно загружена из {model_path}")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        raise


def heatmaps_to_keypoints(heatmaps):
    """
    Преобразует тепловые карты в координаты ключевых точек
    
    Args:
        heatmaps (torch.Tensor): Тензор тепловых карт [B, num_keypoints, H, W]
        
    Returns:
        numpy.ndarray: Массив координат ключевых точек [B, num_keypoints, 2]
    """
    B, K, H, W = heatmaps.shape
    keypoints = []
    for i in range(B):
        kp = []
        for j in range(K):
            hmap = heatmaps[i, j]
            y, x = torch.argmax(hmap.view(-1), dim=0) // W, torch.argmax(hmap.view(-1), dim=0) % W
            kp.append([x.item(), y.item()])
        keypoints.append(kp)
    return np.array(keypoints)


def preprocess_image(image, image_size=256):
    """
    Предобработка изображения для подачи в модель
    
    Args:
        image (numpy.ndarray): Входное изображение BGR
        image_size (int): Размер стороны квадратного изображения для модели
        
    Returns:
        torch.Tensor: Предобработанный тензор изображения [1, 3, H, W]
    """
    # Конвертация из BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Изменение размера
    image_resized = cv2.resize(image_rgb, (image_size, image_size))
    
    # Преобразование в тензор и нормализация
    image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
    
    # Применяем нормализацию как в обучении
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    # Добавляем размерность батча
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def detect_keypoints(model, image, device=None, image_size=256):
    """
    Обнаруживает ключевые точки на изображении с помощью предобученной модели
    
    Args:
        model (torch.nn.Module): Модель для обнаружения ключевых точек
        image (numpy.ndarray): Входное изображение BGR
        device (torch.device, optional): Устройство для вычислений
        image_size (int): Размер изображения для модели
        
    Returns:
        numpy.ndarray: Массив координат ключевых точек [num_keypoints, 2]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Предобработка изображения
    image_tensor = preprocess_image(image, image_size)
    image_tensor = image_tensor.to(device)
    
    # Получение прогноза
    with torch.no_grad():
        heatmaps = model(image_tensor)
    
    # Преобразование тепловых карт в координаты
    keypoints = heatmaps_to_keypoints(heatmaps.cpu())[0]
    
    # Масштабирование координат обратно к оригинальному размеру
    h, w = image.shape[:2]
    keypoints = keypoints * np.array([w / image_size, h / image_size])
    
    return keypoints


def draw_keypoints(image, keypoints, radius=3, color=(0, 255, 0)):
    """
    Рисует ключевые точки на изображении
    
    Args:
        image (numpy.ndarray): Входное изображение BGR
        keypoints (numpy.ndarray): Координаты ключевых точек [num_keypoints, 2]
        radius (int): Радиус кругов для отображения точек
        color (tuple): Цвет точек (BGR)
        
    Returns:
        numpy.ndarray: Изображение с отрисованными точками
    """
    img = image.copy()
    for (x, y) in keypoints:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
    return img


def draw_hand_connections(image, keypoints, connections=None, color=(0, 255, 0), thickness=2):
    """
    Рисует соединения между ключевыми точками руки
    
    Args:
        image (numpy.ndarray): Входное изображение BGR
        keypoints (numpy.ndarray): Координаты ключевых точек [num_keypoints, 2]
        connections (list): Список пар индексов точек для соединения
        color (tuple): Цвет линий (BGR)
        thickness (int): Толщина линий
        
    Returns:
        numpy.ndarray: Изображение с отрисованными соединениями
    """
    img = image.copy()
    
    # Стандартная схема соединений для руки (по 21 точке)
    if connections is None:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),    # большой палец
            (0, 5), (5, 6), (6, 7), (7, 8),    # указательный палец
            (0, 9), (9, 10), (10, 11), (11, 12),  # средний палец
            (0, 13), (13, 14), (14, 15), (15, 16),  # безымянный палец
            (0, 17), (17, 18), (18, 19), (19, 20),  # мизинец
            (5, 9), (9, 13), (13, 17),  # поперечные соединения ладони
        ]
    
    # Рисуем соединения
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            pt1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            pt2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            cv2.line(img, pt1, pt2, color, thickness)
    
    return img