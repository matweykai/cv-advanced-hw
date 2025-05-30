# Отчет о реализации YOLOv1: Детекция свиней и людей

## 1. Введение

В данном отчете подробно описывается реализация модели обнаружения объектов YOLOv1 (You Only Look Once) с нуля, как описано в статье [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640). Основная цель заключалась в обнаружении свиней и людей с использованием предоставленного набора данных.

## 2. Подготовка данных

*   **Источник данных:** Набор данных был получен по ссылке [https://disk.yandex.ru/d/qXFgvtO3y-ey_A](https://disk.yandex.ru/d/qXFgvtO3y-ey_A).
*   **Формат аннотаций:** Pascal VOC XML (файлы `.xml` или `.rf.xml`).
*   **Шаги предобработки:** (`data_roboflow.py`)
    *   Загрузка изображений (RGB) и соответствующих XML аннотаций.
    *   Генерация словаря классов (`classes.json`) при первом запуске, если он отсутствует. Найдено 2 класса (предположительно, свиньи и люди).
    *   Изменение размера изображений до `(448, 448)` пикселей (`config.IMAGE_SIZE`).
    *   **Аугментация данных** (если `augment=True` при создании `YoloRoboflowDataset`):
        *   Случайное масштабирование (от 1.0x до 1.2x).
        *   Случайный сдвиг по горизонтали/вертикали (от -10% до +10% размера изображения).
        *   Изменение цветовых характеристик (`ColorJitter`: яркость, контрастность, насыщенность, оттенок).
        *   Соответствующее преобразование координат ограничивающих рамок.
    *   Преобразование изображений в тензоры PyTorch.
    *   Нормализация изображений с использованием средних значений и стандартных отклонений ImageNet (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).
    *   Преобразование аннотаций в выходной тензор ground truth формата `(S, S, 5*B + C)`, где `S=7`, `B=2`, `C=2`.
        *   Для каждой ячейки сетки `(S, S)` кодируется: вектор one-hot для класса (`C`), координаты центра `(x, y)`, ширина `(w)`, высота `(h)` и уверенность (`confidence`) для `B` рамок. Координаты и размеры нормализуются относительно размеров ячейки и изображения соответственно.

## 3. Реализация модели

### 3.1. Архитектура YOLOv1 (`models.py`, `config.py`)

*   Реализована архитектура сверточной нейронной сети в классе `YOLOv1`, основанная на статье YOLOv1.
*   **Основные компоненты:**
    *   Последовательность сверточных слоев (`nn.Conv2d`) с различными размерами ядер и количеством фильтров.
    *   Функция активации LeakyReLU (`nn.LeakyReLU(negative_slope=0.1)`) после большинства сверточных слоев.
    *   Слои Max Pooling (`nn.MaxPool2d`) для уменьшения пространственной размерности.
    *   Слой `nn.Flatten` перед полносвязными слоями.
    *   Два полносвязных слоя (`nn.Linear`): `S*S*1024 -> 4096` и `4096 -> S*S*(B*5+C)`.
    *   Слой Dropout (`nn.Dropout`) после первого полносвязного слоя для регуляризации.
*   **Параметры сетки:** Изображение делится на сетку `S x S`, где `S = 7`.
*   **Предсказания:** Каждая ячейка сетки предсказывает `B` ограничивающих рамок и `C` вероятностей классов. `B = 2`, `C = 2`.
*   **Выходной тензор:** Модель возвращает тензор формы `(batch_size, S, S, B*5 + C)`, т.е. `(batch_size, 7, 7, 12)`. Каждый из `B=2` предсказанных боксов содержит 5 значений: `(x, y, w, h, confidence)`.

### 3.2. Процедура обучения (`train.py`, `loss.py`, `config.py`)

*   Реализована функция потерь `SumSquaredErrorLoss`, основанная на сумме квадратов ошибок (MSE).
    *   **Компоненты потерь:**
        *   **Локализация:** MSE между предсказанными (`x, y`) и реальными координатами центра; MSE между предсказанными (`sqrt(w)`, `sqrt(h)`) и реальными корнями из размеров рамки. Учитывается только для "ответственного" предсказателя (`obj_ij`) в ячейке. Взвешивается с коэффициентом `l_coord = 5`.
        *   **Уверенность (Confidence):** MSE между предсказанной и реальной уверенностью (которая равна IoU предсказанной рамки с реальной рамкой, если объект есть, и 0, если нет). Рассчитывается отдельно для рамок, ответственных за объект (`obj_ij`), и для рамок, не ответственных или в ячейках без объекта (`noobj_ij`). Потери для `noobj_ij` взвешиваются с коэффициентом `l_noobj = 0.5`.
        *   **Классификация:** MSE между предсказанными и реальными вероятностями классов (one-hot). Учитывается только для ячеек, содержащих объект (`obj_i`).
    *   **Ответственный предсказатель:** В каждой ячейке, содержащей объект, предсказатель (bounding box `j`) с наибольшим IoU с реальной рамкой считается "ответственным" (`obj_ij`).
*   **Оптимизатор:** SGD (`torch.optim.SGD`) с параметрами: `learning_rate=1E-4`, `momentum=0.9`, `weight_decay=5E-4`.
*   **Расписание скорости обучения:** Не используется (закомментировано в `train.py`).
*   **Параметры обучения:**
    *   `BATCH_SIZE = 64`
    *   `EPOCHS = 135`
    *   `WARMUP_EPOCHS = 0`
*   **Логика обучения:** Стандартный цикл обучения PyTorch с использованием `DataLoader` (8 воркеров). Выполняется шаг оптимизатора после расчета потерь для каждого батча. Используется `torch.autograd.set_detect_anomaly(True)` для отладки NaN в потерях.
*   **Логирование:** Используется `SummaryWriter` (TensorBoard) для логирования потерь на обучающем (`Loss/train`) и валидационном (`Loss/test`) наборах каждые 4 эпохи. Метрики также сохраняются в `.npy` файлы.

### 3.3. Non-Maximum Suppression (NMS) (`utils.py`)

*   Реализован в функции `plot_boxes` (используется для визуализации/инференса).
*   **Алгоритм:**
    1.  Отфильтровываются все предсказания с уверенностью (`pr(class) * confidence`) ниже порога (`min_confidence=0.2`).
    2.  Оставшиеся рамки сортируются по уверенности в порядке убывания.
    3.  Итеративно выбирается рамка с наивысшей уверенностью.
    4.  Все остальные рамки того же класса, имеющие значительное пересечение (IoU > `max_overlap=0.5`) с выбранной рамкой, удаляются (добавляются в `discarded set`).
    5.  Процесс повторяется для следующей рамки с наивысшей уверенностью, которая еще не была удалена.
*   **Расчет IoU:** Используются функции `get_iou` (тензорная версия) и `get_overlap` (версия для отдельных рамок).

## 4. Оценка

### 4.1. Метрика Mean Average Precision (mAP)



* Реализована метрика Mean Average Precision (mAP) для оценки качества детекции объектов в `evaluate.py`.
* Процесс вычисления mAP включает:
  * Сбор предсказаний модели и ground truth аннотаций на валидационном наборе
  * Фильтрацию предсказаний по порогу уверенности и применение NMS
  * Сопоставление предсказаний с ground truth на основе IoU
  * Расчет кривых Precision-Recall для каждого класса
  * Вычисление Average Precision как площади под PR-кривой с интерполяцией
  * Усреднение AP по всем классам для получения mAP
* Реализована функция `calculate_iou_center_format` для вычисления IoU между боксами в формате [x_center, y_center, width, height]
* Результаты оценки сохраняются в JSON-файл с детализацией по классам


## 5. Результаты и эксперименты

### 5.1. Подбор гиперпараметров

*   **TODO:** Провести систематический подбор гиперпараметров для оптимизации производительности модели.
    *   [Перечислите гиперпараметры для подбора, например: скорость обучения, коэффициент регуляризации весов (weight decay), коэффициенты dropout, параметры аугментации данных, веса компонентов функции потерь]
    *   [Опишите стратегию подбора, например: поиск по сетке (grid search), случайный поиск (random search), байесовская оптимизация]

### 5.2. Логирование экспериментов

*   **TODO:** Настроить всестороннее логирование экспериментов.
    *   [Укажите инструменты/фреймворки для использования, например: TensorBoard, MLflow, Weights & Biases]
    *   Логировать ключевые метрики (mAP, компоненты потерь), использованные гиперпараметры и визуализации (например, примеры предсказаний, кривые потерь).

## 6. Заключение

Обобщена успешная реализация основной архитектуры YOLOv1, процедур обучения, NMS и метрики оценки mAP. Модель способна обрабатывать изображения и генерировать детекции для свиней и людей.

## 7. Дальнейшая работа / TODO

*   Завершить подбор гиперпараметров для поиска оптимальных настроек.
*   Реализовать надежное логирование и отслеживание экспериментов.
*   [Добавьте любые другие потенциальные улучшения или следующие шаги, например: дальнейшее исследование аугментации данных, сравнение с предварительно обученными базовыми сетями (backbones), оптимизация скорости инференса]

## 8. Результаты работы

*   Исходный код реализации YOLOv1.
*   Данный отчет, детализирующий проделанную работу.
