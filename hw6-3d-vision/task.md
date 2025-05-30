# Задание по теме 3D компьютерное зрение

### Разработка простейшей системы помощи водителю

1. Скачайте датасет NuScenes Mini (https://www.nuscenes.org/nuscenes)

   Для этого перейдите на вкладку Downloads, зарегистрируйтесь и скачайте Mini часть в разделе Full dataset.
   <img src="./download_window.png" alt="Окно со cкачиванием" style="zoom:40%;" />

   Изучите информацию о датасете, представленную на сайте, а также приложенный [ноутбук](./nuscenes_dataset.ipynb), демонстрирующий принципы работы с датасетом, необходимые для выполнения задания, и как можно считывать данные.

   Requirements для работы ноутбука:

   - Python <= 3.10
   - [requirements.txt](./requirements.txt)

2. Выберите оттуда одну из понравившихся дорожных сцен, и реализуйте для нее алгоритм, вычисляющий расстояние от ego vehicle до окружающих его движущихся объектов, по данным камеры и лидара.

   Описание алгоритма:

   - С помощью какого-либо 2D детектора (например, yolo из [ultralytics](https://docs.ultralytics.com/modes/predict/#inference-sources)) получите боксы для объектов, которые могут появиться на дороге, на изображении с камеры.
   - Спроецируйте лидарное облако точек на изображение с камеры. Код, осуществляющий проекцию, должен быть написан самостоятельно, а не просто откуда-то импортирован, разрешается использовать numpy/torch для матричных операций и [scipy](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.spatial.transform.Rotation.from_quat.html)/[pyquaternion](https://kieranwynn.github.io/pyquaternion/#basic-usage) для работы с углами.
   - Для каждого обнаруженного объекта оцените расстояние от ego vehicle до него по лидарным точкам, попавшим внутрь соответствующего бокса.


3. Соберите полученные результаты в демо-видео, на котором должны присутствовать изображение с камеры, найденные объекты, спроецированные лидарные точки, попавшие внутрь боксов объектов, полученные расстояния для найденных объектов, для каждого кадра выбранной сцены.
   Пример (https://github.com/thegoldenbeetle/3d-cv-assignment/blob/master/output.gif):

   ![Видео-демо с результатами](./output.gif)

4. По найденным 2D боксам и попавшим внутрь лидарным точкам придумайте способ получить 3D боксы объектов и оцените качество получившегося  3D детектора. Ничего страшного, если метрики получились невысокие и детектор не очень хорошо работает, главное разобраться в том, что представляет собой задача, в каком формате можно сохранить результат, и какие используются метрики оценки качества.
  <details>
  <summary> <b>Подсказки и идеи, которые могут помочь:</b> </summary>
  <ul>
      <li>Воспользуйтесь какой-либо моделью сегментации, чтоб точнее определить лидарные точки, принадлежащие объекту.
      <li>Воспользуйтесь какими-либо эвристиками, статистиками, методами кластеризации лидарных точек.
  	  <li>Воспользуйтесь методом поиска главных компонент (PCA) для нахождения угла yaw.
      <li>Для подсчета качества можно воспользоваться <a href="https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py">скриптом</a> из nuscenes-devkit.
  <ul>
  </details>

### Критерии оценивания

- Получены 2D боксы с помощью какого-либо 2D детектора - 2б
- Лидарное облако точек спроецировано на изображение одной из камер - 3б
- Для обнаруженных объектов получено расстояние от ego vehicle до этих объектов - 4б
- Полученные результаты собраны в демо-видео для одной из камер - 2б
- Код и демо расширено для работы со всеми камерами (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGTH, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGTH) - 2б
- По найденным 2D боксам и попавшим внутрь лидарным точкам получены 3D боксы - 2б,
  бонусом оцените качество полученного 3D детектора

В качестве отчетности необходимо предоставить код (ссылку на ваш репозиторий) с инструкциями по его запуску, комментариями, полученные демо-видео, в случае реализации получения 3D боксов - описание алгоритма.   
