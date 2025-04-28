from fastapi import FastAPI, WebSocket
from track_4 import track_data, country_balls_amount
import asyncio
import glob
import random
from PIL import Image
from io import BytesIO
import re
import base64
import os
from models import FramePrediction
from metrics import TrackerMetricCollector


from soft_tracker import SoftTracker

app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')

soft_tracker = SoftTracker(forget_time=5, max_match_dist=100)
metrics_collector = TrackerMetricCollector()


def tracker_soft(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """
    # print('IN SOFT TRACKER')
    # print(el)

    det_box_ind_to_data = []
    det_boxes = []

    for temp_ind, temp_data in enumerate(el['data']):
        if len(temp_data['bounding_box']) != 0:
            det_box_ind_to_data.append(temp_ind)
            det_boxes.append(temp_data['bounding_box'])
    
    track_ids = soft_tracker.track(det_boxes)

    for data_ind, temp_track_id in zip(det_box_ind_to_data, track_ids):
        el['data'][data_ind]['track_id'] = temp_track_id

    metrics_collector.update(FramePrediction.model_validate(el))

    return el


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true и воспользуйтесь нижним закомментированным кодом в этом файле для первого прогона, 
    на повторном прогоне можете читать сохраненные фреймы из папки
    и по координатам вырезать необходимые регионы.
    """
    return el


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))
    for el in track_data:
        await asyncio.sleep(0.5)
        # TODO: part 1
        el = tracker_soft(el)
        # TODO: part 2
        # el = tracker_strong(el)
        # отправка информации по фрейму
        # json_el = json.dumps(el)
        print(el)
        await websocket.send_json(el)
    # добавьте сюда код рассчета метрики
    track_metric_val = metrics_collector.calculate()

    print(f'Track metric: {round(track_metric_val, 3)}')

    print('Bye..')


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     print('Accepting client connection...')
#     await websocket.accept()
#     await websocket.receive_text()
#     # отправка служебной информации для инициализации объектов
#     # класса CountryBall на фронте

#     dir = "save_frames_dir"
#     if not os.path.exists(dir):
#         os.makedirs(dir)

#     await websocket.send_text(str(country_balls))
#     for el in track_data:
#         await asyncio.sleep(0.5)
#         image_data = await websocket.receive_text()
#         # print(image_data)
#         try:
#             image_data = re.sub('^data:image/.+;base64,', '', image_data)
#             image = Image.open(BytesIO(base64.b64decode(image_data)))
#             image = image.resize((1000, 800), Image.Resampling.LANCZOS)
#             frame_id = el['frame_id'] - 1
#             image.save(f"{dir}/frame_{frame_id}.png")
#             # print(image)
#         except Exception as e:
#             print(e)
    
#         # отправка информации по фрейму
#         await websocket.send_json(el)

#     await websocket.send_json(el)
#     await asyncio.sleep(0.5)
#     image_data = await websocket.receive_text()
#     try:
#         image_data = re.sub('^data:image/.+;base64,', '', image_data)
#         image = Image.open(BytesIO(base64.b64decode(image_data)))
#         image = image.resize((1000, 800), Image.Resampling.LANCZOS)
#         image.save(f"{dir}/frame_{el['frame_id']}.png")
#     except Exception as e:
#         print(e)

#     print('Bye..')
