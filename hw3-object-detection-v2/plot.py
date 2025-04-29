import torch
import config
import os
import utils
from tqdm import tqdm
from data_roboflow import YoloRoboflowDataset
from models import YOLOv1
from torch.utils.data import DataLoader


base_dir = 'models/yolo_v1'
if os.path.exists(base_dir):
    dates = os.listdir(base_dir)
    if dates:
        latest_date = max(dates)
        times = os.listdir(os.path.join(base_dir, latest_date))
        if times:
            latest_time = max(times)
            MODEL_DIR = os.path.join(base_dir, latest_date, latest_time)
            print(f"Using most recent model directory: {MODEL_DIR}")

def plot_test_images():
    classes = utils.load_class_array()

    dataset = YoloRoboflowDataset('valid', normalize=True, augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = YOLOv1()
    model.eval()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'weights', 'final'), map_location=utils.device))

    count = 0
    with torch.no_grad():
        for image, labels, original in tqdm(loader):
            predictions = model.forward(image)
            for i in range(image.size(dim=0)):
                utils.plot_boxes(
                    original[i, :, :, :],
                    predictions[i, :, :, :],
                    classes,
                    file=os.path.join('results', f'{count}')
                )
                count += 1


if __name__ == '__main__':
    plot_test_images()
