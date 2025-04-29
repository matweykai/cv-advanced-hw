import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import random
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import config

class VOCDataset(Dataset):

    def __init__(self, is_train, file_names, base_dir, image_size=448, grid_size=7, num_bboxes=2, num_classes=config.C):
        self.is_train = is_train
        self.image_size = image_size

        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        mean = [74.8052, 76.4244, 74.4321]
        self.mean = np.array(mean, dtype=np.float32)

        self.to_tensor = transforms.ToTensor()

        self.paths, self.boxes, self.labels = [], [], []

        for line in file_names:
            label_path = f"{base_dir}/{'train' if is_train else 'valid'}/{line}.xml"
            image_path = f"{base_dir}/{'train' if is_train else 'valid'}/{line}.jpg"
            
            # Parse XML file in VOC format
            tree = ET.parse(label_path)
            root = tree.getroot()
            
            box = []
            label = []
            
            # Extract object information from XML
            for obj in root.findall('object'):
                name = obj.find('name').text
                # Map class names to class indices (you may need to customize this)
                class_idx = 0 if name == 'worker' else 1  # Assuming 'worker' is 0, 'pig' is 1
                
                bbox = obj.find('bndbox')
                x1 = int(float(bbox.find('xmin').text))
                y1 = int(float(bbox.find('ymin').text))
                x2 = int(float(bbox.find('xmax').text))
                y2 = int(float(bbox.find('ymax').text))
                
                box.append([x1, y1, x2, y2])
                label.append(class_idx)
                
                if len(box) > 0:
                    self.boxes.append(torch.Tensor(box))
                    self.labels.append(torch.LongTensor(label))
                    self.paths.append(image_path)

        self.num_samples = len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        boxes = self.boxes[idx].clone()  # [n, 4]
        labels = self.labels[idx].clone()  # [n,]

        if self.is_train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.random_scale(img, boxes)

            img = self.random_blur(img)
            img = self.random_brightness(img)
            img = self.random_hue(img)
            img = self.random_saturation(img)

            img, boxes, labels = self.random_shift(img, boxes, labels)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        # # For debug.
        # debug_dir = 'tmp/voc_tta'
        # os.makedirs(debug_dir, exist_ok=True)
        # img_show = img.copy()
        # box_show = boxes.numpy().reshape(-1)
        # n = len(box_show) // 4
        # for b in range(n):
        #     pt1 = (int(box_show[4 * b + 0]), int(box_show[4 * b + 1]))
        #     pt2 = (int(box_show[4 * b + 2]), int(box_show[4 * b + 3]))
        #     cv2.rectangle(img_show, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=1)
        # cv2.imwrite(os.path.join(debug_dir, 'test_{}.jpg'.format(idx)), img_show)

        h, w, _ = img.shape
        boxes /= torch.Tensor([[w, h, w, h]]).expand_as(boxes)  # normalize (x1, y1, x2, y2) w.r.t. image width/height.
        target = self.encode(boxes, labels)  # [S, S, 5 x B + C]

        img = cv2.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # assuming the model is pretrained with RGB images.
        img = (img - self.mean) / 255.0  # normalize from -1.0 to 1.0.
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = self.to_tensor(img)

        return img, target

    def __len__(self):
        return self.num_samples

    def encode(self, boxes, labels):

        S, B, C = self.S, self.B, self.C
        N = 5 * B + C

        target = torch.zeros(S, S, N)
        cell_size = 1.0 / float(S)
        boxes_wh = boxes[:, 2:] - boxes[:, :2]  # width and height for each box, [n, 2]
        boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0  # center x & y for each box, [n, 2]
        for b in range(boxes.size(0)):
            xy, wh, label = boxes_xy[b], boxes_wh[b], int(labels[b])

            ij = (xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1])  # y & x index which represents its location on the grid.
            x0y0 = ij * cell_size  # x & y of the cell left-top corner.
            xy_normalized = (xy - x0y0) / cell_size  # x & y of the box on the cell, normalized from 0.0 to 1.0.

            # TBM, remove redundant dimensions from target tensor.
            # To remove these, loss implementation also has to be modified.
            for k in range(B):
                S = 5 * k
                target[j, i, S:S + 2] = xy_normalized
                target[j, i, S + 2:S + 4] = wh
                target[j, i, S + 4] = 1.0
            target[j, i, 5 * B + label] = 1.0

        return target

    @staticmethod
    def random_flip(img, boxes):
        if random.random() < 0.5:
            return img, boxes

        h, w, _ = img.shape

        img = np.fliplr(img)

        x1, x2 = boxes[:, 0], boxes[:, 2]
        x1_new = w - x2
        x2_new = w - x1
        boxes[:, 0], boxes[:, 2] = x1_new, x2_new

        return img, boxes

    @staticmethod
    def random_scale(img, boxes):
        if random.random() < 0.5:
            return img, boxes

        scale = random.uniform(0.8, 1.2)
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(int(w * scale), h), interpolation=cv2.INTER_LINEAR)
        scale_tensor = torch.FloatTensor([[scale, 1.0, scale, 1.0]]).expand_as(boxes)
        boxes = boxes * scale_tensor

        return img, boxes

    @staticmethod
    def random_blur(bgr):
        if random.random() < 0.5:
            return bgr

        ksize = random.choice([2, 3, 4, 5])
        bgr = cv2.blur(bgr, (ksize, ksize))
        return bgr

    @staticmethod
    def random_brightness(bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    @staticmethod
    def random_hue(bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.8, 1.2)
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    @staticmethod
    def random_saturation(bgr):
        if random.random() < 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    def random_shift(self, img, boxes, labels):
        if random.random() < 0.5:
            return img, boxes, labels

        center = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        h, w, c = img.shape
        img_out = np.zeros((h, w, c), dtype=img.dtype)
        mean_bgr = self.mean[::-1]
        img_out[:, :] = mean_bgr

        dx = random.uniform(-w * 0.2, w * 0.2)
        dy = random.uniform(-h * 0.2, h * 0.2)
        dx, dy = int(dx), int(dy)

        if dx >= 0 and dy >= 0:
            img_out[dy:, dx:] = img[:h - dy, :w - dx]
        elif dx >= 0 and dy < 0:
            img_out[:h + dy, dx:] = img[-dy:, :w - dx]
        elif dx < 0 and dy >= 0:
            img_out[dy:, :w + dx] = img[:h - dy, -dx:]
        elif dx < 0 and dy < 0:
            img_out[:h + dy, :w + dx] = img[-dy:, -dx:]

        center = center + torch.FloatTensor([[dx, dy]]).expand_as(center)  # [n, 2]
        mask_x = (center[:, 0] >= 0) & (center[:, 0] < w)  # [n,]
        mask_y = (center[:, 1] >= 0) & (center[:, 1] < h)  # [n,]
        mask = (mask_x & mask_y).view(-1, 1)  # [n, 1], mask for the boxes within the image after shift.

        boxes_out = boxes[mask.expand_as(boxes)].view(-1, 4)  # [m, 4]
        if len(boxes_out) == 0:
            return img, boxes, labels
        shift = torch.FloatTensor([[dx, dy, dx, dy]]).expand_as(boxes_out)  # [m, 4]

        boxes_out = boxes_out + shift
        boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
        boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
        boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
        boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

        labels_out = labels[mask.view(-1)]

        return img_out, boxes_out, labels_out

    def random_crop(self, img, boxes, labels):
        if random.random() < 0.5:
            return img, boxes, labels

        center = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        h_orig, w_orig, _ = img.shape
        h = random.uniform(0.6 * h_orig, h_orig)
        w = random.uniform(0.6 * w_orig, w_orig)
        y = random.uniform(0, h_orig - h)
        x = random.uniform(0, w_orig - w)
        h, w, x, y = int(h), int(w), int(x), int(y)

        center = center - torch.FloatTensor([[x, y]]).expand_as(center)  # [n, 2]
        mask_x = (center[:, 0] >= 0) & (center[:, 0] < w)  # [n,]
        mask_y = (center[:, 1] >= 0) & (center[:, 1] < h)  # [n,]
        mask = (mask_x & mask_y).view(-1, 1)  # [n, 1], mask for the boxes within the image after crop.

        boxes_out = boxes[mask.expand_as(boxes)].view(-1, 4)  # [m, 4]
        if len(boxes_out) == 0:
            return img, boxes, labels
        shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_out)  # [m, 4]

        boxes_out = boxes_out - shift
        boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
        boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
        boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
        boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

        labels_out = labels[mask.view(-1)]
        img_out = img[y:y + h, x:x + w, :]

        return img_out, boxes_out, labels_out


def test():
    from torch.utils.data import DataLoader

    base_dir = 'data'

    import os
    
    train_names = [f.rsplit('.jpg', 1)[0] for f in os.listdir(base_dir + '/train') if f.endswith('.jpg')]

    dataset = VOCDataset(is_train=True, file_names=train_names, base_dir=base_dir)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    data_iter = iter(data_loader)
    for i in range(100):
        img, target = next(data_iter)
        print(img.size(), target.size())
def calculate_mean_std():
    """
    Calculate the mean and standard deviation of the dataset images.
    
    Returns:
        mean: Tensor of mean values for each channel
        std: Tensor of standard deviation values for each channel
    """
    import numpy as np
    import cv2
    from tqdm import tqdm
    import os
    
    # Get train file names
    base_dir = 'data'
    train_names = [f.rsplit('.jpg', 1)[0] for f in os.listdir(base_dir + '/train') if f.endswith('.jpg')]
    
    # Initialize variables to store sum and sum of squares
    sum_rgb = np.zeros(3, dtype=np.float64)  # Use float64 to prevent overflow
    sum_rgb_squared = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    
    print("Calculating dataset mean and std...")
    for name in tqdm(train_names):
        # Load image
        img_path = f"{base_dir}/train/{name}.jpg"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Convert to float64 to prevent overflow during squaring
        img = img.astype(np.float64)
        
        # Update sums
        sum_rgb += np.sum(img, axis=(0, 1))
        sum_rgb_squared += np.sum(np.square(img), axis=(0, 1))
        pixel_count += img.shape[0] * img.shape[1]
    
    # Calculate mean and std
    mean = sum_rgb / pixel_count
    # Use a more numerically stable method to calculate standard deviation
    var = (sum_rgb_squared / pixel_count) - (mean ** 2)
    # Ensure variance is non-negative due to potential floating-point errors
    var = np.maximum(var, 0)
    std = np.sqrt(var)
    
    # Convert to PyTorch tensors and normalize to [0, 1]
    mean_tensor = torch.FloatTensor(mean)
    std_tensor = torch.FloatTensor(std)
    
    print(f"Dataset mean: {mean_tensor}")
    print(f"Dataset std: {std_tensor}")
    
    return mean_tensor, std_tensor

if __name__ == '__main__':
    # test()
    calculate_mean_std()