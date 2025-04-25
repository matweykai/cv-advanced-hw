import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import random
import numpy as np
import cv2
import os


class PigDataset(Dataset):

    def __init__(self, is_train, file_names=None, base_dir=None, image_size=448, grid_size=7, num_bboxes=2, num_classes=20, 
                 labels_dir='labels', images_dir='extracted_frame', split_ratio=0.8):
        self.is_train = is_train
        self.image_size = image_size

        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        mean = [122.67891434, 116.66876762, 104.00698793]
        self.mean = np.array(mean, dtype=np.float32)

        self.to_tensor = transforms.ToTensor()

        self.paths, self.boxes, self.labels = [], [], []

        if file_names is not None:
            # If specific file names are provided
            for line in file_names:
                file_name = line.rstrip()
                label_path = os.path.join(base_dir, labels_dir, f"{file_name}.txt")
                image_path = os.path.join(base_dir, images_dir, f"{file_name}.jpg")
                self._process_file(label_path, image_path)
        else:
            # If no specific files, load all files from the directories and split by ratio
            label_files = [f for f in os.listdir(os.path.join(base_dir, labels_dir)) if f.endswith('.txt')]
            # Sort to ensure same order in train/val splits
            label_files.sort()
            
            # Determine split point
            split_idx = int(len(label_files) * split_ratio)
            
            # Select appropriate subset based on is_train flag
            selected_files = label_files[:split_idx] if is_train else label_files[split_idx:]
            
            for label_file in selected_files:
                file_base = os.path.splitext(label_file)[0]
                label_path = os.path.join(base_dir, labels_dir, label_file)
                image_path = os.path.join(base_dir, images_dir, f"{file_base}.jpg")
                self._process_file(label_path, image_path)

        self.num_samples = len(self.paths)
        
    def _process_file(self, label_path, image_path):
        # Only add the file if both label and image exist
        if os.path.exists(label_path) and os.path.exists(image_path):
            self.paths.append(image_path)
            with open(label_path) as f:
                objects = f.readlines()
                box = []
                label = []
                for object in objects:
                    parts = object.rstrip().split()
                    if len(parts) >= 5:  # Ensure we have enough parts
                        c = int(parts[0])
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        box.append([x1, y1, x2, y2])
                        label.append(c)
                if box:  # Only add if we found valid boxes
                    self.boxes.append(torch.Tensor(box))
                    self.labels.append(torch.LongTensor(label))
                else:
                    # Remove the path if no valid boxes were found
                    self.paths.pop()

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
        # img = (img - self.mean) / 255.0  # normalize from -1.0 to 1.0.
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

    base_dir = './'  # Adjust to your base directory
    
    # Test with automatic split
    train_dataset = PigDataset(is_train=True, base_dir=base_dir)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    val_dataset = PigDataset(is_train=False, base_dir=base_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Iterate through a few batches
    data_iter = iter(train_loader)
    for i in range(min(3, len(train_loader))):
        img, target = next(data_iter)
        print(f"Batch {i}: Image shape: {img.shape}, Target shape: {target.shape}")

if __name__ == "__main__":
    test()
