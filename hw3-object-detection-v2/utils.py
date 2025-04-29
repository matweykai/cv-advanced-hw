import torch
import json
import os
import config
import matplotlib.patches as patches
import torchvision
import torchvision.transforms as T
from PIL import ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def get_iou(p, a):
    p_tl, p_br = bbox_to_coords(p)          # (batch, S, S, B, 2)
    a_tl, a_br = bbox_to_coords(a)

    # Largest top-left corner and smallest bottom-right corner give the intersection
    coords_join_size = (-1, -1, -1, config.B, config.B, 2)
    tl = torch.max(
        p_tl.unsqueeze(4).expand(coords_join_size),         # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
        a_tl.unsqueeze(3).expand(coords_join_size)          # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coords_join_size),
        a_br.unsqueeze(3).expand(coords_join_size)
    )

    intersection_sides = torch.clamp(br - tl, min=0.0)
    intersection = intersection_sides[..., 0] \
                   * intersection_sides[..., 1]       # (batch, S, S, B, B)

    p_area = bbox_attr(p, 2) * bbox_attr(p, 3)                  # (batch, S, S, B)
    p_area = p_area.unsqueeze(4).expand_as(intersection)        # (batch, S, S, B, 1) -> (batch, S, S, B, B)

    a_area = bbox_attr(a, 2) * bbox_attr(a, 3)                  # (batch, S, S, B)
    a_area = a_area.unsqueeze(3).expand_as(intersection)        # (batch, S, S, 1, B) -> (batch, S, S, B, B)

    union = p_area + a_area - intersection

    # Catch division-by-zero
    zero_unions = (union == 0.0)
    union[zero_unions] = config.EPSILON
    intersection[zero_unions] = 0.0

    return intersection / union


def bbox_to_coords(t):
    """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)


def scheduler_lambda(epoch):
    if epoch < config.WARMUP_EPOCHS + 75:
        return 1
    elif epoch < config.WARMUP_EPOCHS + 105:
        return 0.1
    else:
        return 0.01


def load_class_dict():
    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, 'r') as file:
            return json.load(file)
    new_dict = {}
    save_class_dict(new_dict)
    return new_dict


def load_class_array():
    classes = load_class_dict()
    result = [None for _ in range(len(classes))]
    for c, i in classes.items():
        result[i] = c
    return result


def save_class_dict(obj):
    folder = os.path.dirname(config.CLASSES_PATH)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(config.CLASSES_PATH, 'w') as file:
        json.dump(obj, file, indent=2)


def get_dimensions(label):
    size = label['annotation']['size']
    return int(size['width']), int(size['height'])


def get_bounding_boxes(label):
    width, height = get_dimensions(label)
    x_scale = config.IMAGE_SIZE[0] / width
    y_scale = config.IMAGE_SIZE[1] / height
    boxes = []
    objects = label['annotation']['object']
    for obj in objects:
        box = obj['bndbox']
        coords = (
            int(int(box['xmin']) * x_scale),
            int(int(box['xmax']) * x_scale),
            int(int(box['ymin']) * y_scale),
            int(int(box['ymax']) * y_scale)
        )
        name = obj['name']
        boxes.append((name, coords))
    return boxes


def bbox_attr(data, i):
    """Returns the Ith attribute of each bounding box in data."""

    attr_start = config.C + i
    return data[..., attr_start::5]


def scale_bbox_coord(coord, center, scale):
    return ((coord - center) * scale) + center


def get_overlap(a, b):
    """Returns proportion overlap between two boxes in the form (tl, width, height, confidence, class)."""

    a_tl, a_width, a_height, _, _ = a
    b_tl, b_width, b_height, _, _ = b

    i_tl = (
        max(a_tl[0], b_tl[0]),
        max(a_tl[1], b_tl[1])
    )
    i_br = (
        min(a_tl[0] + a_width, b_tl[0] + b_width),
        min(a_tl[1] + a_height, b_tl[1] + b_height),
    )

    intersection = max(0, i_br[0] - i_tl[0]) \
                   * max(0, i_br[1] - i_tl[1])

    a_area = a_width * a_height
    b_area = b_width * b_height

    a_intersection = b_intersection = intersection
    if a_area == 0:
        a_intersection = 0
        a_area = config.EPSILON
    if b_area == 0:
        b_intersection = 0
        b_area = config.EPSILON

    return torch.max(
        a_intersection / a_area,
        b_intersection / b_area
    ).item()


def non_max_suppression(boxes_xywh, scores, iou_threshold):
    """
    Performs Non-Maximum Suppression (NMS) on bounding boxes using torchvision.

    Args:
        boxes_xywh (torch.Tensor): Bounding boxes in [x_center, y_center, width, height] format. Shape: (N, 4).
        scores (torch.Tensor): Confidence scores for each box. Shape: (N,).
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        torch.Tensor: Indices of the boxes to keep. Shape: (K,).
    """
    if boxes_xywh.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes_xywh.device)

    # Convert [xc, yc, w, h] to [x1, y1, x2, y2]
    x_center, y_center, widths, heights = boxes_xywh.unbind(dim=1)
    x1 = x_center - widths / 2
    y1 = y_center - heights / 2
    x2 = x_center + widths / 2
    y2 = y_center + heights / 2
    boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=1)

    # Ensure tensors are on the same device if not already
    # boxes_xyxy = boxes_xyxy.to(device)
    # scores = scores.to(device)

    keep_indices = torchvision.ops.nms(boxes_xyxy, scores, iou_threshold)
    return keep_indices


def plot_boxes(data, labels, classes, color='orange', min_confidence=0.2, max_overlap=0.5, file=None):
    """Plots bounding boxes on the given image."""

    grid_size_x = data.size(dim=2) / config.S
    grid_size_y = data.size(dim=1) / config.S
    m = labels.size(dim=0)
    n = labels.size(dim=1)

    bboxes = []
    for i in range(m):
        for j in range(n):
            for k in range((labels.size(dim=2) - config.C) // 5):
                bbox_start = 5 * k + config.C
                bbox_end = 5 * (k + 1) + config.C
                bbox = labels[i, j, bbox_start:bbox_end]
                class_index = torch.argmax(labels[i, j, :config.C]).item()
                confidence = labels[i, j, class_index].item() * bbox[4].item()          # pr(c) * IOU
                if confidence > min_confidence:
                    width = bbox[2] * config.IMAGE_SIZE[0]
                    height = bbox[3] * config.IMAGE_SIZE[1]
                    tl = (
                        bbox[0] * config.IMAGE_SIZE[0] + j * grid_size_x - width / 2,
                        bbox[1] * config.IMAGE_SIZE[1] + i * grid_size_y - height / 2
                    )
                    bboxes.append([tl, width, height, confidence, class_index])

    # Sort by highest to lowest confidence
    bboxes = sorted(bboxes, key=lambda x: x[3], reverse=True)

    # Non-maximum suppression and render image
    image = T.ToPILImage()(data)
    draw = ImageDraw.Draw(image)
    keep_indices_all_classes = set()
    bboxes_by_class = defaultdict(list)
    indices_by_class = defaultdict(list)

    # Group boxes by class index and store original index
    for idx, box_data in enumerate(bboxes):
        class_idx = box_data[4]
        bboxes_by_class[class_idx].append(box_data)
        indices_by_class[class_idx].append(idx)

    # Apply NMS for each class separately
    for class_idx, class_boxes_data in bboxes_by_class.items():
        if not class_boxes_data:
            continue

        # Prepare tensors for NMS function
        # Convert [[tl, w, h, conf, cls], ...] to tensors
        # Box format needs conversion: [tl_x, tl_y, w, h] -> [center_x, center_y, w, h]
        # Assuming tl coords are relative to image top-left (0,0) and w,h are relative widths/heights
        # Let's assume the coordinates were already scaled to image size in pixels earlier
        boxes_for_nms = []
        scores_for_nms = []
        original_indices_for_class = indices_by_class[class_idx]

        for box_data in class_boxes_data:
            tl, width, height, confidence, _ = box_data
            # Convert tl, w, h (pixels) to xc, yc, w, h (pixels)
            xc = tl[0] + width / 2
            yc = tl[1] + height / 2
            boxes_for_nms.append([xc, yc, width, height])
            scores_for_nms.append(confidence)

        if not boxes_for_nms:
            continue

        boxes_tensor = torch.tensor(boxes_for_nms, dtype=torch.float32, device=device)
        scores_tensor = torch.tensor(scores_for_nms, dtype=torch.float32, device=device)

        # Apply NMS using the new function
        keep_local_indices = non_max_suppression(boxes_tensor, scores_tensor, max_overlap) # Use max_overlap from plot_boxes args

        # Map local kept indices back to original indices from the initial bboxes list
        for local_idx in keep_local_indices:
            original_idx = original_indices_for_class[local_idx.item()]
            keep_indices_all_classes.add(original_idx)

    # Draw only the kept boxes
    for i in range(len(bboxes)):
        if i in keep_indices_all_classes:
            tl, width, height, confidence, class_index = bboxes[i]

            # Annotate image
            draw.rectangle((tl, (tl[0] + width, tl[1] + height)), outline='orange')
            text_pos = (max(0, tl[0]), max(0, tl[1] - 11))
            text = f'{classes[class_index]} {round(confidence * 100, 1)}%'
            text_bbox = draw.textbbox(text_pos, text)
            draw.rectangle(text_bbox, fill='orange')
            draw.text(text_pos, text)
    if file is None:
        image.show()
    else:
        output_dir = os.path.dirname(file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not file.endswith('.png'):
            file += '.png'
        image.save(file)