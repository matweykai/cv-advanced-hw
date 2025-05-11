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
from collections import defaultdict, Counter
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def get_iou(predictions, a):
    p_tl, p_br = bbox_to_coords(predictions)          # (batch, S, S, B, 2)
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

    p_area = bbox_attr(predictions, 2) * bbox_attr(predictions, 3)                  # (batch, S, S, B)
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


def non_max_suppression(boxes_xywh, scores, iou_threshold, confidence_threshold=0.2):
    """
    Performs Non-Maximum Suppression (NMS) following the exact algorithm.

    Args:
        boxes_xywh (torch.Tensor): Bounding boxes in [x_center, y_center, width, height] format. Shape: (N, 4).
        scores (torch.Tensor): Confidence scores for each box. Shape: (N,).
        iou_threshold (float): IoU threshold for suppression.
        confidence_threshold (float): Minimum confidence score to keep a box.

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

    # Filter boxes by confidence threshold (Step 2)
    mask = scores >= confidence_threshold
    boxes = boxes_xyxy[mask]
    scores = scores[mask]
    
    # Sort boxes by confidence scores (Step 3)
    _, indices = scores.sort(descending=True)
    boxes = boxes[indices]
    scores = scores[indices]
    
    keep = []
    
    while boxes.size(0) > 0:
        keep.append(indices[0].item())
        
        if boxes.size(0) == 1:
            break
            
        iou = torchvision.ops.box_iou(boxes[0:1], boxes[1:])
        
        mask = iou[0] < iou_threshold
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
        indices = indices[1:][mask]
    
    return torch.tensor(keep, dtype=torch.int64, device=boxes_xywh.device)


def calculate_iou_midpoint(box1, box2):
    """
    Calculates IoU between two boxes in [x_center, y_center, width, height] format.
    
    Args:
        box1: First box coordinates [x_center, y_center, width, height]
        box2: Second box coordinates [x_center, y_center, width, height]
        
    Returns:
        Intersection over Union (IoU) value
    """
    # Convert from midpoint to corner format
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # Calculate intersection area
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = 0
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Calculate box areas
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    # Calculate union area and IoU
    union_area = box1_area + box2_area - inter_area + config.EPSILON  # Add epsilon to prevent division by zero
    
    return inter_area / union_area


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=config.C
):
    """
    Calculates mean average precision
    
    Parameters:
        pred_boxes: Can be either:
            - List of lists [train_idx, class_prediction, prob_score, x, y, w, h]
            - List of dictionaries with keys: 'image_idx', 'class_idx', 'confidence', 'bbox'
        true_boxes: Similar format as pred_boxes
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
        
    Returns:
        float: mAP value across all classes given a specific IoU threshold
        dict: AP values for each class (when dict format is used for boxes)
    """
    
    # Determine input format (list or dict)
    is_dict_format = isinstance(pred_boxes[0], dict) if pred_boxes else False
    
    # Convert dictionary format to list format if needed
    if is_dict_format:
        # Each dict has 'image_idx', 'class_idx', 'confidence', 'bbox' keys
        epsilon = 1e-6  # small value to avoid division by zero
        average_precisions = {}
        
        # Group ground truths by image index and class index for efficient lookup
        gt_by_image_class = defaultdict(list)
        # Count total number of ground truths for each class
        total_gts_per_class = defaultdict(int)
        for gt in true_boxes:
            key = (gt['image_idx'], gt['class_idx'])
            gt_by_image_class[key].append(gt)
            total_gts_per_class[gt['class_idx']] += 1
            # Reset used flag if it exists
            gt['used'] = False
        
        # Calculate AP for each class
        for class_idx in range(num_classes):
            # Get all detections for the current class
            class_detections = [d for d in pred_boxes if d['class_idx'] == class_idx]
            
            # Get the total number of ground truth objects for this class
            num_gts = total_gts_per_class[class_idx]
            
            # Skip if no ground truths for this class
            if num_gts == 0:
                average_precisions[class_idx] = float('nan')
                continue
                
            # Sort detections by confidence score (highest first)
            class_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Initialize TP and FP arrays
            tp = torch.zeros(len(class_detections))
            fp = torch.zeros(len(class_detections))
            
            # Reset 'used' flag for all ground truths of this class
            for gt_list in gt_by_image_class.values():
                for gt in gt_list:
                    gt['used'] = False
            
            # Match detections to ground truths
            for det_idx, detection in enumerate(class_detections):
                img_idx = detection['image_idx']
                # Get ground truths for this image and class
                gts_in_image = gt_by_image_class.get((img_idx, class_idx), [])
                
                best_iou = -1
                best_gt_idx = -1
                
                # Find best matching ground truth based on IoU
                for gt_idx, gt in enumerate(gts_in_image):
                    iou = calculate_iou_midpoint(detection['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Assign TP or FP based on IoU and whether GT was already used
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    matched_gt = gts_in_image[best_gt_idx]
                    if not matched_gt['used']:
                        tp[det_idx] = 1
                        matched_gt['used'] = True
                    else:
                        fp[det_idx] = 1
                else:
                    fp[det_idx] = 1
            
            # Calculate precision and recall
            tp_cumsum = torch.cumsum(tp, dim=0)
            fp_cumsum = torch.cumsum(fp, dim=0)
            
            recalls = tp_cumsum / (num_gts + epsilon)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + epsilon)
            
            # Add start and end points for AUC calculation
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            
            # Ensure precision is monotonically decreasing
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = max(precisions[i], precisions[i + 1])
            
            # Calculate area under PR curve
            ap = torch.trapz(precisions, recalls)
            average_precisions[class_idx] = ap.item()
        
        # Calculate mAP (mean of valid AP values)
        valid_aps = [ap for ap in average_precisions.values() if not np.isnan(ap)]
        mAP = sum(valid_aps) / (len(valid_aps) + epsilon) if valid_aps else 0.0
        
        return mAP, average_precisions
    
    # Original list format implementation
    else:
        avg_precisions = []
        epsilon = 1e-6 # used for numerical stability
        
        for c in range(num_classes):
            detections = [detection for detection in pred_boxes if detection[1] == c]
            ground_truths = [true_box for true_box in true_boxes if true_box[1] == c]
            
            amount_bboxes = Counter([gt[0] for gt in ground_truths])
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)
                
            # Sort by confidence score
            detections.sort(key=lambda x: x[2], reverse=True)
            
            TP = torch.zeros(len(detections))
            FP = torch.zeros(len(detections))
            total_true_bboxes = len(ground_truths)
            
            if total_true_bboxes == 0:
                continue
                
            for detection_idx, detection in enumerate(detections):
                # Get ground truths with the same image index
                ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
                
                best_iou = 0
                best_gt_idx = -1
                
                for idx, gt in enumerate(ground_truth_img):
                    # Use the boxes in midpoint format
                    if box_format == "midpoint":
                        # Boxes are in midpoint format: [x, y, w, h]
                        box1 = detection[3:7]
                        box2 = gt[3:7]
                        
                        iou = calculate_iou_midpoint(box1, box2)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx
                
                if best_iou > iou_threshold and best_gt_idx >= 0:
                    # Check if this ground truth has already been matched
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
            
            # Calculate precision and recall
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            
            # Add start and end points for AUC calculation
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            
            # Calculate area under PR curve
            avg_precisions.append(torch.trapz(precisions, recalls))

        return sum(avg_precisions) / (len(avg_precisions) + epsilon)

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
    print(len(bboxes))

    # Calculate IOUs between each pair of boxes
    num_boxes = len(bboxes)
    iou = [[0 for _ in range(num_boxes)] for _ in range(num_boxes)]
    for i in range(num_boxes):
        for j in range(num_boxes):
            iou[i][j] = get_overlap(bboxes[i], bboxes[j])

    # Non-maximum suppression and render image
    image = T.ToPILImage()(data)
    draw = ImageDraw.Draw(image)
    discarded = set()
    for i in range(num_boxes):
        if i not in discarded:
            tl, width, height, confidence, class_index = bboxes[i]

            # Decrease confidence of other conflicting bboxes
            for j in range(num_boxes):
                other_class = bboxes[j][4]
                if j != i and other_class == class_index and iou[i][j] > max_overlap:
                    discarded.add(j)

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

def convert_cell_to_image_coords(i, j, x, y, w, h):
    """
    Convert cell-relative coordinates to image-relative coordinates.
    
    Args:
        i, j: Cell indices in the grid
        x, y, w, h: Cell-relative coordinates
        
    Returns:
        Tuple of (x_center, y_center, width, height) in image-relative coordinates [0,1]
    """
    grid_size_x = 1.0 / config.S
    grid_size_y = 1.0 / config.S
    x_center = (j + x) * grid_size_x
    y_center = (i + y) * grid_size_y
    return x_center, y_center, w, h


def extract_predictions(predictions, class_conf_threshold=0.5):
    """
    Extract bounding box predictions from model output.
    
    Args:
        predictions: Model prediction tensor of shape [batch_size, S, S, C+B*5]
        class_conf_threshold: Minimum class confidence threshold
        
    Returns:
        List of dictionaries with detection information per image in batch
    """
    batch_size = predictions.shape[0]
    all_detections = []
    
    for batch_idx in range(batch_size):
        image_detections = []
        for i in range(config.S):
            for j in range(config.S):
                # Get class probabilities
                class_probs = predictions[batch_idx, i, j, :config.C]
                class_idx = torch.argmax(class_probs).item()
                class_prob = class_probs[class_idx].item()
                
                # Process each bounding box
                for b in range(config.B):
                    bbox_start = config.C + b * 5
                    
                    # Get confidence and compute final score
                    box_conf = predictions[batch_idx, i, j, bbox_start + 4].item()
                    confidence = class_prob * box_conf
                    
                    if confidence > class_conf_threshold:
                        # Extract box coordinates
                        x = predictions[batch_idx, i, j, bbox_start].item()
                        y = predictions[batch_idx, i, j, bbox_start + 1].item()
                        w = predictions[batch_idx, i, j, bbox_start + 2].item()
                        h = predictions[batch_idx, i, j, bbox_start + 3].item()
                        
                        # Convert to image coordinates
                        x_center, y_center, width, height = convert_cell_to_image_coords(i, j, x, y, w, h)
                        
                        image_detections.append({
                            'image_idx': batch_idx,
                            'class_idx': class_idx,
                            'confidence': confidence,
                            'bbox': [x_center, y_center, width, height]
                        })
        
        all_detections.append(image_detections)
    
    return all_detections


def extract_ground_truths(labels):
    """
    Extract ground truth boxes from labels tensor.
    
    Args:
        labels: Ground truth labels tensor of shape [batch_size, S, S, C+B*5]
        
    Returns:
        List of dictionaries with ground truth information per image in batch
    """
    batch_size = labels.shape[0]
    all_ground_truths = []
    
    for batch_idx in range(batch_size):
        image_gts = []
        for i in range(config.S):
            for j in range(config.S):
                # Check if any object exists in this cell
                has_object = False
                for b in range(config.B):
                    bbox_start = config.C + b * 5
                    if labels[batch_idx, i, j, bbox_start + 4].item() > 0:
                        has_object = True
                        break
                
                if has_object:
                    # Get class index
                    class_idx = torch.argmax(labels[batch_idx, i, j, :config.C]).item()
                    
                    # Extract box coordinates (use first box only since ground truth has one box per cell)
                    bbox_start = config.C
                    x = labels[batch_idx, i, j, bbox_start].item()
                    y = labels[batch_idx, i, j, bbox_start + 1].item()
                    w = labels[batch_idx, i, j, bbox_start + 2].item()
                    h = labels[batch_idx, i, j, bbox_start + 3].item()
                    
                    # Convert to image coordinates
                    x_center, y_center, width, height = convert_cell_to_image_coords(i, j, x, y, w, h)
                    
                    image_gts.append({
                        'image_idx': batch_idx,
                        'class_idx': class_idx,
                        'bbox': [x_center, y_center, width, height],
                        'used': False
                    })
        
        all_ground_truths.append(image_gts)
    
    return all_ground_truths


def convert_to_evaluation_format(boxes, image_offset=0):
    """
    Convert detection/ground truth boxes to the format expected by mean_average_precision:
    [image_idx, class_idx, confidence, x, y, w, h]
    
    Args:
        boxes: List of detection/ground truth dictionaries
        image_offset: Offset to add to image_idx (useful for batch processing)
        
    Returns:
        List of lists in format [image_idx, class_idx, confidence, x, y, w, h]
    """
    result = []
    
    for box in boxes:
        # Create entry in the format [image_idx, class_idx, confidence, x, y, w, h]
        entry = [
            box['image_idx'] + image_offset,
            box['class_idx'],
            box.get('confidence', 1.0),  # Use 1.0 for ground truths
            *box['bbox']  # Unpack bbox coordinates
        ]
        result.append(entry)
    
    return result


def apply_nms(detections, iou_threshold=0.5, conf_threshold=0.0):
    """
    Apply Non-Maximum Suppression to detections by image and class.
    
    Args:
        detections: List of detection dictionaries with 'image_idx', 'class_idx', 'confidence', 'bbox' keys
        iou_threshold: IoU threshold for suppression
        conf_threshold: Confidence threshold to filter detections
        
    Returns:
        List of detections after NMS
    """
    result = []
    filtered_detections = [d for d in detections if d['confidence'] >= conf_threshold]
    
    # Group detections by image index
    detections_by_image = defaultdict(list)
    for det in filtered_detections:
        detections_by_image[det['image_idx']].append(det)

    for img_detections in detections_by_image.values():
        # Group detections in this image by class index
        detections_by_class = defaultdict(list)
        for det in img_detections:
            detections_by_class[det['class_idx']].append(det)

        # Apply NMS for each class in this image
        for _, class_dets in detections_by_class.items():
            if not class_dets:
                continue

            # Extract boxes and scores
            boxes = torch.tensor([d['bbox'] for d in class_dets], dtype=torch.float32, device=device)
            scores = torch.tensor([d['confidence'] for d in class_dets], dtype=torch.float32, device=device)

            # Perform NMS
            keep_indices = non_max_suppression(boxes, scores, iou_threshold)
            for idx in keep_indices:
                result.append(class_dets[idx.item()])
                
    return result


def predict_and_process(model, images, conf_threshold=0.5, iou_threshold=0.5):
    """
    Perform prediction and post-processing on images.
    
    Args:
        model: The trained YOLO model
        images: Tensor of images [batch_size, channels, height, width]
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of lists of detections per image, each in format:
        [{'class_idx': idx, 'confidence': score, 'bbox': [x, y, w, h]}, ...]
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        predictions = model(images)
    
    # Extract detections for each image
    batch_detections = extract_predictions(predictions, conf_threshold)
    processed_detections = []
    
    # Process each image's detections
    for image_idx, detections in enumerate(batch_detections):
        # Apply NMS per image
        detections_after_nms = apply_nms(detections, iou_threshold, conf_threshold)
        processed_detections.append(detections_after_nms)
    
    return processed_detections


def evaluate_model(model, dataloader, num_classes, iou_threshold=0.5, confidence_threshold=0.1):
    """
    Calculate mean Average Precision (mAP) for the model on a dataset.
    
    Args:
        model: Trained YOLO model
        dataloader: DataLoader for evaluation dataset
        num_classes: Number of classes
        iou_threshold: IoU threshold for mAP calculation
        confidence_threshold: Confidence threshold for detections
        
    Returns:
        mAP: Mean Average Precision value
        aps: Dictionary of Average Precision per class
    """
    model.eval()
    all_detections = []
    all_ground_truths = []
    image_count = 0
    
    # Collect predictions and ground truth
    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="Collecting data"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Get predictions and extract detections
            predictions = model(images)
            batch_size = images.shape[0]
            
            # Extract detections and ground truths
            batch_dets = extract_predictions(predictions, confidence_threshold)
            batch_gts = extract_ground_truths(targets)
            
            # Flatten and add image offset
            for image_idx in range(batch_size):
                # Add all detections for this image
                for det in batch_dets[image_idx]:
                    det['image_idx'] = image_count + image_idx
                    all_detections.append(det)
                
                # Add all ground truths for this image
                for gt in batch_gts[image_idx]:
                    gt['image_idx'] = image_count + image_idx
                    all_ground_truths.append(gt)
                
            image_count += batch_size
    
    # Apply NMS
    all_detections = apply_nms(all_detections, iou_threshold, confidence_threshold)
    
    # Calculate mAP
    mAP, average_precisions = mean_average_precision(
        all_detections,
        all_ground_truths,
        iou_threshold=iou_threshold,
        box_format="midpoint",
        num_classes=num_classes
    )
    
    return mAP, average_precisions


def boxes_to_corners(boxes):
    """
    Convert boxes from [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max].
    
    Args:
        boxes: Tensor or list of [x_center, y_center, width, height] format
        
    Returns:
        Boxes in [x_min, y_min, x_max, y_max] format
    """
    if isinstance(boxes, torch.Tensor):
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        return torch.stack((x_min, y_min, x_max, y_max), dim=1)
    else:
        # Handle list or single box
        if not isinstance(boxes[0], (list, tuple)):
            # Single box
            x_center, y_center, width, height = boxes
            return [
                x_center - width / 2,
                y_center - height / 2,
                x_center + width / 2,
                y_center + height / 2
            ]
        else:
            # List of boxes
            return [boxes_to_corners(box) for box in boxes]


def corners_to_boxes(corners):
    """
    Convert boxes from [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height].
    
    Args:
        corners: Tensor or list of [x_min, y_min, x_max, y_max] format
        
    Returns:
        Boxes in [x_center, y_center, width, height] format
    """
    if isinstance(corners, torch.Tensor):
        x_min, y_min, x_max, y_max = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        return torch.stack((x_center, y_center, width, height), dim=1)
    else:
        # Handle list or single corner
        if not isinstance(corners[0], (list, tuple)):
            # Single corner
            x_min, y_min, x_max, y_max = corners
            width = x_max - x_min
            height = y_max - y_min
            return [
                x_min + width / 2,
                y_min + height / 2,
                width,
                height
            ]
        else:
            # List of corners
            return [corners_to_boxes(corner) for corner in corners]