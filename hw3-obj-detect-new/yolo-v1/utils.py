import torch
import json
import os
import config
import matplotlib.patches as patches
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


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes in format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Handle division by zero
    if union_area < config.EPSILON:
        return 0.0
    
    return intersection_area / union_area


def apply_nms(bboxes, iou_threshold=0.5, format_type="visualization"):
    """
    Unified non-maximum suppression function that can handle different bbox formats.
    
    Args:
        bboxes: List of bounding boxes
        iou_threshold: IoU threshold for suppression
        format_type: Type of format for bboxes, either "visualization" or "evaluation"
            - visualization: [tl, width, height, confidence, class_index]
            - evaluation: [class_idx, confidence, x1, y1, x2, y2]
    
    Returns:
        List of indices of kept boxes (for visualization format)
        OR List of kept boxes (for evaluation format)
    """
    if not bboxes:
        return [] if format_type == "evaluation" else []
    
    # Sort by confidence (in both formats, confidence is at index 3 for viz or 1 for eval)
    confidence_idx = 3 if format_type == "visualization" else 1
    indices = sorted(range(len(bboxes)), key=lambda i: bboxes[i][confidence_idx], reverse=True)
    
    # Initialize list to track which boxes to keep
    kept_indices = []
    
    for i, box_idx in enumerate(indices):
        # Add current box to kept boxes
        kept_indices.append(box_idx)
        
        # For remaining boxes, check if they should be suppressed
        for j in range(i + 1, len(indices)):
            compare_idx = indices[j]
            
            # Check if boxes are the same class
            if format_type == "visualization":
                curr_class = bboxes[box_idx][4]
                compare_class = bboxes[compare_idx][4]
                
                # Calculate IoU - using get_overlap for visualization format
                iou = get_overlap(bboxes[box_idx], bboxes[compare_idx])
            else:  # evaluation format
                curr_class = bboxes[box_idx][0]
                compare_class = bboxes[compare_idx][0]
                
                # Calculate IoU - using calculate_iou for evaluation format
                iou = calculate_iou(bboxes[box_idx][2:], bboxes[compare_idx][2:])
            
            # Suppress box if same class and high IoU
            if curr_class == compare_class and iou > iou_threshold:
                # Remove this index by marking it (setting to None)
                indices[j] = None
        
        # Clean up None values
        indices = [idx for idx in indices if idx is not None]
    
    if format_type == "evaluation":
        # Return the actual boxes for evaluation format
        return [bboxes[idx] for idx in kept_indices]
    else:
        # Return the indices for visualization format
        return kept_indices


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

    # Sort by highest to lowest confidence (done within apply_nms)
    kept_indices = apply_nms(bboxes, max_overlap, "visualization")

    # Non-maximum suppression and render image
    image = T.ToPILImage()(data)
    draw = ImageDraw.Draw(image)
    
    for idx in kept_indices:
        tl, width, height, confidence, class_index = bboxes[idx]
        
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


def extract_bboxes_from_yolo_output(predictions, min_confidence=0.1):
    """
    Extract bounding boxes from YOLO output format.
    
    Args:
        predictions: Tensor with shape (S, S, B*5+C)
        min_confidence: Minimum confidence threshold
    
    Returns:
        List of [class_idx, confidence, x1, y1, x2, y2] for each detection
    """
    S = predictions.size(0)
    bboxes = []
    
    # Calculate grid cell size
    grid_size_x = config.IMAGE_SIZE[0] / S
    grid_size_y = config.IMAGE_SIZE[1] / S
    
    for i in range(S):
        for j in range(S):
            for k in range((predictions.size(-1) - config.C) // 5):
                bbox_start = 5 * k + config.C
                bbox_end = 5 * (k + 1) + config.C
                
                bbox = predictions[i, j, bbox_start:bbox_end]
                class_scores = predictions[i, j, :config.C]
                class_idx = torch.argmax(class_scores).item()
                class_confidence = class_scores[class_idx].item()
                
                # Combined confidence (class probability * objectness)
                confidence = class_confidence * bbox[4].item()
                
                if confidence > min_confidence:
                    # Convert to absolute coordinates
                    cx = (bbox[0].item() + j) * grid_size_x
                    cy = (bbox[1].item() + i) * grid_size_y
                    w = bbox[2].item() * config.IMAGE_SIZE[0]
                    h = bbox[3].item() * config.IMAGE_SIZE[1]
                    
                    # Convert to x1, y1, x2, y2 format
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                    
                    bboxes.append([class_idx, confidence, x1, y1, x2, y2])
    
    return bboxes


def non_max_suppression(bboxes, iou_threshold=0.5):
    """
    Apply non-maximum suppression to remove overlapping bounding boxes.
    
    Args:
        bboxes: List of [class_idx, confidence, x1, y1, x2, y2]
        iou_threshold: IoU threshold for suppression
    
    Returns:
        List of bounding boxes after NMS
    """
    return apply_nms(bboxes, iou_threshold, "evaluation")


def calculate_map(model, data_loader, iou_threshold=0.5, confidence_threshold=0.1):
    """
    Calculate mAP (mean Average Precision) for a YOLO model.
    
    Args:
        model: YOLO model
        data_loader: DataLoader with validation/test data
        iou_threshold: IoU threshold for considering a prediction correct
        confidence_threshold: Confidence threshold for predictions
    
    Returns:
        mAP value and per-class AP values
    """
    model.eval()
    
    # Initialize storage for predictions and ground truth
    all_predictions = []
    all_ground_truths = []
    
    print("Collecting predictions...")
    with torch.no_grad():
        for images, labels, _ in tqdm(data_loader):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Forward pass
            predictions = model(images)
            
            # Process each image in the batch
            for i in range(batch_size):
                pred = predictions[i]
                label = labels[i]
                
                # Extract predicted bounding boxes
                pred_bboxes = extract_bboxes_from_yolo_output(pred, confidence_threshold)
                pred_bboxes = non_max_suppression(pred_bboxes, iou_threshold)
                all_predictions.append(pred_bboxes)
                
                # Extract ground truth bounding boxes
                gt_bboxes = extract_bboxes_from_yolo_output(label, 0.0)  # No confidence threshold for GT
                all_ground_truths.append(gt_bboxes)
    
    # Calculate AP for each class
    ap_per_class = {}
    num_classes = config.C
    
    print("Calculating AP for each class...")
    for class_idx in range(num_classes):
        # Flatten predictions for this class
        predictions_class = []
        for img_idx, pred_boxes in enumerate(all_predictions):
            for box in pred_boxes:
                if box[0] == class_idx:
                    # [img_idx, confidence, x1, y1, x2, y2]
                    predictions_class.append([img_idx] + box[1:])
        
        # Sort by confidence
        predictions_class.sort(key=lambda x: x[1], reverse=True)
        
        # Count ground truths for this class
        gt_counter = defaultdict(int)
        for img_idx, gt_boxes in enumerate(all_ground_truths):
            for box in gt_boxes:
                if box[0] == class_idx:
                    gt_counter[img_idx] += 1
        
        total_gt = sum(gt_counter.values())
        if total_gt == 0:
            ap_per_class[class_idx] = 0.0
            continue
        
        # Initialize detection status for ground truths
        gt_detected = {img_idx: [False] * count for img_idx, count in gt_counter.items()}
        
        # Calculate precision and recall points
        tp = 0
        fp = 0
        precision_points = []
        recall_points = []
        
        for pred in predictions_class:
            img_idx = pred[0]
            pred_box = pred[2:]  # x1, y1, x2, y2
            
            # Check if there's any matching ground truth
            matched_gt = False
            
            if img_idx in gt_counter and gt_counter[img_idx] > 0:
                # Get ground truth boxes for this image
                gt_boxes = []
                for gt in all_ground_truths[img_idx]:
                    if gt[0] == class_idx:
                        gt_boxes.append(gt[2:])  # x1, y1, x2, y2
                
                # Calculate IoU with each ground truth box
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou and not gt_detected[img_idx][i]:
                        best_iou = iou
                        best_gt_idx = i
                
                # If IoU exceeds threshold, it's a true positive
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    gt_detected[img_idx][best_gt_idx] = True
                    matched_gt = True
            
            # Update counters
            if matched_gt:
                tp += 1
            else:
                fp += 1
            
            # Calculate precision and recall
            precision = tp / (tp + fp)
            recall = tp / total_gt
            
            precision_points.append(precision)
            recall_points.append(recall)
        
        # Add sentinel points
        precision_points = np.array([0] + precision_points + [0])
        recall_points = np.array([0] + recall_points + [1])
        
        # Make precision monotonically decreasing
        for i in range(len(precision_points) - 1, 0, -1):
            precision_points[i-1] = max(precision_points[i-1], precision_points[i])
        
        # Find recall change points
        recall_changes = []
        for i in range(1, len(recall_points)):
            if recall_points[i] != recall_points[i-1]:
                recall_changes.append(i)
        
        # Calculate AP using the 11-point interpolation method
        ap = 0
        for recall_level in np.arange(0, 1.1, 0.1):
            # Find max precision at recall >= recall_level
            precision_at_recall = 0
            for i in range(len(precision_points)):
                if recall_points[i] >= recall_level:
                    precision_at_recall = max(precision_at_recall, precision_points[i])
            
            ap += precision_at_recall / 11
        
        ap_per_class[class_idx] = ap
    
    # Calculate mAP
    mean_ap = sum(ap_per_class.values()) / len(ap_per_class)
    
    return mean_ap, ap_per_class


def evaluate_model(model, data_loader, classes=None):
    """
    Evaluate YOLO model performance using mAP at various IoU thresholds.
    
    Args:
        model: YOLO model
        data_loader: DataLoader with validation/test data
        classes: Optional list of class names
    
    Returns:
        Dictionary with evaluation results
    """
    if classes is None:
        classes = load_class_array()
    
    print("Evaluating model...")
    results = {}
    
    # Calculate mAP at different IoU thresholds
    iou_thresholds = [0.5, 0.75]
    for iou_threshold in iou_thresholds:
        print(f"Calculating mAP at IoU={iou_threshold}")
        map_value, ap_per_class = calculate_map(
            model, data_loader, 
            iou_threshold=iou_threshold, 
            confidence_threshold=0.1
        )
        
        results[f"mAP@{iou_threshold}"] = map_value
        
        # Store per-class results
        for class_idx, ap in ap_per_class.items():
            class_name = classes[class_idx] if class_idx < len(classes) else f"class_{class_idx}"
            results[f"AP@{iou_threshold}_{class_name}"] = ap
    
    # Calculate mAP@0.5:0.95 (COCO style - average over IoU thresholds)
    detailed_iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_sum = 0
    count = 0
    
    print("Calculating COCO-style mAP@0.5:0.95")
    for iou_threshold in detailed_iou_thresholds:
        map_value, _ = calculate_map(
            model, data_loader,
            iou_threshold=iou_threshold,
            confidence_threshold=0.1
        )
        ap_sum += map_value
        count += 1
    
    results["mAP@0.5:0.95"] = ap_sum / count
    
    return results

