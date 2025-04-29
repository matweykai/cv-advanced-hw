import torch
import numpy as np
from collections import defaultdict
from .nms import nms

def bbox_iou(box1, box2):
    """
    Calculate IoU between box1 and box2
    
    Args:
        box1: tensor of shape (4,) representing (x1, y1, x2, y2)
        box2: tensor of shape (4,) representing (x1, y1, x2, y2)
    
    Returns:
        IoU: scalar value
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    
    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union_area = b1_area + b2_area - inter_area
    
    iou = inter_area / union_area
    
    return iou

def convert_yolo_predictions(predictions, S, B, C, conf_thresh=0.1, prob_thresh=0.1):
    """
    Convert YOLO predictions to bounding boxes
    
    Args:
        predictions: tensor of shape (batch_size, S*S*(5*B+C))
        S: grid size
        B: number of bounding boxes per grid cell
        C: number of classes
        conf_thresh: confidence threshold
        prob_thresh: probability threshold
    
    Returns:
        boxes: list of tensors, each tensor contains bounding boxes for one image
        class_indices: list of tensors, each tensor contains class indices for one image
        probs: list of tensors, each tensor contains probabilities for one image
    """
    batch_size = predictions.size(0)
    predictions = predictions.view(batch_size, S, S, 5*B+C)
    
    boxes = []
    class_indices = []
    probs = []
    
    for i in range(batch_size):
        boxes_img = []
        class_indices_img = []
        probs_img = []
        
        for row in range(S):
            for col in range(S):
                # Get class probabilities
                class_probs = predictions[i, row, col, 5*B:]
                
                # Get box confidences
                box_confidences = []
                for b in range(B):
                    box_confidences.append(predictions[i, row, col, 5*b + 4])
                
                # Find the box with highest confidence
                box_idx = torch.argmax(torch.tensor(box_confidences))
                box_confidence = box_confidences[box_idx]
                
                # Get class with highest probability
                class_idx = torch.argmax(class_probs)
                class_prob = class_probs[class_idx]
                
                # Calculate final probability
                prob = box_confidence * class_prob
                
                # Filter based on thresholds
                if box_confidence > conf_thresh and prob > prob_thresh:
                    # Get box coordinates
                    x = predictions[i, row, col, 5*box_idx]
                    y = predictions[i, row, col, 5*box_idx + 1]
                    w = predictions[i, row, col, 5*box_idx + 2]
                    h = predictions[i, row, col, 5*box_idx + 3]
                    
                    # Convert to absolute coordinates
                    x = (col + x) / S
                    y = (row + y) / S
                    w = w
                    h = h
                    
                    # Convert to corner coordinates
                    x1 = x - w/2
                    y1 = y - h/2
                    x2 = x + w/2
                    y2 = y + h/2
                    
                    # Add to lists
                    boxes_img.append(torch.tensor([x1, y1, x2, y2]))
                    class_indices_img.append(class_idx)
                    probs_img.append(prob)
        
        # Convert lists to tensors
        if boxes_img:
            boxes.append(torch.stack(boxes_img))
            class_indices.append(torch.tensor(class_indices_img))
            probs.append(torch.tensor(probs_img))
        else:
            boxes.append(torch.tensor([]))
            class_indices.append(torch.tensor([]))
            probs.append(torch.tensor([]))
    
    return boxes, class_indices, probs

def calculate_ap(recalls, precisions):
    """
    Calculate Average Precision using the 11-point interpolation
    
    Args:
        recalls: list of recall values
        precisions: list of precision values
    
    Returns:
        ap: average precision
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap

def calculate_mAP(pred_boxes, pred_classes, pred_scores, true_boxes, true_classes, num_classes, iou_threshold=0.5):
    """
    Calculate mean Average Precision
    
    Args:
        pred_boxes: list of tensors, each tensor contains predicted bounding boxes for one image
        pred_classes: list of tensors, each tensor contains predicted class indices for one image
        pred_scores: list of tensors, each tensor contains predicted scores for one image
        true_boxes: list of tensors, each tensor contains ground truth bounding boxes for one image
        true_classes: list of tensors, each tensor contains ground truth class indices for one image
        num_classes: number of classes
        iou_threshold: IoU threshold for a prediction to be considered correct
    
    Returns:
        mAP: mean Average Precision
    """
    # Initialize dictionaries to store detections and ground truths
    all_detections = defaultdict(list)
    all_ground_truths = defaultdict(list)
    
    # Process all images
    for img_idx in range(len(pred_boxes)):
        # Skip if no predictions for this image
        if len(pred_boxes[img_idx]) == 0:
            continue
        
        # Get predictions for this image
        boxes = pred_boxes[img_idx]
        classes = pred_classes[img_idx]
        scores = pred_scores[img_idx]
        
        # Store predictions by class
        for box_idx in range(len(boxes)):
            class_idx = classes[box_idx].item()
            all_detections[class_idx].append([img_idx, scores[box_idx].item(), boxes[box_idx]])
        
        # Get ground truths for this image
        gt_boxes = true_boxes[img_idx]
        gt_classes = true_classes[img_idx]
        
        # Store ground truths by class
        for box_idx in range(len(gt_boxes)):
            class_idx = gt_classes[box_idx].item()
            all_ground_truths[class_idx].append([img_idx, gt_boxes[box_idx]])
    
    # Calculate AP for each class
    average_precisions = []
    
    for class_idx in range(num_classes):
        # Skip if no ground truths for this class
        if class_idx not in all_ground_truths:
            continue
        
        # Get detections and ground truths for this class
        detections = all_detections[class_idx]
        ground_truths = all_ground_truths[class_idx]
        
        # Sort detections by score
        detections.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize variables for precision and recall calculation
        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))
        
        # Count total number of ground truths
        total_gt = len(ground_truths)
        
        # Create a dictionary to keep track of detected ground truths
        detected_gt = defaultdict(list)
        
        # Process each detection
        for det_idx, detection in enumerate(detections):
            img_idx, score, pred_box = detection
            
            # Get ground truths for this image
            img_gts = [gt for gt in ground_truths if gt[0] == img_idx]
            
            # Initialize variables
            best_iou = -1
            best_gt_idx = -1
            
            # Find the ground truth with highest IoU
            for gt_idx, gt in enumerate(img_gts):
                gt_img_idx, gt_box = gt
                
                # Skip if this ground truth has already been detected
                if gt_idx in detected_gt[img_idx]:
                    continue
                
                # Calculate IoU
                iou = bbox_iou(pred_box, gt_box)
                
                # Update best IoU
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if detection is correct
            if best_iou >= iou_threshold:
                # Mark this ground truth as detected
                detected_gt[img_idx].append(best_gt_idx)
                TP[det_idx] = 1
            else:
                FP[det_idx] = 1
        
        # Calculate cumulative TP and FP
        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)
        
        # Calculate precision and recall
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        recalls = TP_cumsum / (total_gt + 1e-6)
        
        # Add a start point (0, 1) for precision-recall curve
        precisions = np.concatenate(([1], precisions))
        recalls = np.concatenate(([0], recalls))
        
        # Calculate AP
        ap = calculate_ap(recalls, precisions)
        average_precisions.append(ap)
    
    # Calculate mAP
    mAP = np.mean(average_precisions) if average_precisions else 0
    
    return mAP
