import torch
import os
import utils
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from data_roboflow import YoloRoboflowDataset
from models import YOLOv1
from torch.utils.data import DataLoader

import numpy as np
from collections import defaultdict
from utils import get_iou, bbox_attr, non_max_suppression
import config


# Helper for IoU calculation
def calculate_iou_center_format(box1, box2):
    """Calculates IoU between two boxes in [x_center, y_center, width, height] format."""
    # box format: [x_center, y_center, width, height]
    pred_x1 = box1[0] - box1[2] / 2
    pred_y1 = box1[1] - box1[3] / 2
    pred_x2 = box1[0] + box1[2] / 2
    pred_y2 = box1[1] + box1[3] / 2

    gt_x1 = box2[0] - box2[2] / 2
    gt_y1 = box2[1] - box2[3] / 2
    gt_x2 = box2[0] + box2[2] / 2
    gt_y2 = box2[1] + box2[3] / 2

    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)

    inter_area = 0
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    pred_area = box1[2] * box1[3]
    gt_area = box2[2] * box2[3]
    # Add epsilon to prevent division by zero
    union_area = pred_area + gt_area - inter_area + 1e-6

    iou = inter_area / union_area
    return iou

# Helper function to extract boxes from predictions/targets for a batch
def extract_batch_data(predictions, ground_truths, batch_size, image_indices):
    """Extracts and formats detections and ground truths from a batch."""
    batch_detections = []
    batch_ground_truths = []

    for b_idx in range(batch_size):
        img_idx = image_indices[b_idx] # Use provided image index

        # Process predictions for image img_idx
        for i in range(config.S):
            for j in range(config.S):
                # Prediction format: [C probabilities, B * (x, y, w, h, conf)]
                class_probs = predictions[b_idx, i, j, :config.C]
                # Find the class with the highest probability in this cell
                class_prob, class_idx = torch.max(class_probs, dim=0)
                class_idx = class_idx.item()
                class_prob = class_prob.item()

                for b in range(config.B):
                    bbox_start = config.C + b * 5
                    box_conf = predictions[b_idx, i, j, bbox_start + 4].item()
                    # Final confidence is P(Class) * P(Object)
                    combined_confidence = class_prob * box_conf

                    # Extract box coordinates relative to cell
                    x = predictions[b_idx, i, j, bbox_start].item()
                    y = predictions[b_idx, i, j, bbox_start + 1].item()
                    # Note: Original YOLO paper predicts sqrt(w) and sqrt(h)
                    # Assuming model output is already w, h directly for simplicity based on existing code
                    # If it's sqrt, need to square them: w = w**2, h = h**2
                    w = predictions[b_idx, i, j, bbox_start + 2].item()
                    h = predictions[b_idx, i, j, bbox_start + 3].item()

                    # Convert cell-relative coordinates to image-relative coordinates [0, 1]
                    grid_size_x = 1.0 / config.S
                    grid_size_y = 1.0 / config.S
                    x_center = (j + x) * grid_size_x
                    y_center = (i + y) * grid_size_y
                    # Width and height are relative to image size

                    batch_detections.append({
                        'image_idx': img_idx,
                        'class_idx': class_idx,
                        'confidence': combined_confidence,
                        'bbox': [x_center, y_center, w, h] # Relative coords [0,1]
                    })

        # Process ground truths for image img_idx
        # GT format: [C class one-hot, x, y, w, h, objectness] (assuming one box per cell)
        for i in range(config.S):
            for j in range(config.S):
                # Check objectness score at the end
                if ground_truths[b_idx, i, j, config.C + 4].item() > 0: # Object present in this cell
                    # Find the class index
                    class_idx = torch.argmax(ground_truths[b_idx, i, j, :config.C]).item()

                    # Extract GT box coordinates relative to cell
                    bbox_start = config.C
                    x = ground_truths[b_idx, i, j, bbox_start].item()
                    y = ground_truths[b_idx, i, j, bbox_start + 1].item()
                    w = ground_truths[b_idx, i, j, bbox_start + 2].item()
                    h = ground_truths[b_idx, i, j, bbox_start + 3].item()

                    # Convert cell-relative coordinates to image-relative coordinates [0, 1]
                    grid_size_x = 1.0 / config.S
                    grid_size_y = 1.0 / config.S
                    x_center = (j + x) * grid_size_x
                    y_center = (i + y) * grid_size_y

                    batch_ground_truths.append({
                        'image_idx': img_idx,
                        'class_idx': class_idx,
                        'bbox': [x_center, y_center, w, h], # Relative coords [0,1]
                        'used': False # Flag to track matching during AP calculation
                    })
    return batch_detections, batch_ground_truths


def calculate_mAP(model, dataloader, num_classes, iou_threshold=0.5, confidence_threshold=0.01):
    """
    Calculate mean Average Precision (mAP) for the model on the given dataset
    using standard PASCAL VOC methodology (AUC with interpolation).

    Args:
        model: The trained YOLO model.
        dataloader: DataLoader for the evaluation dataset.
        num_classes: Total number of object classes.
        iou_threshold: IoU threshold for considering a detection as True Positive (TP).
        confidence_threshold: Minimum confidence score for a detection to be considered.

    Returns:
        mAP: Mean Average Precision across all classes with ground truths.
        aps: Dictionary containing Average Precision (AP) for each class index.
             Classes with no ground truths will have NaN as AP.
    """
    model.eval()
    all_detections = []
    all_ground_truths = []
    image_count = 0

    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="Collecting data"):
            images = images.to(utils.device)
            targets = targets.to(utils.device)
            predictions = model(images)

            batch_size = images.shape[0]
            # Generate unique image indices for this batch
            image_indices = list(range(image_count, image_count + batch_size))
            image_count += batch_size

            # Extract detections and ground truths for this batch
            batch_dets, batch_gts = extract_batch_data(predictions, targets, batch_size, image_indices)
            all_detections.extend(batch_dets)
            all_ground_truths.extend(batch_gts)

    # Filter detections based on the confidence threshold
    # This should happen *after* collecting all detections
    all_detections = [d for d in all_detections if d['confidence'] >= confidence_threshold]

    # --- Apply Non-Max Suppression (NMS) per image, per class ---
    print("Applying Non-Max Suppression...")
    detections_after_nms = []
    # Group detections by image index
    detections_by_image = defaultdict(list)
    for det in all_detections:
        detections_by_image[det['image_idx']].append(det)

    for img_idx, img_detections in tqdm(detections_by_image.items(), desc="NMS per image"):
        # Group detections in this image by class index
        detections_by_class = defaultdict(list)
        for det in img_detections:
            detections_by_class[det['class_idx']].append(det)

        # Apply NMS for each class in this image
        for class_idx, class_dets in detections_by_class.items():
            if not class_dets:
                continue

            # Extract boxes ([xc, yc, w, h] format, relative) and scores
            boxes = torch.tensor([d['bbox'] for d in class_dets], dtype=torch.float32, device=utils.device)
            scores = torch.tensor([d['confidence'] for d in class_dets], dtype=torch.float32, device=utils.device)

            # Perform NMS
            keep_indices = non_max_suppression(boxes, scores, iou_threshold) # Use the main iou_threshold

            # Keep only the detections that survived NMS
            for idx in keep_indices:
                detections_after_nms.append(class_dets[idx.item()])

    print(f"Detections after NMS: {len(detections_after_nms)}")
    all_detections = detections_after_nms # Use NMS results for AP calculation
    # --- End NMS ---

    # --- mAP Calculation ---

    # Group ground truths by image index and class index for efficient lookup
    gt_by_image_class = defaultdict(list)
    # Count total number of ground truths for each class
    total_gts_per_class = defaultdict(int)
    for gt in all_ground_truths:
        key = (gt['image_idx'], gt['class_idx'])
        gt_by_image_class[key].append(gt)
        total_gts_per_class[gt['class_idx']] += 1


    # Calculate AP for each class
    average_precisions = {}
    epsilon = 1e-6 # Small value to avoid division by zero

    print(f"Calculating AP for {num_classes} classes...")
    for class_idx in range(num_classes):
        # Get all detections for the current class
        class_detections = [d for d in all_detections if d['class_idx'] == class_idx]

        # Get the total number of ground truth objects for this class
        num_gts = total_gts_per_class[class_idx]

        # Handle cases with no ground truths for the class
        if num_gts == 0:
            # If there are no ground truths, AP is undefined (NaN).
            # If there are also no detections, some might argue AP=1, but NaN is safer.
            average_precisions[class_idx] = float('nan')
            continue # Skip AP calculation for this class


        # Sort detections for this class by confidence score in descending order
        class_detections.sort(key=lambda x: x['confidence'], reverse=True)

        # Initialize True Positive (TP) and False Positive (FP) arrays
        tp = np.zeros(len(class_detections))
        fp = np.zeros(len(class_detections))

        # Reset the 'used' flag for all ground truths before processing each class
        # This is crucial because a GT can only be matched once per class evaluation
        for gt_list in gt_by_image_class.values():
             for gt in gt_list:
                 gt['used'] = False

        # Match detections to ground truths for the current class
        for det_idx, detection in enumerate(class_detections):
            img_idx = detection['image_idx']
            # Get potential ground truth matches (same image, same class)
            gts_in_image = gt_by_image_class.get((img_idx, class_idx), [])

            best_iou = -1
            best_gt_match_index_in_list = -1 # Index within the gts_in_image list

            # Find the best GT match for this detection based on IoU
            for gt_idx, gt in enumerate(gts_in_image):
                iou = calculate_iou_center_format(detection['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_match_index_in_list = gt_idx

            # Assign TP or FP based on IoU threshold and whether GT was already used
            if best_iou >= iou_threshold:
                # Retrieve the actual GT object using the index
                matched_gt = gts_in_image[best_gt_match_index_in_list]
                if not matched_gt['used']:
                    tp[det_idx] = 1
                    matched_gt['used'] = True # Mark this GT as used for this class
                else:
                    # Detected an object already matched by a higher confidence detection
                    fp[det_idx] = 1
            else:
                # Detection did not match any GT above the threshold
                fp[det_idx] = 1


        # Calculate cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Calculate Precision and Recall curves
        recalls = tp_cumsum / (num_gts + epsilon)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + epsilon)

        # --- Calculate Average Precision (AP) using PASCAL VOC method ---
        # (Area under the PR curve with interpolation)

        # Append sentinel values for start and end
        precisions = np.concatenate(([0.], precisions, [0.]))
        recalls = np.concatenate(([0.], recalls, [1.])) # Recall ends at 1.0

        # Ensure precision is monotonically decreasing
        # (Precision at recall R should be max precision at recalls >= R)
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])

        # Find indices where recall changes
        recall_change_indices = np.where(recalls[1:] != recalls[:-1])[0]

        # Calculate AP as the sum of areas of rectangles under the curve
        # Area = (recall_at_i+1 - recall_at_i) * precision_at_i+1
        ap = np.sum((recalls[recall_change_indices + 1] - recalls[recall_change_indices]) *
                    precisions[recall_change_indices + 1])

        average_precisions[class_idx] = ap

    # --- Calculate mean Average Precision (mAP) ---
    # Average the AP scores only for classes that had ground truths (non-NaN AP)
    valid_aps = [ap for ap in average_precisions.values() if not np.isnan(ap)]
    mAP = np.mean(valid_aps) if valid_aps else 0.0 # Return 0.0 if no classes had GTs

    print("AP calculation complete.")
    return mAP, average_precisions


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO v1 model using mAP')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory containing model weights')
    parser.add_argument('--weights', type=str, default='final', help='Name of weights file to use (e.g., final.pt)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for mAP calculation (usually 0.5 for PASCAL VOC)')
    parser.add_argument('--confidence_threshold', type=float, default=0.01, help='Confidence threshold for considering detections')
    args = parser.parse_args()

    # Set model directory if not provided
    if args.model_dir is None:
        base_dir = 'models/yolo_v1'
        if os.path.exists(base_dir):
            try:
                latest_date = max(os.listdir(base_dir))
                times = os.listdir(os.path.join(base_dir, latest_date))
                if times:
                    latest_time = max(times)
                    args.model_dir = os.path.join(base_dir, latest_date, latest_time)
                    print(f"Using most recent model directory found: {args.model_dir}")
            except ValueError: # Handle empty base_dir
                 print(f"Warning: No subdirectories found in {base_dir}. Cannot automatically find latest model.")
                 pass # Keep args.model_dir as None

    if args.model_dir is None or not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found or specified. Please use --model_dir.")
        print(f"Checked path: {args.model_dir}")
        return

    weights_path = os.path.join(args.model_dir, 'weights', args.weights)
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        # Suggest looking for files if the exact name doesn't match
        weights_dir = os.path.join(args.model_dir, 'weights')
        if os.path.exists(weights_dir):
            available_files = os.listdir(weights_dir)
            print(f"Available files in {weights_dir}: {available_files}")
            # Check if there's a file without extension that matches the name without .pt
            if args.weights.replace('.pt', '') in available_files:
                print(f"Found matching file without extension. Try using: --weights {args.weights.replace('.pt', '')}")
        return

    # Load class names
    try:
        classes = utils.load_class_array()
        num_classes = len(classes)
        print(f"Loaded {num_classes} classes: {classes}")
    except Exception as e:
        print(f"Error loading class names: {e}. Make sure class file exists and is configured.")
        return

    # Create dataset and dataloader
    print("Loading validation dataset...")
    try:
        # Ensure 'valid' split exists and data is accessible
        dataset = YoloRoboflowDataset(set_type='valid', normalize=True, augment=False)
        if len(dataset) == 0:
             print("Error: Validation dataset is empty. Check data source and split name.")
             return
        # shuffle=False is important for consistent image indexing during evaluation
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        print(f"Dataset size: {len(dataset)} images")
    except Exception as e:
        print(f"Error creating dataset/dataloader: {e}")
        return

    # Load model
    print(f"Loading model state from {weights_path}...")
    try:
        model = YOLOv1()
        model.load_state_dict(torch.load(weights_path, map_location=utils.device))
        model = model.to(utils.device)
        model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create results directory if needed
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_subdir_name = f"{os.path.basename(args.model_dir)}_{args.weights.replace('.pt','')}_{timestamp}"
    results_dir = os.path.join('results', 'evaluation', results_subdir_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    # --- Evaluate model ---
    print(f"Evaluating model using mAP@{args.iou_threshold:.2f} IoU threshold...")
    mAP, aps_per_class = calculate_mAP(
        model,
        loader,
        num_classes=num_classes,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold
    )

    # --- Print results ---
    print("--- Evaluation Results ---")
    print(f"Confidence Threshold: {args.confidence_threshold}")
    print(f"IoU Threshold:        {args.iou_threshold}")
    print("-" * 30)
    print(f"mAP@{args.iou_threshold:.2f}: {mAP:.4f}")
    print("-" * 30)
    print("AP per class:")
    for class_idx, ap in aps_per_class.items():
        class_name = classes[class_idx] if class_idx < len(classes) else f"Class_{class_idx}"
        if np.isnan(ap):
            print(f"  {class_name:<15}: N/A (No ground truths)")
        else:
            print(f"  {class_name:<15}: {ap:.4f}")
    print("-" * 30)

    # --- Save results to JSON ---
    results_data = {
        'model_dir': args.model_dir,
        'weights_file': args.weights,
        'timestamp': timestamp,
        'config': {
            'iou_threshold': args.iou_threshold,
            'confidence_threshold': args.confidence_threshold,
            'batch_size': args.batch_size,
            'num_validation_images': len(dataset),
        },
        'metrics': {
            f'mAP_{args.iou_threshold:.2f}': mAP,
            'ap_per_class': {classes[idx]: ap for idx, ap in aps_per_class.items()}
        }
    }
    results_file_path = os.path.join(results_dir, 'evaluation_results.json')
    try:
        with open(results_file_path, 'w') as f:
            # Use custom encoder for NaN values if needed, or convert them
            json.dump(results_data, f, indent=2, default=lambda x: str(x) if np.isnan(x) else x)
        print(f"Results saved to {results_file_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")


if __name__ == "__main__":
    main() 