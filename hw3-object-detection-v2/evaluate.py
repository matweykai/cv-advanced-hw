import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import config
import numpy as np
import torch
import utils
from data_loader import YoloRoboflowDataset
from models import YOLOv1
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (mean_average_precision,
                   extract_predictions, extract_ground_truths, apply_nms, evaluate_model)


def convert_cell_to_image_coords(i, j, x, y, w, h):
    """Convert cell-relative coordinates to image-relative coordinates."""
    grid_size_x = 1.0 / config.S
    grid_size_y = 1.0 / config.S
    x_center = (j + x) * grid_size_x
    y_center = (i + y) * grid_size_y
    return x_center, y_center, w, h

# Helper function to extract boxes from predictions/targets for a batch
def extract_batch_data(predictions, ground_truths, batch_size, image_indices):
    """Extracts and formats detections and ground truths from a batch."""
    batch_detections = []
    batch_ground_truths = []

    for b_idx in range(batch_size):
        img_idx = image_indices[b_idx]
        
        # Extract predictions and ground truths for this image
        batch_detections.extend(extract_predictions(predictions, b_idx, img_idx))
        batch_ground_truths.extend(extract_ground_truths(ground_truths, b_idx, img_idx))
            
    return batch_detections, batch_ground_truths



def calculate_mAP(model, dataloader, num_classes, iou_threshold=0.5, confidence_threshold=0.1):
    """
    Calculate mean Average Precision (mAP) for the model on the given dataset

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

    # Apply NMS
    all_detections = apply_nms(all_detections, iou_threshold)
    
    # Calculate mAP using the utility function from utils.py
    print(f"Calculating AP for {num_classes} classes...")
    mAP, average_precisions = mean_average_precision(
        all_detections, 
        all_ground_truths,
        iou_threshold=iou_threshold,
        box_format="midpoint",
        num_classes=num_classes
    )
    
    print("AP calculation complete.")
    return mAP, average_precisions


def find_latest_model():
    """Find the most recent model directory if not specified."""
    base_dir = 'models/yolo_v1'
    if not os.path.exists(base_dir):
        return None
        
    try:
        latest_date = max(os.listdir(base_dir))
        times = os.listdir(os.path.join(base_dir, latest_date))
        if times:
            latest_time = max(times)
            model_dir = os.path.join(base_dir, latest_date, latest_time)
            print(f"Using most recent model directory found: {model_dir}")
            return model_dir
    except (ValueError, FileNotFoundError):
        print(f"Warning: No valid subdirectories found in {base_dir}.")
    
    return None


def setup_results_dir(model_dir, weights_name):
    """Create and return the path to the results directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_subdir_name = f"{os.path.basename(model_dir)}_{weights_name.replace('.pt', '')}_{timestamp}"
    results_dir = os.path.join('results', 'evaluation', results_subdir_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    return results_dir


def save_results(results_dir, results_data):
    """Save evaluation results to a JSON file."""
    results_file_path = os.path.join(results_dir, 'evaluation_results.json')
    try:
        with open(results_file_path, 'w') as f:
            # Handle NaN values
            json.dump(results_data, f, indent=2, default=lambda x: str(x) if np.isnan(x) else x)
        print(f"Results saved to {results_file_path}")
        return True
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
        return False


def print_results(mAP, aps_per_class, classes, iou_threshold, confidence_threshold):
    """Print formatted evaluation results."""
    print("--- Evaluation Results ---")
    print(f"Confidence Threshold: {confidence_threshold}")
    print(f"IoU Threshold:        {iou_threshold}")
    print("-" * 30)
    print(f"mAP@{iou_threshold:.2f}: {mAP:.4f}")
    print("-" * 30)
    print("AP per class:")
    for class_idx, ap in aps_per_class.items():
        class_name = classes[class_idx] if class_idx < len(classes) else f"Class_{class_idx}"
        if np.isnan(ap):
            print(f"  {class_name:<15}: N/A (No ground truths)")
        else:
            print(f"  {class_name:<15}: {ap:.4f}")
    print("-" * 30)


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO v1 model using mAP')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory containing model weights')
    parser.add_argument('--weights', type=str, default='final', help='Name of weights file to use (e.g., final.pt)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for mAP calculation (usually 0.5 for PASCAL VOC)')
    parser.add_argument('--confidence_threshold', type=float, default=0.2, help='Confidence threshold for considering detections')
    args = parser.parse_args()

    # Find or validate model directory
    if args.model_dir is None:
        args.model_dir = find_latest_model()
    
    if args.model_dir is None or not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found or specified. Please use --model_dir.")
        print(f"Checked path: {args.model_dir}")
        return

    # Locate and validate weights file
    weights_path = os.path.join(args.model_dir, 'weights', args.weights)
    if not os.path.exists(weights_path) and not os.path.exists(weights_path + '.pt'):
        if os.path.exists(weights_path + '.pt'):
            weights_path += '.pt'
        else:
            print(f"Error: Weights file not found at {weights_path}")
            weights_dir = os.path.join(args.model_dir, 'weights')
            if os.path.exists(weights_dir):
                available_files = os.listdir(weights_dir)
                print(f"Available files in {weights_dir}: {available_files}")
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
        dataset = YoloRoboflowDataset(set_type='valid', normalize=True, augment=False)
        if len(dataset) == 0:
            print("Error: Validation dataset is empty. Check data source and split name.")
            return
        
        loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,  # Important for consistent image indexing
            num_workers=4, 
            pin_memory=True
        )
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
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Set up results directory
    results_dir = setup_results_dir(args.model_dir, args.weights)

    # Evaluate model using the utility function
    print(f"Evaluating model using mAP@{args.iou_threshold:.2f} IoU threshold...")
    mAP, aps_per_class = evaluate_model(
        model,
        loader,
        num_classes=num_classes,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold
    )

    # Print results
    print_results(mAP, aps_per_class, classes, args.iou_threshold, args.confidence_threshold)

    # Save results to JSON
    results_data = {
        'model_dir': args.model_dir,
        'weights_file': args.weights,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
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
    save_results(results_dir, results_data)


if __name__ == "__main__":
    main() 