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


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO v1 model')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory containing model weights')
    parser.add_argument('--weights', type=str, default='final', help='Name of weights file to use (without extension)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for mAP calculation')
    parser.add_argument('--confidence_threshold', type=float, default=0.1, help='Confidence threshold for detections')
    args = parser.parse_args()
    
    # Set model directory if not provided
    if args.model_dir is None:
        # Find most recent model directory
        base_dir = 'models/yolo_v1'
        if os.path.exists(base_dir):
            dates = os.listdir(base_dir)
            if dates:
                latest_date = max(dates)
                times = os.listdir(os.path.join(base_dir, latest_date))
                if times:
                    latest_time = max(times)
                    args.model_dir = os.path.join(base_dir, latest_date, latest_time)
                    print(f"Using most recent model directory: {args.model_dir}")
    
    if args.model_dir is None or not os.path.exists(args.model_dir):
        print("No model directory found. Please specify using --model_dir")
        return
    
    # Load class names
    classes = utils.load_class_array()
    print(f"Loaded {len(classes)} classes: {classes}")
    
    # Create dataset and dataloader
    print("Loading validation dataset...")
    dataset = YoloRoboflowDataset('valid', normalize=True, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
    model = YOLOv1()
    model.eval()
    weights_path = os.path.join(args.model_dir, 'weights', args.weights)
    model.load_state_dict(torch.load(weights_path, map_location=utils.device))
    model = model.to(utils.device)
    
    # Create results directory if needed
    results_dir = os.path.join('results', 'evaluation', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate model
    print(f"Evaluating model on {len(dataset)} validation images...")
    results = utils.evaluate_model(model, loader, classes)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    
    # Print mAP metrics
    mAP_metrics = [key for key in results.keys() if key.startswith('mAP')]
    for metric in mAP_metrics:
        print(f"{metric}: {results[metric]:.4f}")
    
    print("\nPer-class Average Precision:")
    print("-" * 50)
    
    # Print AP metrics for each class
    for class_idx, class_name in enumerate(classes):
        ap_05 = results.get(f"AP@0.5_{class_name}", 0)
        ap_075 = results.get(f"AP@0.75_{class_name}", 0)
        print(f"{class_name}: AP@0.5={ap_05:.4f}, AP@0.75={ap_075:.4f}")
    
    # Save results to JSON
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {os.path.join(results_dir, 'evaluation_results.json')}")


if __name__ == "__main__":
    main() 