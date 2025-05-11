import torch
import torchvision.transforms as T
import cv2

import config
import utils
from utils import extract_predictions, apply_nms

def predict(model, image, confidence_threshold=0.5, iou_threshold=0.7):
    """
    Predicts objects in an image using a YOLO model.
    
    Args:
        model: The YOLO model to use for prediction
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score to consider a detection
        iou_threshold: IoU threshold for non-maximum suppression
        
    Returns:
        List of detections in format [(class_name, score, (x_min, y_min, x_max, y_max)), ...]
    """
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    
    # Convert to tensor
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(utils.device)
    
    # Get predictions using shared code
    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Extract detections
    detections = extract_predictions(predictions, confidence_threshold)[0]  # Get first image from batch
    
    # Apply NMS
    filtered_detections = apply_nms(detections, iou_threshold, confidence_threshold)
    
    # Format results
    result = []
    
    if filtered_detections:
        # Get class names
        classes = utils.load_class_array()
        
        # Prepare detection results
        for det in filtered_detections:
            class_idx = det['class_idx']
            score = det['confidence']
            box = det['bbox']
            
            # Convert normalized coordinates to pixel values and to corner format
            x_center, y_center, width, height = box
            x_min = int((x_center - width/2) * original_width)
            y_min = int((y_center - height/2) * original_height)
            x_max = int((x_center + width/2) * original_width)
            y_max = int((y_center + height/2) * original_height)
            
            class_name = classes[class_idx]
            result.append((class_name, score, (x_min, y_min, x_max, y_max)))
    
    return result