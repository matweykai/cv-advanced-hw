import torch
import torchvision.transforms as transforms

import os
import cv2
import argparse
import numpy as np
import config

from model import YOLOv1
from utils.nms import nms

# VOC class names and BGR color.
VOC_CLASS_BGR = {
    'pig': (128, 0, 0),
    'worker': (0, 128, 0)
}


def visualize_boxes(image_bgr, boxes, class_names, probs, name_bgr_dict=None, line_thickness=2):
    if name_bgr_dict is None:
        name_bgr_dict = VOC_CLASS_BGR

    image_boxes = image_bgr.copy()
    for box, class_name, prob in zip(boxes, class_names, probs):
        # Draw box on the image.
        left_top, right_bottom = box
        left, top = int(left_top[0]), int(left_top[1])
        right, bottom = int(right_bottom[0]), int(right_bottom[1])
        bgr = name_bgr_dict[class_name]
        cv2.rectangle(image_boxes, (left, top), (right, bottom), bgr, thickness=line_thickness)

        # Draw text on the image.
        text = '%s %.2f' % (class_name, prob)
        size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
        text_w, text_h = size

        x, y = left, top
        x1y1 = (x, y)
        x2y2 = (x + text_w + line_thickness, y + text_h + line_thickness + baseline)
        cv2.rectangle(image_boxes, x1y1, x2y2, bgr, -1)
        cv2.putText(image_boxes, text, (x + line_thickness, y + 2 * baseline + line_thickness),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=8)

    return image_boxes


class YOLODetector:
    def __init__(self,
                 model_path, 
                 class_name_list=None, 
                 mean_rgb=[74.66735538, 76.50005623, 74.29610217],
                 conf_thresh=0.1, 
                 prob_thresh=0.1, 
                 nms_thresh=0.5,
                 use_cuda=False):

        # Load YOLO model.
        print("Loading YOLO model...")
        yolo = YOLOv1()
        
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.yolo = yolo
        
        # Load weights from model path
        state_dict = torch.load(model_path, map_location=self.device)

        # Remove 'module.' prefix from state_dict keys if present
        if 'state_dict' in state_dict:
            new_state_dict = {}
            for k, v in state_dict['state_dict'].items():
                name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
                new_state_dict[name] = v
            self.yolo.load_state_dict(new_state_dict)
        else:
            self.yolo.load_state_dict(state_dict)
        
        # Move model to device
        self.yolo = self.yolo.to(self.device)
        
        # Use DataParallel if multiple GPUs available
        if use_cuda and torch.cuda.device_count() > 1:
            self.yolo = torch.nn.DataParallel(self.yolo)
            
        print(f"Model loaded on {self.device}!")

        self.yolo.eval()

        # Get model parameters from config
        self.S = config.S
        self.B = config.B
        self.C = config.C

        self.class_name_list = class_name_list if (class_name_list is not None) else list(VOC_CLASS_BGR.keys())
        # assert len(self.class_name_list) == self.C

        self.mean = np.array(mean_rgb, dtype=np.float32)
        assert self.mean.shape == (3,)

        # Detection thresholds
        self.conf_thresh = conf_thresh
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

        self.to_tensor = transforms.ToTensor()

        # Warm up the model
        self._warmup()
        
    def _warmup(self, iterations=10):
        """Warm up the model with dummy inputs"""
        dummy_input = torch.zeros((1, 3, 448, 448)).to(self.device)
        with torch.no_grad():
            for _ in range(iterations):
                self.yolo(dummy_input)

    def detect(self, image_bgr, image_size=448):
        h, w, _ = image_bgr.shape
        
        # Preprocess image
        img = self._preprocess_image(image_bgr, image_size)
        
        # Forward pass
        with torch.no_grad():
            pred_tensor = self.yolo(img)
            
        pred_tensor = pred_tensor.cpu().data
        pred_tensor = pred_tensor.squeeze(0)  # squeeze batch dimension.

        # Get detected boxes_detected, labels, confidences, class-scores.
        boxes_normalized_all, class_labels_all, confidences_all, class_scores_all = self.decode(pred_tensor)
        if boxes_normalized_all.size(0) == 0:
            return [], [], []  # if no box found, return empty lists.

        # Apply non maximum supression for boxes of each class.
        boxes_normalized, class_labels, probs = self._apply_nms(
            boxes_normalized_all, 
            class_labels_all, 
            confidences_all, 
            class_scores_all
        )

        # Convert normalized coordinates to image coordinates
        boxes_detected, class_names_detected, probs_detected = self._postprocess_detections(
            boxes_normalized, class_labels, probs, w, h
        )

        return boxes_detected, class_names_detected, probs_detected
        
    def _preprocess_image(self, image_bgr, image_size):
        """Preprocess an image for detection"""
        img = cv2.resize(image_bgr, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # assuming the model is trained with RGB images.
        img = (img - self.mean) / 255.0
        img = self.to_tensor(img)  # [image_size, image_size, 3] -> [3, image_size, image_size]
        img = img[None, :, :, :]  # [3, image_size, image_size] -> [1, 3, image_size, image_size]
        img = img.to(self.device)
        return img
    
    def _apply_nms(self, boxes_normalized_all, class_labels_all, confidences_all, class_scores_all):
        """Apply non-maximum suppression to detection results"""
        boxes_normalized, class_labels, probs = [], [], []

        for class_label in range(len(self.class_name_list)):
            mask = (class_labels_all == class_label)
            if torch.sum(mask) == 0:
                continue  # if no box found, skip that class.

            boxes_normalized_masked = boxes_normalized_all[mask]
            class_labels_maked = class_labels_all[mask]
            confidences_masked = confidences_all[mask]
            class_scores_masked = class_scores_all[mask]

            ids = nms(boxes_normalized_masked, confidences_masked, nms_thresh=self.nms_thresh)

            boxes_normalized.append(boxes_normalized_masked[ids])
            class_labels.append(class_labels_maked[ids])
            probs.append(confidences_masked[ids] * class_scores_masked[ids])

        if not boxes_normalized:  # Check if all lists are empty
            return torch.FloatTensor(0, 4), torch.LongTensor(0), torch.FloatTensor(0)
            
        boxes_normalized = torch.cat(boxes_normalized, 0)
        class_labels = torch.cat(class_labels, 0)
        probs = torch.cat(probs, 0)
        
        return boxes_normalized, class_labels, probs
    
    def _postprocess_detections(self, boxes_normalized, class_labels, probs, image_width, image_height):
        """Convert normalized coordinates to image coordinates and class indices to names"""
        boxes_detected, class_names_detected, probs_detected = [], [], []
        
        for b in range(boxes_normalized.size(0)):
            box_normalized = boxes_normalized[b]
            class_label = class_labels[b]
            prob = probs[b]

            # Unnormalize coordinates with image dimensions
            x1, x2 = image_width * box_normalized[0], image_width * box_normalized[2]
            y1, y2 = image_height * box_normalized[1], image_height * box_normalized[3]
            boxes_detected.append(((x1, y1), (x2, y2)))

            # Convert class index to class name
            class_label = int(class_label)
            class_name = self.class_name_list[class_label]
            class_names_detected.append(class_name)

            # Convert probability from tensor to float
            prob = float(prob)
            probs_detected.append(prob)
            
        return boxes_detected, class_names_detected, probs_detected

    def decode(self, pred_tensor):
        """Decode prediction tensor to boxes, labels, confidences and scores"""
        S, B, C = self.S, self.B, self.C
        boxes, labels, confidences, class_scores = [], [], [], []

        cell_size = 1.0 / float(S)

        conf = pred_tensor[:, :, 4].unsqueeze(2)  # [S, S, 1]
        for b in range(1, B):
            conf = torch.cat((conf, pred_tensor[:, :, 5 * b + 4].unsqueeze(2)), 2)
        conf_mask = conf > self.conf_thresh  # [S, S, B]

        for i in range(S):  # for x-dimension.
            for j in range(S):  # for y-dimension.
                class_score, class_label = torch.max(pred_tensor[j, i, 5 * B:], 0)

                for b in range(B):
                    conf = pred_tensor[j, i, 5 * b + 4]
                    prob = conf * class_score
                    if float(prob) < self.prob_thresh:
                        continue

                    # Compute box corner (x1, y1, x2, y2) from tensor.
                    box = pred_tensor[j, i, 5 * b: 5 * b + 4]
                    x0y0_normalized = torch.FloatTensor([i, j]) * cell_size  # cell left-top corner
                    xy_normalized = box[:2] * cell_size + x0y0_normalized  # box center
                    wh_normalized = box[2:]  # Box width and height
                    box_xyxy = torch.FloatTensor(4)
                    box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized  # left-top corner (x1, y1)
                    box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized  # right-bottom corner (x2, y2)

                    # Append result to the lists.
                    boxes.append(box_xyxy)
                    labels.append(class_label)
                    confidences.append(conf)
                    class_scores.append(class_score)

        if len(boxes) > 0:
            boxes = torch.stack(boxes, 0)  # [n_boxes, 4]
            labels = torch.stack(labels, 0)  # [n_boxes, ]
            confidences = torch.stack(confidences, 0)  # [n_boxes, ]
            class_scores = torch.stack(class_scores, 0)  # [n_boxes, ]
        else:
            # If no box found, return empty tensors.
            boxes = torch.FloatTensor(0, 4)
            labels = torch.LongTensor(0)
            confidences = torch.FloatTensor(0)
            class_scores = torch.FloatTensor(0)

        return boxes, labels, confidences, class_scores


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv1 implementation using PyTorch')
    parser.add_argument('--weight', default='weights/best.pth', help='Model path')
    parser.add_argument('--in_path', default='data/train/Movie_1_frame_000045s_jpg.rf.2991775f3f0b9901244fce6514ce133b.jpg', help='Input image path')
    parser.add_argument('--out_path', default='result.jpg', help='Output image path')
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--prob_thresh', type=float, default=0.1, help='Probability threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.35, help='NMS threshold')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    # Load model
    yolo = YOLODetector(
        args.weight, 
        conf_thresh=args.conf_thresh, 
        prob_thresh=args.prob_thresh, 
        nms_thresh=args.nms_thresh,
        use_cuda=args.use_cuda
    )

    # Check if input file exists
    if not os.path.isfile(args.in_path):
        print(f"Error: Input file '{args.in_path}' does not exist.")
        exit(1)
        
    # Load image
    image = cv2.imread(args.in_path)
    if image is None:
        print(f"Error: Could not read image '{args.in_path}'.")
        exit(1)

    # Detect objects
    boxes, class_names, probs = yolo.detect(image)

    print(boxes, class_names, probs)

    # Visualize
    image_boxes = visualize_boxes(image, boxes, class_names, probs)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.out_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Output detection result as an image
    cv2.imwrite(args.out_path, image_boxes)
    print(f"Detection result saved to '{args.out_path}'")