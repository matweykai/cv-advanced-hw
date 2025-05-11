import torch
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from data_loader import YoloRoboflowDataset
from loss import SumSquaredErrorLoss
from models import YOLOv1
import config
from utils import (
    device, mean_average_precision, extract_predictions, extract_ground_truths,
    convert_to_evaluation_format
)

import wandb


if __name__ == '__main__':   

    wandb.init(project="yolo")
   # Prevent recursive subprocess creation
    torch.autograd.set_detect_anomaly(True)         # Check for nan loss
    now = datetime.now()

    model = YOLOv1().to(device)
    loss_function = SumSquaredErrorLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=5E-4
    )

    # Load the dataset
    train_set = YoloRoboflowDataset('train', normalize=True, augment=False)
    test_set = YoloRoboflowDataset('valid', normalize=True, augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True,
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=24,
        num_workers=8,
        persistent_workers=True,
        drop_last=True
    )

    # Print dataset information
    print(f"Training dataset: {len(train_set)} images")
    print(f"Training batches: {len(train_loader)} batches of size {config.BATCH_SIZE}")
    print(f"Validation dataset: {len(test_set)} images")
    print(f"Validation batches: {len(test_loader)} batches of size {config.BATCH_SIZE}")
    
    # # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 if epoch < config.WARMUP_EPOCHS else 
                               0.1 if epoch < config.WARMUP_EPOCHS + 30 else 0.01
    )

    # Create folders
    root = os.path.join(
        'models',
        'yolo_v1',
        now.strftime('%m_%d_%Y'),
        now.strftime('%H_%M_%S')
    )
    weight_dir = os.path.join(root, 'weights')
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)

    # Metrics
    train_losses = np.empty((2, 0))
    test_losses = np.empty((2, 0))
    train_errors = np.empty((2, 0))
    test_errors = np.empty((2, 0))


    def save_metrics():
        np.save(os.path.join(root, 'train_losses'), train_losses)
        np.save(os.path.join(root, 'test_losses'), test_losses)
        np.save(os.path.join(root, 'train_errors'), train_errors)
        np.save(os.path.join(root, 'test_errors'), test_errors)

    #####################
    #       Train       #
    #####################
    for epoch in tqdm(range(config.WARMUP_EPOCHS + config.EPOCHS), desc='Epoch'):
        model.train()
        train_loss = 0
        all_pred_boxes = []
        all_true_boxes = []
        
        for batch_idx, (data, labels, _) in enumerate(tqdm(train_loader, desc='Train', leave=False)):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model.forward(data)
            loss = loss_function(predictions, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)
            
            # Get bounding boxes for mAP calculation
            batch_size = data.shape[0]
            
            batch_preds = extract_predictions(predictions, class_conf_threshold=0.5)
            batch_gts = extract_ground_truths(labels)
            
            # Convert to list format for mAP calculation
            batch_offset = batch_idx * batch_size
            for i in range(batch_size):
                # Add predictions for this image
                pred_boxes_list = convert_to_evaluation_format(batch_preds[i], batch_offset + i)
                all_pred_boxes.extend(pred_boxes_list)
                
                # Add ground truths for this image
                gt_boxes_list = convert_to_evaluation_format(batch_gts[i], batch_offset + i)
                all_true_boxes.extend(gt_boxes_list)

        # Calculate mAP
        train_map = mean_average_precision(
            all_pred_boxes, 
            all_true_boxes, 
            iou_threshold=0.5, 
            box_format="midpoint", 
            num_classes=config.C
        )
        print(f"Epoch {epoch} - Train mAP: {train_map:.4f}")

        # # Step and graph scheduler once an epoch
        # writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        # scheduler.step()

        train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
        wandb.log({"train_loss": loss.item(), "train_mAP": train_map})

        if epoch % 4 == 0:
            model.eval()

            with torch.no_grad():
                test_loss = 0
                all_pred_boxes = []
                all_true_boxes = []
                
                for test_batch_idx, (data, labels, _) in enumerate(tqdm(test_loader, desc='Test', leave=False)):
                    data = data.to(device)
                    labels = labels.to(device)

                    predictions = model.forward(data)
                    loss = loss_function(predictions, labels)

                    test_loss += loss.item() / len(test_loader)
                    
                    # Get bounding boxes for mAP calculation using shared functions
                    batch_size = data.shape[0]
                    batch_preds = extract_predictions(predictions, class_conf_threshold=0.5)
                    batch_gts = extract_ground_truths(labels)
                    
                    # Convert to list format for mAP calculation
                    batch_offset = test_batch_idx * batch_size
                    for i in range(batch_size):
                        # Add predictions for this image
                        pred_boxes_list = convert_to_evaluation_format(batch_preds[i], batch_offset + i)
                        all_pred_boxes.extend(pred_boxes_list)
                        
                        # Add ground truths for this image
                        gt_boxes_list = convert_to_evaluation_format(batch_gts[i], batch_offset + i)
                        all_true_boxes.extend(gt_boxes_list)
                    
                    del data, labels
                
                # Calculate test mAP
                test_map = mean_average_precision(
                    all_pred_boxes, 
                    all_true_boxes, 
                    iou_threshold=0.5, 
                    box_format="midpoint", 
                    num_classes=config.C
                )
                print(f"Epoch {epoch} - Test mAP: {test_map:.4f}")
                
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            wandb.log({"test_loss": test_loss, "test_mAP": test_map})
            save_metrics()
    save_metrics()
    torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))
