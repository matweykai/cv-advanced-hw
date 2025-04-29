import torch
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data_roboflow import YoloRoboflowDataset
from loss import SumSquaredErrorLoss
from models import YOLOv1
import config
from utils import device
import wandb


if __name__ == '__main__':   

    wandb.init(project="yolo")
   # Prevent recursive subprocess creation
    torch.autograd.set_detect_anomaly(True)         # Check for nan loss
    writer = SummaryWriter()
    now = datetime.now()

    model = YOLOv1().to(device)
    loss_function = SumSquaredErrorLoss()

    # # Adam works better
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=config.LEARNING_RATE,
    #     weight_decay=5E-4
    # )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=5E-4
    )

    # Load the dataset
    train_set = YoloRoboflowDataset('train', normalize=True, augment=True)
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
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True
    )

    # Print dataset information
    print(f"Training dataset: {len(train_set)} images")
    print(f"Training batches: {len(train_loader)} batches of size {config.BATCH_SIZE}")
    print(f"Validation dataset: {len(test_set)} images")
    print(f"Validation batches: {len(test_loader)} batches of size {config.BATCH_SIZE}")
    
    # Learning rate scheduler
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
        for batch_idx, (data, labels, _) in enumerate(tqdm(train_loader, desc='Train', leave=False)):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model.forward(data)
            loss = loss_function(predictions, labels)
            
            # Debug info for loss components
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    # Check if predictions have any reasonable values
                    pred_confidence = predictions[..., config.C + 4].detach()
                    gt_confidence = labels[..., config.C + 4].detach()
                    print(f"Batch {batch_idx}:")
                    print(f"  Pred confidence: min={pred_confidence.min().item():.4f}, max={pred_confidence.max().item():.4f}, mean={pred_confidence.mean().item():.4f}")
                    print(f"  GT confidence: min={gt_confidence.min().item():.4f}, max={gt_confidence.max().item():.4f}, mean={gt_confidence.mean().item():.4f}")
                    print(f"  Batch loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)
            wandb.log({"train_loss": loss.item()})
            del data, labels

        # Step and graph scheduler once an epoch
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        scheduler.step()

        train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
        writer.add_scalar('Loss/train', train_loss, epoch)

        if epoch % 4 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for data, labels, _ in tqdm(test_loader, desc='Test', leave=False):
                    data = data.to(device)
                    labels = labels.to(device)

                    predictions = model.forward(data)
                    loss = loss_function(predictions, labels)

                    test_loss += loss.item() / len(test_loader)
                    del data, labels
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            writer.add_scalar('Loss/test', test_loss, epoch)
            save_metrics()
    save_metrics()
    torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))
