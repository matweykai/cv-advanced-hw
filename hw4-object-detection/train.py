import glob
import os

import lightning as L
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from model import YOLO
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transforms import get_train_transforms, get_val_transforms


class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        
        # Ensure we have corresponding labels
        self.image_files = [img for img in self.image_files if os.path.exists(
            os.path.join(label_dir, os.path.basename(img).replace('.jpg', '.txt')))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        # Get corresponding label file
        base_name = os.path.basename(img_path)
        # Handle the special format with seconds suffix
        label_name = base_name.replace('.00s.jpg', '.txt')
        
        # If this pattern doesn't work, try the standard one
        if not os.path.exists(os.path.join(self.label_dir, label_name)):
            # Try extracting frame number and movie
            parts = base_name.split('_')
            if len(parts) >= 4:
                # Format: Movie_X_frame_NNNNNN_YY.00s.jpg -> Movie_X_frame_NNNNNN_YY.txt
                label_name = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3].split('.')[0]}.txt"
        
        label_path = os.path.join(self.label_dir, label_name)
        
        # Load labels (class, x, y, width, height)
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    values = line.strip().split()
                    class_id = int(values[0])
                    x, y, width, height = map(float, values[1:5])
                    boxes.append([class_id, x, y, width, height])
        
        boxes = np.array(boxes)
        
        # Apply transformations if any
        if self.transform:
            transformed = self.transform(image=image_np, bboxes=boxes[:, 1:], class_ids=boxes[:, 0])
            image_np = transformed['image']
            
            # Reconstruct boxes with class ids
            if len(transformed['bboxes']) > 0:
                transformed_boxes = np.column_stack((
                    transformed['class_ids'],
                    np.array(transformed['bboxes'])
                ))
                boxes = transformed_boxes
            else:
                boxes = np.array([])
        
        # Make sure image is in the correct format for training
        if isinstance(image_np, np.ndarray):
            # Convert to tensor and normalize
            image_tensor = torch.tensor(image_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        else:
            # If still a PIL image
            image_tensor = torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32) / 255.0
        
        return {
            'image': image_tensor,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 5), dtype=torch.float32)
        }


class YOLODataModule(L.LightningDataModule):
    def __init__(self, image_dir="extracted_frames", label_dir="labels", batch_size=16, num_workers=4, val_split=0.2, img_size=416):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.img_size = img_size
        
        # Initialize transforms
        self.train_transform = get_train_transforms(height=img_size, width=img_size)
        self.val_transform = get_val_transforms(height=img_size, width=img_size)
    
    def prepare_data(self):
        # This method is called only once and on only one GPU
        # Check if directories exist
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory {self.image_dir} does not exist.")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Label directory {self.label_dir} does not exist.")
    
    def setup(self, stage=None):
        # This method is called on every GPU
        # Get all image files
        all_image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        
        # Ensure we have corresponding labels
        valid_image_files = []
        for img in all_image_files:
            # Extract base name without extension
            base_name = os.path.basename(img)
            # Handle the special format with seconds suffix
            label_name = base_name.replace('.00s.jpg', '.txt')
            
            # If this pattern doesn't work, try the standard one
            if not os.path.exists(os.path.join(self.label_dir, label_name)):
                # Try extracting frame number and movie
                parts = base_name.split('_')
                if len(parts) >= 4:
                    # Format: Movie_X_frame_NNNNNN_YY.00s.jpg -> Movie_X_frame_NNNNNN_YY.txt
                    label_name = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3].split('.')[0]}.txt"
            
            label_path = os.path.join(self.label_dir, label_name)
            if os.path.exists(label_path):
                # Only include images that have corresponding label files
                valid_image_files.append(img)
        
        print(f"Found {len(all_image_files)} images, {len(valid_image_files)} with valid labels")
        
        if len(valid_image_files) == 0:
            raise ValueError(f"No valid image-label pairs found in {self.image_dir} and {self.label_dir}")
        
        # Split into train and validation sets
        self.train_files, self.val_files = train_test_split(
            valid_image_files, test_size=self.val_split, random_state=42
        )
        
        # Create datasets based on stage
        if stage == 'fit' or stage is None:
            self.train_dataset = YOLODataset(
                image_dir=self.image_dir,
                label_dir=self.label_dir,
                transform=self.train_transform
            )
            
            self.val_dataset = YOLODataset(
                image_dir=self.image_dir,
                label_dir=self.label_dir,
                transform=self.val_transform
            )
            
            # Filter datasets to use only train/val files
            self.train_dataset.image_files = self.train_files
            self.val_dataset.image_files = self.val_files
        
        if stage == 'test' or stage is None:
            # For testing, you can use the validation set or a separate test set
            self.test_dataset = YOLODataset(
                image_dir=self.image_dir,
                label_dir=self.label_dir,
                transform=self.val_transform
            )
            self.test_dataset.image_files = self.val_files
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def teardown(self, stage=None):
        # Clean up after fit or test
        pass
    
    def on_exception(self, exception):
        # Handle any exceptions
        print(f"An exception occurred during training: {exception}")


def train_yolo(
    image_dir="extracted_frames",
    label_dir="labels",
    batch_size=16,
    num_workers=4,
    img_size=416,
    max_epochs=10,
    num_classes=2,
    log_dir="logs",
    checkpoint_dir="checkpoints"
):
    """
    Train a YOLO model with the given parameters.
    
    Args:
        image_dir (str): Directory containing training images
        label_dir (str): Directory containing label files
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        img_size (int): Image size for training (square)
        max_epochs (int): Maximum number of training epochs
        num_classes (int): Number of classes to detect
        log_dir (str): Directory to save logs
        checkpoint_dir (str): Directory to save checkpoints
        
    Returns:
        tuple: Trained model and trainer instance
    """
    # Initialize data module
    data_module = YOLODataModule(
        image_dir=image_dir,
        label_dir=label_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size
    )
    
    # Make sure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="yolo-{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                mode="min",
                monitor="val_loss"
            )
        ],
        logger=CSVLogger(log_dir),
    )
    
    # Initialize model
    model = YOLO(num_classes=num_classes, num_anchors=3, num_features=3)
    
    # Train model
    trainer.fit(model, datamodule=data_module)
    
    return model, trainer


if __name__ == "__main__":
    # This will run when the script is executed directly
    train_yolo()