import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms(height=416, width=416):
    """
    Get training transformations for object detection.
    
    Args:
        height (int): Target height for resizing.
        width (int): Target width for resizing.
        
    Returns:
        albumentations.Compose: Composition of transformations.
    """
    return A.Compose(
        [
            A.Resize(height=height, width=width, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomRotate90(p=0.2),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
            A.Blur(blur_limit=3, p=0.1),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
        ],
        bbox_params=A.BboxParams(
            format='yolo',  # YOLO format: [x_center, y_center, width, height]
            min_area=0,
            min_visibility=0.1,
            label_fields=['class_ids']
        )
    )


def get_val_transforms(height=416, width=416):
    """
    Get validation transformations for object detection.
    
    Args:
        height (int): Target height for resizing.
        width (int): Target width for resizing.
        
    Returns:
        albumentations.Compose: Composition of transformations.
    """
    return A.Compose(
        [
            A.Resize(height=height, width=width, always_apply=True),
        ],
        bbox_params=A.BboxParams(
            format='yolo',  # YOLO format: [x_center, y_center, width, height]
            min_area=0,
            min_visibility=0.1,
            label_fields=['class_ids']
        )
    )


def get_strong_transforms(height=416, width=416):
    """
    Get stronger augmentations for training when more augmentation is needed.
    
    Args:
        height (int): Target height for resizing.
        width (int): Target width for resizing.
        
    Returns:
        albumentations.Compose: Composition of transformations.
    """
    return A.Compose(
        [
            A.Resize(height=height, width=width, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomRotate90(p=0.5),
            A.GridDistortion(p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format='yolo',  # YOLO format: [x_center, y_center, width, height]
            min_area=0,
            min_visibility=0.1,
            label_fields=['class_ids']
        )
    )


def get_mosaic_transforms(height=416, width=416):
    """
    Get transformations for mosaic augmentation (inspired by YOLOv5).
    This needs to be combined with custom mosaic implementation.
    
    Args:
        height (int): Target height for resizing.
        width (int): Target width for resizing.
        
    Returns:
        albumentations.Compose: Composition of transformations.
    """
    return A.Compose(
        [
            A.Resize(height=height, width=width, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
        ],
        bbox_params=A.BboxParams(
            format='yolo',  # YOLO format: [x_center, y_center, width, height]
            min_area=0,
            min_visibility=0.1,
            label_fields=['class_ids']
        )
    ) 