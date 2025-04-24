import argparse
import os
from train import train_yolo


def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection Training and Inference")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    train_parser.add_argument("--image-dir", type=str, default="extracted_frames", 
                              help="Directory containing training images")
    train_parser.add_argument("--label-dir", type=str, default="labels", 
                              help="Directory containing label files")
    train_parser.add_argument("--batch-size", type=int, default=16, 
                              help="Batch size for training")
    train_parser.add_argument("--num-workers", type=int, default=4, 
                              help="Number of workers for data loading")
    train_parser.add_argument("--img-size", type=int, default=416, 
                              help="Image size for training (square)")
    train_parser.add_argument("--max-epochs", type=int, default=10, 
                              help="Maximum number of training epochs")
    train_parser.add_argument("--num-classes", type=int, default=2, 
                              help="Number of classes to detect")
    train_parser.add_argument("--log-dir", type=str, default="logs", 
                              help="Directory to save logs")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", 
                              help="Directory to save checkpoints")
    
    # TODO: Add inference command parser once implemented
    
    args = parser.parse_args()
    
    if args.command == "train":
        print(f"Starting training with {args.batch_size} batch size and {args.max_epochs} epochs...")
        model, trainer = train_yolo(
            image_dir=args.image_dir,
            label_dir=args.label_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            max_epochs=args.max_epochs,
            num_classes=args.num_classes,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir
        )
        print(f"Training completed. Checkpoints saved to {args.checkpoint_dir}")
    elif args.command is None:
        parser.print_help()


if __name__ == "__main__":
    main()
