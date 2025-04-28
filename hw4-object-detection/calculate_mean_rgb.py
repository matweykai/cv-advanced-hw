import os
import cv2
import numpy as np
from tqdm import tqdm

def calculate_mean_rgb(image_dir):
    # Get list of all jpg files
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return None
    
    # Initialize arrays to store channel sums and count pixels
    rgb_sums = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    for img_path in tqdm(image_files):
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
                
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Sum all pixel values
            rgb_sums += np.sum(img, axis=(0, 1))
            
            # Count number of pixels
            pixel_count += img.shape[0] * img.shape[1]
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate mean
    if pixel_count > 0:
        mean_rgb = rgb_sums / pixel_count
        return mean_rgb
    else:
        return None

if __name__ == "__main__":
    image_dir = "data/extracted_frames"
    mean_rgb = calculate_mean_rgb(image_dir)
    
    if mean_rgb is not None:
        print(f"Dataset mean RGB: {mean_rgb}")
        print(f"For use in code: mean_rgb=[{mean_rgb[0]}, {mean_rgb[1]}, {mean_rgb[2]}]")
    else:
        print("Failed to calculate mean RGB values") 