import torch
import config
import utils
import random
import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.data import Dataset


def parse_voc_xml(xml_path):
    """Parses a Pascal VOC XML file and returns a dictionary representation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size_elem = root.find('size')
    size_dict = {
        'width': size_elem.find('width').text,
        'height': size_elem.find('height').text,
        'depth': size_elem.find('depth').text,
    }
    
    objects_list = []
    for obj_elem in root.findall('object'):
        bndbox_elem = obj_elem.find('bndbox')
        bndbox_dict = {
            'xmin': bndbox_elem.find('xmin').text,
            'ymin': bndbox_elem.find('ymin').text,
            'xmax': bndbox_elem.find('xmax').text,
            'ymax': bndbox_elem.find('ymax').text,
        }
        obj_dict = {
            'name': obj_elem.find('name').text,
            'pose': obj_elem.find('pose').text,
            'truncated': obj_elem.find('truncated').text,
            'difficult': obj_elem.find('difficult').text,
            'bndbox': bndbox_dict,
        }
        objects_list.append(obj_dict)
        
    annotation_dict = {
        'folder': root.find('folder').text,
        'filename': root.find('filename').text,
        'path': root.find('path').text if root.find('path') is not None else '', # Handle optional path
        'source': {'database': root.find('source/database').text},
        'size': size_dict,
        'segmented': root.find('segmented').text,
        'object': objects_list,
    }
    return {'annotation': annotation_dict}


class YoloRoboflowDataset(Dataset):
    def __init__(self, set_type, normalize=False, augment=False):
        assert set_type in {'train', 'valid'}
        self.set_type = set_type
        self.data_dir = os.path.join(config.DATA_PATH, set_type)
        self.normalize = normalize
        self.augment = augment
        self.classes = utils.load_class_dict()

        # Find image and annotation files
        self.image_files = sorted(glob.glob(os.path.join(self.data_dir, '*.jpg')))
        self.file_pairs = []
        for img_path in self.image_files:
            # Try finding .rf.xml first, then fallback to .xml
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            ann_path_rf = os.path.join(self.data_dir, base_name + '.rf.xml')
            ann_path_plain = os.path.join(self.data_dir, base_name + '.xml')

            if os.path.exists(ann_path_rf):
                 self.file_pairs.append((img_path, ann_path_rf))
            elif os.path.exists(ann_path_plain):
                 self.file_pairs.append((img_path, ann_path_plain))
            else:
                 print(f"Warning: Annotation not found for image {img_path}")


        # Generate class index if needed
        if len(self.classes) == 0 and len(self.file_pairs) > 0:
            print("Generating class dictionary...")
            index = 0
            # Use file_pairs for iteration
            for img_path, ann_path in tqdm(self.file_pairs, desc=f'Generating class dict for {set_type}'):
                try:
                    label_dict = parse_voc_xml(ann_path) # Parse XML
                    # Assuming get_bounding_boxes can handle the parsed dict directly
                    for _, bbox_pair in enumerate(utils.get_bounding_boxes(label_dict)):
                        name, _ = bbox_pair
                        if name not in self.classes:
                            self.classes[name] = index
                            index += 1
                except ET.ParseError:
                    print(f"Warning: Could not parse XML file {ann_path}")
                except Exception as e:
                     print(f"Warning: Error processing annotation {ann_path}: {e}")

            if len(self.classes) > 0:
                 # Update config.C if it was placeholder
                 if config.C != len(self.classes):
                     print(f"Updating config.C from {config.C} to {len(self.classes)}")
                     # Note: This won't permanently change config.py, just for this run
                     config.C = len(self.classes)
                 utils.save_class_dict(self.classes)
            else:
                 print("Warning: No classes found. Check dataset and annotations.")


        # Define transforms here
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(config.IMAGE_SIZE, antialias=True) # Added antialias
        ])


    def __getitem__(self, i):
        img_path, ann_path = self.file_pairs[i]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
             print(f"Error loading image {img_path}: {e}")
             # Return dummy data or raise error
             return torch.zeros((3, *config.IMAGE_SIZE)), torch.zeros((config.S, config.S, 5 * config.B + config.C)), torch.zeros((3, *config.IMAGE_SIZE))


        # Load annotation
        try:
            label = parse_voc_xml(ann_path)
        except ET.ParseError:
            print(f"Warning: Could not parse XML file {ann_path}. Returning empty label.")
            label = {'annotation': {'object': []}} # Provide dummy structure
        except Exception as e:
             print(f"Warning: Error processing annotation {ann_path}: {e}. Returning empty label.")
             label = {'annotation': {'object': []}} # Provide dummy structure


        # Apply initial transforms
        data = self.transform(image)
        original_data = data.clone() # Clone before augmentations

        x_shift = 0
        y_shift = 0
        scale = 1.0
        # Augment images
        if self.augment:
            x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0])
            y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1])
            scale = 1 + 0.2 * random.random()
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            # Add Color Jitter for more robust augmentation
            color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            data = color_jitter(data)
            # Removed individual hue/saturation adjustments, replaced by ColorJitter

        # Normalize image
        if self.normalize:
            # Ensure data is float before normalizing
            data = data.float() if data.dtype != torch.float32 else data
            normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            data = normalizer(data)


        # Calculate grid cell size (using config.IMAGE_SIZE directly is more robust)
        grid_size_x = config.IMAGE_SIZE[0] / config.S
        grid_size_y = config.IMAGE_SIZE[1] / config.S

        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        boxes = {}
        class_names = {}                    # Track what class each grid cell has been assigned to
        
        # Use the potentially updated config.C
        local_C = len(self.classes) if len(self.classes) > 0 else config.C
        depth = 5 * config.B + local_C     # 5 numbers per bbox, then one-hot encoding of label
        ground_truth = torch.zeros((config.S, config.S, depth))

        # Use utils.get_bounding_boxes which expects the parsed dict
        bounding_boxes = utils.get_bounding_boxes(label)

        for j, bbox_pair in enumerate(bounding_boxes):
            name, coords = bbox_pair
            if name not in self.classes:
                # This case should ideally not happen if class dict is generated correctly
                print(f"Warning: Class '{name}' found in {ann_path} but not in generated class dict. Skipping box.")
                continue

            class_index = self.classes[name]
            x_min, x_max, y_min, y_max = coords # These are already scaled by get_bounding_boxes

            # Augment labels (applies shift/scale from image augmentation)
            if self.augment:
                half_width = config.IMAGE_SIZE[0] / 2
                half_height = config.IMAGE_SIZE[1] / 2
                # Apply the same shift/scale used for the image
                x_min = utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = utils.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = utils.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = utils.scale_bbox_coord(y_max, half_height, scale) + y_shift

                # Clamp coordinates to be within image bounds after augmentation
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(config.IMAGE_SIZE[0] - 1, x_max)
                y_max = min(config.IMAGE_SIZE[1] - 1, y_max)


            # Calculate the position of center of bounding box
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2

            # Check if bounding box is valid after augmentation
            if x_max <= x_min or y_max <= y_min:
                 continue # Skip invalid boxes

            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            # Ensure row/col are within grid bounds
            if 0 <= col < config.S and 0 <= row < config.S:
                cell = (row, col)
                if cell not in class_names or name == class_names[cell]:
                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(local_C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, :local_C] = one_hot
                    class_names[cell] = name

                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(cell, 0)
                    if bbox_index < config.B:
                        # Calculate relative coordinates and dimensions
                        # Ensure grid_size is not zero
                        rel_x = (mid_x - col * grid_size_x) / grid_size_x if grid_size_x > 0 else 0
                        rel_y = (mid_y - row * grid_size_y) / grid_size_y if grid_size_y > 0 else 0
                        width = (x_max - x_min) / config.IMAGE_SIZE[0]
                        height = (y_max - y_min) / config.IMAGE_SIZE[1]

                        # Clamp relative coords and dimensions to [0, 1]
                        rel_x = max(0.0, min(1.0, rel_x))
                        rel_y = max(0.0, min(1.0, rel_y))
                        width = max(0.0, min(1.0, width))
                        height = max(0.0, min(1.0, height))


                        bbox_truth = (
                            rel_x,     # X coord relative to grid square [0, 1]
                            rel_y,     # Y coord relative to grid square [0, 1]
                            width,     # Width relative to image size [0, 1]
                            height,    # Height relative to image size [0, 1]
                            1.0        # Confidence (objectness score)
                        )

                        # Fill all bbox slots with current bbox (starting from current bbox slot, avoid overriding prev)
                        bbox_start = 5 * bbox_index + local_C
                        # Ensure slice does not exceed tensor bounds
                        if bbox_start < ground_truth.shape[2]:
                            ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(config.B - bbox_index)
                        boxes[cell] = bbox_index + 1
            # else: # Optional: Log boxes falling outside grid
            #     print(f"Warning: Box center ({mid_x:.1f}, {mid_y:.1f}) falls outside grid {config.S}x{config.S} for image {img_path}")


        return data, ground_truth, original_data

    def __len__(self):
        return len(self.file_pairs) # Use the length of file pairs


if __name__ == '__main__':
    # Display data
    print("Loading class array...")
    obj_classes = utils.load_class_array()
    if not obj_classes:
         print("Class array is empty. Running dataset init to generate classes...")
         # Initialize dataset once to generate class file if needed
         temp_train_set = YoloRoboflowDataset('train', normalize=False, augment=False)
         obj_classes = utils.load_class_array() # Try loading again
         if not obj_classes:
              print("Failed to load/generate classes. Exiting.")
              exit()
         else:
              print(f"Generated/Loaded {len(obj_classes)} classes: {obj_classes}")


    print("Initializing train dataset...")
    train_set = YoloRoboflowDataset('train', normalize=True, augment=True)
    print(f"Train set size: {len(train_set)}")

    # Optional: Initialize and check valid dataset
    # print("Initializing valid dataset...")
    # valid_set = YoloRoboflowDataset('valid', normalize=True, augment=False)
    # print(f"Valid set size: {len(valid_set)}")

    # Basic check on the first item
    if len(train_set) > 0:
        print("Checking first item from train set...")
        data, label, orig_data = train_set[0]
        print(f"Data shape: {data.shape}, Label shape: {label.shape}, Original data shape: {orig_data.shape}")
        print(f"Label non-zero elements: {torch.sum(label != 0)}")
        # Add more checks if needed, e.g., min/max values
        print(f"Data min: {torch.min(data)}, max: {torch.max(data)}")

        # You can uncomment plotting, but ensure classes are loaded correctly
        # if obj_classes:
        #      print("Plotting boxes for the first item...")
        #      # Denormalize data for plotting if normalized
        #      # plot_data = data.clone() # Avoid modifying original tensor
        #      # if train_set.normalize:
        #      #     inv_normalize = T.Normalize(
        #      #         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        #      #         std=[1/0.229, 1/0.224, 1/0.225]
        #      #     )
        #      #     plot_data = inv_normalize(plot_data)
        #      # Need to use original_data for plotting as augmentations/normalization change it
        #      utils.plot_boxes(orig_data, label, obj_classes, max_overlap=float('inf'))
        # else:
        #      print("Cannot plot boxes, class array not loaded.")
    else:
        print("Train set is empty. Check data directory and file matching logic.")

    # Removed old check loop
    # negative_labels = 0
    # smallest = 0
    # largest = 0
    # print("Iterating through train set for stats (first 10 items)...")
    # for i, (data, label, _) in enumerate(train_set):
    #     if i >= 10: break # Limit check for speed
    #     # negative_labels += torch.sum(label < 0).item() # This check might not be relevant anymore
    #     smallest = min(smallest, torch.min(data).item())
    #     largest = max(largest, torch.max(data).item())
    # # print('num_negatives', negative_labels) # Commented out irrelevant check
    # print(f'Checked {min(10, len(train_set))} items. Data range: [{smallest}, {largest}]')
