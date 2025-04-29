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
            T.Resize(config.IMAGE_SIZE) # Added antialias
        ])


    def __getitem__(self, i):
        img_path, ann_path = self.file_pairs[i]

        image = Image.open(img_path).convert('RGB')

        label = parse_voc_xml(ann_path)

        # Apply initial transforms
        data = self.transform(image)
        original_data = data.clone() # Clone before augmentations

        x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0])
        y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1])
        scale = 1 + 0.2 * random.random()
        # Augment images
        if self.augment:
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            data = TF.adjust_hue(data, 0.2 * random.random() - 0.1)
            data = TF.adjust_saturation(data, 0.2 * random.random() + 0.9)
        # Normalize image
        if self.normalize:
            data = TF.normalize(data, mean=[0.2934, 0.2997, 0.2919], std=[0.2473, 0.2431, 0.2438])

        # Calculate grid cell size (using config.IMAGE_SIZE directly is more robust)
        grid_size_x = config.IMAGE_SIZE[0] / config.S
        grid_size_y = config.IMAGE_SIZE[1] / config.S

        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        boxes = {}
        class_names = {}                    # Track what class each grid cell has been assigned to
        depth = 5 * config.B + config.C     # 5 numbers per bbox, then one-hot encoding of label
        ground_truth = torch.zeros((config.S, config.S, depth))
        for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):
            name, coords = bbox_pair
            assert name in self.classes, f"Unrecognized class '{name}'"
            class_index = self.classes[name]
            x_min, x_max, y_min, y_max = coords

            # Augment labels
            if self.augment:
                half_width = config.IMAGE_SIZE[0] / 2
                half_height = config.IMAGE_SIZE[1] / 2
                x_min = utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = utils.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = utils.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = utils.scale_bbox_coord(y_max, half_height, scale) + y_shift

            # Calculate the position of center of bounding box
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            if 0 <= col < config.S and 0 <= row < config.S:
                cell = (row, col)
                if cell not in class_names or name == class_names[cell]:
                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(config.C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, :config.C] = one_hot
                    class_names[cell] = name

                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(cell, 0)
                    if bbox_index < config.B:
                        bbox_truth = (
                            (mid_x - col * grid_size_x) / grid_size_x,              # X coord relative to grid square (normalized 0-1)
                            (mid_y - row * grid_size_y) / grid_size_y,              # Y coord relative to grid square (normalized 0-1)
                            (x_max - x_min) / config.IMAGE_SIZE[0],                 # Width relative to image size
                            (y_max - y_min) / config.IMAGE_SIZE[1],                 # Height relative to image size
                            1.0                                                     # Confidence
                        )

                        # Fill only the appropriate bbox slot
                        bbox_start = 5 * bbox_index + config.C
                        ground_truth[row, col, bbox_start:bbox_start + 5] = torch.tensor(bbox_truth)
                        boxes[cell] = bbox_index + 1

        return data, ground_truth, original_data

    def __len__(self):
        return len(self.file_pairs)


if __name__ == '__main__':
    # Display data
    obj_classes = utils.load_class_array()
    train_set = YoloRoboflowDataset('train', normalize=False, augment=False)

    negative_labels = 0
    smallest = 0
    largest = 0
    for data, label, _ in train_set:
        negative_labels += torch.sum(label < 0).item()
        smallest = min(smallest, torch.min(data).item())
        largest = max(largest, torch.max(data).item())
        # utils.plot_boxes(data, label, obj_classes, max_overlap=float('inf'))
    # Calculate mean and std of the training dataset
    print("Calculating mean and std of the training dataset...")
    
    # Initialize variables for mean and std calculation
    sum_pixels = 0
    sum_squared_pixels = 0
    num_pixels = 0
    
    # Create a DataLoader to efficiently process the dataset
    from torch.utils.data import DataLoader
    loader = DataLoader(train_set, batch_size=32, num_workers=4, shuffle=False)
    
    # Iterate through the dataset
    from tqdm import tqdm
    for batch, _, _ in tqdm(loader):
        # batch shape: [B, C, H, W]
        batch_size = batch.size(0)
        channels = batch.size(1)
        height = batch.size(2)
        width = batch.size(3)
        
        # Reshape to [B, C, H*W]
        batch = batch.view(batch_size, channels, -1)
        
        # Sum all pixel values and squared pixel values
        sum_pixels += torch.sum(batch, dim=[0, 2])
        sum_squared_pixels += torch.sum(batch ** 2, dim=[0, 2])
        num_pixels += batch_size * height * width
    
    # Calculate mean and std
    mean = sum_pixels / num_pixels
    var = (sum_squared_pixels / num_pixels) - (mean ** 2)
    std = torch.sqrt(var)
    
    print(f"Dataset mean: {mean}")
    print(f"Dataset std: {std}")
    # print('num_negatives', negative_labels)
    # print('dist', smallest, largest)