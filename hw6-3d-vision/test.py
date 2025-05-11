# %%
# !mkdir -p v1.0-mini
# !tar -xzf 'nuScenes v1.0 mini.tgz' -C v1.0-mini

# %%
# !pip install -U 'nbformat>=4.2.0'

# %%
from nuscenes.nuscenes import NuScenes
from pathlib import Path
from utils import read_image, show_image, read_lidar, show_lidar_pointcloud
from pyquaternion import Quaternion

# %%
nuscenes_root_dir = Path('./v1.0-mini')

# %% [markdown]
# ## Get samples for one of the scenes

# %%
nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_root_dir)

# %%
nusc.list_scenes()

# %%
my_scene = nusc.scene[0]
my_scene

# %%
my_scene_samples_tokens = [sample['token'] for sample in nusc.sample if sample['scene_token'] == my_scene['token']]

# %%
my_scene_samples_tokens

# %% [markdown]
# ## Let's look at the data for one sample in the scene

# %%
sample_token = my_scene_samples_tokens[3]  #'e0845f5322254dafadbbed75aaa07969'

sample_info = nusc.get('sample', sample_token)

# %%
sample_info

# %% [markdown]
# ### CAM_FRONT data for the sample

# %%
camera_front_data = nusc.get('sample_data', sample_info['data']['CAM_FRONT'])

# %%
camera_front_data

# %%
camera_front_image = read_image(nuscenes_root_dir / camera_front_data['filename'])
show_image(camera_front_image)

# %%
camera_calibration = nusc.get('calibrated_sensor', camera_front_data['calibrated_sensor_token'])
camera_calibration

# %% [markdown]
# ### LIDAR_TOP data for the sample

# %%
lidar_top_data = nusc.get('sample_data', sample_info['data']['LIDAR_TOP'])
lidar_top_data

# %%
lidar_top_pointcloud = read_lidar(nuscenes_root_dir / lidar_top_data['filename'])
show_lidar_pointcloud(lidar_top_pointcloud)

# %%
lidar_calibration = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])
lidar_calibration

# %% [markdown]
# ### Transform quaternion to rotation matrix

# %%
q = Quaternion(lidar_calibration['rotation'])

# %%
q.rotation_matrix

# %%
# !uv pip install ultralytics inference'[yolo-world]' "git+https://github.com/openai/CLIP.git"

# %%
from ultralytics import YOLO
from inference.models.yolo_world.yolo_world import YOLOWorld
import supervision as sv
import cv2

# Initialize a YOLO-World model
model = YOLOWorld(model_id='yolo_world/v2-x') 

# %%
from matplotlib import pyplot as plt

road_objects = [
        "car", "truck", "bus", "motorcycle", "bicycle", 
        "person", "traffic light", "stop sign", "fire hydrant", "cone"
    ]

def detect_objects_in_camera_image(nusc, sample_info, camera_channel, nuscenes_root_dir, model: YOLOWorld, iou=0.5, confidence=0.3):
    cam_data = nusc.get('sample_data', sample_info['data'][camera_channel])
    cam_path = nuscenes_root_dir / cam_data['filename']

    # Read the image
    image = cv2.imread(str(cam_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Run YOLO-World model on the image with text prompts
    results = model.infer(
        image,
        text=road_objects,
        confidence=confidence,  # Confidence threshold
        iou=iou          # IoU threshold for NMS
    )
    
    return image, results


def display_detections(results, image, camera_channel, show_image=False):
    # Extract detections
    detections = sv.Detections.from_inference(results)

    # Create annotator for visualization
    box_annotator = sv.BoxAnnotator()

    # Annotate the image with bounding boxes
    annotated_image = box_annotator.annotate(
        scene=image.copy(), 
        detections=detections
    )
    
    return annotated_image, detections

camera_channel = 'CAM_FRONT'
# Example usage for CAM_FRONT
image, results = detect_objects_in_camera_image(
    nusc, sample_info, camera_channel, nuscenes_root_dir, model
)

annotated_image, detections = display_detections(results, image, camera_channel, show_image=True)

plt.figure(figsize=(12, 8))
plt.imshow(annotated_image)
plt.axis('off')
plt.title(f'Object Detection on {camera_channel}')
plt.show()

# %%
# Define all camera channels
camera_channels = [
    'CAM_FRONT', 
    'CAM_FRONT_LEFT', 
    'CAM_FRONT_RIGHT', 
    'CAM_BACK', 
    'CAM_BACK_LEFT', 
    'CAM_BACK_RIGHT'
]

# Create a figure to display all camera views
plt.figure(figsize=(20, 15))

# Process each camera channel
for idx, camera_channel in enumerate(camera_channels):
    # Run object detection on the camera image
    image, results = detect_objects_in_camera_image(
        nusc, sample_info, camera_channel, nuscenes_root_dir, model
    )
    
    # Extract detections
    annotated_image, detections = display_detections(results, image, camera_channel, show_image=False)
    
    # Add to subplot
    plt.subplot(2, 3, idx+1)
    plt.imshow(annotated_image)
    plt.title(f'{camera_channel} - {len(detections)} objects')
    plt.axis('off')

plt.tight_layout()
plt.show()


# %%

import numpy as np

# Define function to project LiDAR points to camera image
def project_lidar_to_camera(lidar_points_sensor_row, cam_calib, lidar_calib):
    # Step 1: LiDAR points (sensor frame) to Ego vehicle frame
    R_lidar_to_ego = Quaternion(lidar_calib['rotation']).rotation_matrix
    T_lidar_to_ego_row = np.array(lidar_calib['translation']).reshape(1, 3)
    
    lidar_points_ego_row = lidar_points_sensor_row @ R_lidar_to_ego.T + T_lidar_to_ego_row

    # Step 2: Ego vehicle frame to Camera frame
    R_cam_to_ego = Quaternion(cam_calib['rotation']).rotation_matrix
    T_cam_in_ego_row = np.array(cam_calib['translation']).reshape(1, 3) # Position of camera origin in ego coords

    points_camera_frame_row = (lidar_points_ego_row - T_cam_in_ego_row) @ R_cam_to_ego
    
    # Step 3: Filter points in front of the camera
    # Points too close to the camera plane or behind are filtered.
    in_front_cam_mask = points_camera_frame_row[:, 2] > 0.1 
    points_in_cam_frame_filtered = points_camera_frame_row[in_front_cam_mask]
    
    if points_in_cam_frame_filtered.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,3)), np.array([], dtype=bool)

    # Step 4: Camera frame to image plane (perspective projection)
    K_cam_intrinsic = np.array(cam_calib['camera_intrinsic'])
    
    # Project points to 2D image plane
    projected_hom_row = points_in_cam_frame_filtered @ K_cam_intrinsic.T
    
    # Perspective divide: divide x and y by z to get image coordinates
    projected_points_img = projected_hom_row[:, :2] / projected_hom_row[:, 2, np.newaxis]
    
    depths_in_cam_frame = projected_hom_row[:, 2] # Or points_in_cam_frame_filtered[:, 2]

    return projected_points_img, depths_in_cam_frame, points_in_cam_frame_filtered, in_front_cam_mask

# Import libraries for creating GIFs
import imageio
import os
# Use tqdm without notebook version to avoid IProgress error
from tqdm import tqdm

# Define all camera channels
camera_channels = [
    'CAM_FRONT', 
    'CAM_FRONT_LEFT', 
    'CAM_FRONT_RIGHT', 
    'CAM_BACK', 
    'CAM_BACK_LEFT', 
    'CAM_BACK_RIGHT'
]

# Create output directory for frames and GIFs
output_dir = "camera_videos"
os.makedirs(output_dir, exist_ok=True)

# Get all sample tokens from the scene
scene_token = sample_info['scene_token']
scene = nusc.get('scene', scene_token)
sample_tokens = []

# Start with the first sample in the scene
current_sample_token = scene['first_sample_token']
while current_sample_token != '':
    sample_tokens.append(current_sample_token)
    current_sample = nusc.get('sample', current_sample_token)
    current_sample_token = current_sample['next']

# Process each camera to create a video
for camera_channel in tqdm(camera_channels, desc="Processing cameras"):
    camera_frames = []
    
    # Create a directory for this camera's frames
    camera_dir = os.path.join(output_dir, camera_channel)
    os.makedirs(camera_dir, exist_ok=True)
    
    # Process each sample (frame) in the scene
    for i, sample_token in enumerate(tqdm(sample_tokens[:20], desc=f"Processing {camera_channel} frames", leave=False)):  # Limit to 20 frames for faster processing
        current_sample = nusc.get('sample', sample_token)
        
        # Get LiDAR data for this sample
        lidar_top_data = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
        lidar_calibration = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])
        
        # Read LiDAR pointcloud
        lidar_path = nuscenes_root_dir / lidar_top_data['filename']
        lidar_points = read_lidar(lidar_path)
        
        # Get camera data and calibration
        cam_data = nusc.get('sample_data', current_sample['data'][camera_channel])
        cam_calibration = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        # Run object detection on the camera image
        image, results = detect_objects_in_camera_image(
            nusc, current_sample, camera_channel, nuscenes_root_dir, model
        )
        
        # Extract detections
        annotated_image, detections = display_detections(results, image, camera_channel, show_image=False)
        
        # Create a copy for LiDAR projection
        annotated_image_with_lidar = image.copy()
        
        # Project LiDAR points to camera image
        projected_points, depths, points_in_cam_frame, valid_projection_mask = project_lidar_to_camera(
            lidar_points[:, :3],
            cam_calibration,
            lidar_calibration
        )
        
        # Filter points that are within the image boundaries
        image_height, image_width = annotated_image.shape[:2]
        in_image_mask = (
            (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_width) &
            (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_height)
        )
        
        visible_projected_points = projected_points[in_image_mask]
        visible_depths = depths[in_image_mask]
        
        # Process each detection to estimate distance
        object_distances = []
        updated_labels = []
        label_annotator = sv.LabelAnnotator()
        box_annotator = sv.BoxAnnotator()
        
        # Draw bounding boxes first
        annotated_image_with_lidar = box_annotator.annotate(
            scene=annotated_image_with_lidar, 
            detections=detections
        )
        
        for j in range(len(detections)):
            box = detections.xyxy[j]
            class_id = detections.class_id[j]
            confidence = detections.confidence[j]
            label_text = f"{model.model.names[class_id]} {confidence:.2f}"
            
            # Filter LiDAR points within the bounding box
            points_in_box_mask = (
                (visible_projected_points[:, 0] >= box[0]) & (visible_projected_points[:, 0] <= box[2]) &
                (visible_projected_points[:, 1] >= box[1]) & (visible_projected_points[:, 1] <= box[3])
            )
            
            lidar_points_in_box = visible_projected_points[points_in_box_mask]
            depths_in_box = visible_depths[points_in_box_mask]
            
            current_label = label_text
            if depths_in_box.shape[0] > 0:
                # Estimate distance (median depth of points in the box)
                distance = np.median(depths_in_box)
                object_distances.append(distance)
                current_label += f" {distance:.2f}m"
                
                # Draw LiDAR points inside the box
                for pt_idx in range(lidar_points_in_box.shape[0]):
                    cv2.circle(annotated_image_with_lidar, 
                              (int(lidar_points_in_box[pt_idx, 0]), int(lidar_points_in_box[pt_idx, 1])), 
                              2, (255, 0, 0), -1)  # Blue for points in box
            else:
                object_distances.append(float('inf'))  # No points in box
            
            updated_labels.append(current_label)
        
        # Annotate with updated labels (including distances)
        if len(detections) > 0:
            annotated_image_with_lidar = label_annotator.annotate(
                scene=annotated_image_with_lidar, 
                detections=detections, 
                labels=updated_labels
            )
        
        # Add timestamp and frame number
        timestamp = cam_data['timestamp']
        cv2.putText(annotated_image_with_lidar, 
                   f"Frame: {i} | Timestamp: {timestamp}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save the frame
        frame_path = os.path.join(camera_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(annotated_image_with_lidar, cv2.COLOR_RGB2BGR))
        
        # Add to frames list
        camera_frames.append(annotated_image_with_lidar)
    
    # Create GIF for this camera
    gif_path = os.path.join(output_dir, f"{camera_channel}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=0.2) as writer:
        for frame in camera_frames:
            writer.append_data(frame)
    
    print(f"Created GIF for {camera_channel}: {gif_path}")

print("All camera GIFs have been created in the 'camera_videos' directory.")
