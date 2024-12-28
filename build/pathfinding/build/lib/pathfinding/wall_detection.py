import cv2
import numpy as np


def preprocess_disparity(self, disparity_image):
    # Convert to float32 for processing
    disparity = disparity_image.astype(np.float32)
    
    # Replace invalid disparity values (e.g., zeros) with NaN
    disparity[disparity <= 0] = np.nan
    
    # Apply a median filter to reduce noise
    disparity_filtered = cv2.medianBlur(disparity, 5)
    
    return disparity_filtered

def exclude_detected_objects(self, disparity_image, detections):
    # Create a mask for detected objects
    object_mask = np.zeros_like(disparity_image, dtype=np.uint8)
    for detection in detections:
        if not hasattr(detection, "xyxy"):
            continue
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        cv2.rectangle(object_mask, (x1, y1), (x2, y2), 255, -1)
    
    # Set disparity values within object regions to NaN
    disparity_image[object_mask > 0] = np.nan
    return disparity_image

def compute_disparity_variance(self, disparity_image):
    # Define window size for variance calculation
    window_size = 5  # Adjust as needed
    
    # Compute local mean and squared mean
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)
    local_mean = cv2.filter2D(disparity_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    local_mean_sq = cv2.filter2D(disparity_image ** 2, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    
    # Compute local variance
    local_variance = local_mean_sq - local_mean ** 2
    
    return local_variance

def filter_by_disparity_value(self, disparity_image, wall_mask):
    # Assuming you have a mapping from disparity to distance
    # For simplicity, let's say higher disparity means closer
    # Define disparity range corresponding to 9 feet (adjust as needed)
    max_disparity_for_9ft = some_value  # You can provide this value
    
    # Create a mask for disparity values within the desired range
    disparity_mask = np.zeros_like(disparity_image, dtype=np.uint8)
    disparity_mask[disparity_image >= max_disparity_for_9ft] = 255
    
    # Combine masks
    wall_mask = cv2.bitwise_and(wall_mask, disparity_mask)
    
    return wall_mask

def map_wall_mask_to_grid(self, wall_mask):
    # Focus on rows 4-6
    grid_rows = [4, 5, 6]
    grid = np.zeros(GRID_SIZE, dtype=int)
    
    # Resize wall mask to match grid dimensions
    wall_mask_resized = cv2.resize(wall_mask, (GRID_SIZE[1], len(grid_rows)), interpolation=cv2.INTER_NEAREST)
    
    # Map to grid rows 4-6
    for i, grid_row in enumerate(grid_rows):
        grid[grid_row] = (wall_mask_resized[i] > 0).astype(int)
    
    return grid
