import os
import cv2
import numpy as np
import heapq
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray  # For publishing path coordinates to motor control
from ultralytics import YOLO
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time  # For timing the command sequences

# Initialize YOLO model
model = YOLO("yolov5nu.pt")
# class_names = model.names
# print(class_names)

# Parameters
GRID_SIZE = (3, 3)  # 5x5 grid for A* pathfinding
MAX_DISPARITY = 255  # Maximum grayscale value in disparity image
SIMPLE_MODE = True  # Set to True to use simplified person-following logic

class NavigationNode(Node):
    def __init__(self):
        super().__init__('pathfinding_node')
        self.bridge = CvBridge()
        
        # Set up QoS profile with Best Effort reliability and a queue size of 10
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to stereo camera outputs with the Best Effort QoS profile
        self.color_sub = self.create_subscription(Image, '/multisense/aux/image_rect_color', self.color_image_callback, qos_profile)
        self.disparity_sub = self.create_subscription(Image, '/multisense/left/disparity', self.disparity_image_callback, qos_profile)

        # Publisher for path points
        self.path_pub = self.create_publisher(Int32MultiArray, 'navigation_path', 10)

        self.grid_size = GRID_SIZE

        # Variables to store images
        self.color_frame = None
        self.disparity_image = None
        self.processing_interval = 0.5
        self.timer = self.create_timer(self.processing_interval, self.process_images)

        # State variables for command sequences
        self.command_sequence = []  # List of (command, duration) tuples
        self.command_start_time = None
        self.current_command_index = 0

    def color_image_callback(self, msg):
        # Convert the color image from ROS to OpenCV format
        self.color_frame = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
        self.color_frame = cv2.resize(self.color_frame, (640, 360))
        # Verify that the image has the correct shape and log it
        if len(self.color_frame.shape) == 2:  # Single channel (grayscale) image
            self.color_frame = cv2.cvtColor(self.color_frame, cv2.COLOR_GRAY2BGR)
            self.get_logger().info("Converted grayscale image to BGR format.")
        
        # Calculate and log the mean intensity for debugging
        mean_intensity = np.mean(self.color_frame)
        self.get_logger().info(f"Color Image Mean Intensity: {mean_intensity}")
        self.get_logger().info(f"Color Image Shape: {self.color_frame.shape}")

    def disparity_image_callback(self, msg):
        # Convert the disparity image from ROS to OpenCV format
        self.disparity_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
        self.disparity_image = cv2.resize(self.disparity_image, (640, 360))
        # Calculate and log the mean intensity for debugging
        self.mean_intensity = np.mean(self.disparity_image)
        self.get_logger().info(f"Disparity Image Mean Intensity: {self.mean_intensity}")

    def process_images(self):
        # Ensure the color image is available before processing
        if self.color_frame is None:
            return

        # Perform Object Detection
        try:
            results = model.predict(self.color_frame, save=False)
            detections = results[0].boxes
            if not detections:
                self.get_logger().warn("Warning: No objects detected in the image.")
                # No detections, set default command sequence
                self.determine_navigation_to_person(None)
                return
        except ValueError as e:
            self.get_logger().error(f"Error during model prediction: {e}")
            return

        # Simplified logic
        # Always check for person detections
        person_detections = self.get_person_detections(detections)
        if not person_detections:
            self.get_logger().info("No person detected. Moving straight.")
            # No person detected, set default command sequence
            self.determine_navigation_to_person(None)
        else:
            person_box = person_detections[0]  # Use the first detected person
            self.determine_navigation_to_person(person_box)

        # Execute command sequence
        if self.command_sequence and self.current_command_index < len(self.command_sequence):
            command, duration = self.command_sequence[self.current_command_index]
            elapsed_time = time.time() - self.command_start_time
            if elapsed_time >= duration:
                # Move to next command
                self.current_command_index += 1
                if self.current_command_index < len(self.command_sequence):
                    self.command_start_time = time.time()
                    command, duration = self.command_sequence[self.current_command_index]
            navigation_command = command
        else:
            # No active command sequence
            navigation_command = [0, 1]  # Default action

        path_msg = Int32MultiArray(data=navigation_command)
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published navigation command: {navigation_command}")


    def get_person_detections(self, detections):
        person_class_id = 0  # Class ID for 'person' in COCO dataset
        person_detections = []
        class_names = model.names  # {class_id: class_name}

        # Loop over each detection
        for detection in detections:
            # Get the class ID
            cls_id = int(detection.cls.item())  # Use .item() to get the scalar value
            cls_name = class_names.get(cls_id, 'Unknown')
            self.get_logger().info(f"Detected class ID: {cls_id}, Name: {cls_name}")

            # Check if it's a person
            if cls_id == person_class_id:
                person_detections.append(detection)

        self.get_logger().info(f"Total person detections: {len(person_detections)}")
        return person_detections

    def determine_navigation_to_person(self, person_box):
        if person_box is None:
            # No person detected, default action
            self.command_sequence = [([0, 1], 1.0)]  # Move straight for 1 second
            self.command_start_time = time.time()
            self.current_command_index = 0
            self.get_logger().info("No person detected. Moving straight.")
            return

        x1, y1, x2, y2 = map(int, person_box.xyxy[0])
        image_width = self.color_frame.shape[1]
        center_x = (x1 + x2) / 2

        if center_x < image_width / 3:
            # Person is on the left side
            self.command_sequence = [
                ([1, 1], 0.5),   # Turn right for 0.5s
                ([-1, 1], 0.5),  # Turn left for 0.5s
                ([0, 1], 1.0)    # Move straight for 1s
            ]
            self.get_logger().info("Person detected on the left. Initiating right turn sequence.")
        elif center_x > 2 * image_width / 3:
            # Person is on the right side
            self.command_sequence = [
                ([-1, 1], 0.5),  # Turn left for 0.5s
                ([1, 1], 0.5),   # Turn right for 0.5s
                ([0, 1], 1.0)    # Move straight for 1s
            ]
            self.get_logger().info("Person detected on the right. Initiating left turn sequence.")
        else:
            # Person is in the center
            self.command_sequence = [
                ([1, 1], 0.5),   # Turn right for 0.5s
                ([-1, 1], 0.5),  # Turn left for 0.5s
                ([0, 1], 1.0)    # Move straight for 1s
            ]
            self.get_logger().info("Person detected in the center. Initiating turn sequence.")

        self.command_start_time = time.time()
        self.current_command_index = 0


    # Existing methods remain unchanged
    def determine_3x3_navigation(self, grid):
        """Determine the navigation command based on the closest detected object's position in a 3x3 grid."""
        
        # 1. Check each column for 3 open squares from bottom to top.
        for col in range(3):
            if all(not grid[row, col] for row in range(3)):
                if col == 1:
                    return (0, 1)  # Move forward
                elif col == 2:
                    return (1, 1)  # Move forward-left
                elif col == 0:
                    return (-1, 1)  # Move forward-right

        # 2. If no columns have 3 open squares, check each column for 2 open squares (rows 2 and 1).
        for col in range(3):
            if not grid[2, col] and not grid[1, col]:  # Check if both rows in front are open
                if col == 1:
                    return (0, 1)  # Move forward
                elif col == 2:
                    return (1, 1)  # Move forward-left
                elif col == 0:
                    return (-1, 1)  # Move forward-right

        # 3. If no columns have 2 open squares, check each column for 1 open square in the row directly in front (row 2).
        for col in range(3):
            if not grid[2, col]:  # Check if the row in front is open
                if col == 1:
                    return (0, 1)  # Move forward
                elif col == 2:
                    return (1, 1)  # Move forward-left
                elif col == 0:
                    return (-1, 1)  # Move forward-right

        # Default to stop if no clear path is found
        return (0, 0)

    def quantize_disparity_to_grid_row(self, disparity_value):
        band_size = MAX_DISPARITY / GRID_SIZE[0]
        row_index = int(disparity_value / band_size)
        return min(row_index, GRID_SIZE[0] - 1)

    def rank_objects_by_distance(self, disparity_image, detections):
        distance_data = []
        for detection in detections:
            if not hasattr(detection, "xyxy"):
                continue

            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            roi = disparity_image[y1:y2, x1:x2]
            if roi.size > 0:
                # Focus on the central region to avoid edges
                center_roi = roi[roi.shape[0] // 4 : -roi.shape[0] // 4, roi.shape[1] // 4 : -roi.shape[1] // 4]
                # Remove invalid disparities
                valid_disparities = center_roi[center_roi > 0]
                if valid_disparities.size == 0:
                    continue  # Skip if no valid disparities
                median_brightness = np.median(valid_disparities)
                distance_data.append((median_brightness, (x1, y1, x2, y2)))
        
        # Sort by median_brightness in descending order (closest objects first)
        distance_data.sort(key=lambda x: x[0], reverse=True)
        return distance_data

    def populate_astar_grid(self, disparity_image, ranked_objects):
        grid = np.zeros(GRID_SIZE, dtype=int)
        for disparity_value, (x1, y1, x2, y2) in ranked_objects:
            row = self.quantize_disparity_to_grid_row(disparity_value)
            col = int((x1 + x2) / 2 / disparity_image.shape[1] * GRID_SIZE[1])
            col = min(col, GRID_SIZE[1] - 1)
            grid[row][col] = 1
        return grid

    def astar(self, start, goal, grid):
        if grid[goal[0]][goal[1]] == 1:
            goal = self.find_nearest_free_cell(goal, grid)
            if goal is None:
                return None

        open_list = []
        closed_list = set()
        heapq.heappush(open_list, (0, start))
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        came_from = {}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            closed_list.add(current)

            for neighbor in [n for n in self.get_neighbors(current, grid) if grid[n[0]][n[1]] == 0]:
                if neighbor in closed_list:
                    continue
                
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None

    def find_nearest_free_cell(self, goal, grid):
        queue = [(0, goal)]
        visited = set([goal])

        while queue:
            distance, (row, col) = heapq.heappop(queue)
            if grid[row][col] == 0:
                return (row, col)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < GRID_SIZE[0] and 0 <= new_col < GRID_SIZE[1] and (new_row, new_col) not in visited:
                    visited.add((new_row, new_col))
                    heapq.heappush(queue, (distance + 1, (new_row, new_col)))
        return None

    def heuristic(self, node, goal):
        return np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def get_neighbors(self, pos, grid):
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for d in directions:
            new_pos = (pos[0] + d[0], pos[1] + d[1])
            if 0 <= new_pos[0] < grid.shape[0] and 0 <= new_pos[1] < grid.shape[1]:
                neighbors.append(new_pos)
        return neighbors

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
