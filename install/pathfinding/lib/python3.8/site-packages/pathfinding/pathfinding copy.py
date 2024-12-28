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

# Initialize YOLO model
model = YOLO("yolov5nu.pt")

# Parameters
GRID_SIZE = (3, 3)  # 5x5 grid for A* pathfinding
MAX_DISPARITY = 255  # Maximum grayscale value in disparity image

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
        self.color_sub = self.create_subscription(Image, '/multisense/left/image_rect', self.color_image_callback, qos_profile)
        self.disparity_sub = self.create_subscription(Image, '/multisense/left/disparity', self.disparity_image_callback, qos_profile)

        # Publisher for path points
        self.path_pub = self.create_publisher(Int32MultiArray, 'navigation_path', 10)

        self.grid_size = GRID_SIZE

        # Variables to store images
        self.color_frame = None
        self.disparity_image = None
        self.processing_interval = .5
        self.timer = self.create_timer(self.processing_interval, self.process_images)

    def color_image_callback(self, msg):
        # Convert the color image from ROS to OpenCV format
        self.color_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
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
        
        # Calculate and log the mean intensity for debugging
        self.mean_intensity = np.mean(self.disparity_image)
        self.get_logger().info(f"Disparity Image Mean Intensity: {self.mean_intensity}")

    def process_images(self):
        # Ensure both images are available before processing
        if self.color_frame is None or self.disparity_image is None:
            return

        # 1. Perform Object Detection
        try:
            results = model.predict(self.color_frame, save=False)
            if not results[0].boxes:
                self.get_logger().warn("Warning: No objects detected in the image.")
                return
        except ValueError as e:
            self.get_logger().error(f"Error during model prediction: {e}")
            return

        # 2. Rank Objects by Distance
        ranked_objects = self.rank_objects_by_distance(self.disparity_image, results[0].boxes)
        if not ranked_objects:
            self.get_logger().warn("Warning: No valid ranked objects found.")
            return

        # 3. Populate A* Grid
        grid = self.populate_astar_grid_with_disparity(self.disparity_image, ranked_objects)

        # 4. Path Planning Logic for 3x3 Grid
        if self.grid_size == (3, 3):
            # Check if any row in front is fully occupied
            for row in grid[:-1]:  # Iterate over rows in front of the robot
                if all(cell == 1 for cell in row):
                    direction = (0, 0)  # Stop if any row in front is fully occupied
                    path_msg = Int32MultiArray(data=direction)
                    self.path_pub.publish(path_msg)
                    self.get_logger().info("Published 3x3 directional command: (0, 0) - Stop, row fully occupied")
                    return

            # Count occupied cells in each column across all rows in front of the robot
            left_occupied = sum(1 for row in grid[:-1] if row[0] == 1)  # Rows in front, left column
            middle_occupied = sum(1 for row in grid[:-1] if row[1] == 1)  # Rows in front, middle column
            right_occupied = sum(1 for row in grid[:-1] if row[2] == 1)  # Rows in front, right column

            # Determine direction based on occupancy counts
            if left_occupied > 0 and right_occupied == 0:
                direction = (1, 1)  # Move right
            elif right_occupied > 0 and left_occupied == 0:
                direction = (-1, 1)  # Move left
            elif middle_occupied > max(left_occupied, right_occupied):
                # Middle is more occupied; go to the side with fewer obstacles
                direction = (-1, 1) if right_occupied < left_occupied else (1, 1)
            elif middle_occupied <= max(left_occupied, right_occupied):
                # Middle is less occupied; go forward in the middle if it's safer
                direction = (0, 1)
            else:
                # Default to moving right if the occupancy is balanced
                direction = (1, 1)
            if self.mean_intensity <= 30:
                direction = (0,0)
            # Publish the directional command as the path
            path_msg = Int32MultiArray(data=direction)
            self.path_pub.publish(path_msg)
            self.get_logger().info(f"Published 3x3 directional command: {direction}")
            return

        # 5. Full Path Calculation for Other Grid Sizes (like 5x5)
        start = (self.grid_size[0] - 1, self.grid_size[1] // 2)  # Start position at bottom middle
        goal = (0, self.grid_size[1] // 2)  # Goal position at top middle
        path = self.astar(start, goal, grid)

        # Apply transformations to path coordinates and publish
        if path:
            transformed_path = [
                (point[1] - 2, abs(point[0] - (self.grid_size[0] - 1)))
                for point in path
            ]
            path_msg = Int32MultiArray(data=[coord for point in transformed_path for coord in point])
            self.path_pub.publish(path_msg)
            self.get_logger().info(f"Published transformed path: {transformed_path}")


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
                center_roi = roi[roi.shape[0]//4: -roi.shape[0]//4, roi.shape[1]//4: -roi.shape[1]//4]
                avg_brightness = cv2.mean(center_roi)[0]
                distance_data.append((avg_brightness, (x1, y1, x2, y2)))
        
        distance_data.sort(key=lambda x: x[0], reverse=True)
        return distance_data

    def populate_astar_grid_with_disparity(self, disparity_image, ranked_objects):
        grid = np.zeros(GRID_SIZE, dtype=int)
        for avg_brightness, (x1, y1, x2, y2) in ranked_objects:
            row = self.quantize_disparity_to_grid_row(avg_brightness)
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
