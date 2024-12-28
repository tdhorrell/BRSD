import os
import cv2
import numpy as np
import heapq
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray  # For publishing movement instructions
from ultralytics import YOLO
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Initialize YOLO model
model = YOLO("yolov5nu.pt")

# Parameters
GRID_SIZE = (7, 7)  # 7x7 grid for A* pathfinding
MAX_DISPARITY = 255  # Maximum grayscale value in disparity image
PROCESSING_INTERVAL = 1.0  # Process images every second

class NavigationNode(Node):
    def __init__(self):
        super().__init__('pathfinding_node')
        self.bridge = CvBridge()
        
        # Set up QoS profile with Best Effort reliability and a queue size of 1
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to stereo camera outputs
        self.color_sub = self.create_subscription(
            Image, '/multisense/aux/image_rect_color', self.color_image_callback, qos_profile)
        self.disparity_sub = self.create_subscription(
            Image, '/multisense/left/disparity', self.disparity_image_callback, qos_profile)

        # Publisher for movement instructions
        self.instructions_pub = self.create_publisher(Int32MultiArray, 'navigation_instructions', 10)

        # Variables to store images
        self.color_frame = None
        self.disparity_image = None
        self.timer = self.create_timer(PROCESSING_INTERVAL, self.process_images)

    def color_image_callback(self, msg):
        # Convert the color image from ROS to OpenCV format
        self.color_frame = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
        self.color_frame = cv2.resize(self.color_frame, (640, 360))
        # Ensure the image is in BGR format
        if len(self.color_frame.shape) == 2:
            self.color_frame = cv2.cvtColor(self.color_frame, cv2.COLOR_GRAY2BGR)
            self.get_logger().info("Converted grayscale image to BGR format.")

    def disparity_image_callback(self, msg):
        # Convert the disparity image from ROS to OpenCV format
        self.disparity_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
        self.disparity_image = cv2.resize(self.disparity_image, (640, 360))

    def process_images(self):
        # Ensure both images are available before processing
        if self.color_frame is None or self.disparity_image is None:
            return

        # 1. Perform Object Detection
        try:
            results = model.predict(self.color_frame, save=False)
            detections = results[0].boxes
            if detections:
                self.log_object_detections(detections)
            else:
                self.get_logger().info("No objects detected.")
        except ValueError as e:
            self.get_logger().error(f"Error during model prediction: {e}")
            return

        # 2. Rank Objects by Distance
        ranked_objects = self.rank_objects_by_distance(self.disparity_image, detections)

        # 3. Populate Grid
        grid = self.populate_astar_grid(self.disparity_image, ranked_objects)

        # Log the populated grid
        self.get_logger().info(f"Populated Grid:\n{grid}")

        # 4. Perform A* Pathfinding
        start = (GRID_SIZE[0] - 1, GRID_SIZE[1] // 2)  # Start at bottom middle
        goal = (0, GRID_SIZE[1] // 2)  # Goal at top middle
        path = self.astar(start, goal, grid)

        # Check the 3 blocks adjacent to the robot (6, 2), (6, 3), and (6, 4)
        if self.are_all_adjacent_blocks_occupied(start, grid):
            # If all 3 adjacent blocks are occupied, stop the robot
            self.publish_instructions([(0, 0)])  # Publish stop instruction
            self.get_logger().info("All adjacent blocks are occupied. Stopping robot.")
            return  # Stop processing further

        # Determine the first command based on block occupancy
        if grid[start[0]][start[1] - 1] == 0:  # Check if `(6, 2)` is free
            first_instruction = (-1, 1)  # Turn left
        elif grid[start[0]][start[1] + 1] == 0:  # Check if `(6, 4)` is free
            first_instruction = (1, 1)  # Turn right
        else:
            first_instruction = (0, 1)  # Move forward through `(6, 3)`

        # If the first step is blocked (after pathfinding), adjust direction
        first_step = path[0] if path else None
        if first_step and grid[first_step[0]][first_step[1]] == 1:
            # Publish first adjustment and recalculate path
            #self.publish_instructions([first_instruction])
            self.get_logger().info(f"Blocked at the start, adjusting direction: {first_instruction}")
            path = self.astar(start, goal, grid)  # Recalculate path after adjustment
            
            if path:  # If a path is found, include the first instruction and publish the rest of the path
                movement_instructions = [first_instruction] + [
                    self.get_movement_instruction(path[i - 1], path[i]) for i in range(1, len(path))
                ]
                self.publish_instructions(movement_instructions)
                self.get_logger().info(f"Published full path: {movement_instructions}")
        else:
            # No obstacle in the first step, publish the entire path
            self.publish_full_path(path)

    def publish_full_path(self, path):
        # Check if the path is valid
        if path is None or len(path) == 0:
            self.get_logger().info("No path found by A* algorithm. Stopping.")
            self.publish_instructions([(0, 0)])  # Stop the robot
            return


        # Generate movement instructions for the entire path
        movement_instructions = []
        for i in range(1, len(path)):
            current = path[i - 1]
            next_step = path[i]
            instruction = self.get_movement_instruction(current, next_step)
            movement_instructions.append(instruction)

        # Publish the full sequence of instructions
        self.publish_instructions(movement_instructions)
        self.get_logger().info(f"Published full path: {movement_instructions}")


    def are_all_adjacent_blocks_occupied(self, start, grid):
        # Check if all of the 3 blocks in front (6, 2), (6, 3), and (6, 4) are occupied
        occupied = True
        for i in range(-1, 2):  # Checking columns (6, 2), (6, 3), and (6, 4)
            block = (start[0], start[1] + i)  # Checking the adjacent blocks in row (6)
            if 0 <= block[1] < GRID_SIZE[1] and grid[block[0]][block[1]] == 0:
                occupied = False  # If any block is free, set occupied to False
                break
        return occupied


    def publish_instructions(self, instructions):
        # Flatten the list of instructions and publish
        instructions_flat = [coord for instr in instructions for coord in instr]
        instructions_msg = Int32MultiArray(data=instructions_flat)
        self.instructions_pub.publish(instructions_msg)
        self.get_logger().info(f"Published instructions: {instructions}")

    def log_object_detections(self, detections):
        class_names = model.names  # {class_id: class_name}
        for detection in detections:
            cls_id = int(detection.cls.item())
            cls_name = class_names.get(cls_id, 'Unknown')
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            self.get_logger().info(f"Detected {cls_name} at [{x1}, {y1}, {x2}, {y2}]")

    def rank_objects_by_distance(self, disparity_image, detections):
        distance_data = []
        for detection in detections:
            if not hasattr(detection, "xyxy"):
                continue

            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            roi = disparity_image[y1:y2, x1:x2]
            if roi.size > 0:
                # Remove invalid disparities
                valid_disparities = roi[roi > 0]
                if valid_disparities.size == 0:
                    continue  # Skip if no valid disparities
                median_disparity = np.median(valid_disparities)
                distance_data.append((median_disparity, (x1, y1, x2, y2)))
        
        # Sort by median_disparity in descending order (closest objects first)
        distance_data.sort(key=lambda x: x[0], reverse=True)
        return distance_data

    def quantize_disparity_to_grid_row(self, disparity_value):
        # Map disparity values to grid rows (4 to 6)
        max_disparity = MAX_DISPARITY
        band_size = max_disparity / 3  # Dividing disparity range into 3 bands
        # Compute row index: 0 (closest), 1, 2 (farthest within reliable range)
        row_index = int((max_disparity - disparity_value) / band_size)
        row_index = min(row_index, 2)  # Limit row_index to 0, 1, 2
        # Map to grid rows 6 (closest) to 4 (farthest)
        grid_row = GRID_SIZE[0] - 1 - row_index  # GRID_SIZE[0] - 1 = 6
        grid_row = max(4, min(grid_row, 6))  # Ensure the row is between 4 and 6
        return grid_row



    def populate_astar_grid(self, disparity_image, ranked_objects):
        grid = np.zeros(GRID_SIZE, dtype=int)

        for disparity_value, (x1, y1, x2, y2) in ranked_objects:
            grid_row = self.quantize_disparity_to_grid_row(disparity_value)

            # Determine the bounding box width
            bbox_width = x2 - x1

            # Determine grid cell width
            grid_cell_width = disparity_image.shape[1] / GRID_SIZE[1]

            # Iterate through all columns in the grid
            for col in range(GRID_SIZE[1]):
                col_start = col * grid_cell_width
                col_end = (col + 1) * grid_cell_width

                # Calculate overlap between bounding box and grid column
                overlap = max(0, min(x2, col_end) - max(x1, col_start))

                # Check if overlap exceeds 20% of the bounding box width
                if overlap >= 0.2 * bbox_width:
                    grid[grid_row][col] = 1  # Mark the cell as occupied

        self.get_logger().info(f"Populated A* Grid:\n{grid}")
        return grid





    def astar(self, start, goal, grid):
        if grid[goal[0]][goal[1]] == 1:
            goal = self.find_nearest_free_cell(goal, grid)
            if goal is None:
                self.get_logger().info("No available goal cell found.")
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

            for neighbor in self.get_neighbors(current, grid):
                if grid[neighbor[0]][neighbor[1]] == 1 or neighbor in closed_list:
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        self.get_logger().info("No path found by A* algorithm.")
        return None

    def get_movement_instruction(self, current, next_step):
        # Determine movement instruction based on next step
        row_diff = current[0] - next_step[0]
        col_diff = next_step[1] - current[1]

        # Map differences to movement instructions
        # (-1, 1): Move left, (0, 1): Move forward, (1, 1): Move right
        if col_diff < 0:
            return (-1, 1)  # Move left
        elif col_diff > 0:
            return (1, 1)   # Move right
        else:
            return (0, 1)   # Move forward

    def find_nearest_free_cell(self, goal, grid):
        # Search for the nearest free cell to the goal
        from collections import deque
        queue = deque()
        queue.append(goal)
        visited = set()
        while queue:
            cell = queue.popleft()
            if cell in visited:
                continue
            visited.add(cell)
            if grid[cell[0]][cell[1]] == 0:
                return cell
            for neighbor in self.get_neighbors(cell, grid):
                if neighbor not in visited:
                    queue.append(neighbor)
        return None

    def heuristic(self, node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def get_neighbors(self, pos, grid):
        neighbors = []
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # Up, Left, Down, Right
        for dr, dc in directions:
            new_row, new_col = pos[0] + dr, pos[1] + dc
            if (0 <= new_row < GRID_SIZE[0] and
                0 <= new_col < GRID_SIZE[1]):
                neighbors.append((new_row, new_col))
        return neighbors

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()