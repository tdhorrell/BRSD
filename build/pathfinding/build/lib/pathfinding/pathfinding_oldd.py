import os
import cv2
import heapq
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray  # For publishing path coordinates to motor control
from ultralytics import YOLO
from cv_bridge import CvBridge

# Initialize YOLO model
model = YOLO("yolov5su.pt")

class NavigationNode(Node):
    def __init__(self):
        super().__init__('pathfinding_node')
        self.bridge = CvBridge()
        
        # Subscribe to stereo camera outputs
        self.color_sub = self.create_subscription(Image, '/multisense/left_image', self.color_image_callback, 10)
        self.disparity_sub = self.create_subscription(Image, '/multisense/disparity', self.disparity_image_callback, 10)

        # Publisher for path points
        self.path_pub = self.create_publisher(Int32MultiArray, 'navigation_path', 10)

        timer = 1
        self.timer = self.create_timer(timer, self.process_images)

        # Variables to store images
        self.color_frame = None
        self.disparity_image = None

    def color_image_callback(self, msg):
        self.color_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_images()

    def disparity_image_callback(self, msg):
        self.disparity_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        self.process_images()

    def process_images(self):
        path_msg = Int32MultiArray()
        path_msg.data = [1,2,3]
        self.path_pub.publish(path_msg)

        # Ensure both images are available before processing
        if self.color_frame is None or self.disparity_image is None:
            return

        # 1. Perform Object Detection
        results = model.predict(self.color_frame, save=False)
        
        # 2. Rank Objects by Distance
        ranked_objects = self.rank_objects_by_distance(self.disparity_image, results[0].boxes)

        # 3. Populate A* Grid
        grid = np.zeros((480, 640))  # Grid representing the environment
        grid = self.update_grid_with_objects(grid, ranked_objects)

        # 4. A* Pathfinding
        start = (0, 0)  # Camera location
        goal = (479, 639)  # Goal location in bottom-right corner of the grid
        path = self.astar(start, goal, grid)

        # 5. Publish Path
        if path:
            path_msg.data = [coord for point in path for coord in point]
            self.path_pub.publish(path_msg)
            print(f"Published path: {path}")
            self.get_logger().info("Published path: {path}")

    def rank_objects_by_distance(self, disparity_image, detections):
        """Rank objects by proximity based on average brightness in the disparity image."""
        distance_data = []
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            roi = disparity_image[y1:y2, x1:x2]
            avg_brightness = cv2.mean(roi)[0]
            distance_data.append((avg_brightness, (x1, y1, x2, y2)))
        
        # Sort objects by brightness (closer objects have higher brightness)
        distance_data.sort(key=lambda x: x[0], reverse=True)
        return distance_data

    def update_grid_with_objects(self, grid, ranked_objects):
        """Populate the A* grid with high-cost cells where objects are detected."""
        for _, (x1, y1, x2, y2) in ranked_objects:
            for i in range(x1, x2):
                for j in range(y1, y2):
                    grid[j][i] = 1  # Assign a high cost to cells with detected objects
        return grid

    def astar(self, start, goal, grid):
        """A* pathfinding to navigate around detected objects."""
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
                if neighbor in closed_list:
                    continue
                
                tentative_g_score = g_score[current] + grid[neighbor[0]][neighbor[1]]
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None  # Return None if no path found

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
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
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
