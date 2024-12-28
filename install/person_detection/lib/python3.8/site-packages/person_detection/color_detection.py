import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import time
import numpy as np
import cv2
import urllib.request

url = 'http://192.168.0.137/cam-hi.jpg'
cv2.namedWindow("ESP32 Feed", cv2.WINDOW_AUTOSIZE)

class PersonDetectionNode(Node):
    def __init__(self):
        super().__init__('color_detector')

        self.publisher_ = self.create_publisher(Bool, 'person_detector', 10)

        self.startTime = time.time()
        self.detectionCount = 0
        self.frameCount = 0
        self.totalDuration = 30  # Total polling duration in seconds
        self.detectionThreshold = 0.2  # Minimum detection rate threshold

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        # Capture frame from URL
        try:
            imgFromURL = urllib.request.urlopen(url)
            npImg = np.array(bytearray(imgFromURL.read()), dtype=np.uint8)
            frame = cv2.imdecode(npImg, -1)
        except Exception as e:
            self.get_logger().error(f"Failed to retrieve image from URL: {e}")
            return

        # Check if blue is detected in the frame
        if self.isColorDetected(frame):
            self.detectionCount += 1
            self.get_logger().info("Blue Detected")
        else:
            self.get_logger().info("No Blue Detected")
        
        self.frameCount += 1
        cv2.imshow("ESP32 Feed", frame)

        # Check if 30 seconds have passed
        elapsedTime = time.time() - self.startTime
        if elapsedTime >= self.totalDuration:
            detectionRate = self.detectionCount / self.frameCount
            self.get_logger().info(f"Detection rate: {detectionRate:.2f}")

            msg = Bool()
            msg.data = detectionRate >= self.detectionThreshold

            if msg.data:
                self.get_logger().info("Detection condition met. Publishing true.")
            else:
                self.get_logger().info("Detection condition not met. Publishing false.")
            
            self.publisher_.publish(msg)

            # Reset the counters for the next 30-second window
            self.startTime = time.time()
            self.detectionCount = 0
            self.frameCount = 0

        # Exit condition for user interruption
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def isColorDetected(self, frame, pixelThreshold=0.2):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of blue color in HSV
        lower = np.array([75, 75, 23])
        upper = np.array([110, 255, 230])

        hsv_blurred = cv2.GaussianBlur(hsv, (11, 11), 0)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv_blurred, lower, upper)

        # Apply morphological operations to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))

        # Count pixels in the mask
        pixelCount = np.sum(mask > 0)
        total_pixel_count = mask.size
        pixelPercentage = pixelCount / total_pixel_count

        cv2.imshow("Masked Feed", mask)

        return pixelPercentage >= pixelThreshold

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
