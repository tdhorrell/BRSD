import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class ImageDisplayNode(Node):
    def __init__(self):
        super().__init__('image_display_node')
        self.bridge = CvBridge()

        # Set up QoS profile with Best Effort reliability and a queue size of 10
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to the left camera image and disparity topics with Best Effort reliability
        self.get_logger().info("Setting up subscriptions with Best Effort QoS...")
        self.color_sub = self.create_subscription(
            Image, 
            '/multisense/left/image_rect', 
            self.color_callback, 
            qos_profile
        )
        self.disparity_sub = self.create_subscription(
            Image, 
            '/multisense/left/disparity', 
            self.disparity_callback, 
            qos_profile
        )
        self.get_logger().info("Subscriptions set up.")

    def color_callback(self, msg):
        self.get_logger().info(f"Received color image with encoding: {msg.encoding}, size: {msg.height}x{msg.width}")
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.get_logger().info("Color image converted successfully.")
            
            # Log some pixel data or the mean intensity as an example
            mean_intensity = np.mean(self.color_image)
            self.get_logger().info(f"Color Image Mean Intensity: {mean_intensity}")

        except Exception as e:
            self.get_logger().error(f"Failed to convert color image: {e}")

    def disparity_callback(self, msg):
        self.get_logger().info(f"Received disparity image with encoding: {msg.encoding}, size: {msg.height}x{msg.width}")
        try:
            self.disparity_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.get_logger().info("Disparity image converted successfully.")
            
            # Log some pixel data or the mean intensity as an example
            mean_disparity = np.mean(self.disparity_image)
            self.get_logger().info(f"Disparity Image Mean Intensity: {mean_disparity}")

        except Exception as e:
            self.get_logger().error(f"Failed to convert disparity image: {e}")

def main(args=None):
    print("Starting ImageDisplayNode...")
    rclpy.init(args=args)
    node = ImageDisplayNode()
    try:
        rclpy.spin(node)  # This will block and continuously process messages
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
