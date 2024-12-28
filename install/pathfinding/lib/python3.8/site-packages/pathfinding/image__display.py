import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageDisplayNode(Node):
    def __init__(self):
        super().__init__('image_display_node')
        self.bridge = CvBridge()

        # Subscribe to the image topics
        self.color_sub = self.create_subscription(Image, '/multisense/right/image_rect', self.color_callback, 10)
        self.disparity_sub = self.create_subscription(Image, '/multisense/right/disparity_image', self.disparity_callback, 10)

    def color_callback(self, msg):
        # Convert the color image from ROS format to OpenCV format
        color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # Display the color image
        cv2.imshow("Color Image", color_image)
        cv2.waitKey(1)  # Add a short delay to allow the display to update

    def disparity_callback(self, msg):
        # Convert the disparity image from ROS format to OpenCV format
        disparity_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        # Display the disparity image
        cv2.imshow("Disparity Image", disparity_image)
        cv2.waitKey(1)  # Add a short delay to allow the display to update

def main(args=None):
    rclpy.init(args=args)
    node = ImageDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Close OpenCV windows when the node is shut down
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
