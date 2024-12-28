import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Int32MultiArray, Int32

class NavigationNode(Node):
    def __init__(self):
        super().__init__('pathfinding_node')
        self.instructions_pub = self.create_publisher(Int32MultiArray, 'navigation_instructions', 10)
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10
        )
        
        # Define a sequence of commands for testing
        self.command_sequence = [
            (0, 1),  # Move forward
            (-1, 1),  # Turn right
            (0, 1),  # Move forward
            (1, 1), # Turn left
            (0, 1),   # Move forward
            (-1, 1),
            (1, 1),
            (1, 0),
            (0, 0)
        ]
        self.command_index = 0  # Start at the first command
        self.command_interval = 1.5  # Interval between commands in seconds
        self.joystick_command = 0  # Initialize joystick command state
        
        # Subscription to joystick command
        self.point_subscriber = self.create_subscription(
            Int32,
            'joystick_command',
            self.update_joystick,
            qos_profile
        )
        
        # Timer to execute commands at regular intervals
        self.timer = self.create_timer(self.command_interval, self.publish_next_command)
    
    def update_joystick(self, msg):
        # Update the joystick command state
        self.joystick_command = msg.data
        self.get_logger().info(f"Joystick command updated to: {self.joystick_command}")

    def publish_next_command(self):
        # Check if joystick command is 1 before publishing
        if self.joystick_command == 1:
            command = self.command_sequence[self.command_index]
            instructions_msg = Int32MultiArray(data=list(command))
            self.instructions_pub.publish(instructions_msg)
            self.get_logger().info(f"Published command: {command}")
            
            # Move to the next command, loop back to the start if at the end
            self.command_index += 1
            if self.command_index >= len(self.command_sequence):
                self.command_index = 0  # Reset to the first command
        elif self.joystick_command == 0:
            self.command_index = 0
        else:
            self.get_logger().info("Joystick command is not 1; skipping command publishing.")

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
