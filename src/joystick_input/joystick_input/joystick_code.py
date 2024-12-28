import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import Jetson.GPIO as GPIO

class JoystickNode(Node):
    def __init__(self):
        super().__init__('joystick_node')

        # GPIO Pins
        self.vrx = 23
        self.vry = 7

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.vrx, GPIO.IN)
        GPIO.setup(self.vry, GPIO.IN)

        self.publisher_ = self.create_publisher(Int32, 'joystick_command', 10)

        timer = .1
        self.timer = self.create_timer(timer, self.timer_callback)

    def read_joystick(self):
        vrxVal = GPIO.input(self.vrx)
        vryVal = GPIO.input(self.vry)

        return vrxVal, vryVal
    
    def determine_direction(self, vrx, vry):
        if((vrx == 0) and (vry == 1)): #Forward
            return 1
        elif((vrx == 1) and (vry == 0)): #Backwards (JOYSTICK TO THE RIGHT MEANS BACKWARDS)
            return 2
        else: #Neutral
            return 0

    def timer_callback(self):
        vrx, vry = self.read_joystick()
        direction = self.determine_direction(vrx, vry)
        self.publisher_.publish(Int32(data=direction))
        self.get_logger().info(f"Publishing joystick data: {direction}")

def main(args=None):
    rclpy.init(args=args)
    node = JoystickNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    GPIO.cleanup()

if __name__ == '__main__':
    main()