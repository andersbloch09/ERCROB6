#!/usr/bin/env python3

import serial
import argparse
import rospy
from gripper.srv import gripperservice 


# This script defines a GripperController class that controls a gripper device.
# It initializes a ROS node, sets up a service for receiving gripper state requests,
# and defines a loop function to send commands to the gripper device based on the requested state.
# The main function creates an instance of the GripperController class and starts the ROS spin loop.

class GripperController:
    def __init__(self):
        rospy.init_node('gripper_control', anonymous=True)
        self.state_server= rospy.Service('gripper_state', gripperservice, self.state_receiver)
        print("Start control")
    
    def state_receiver(self, req):
        # req = request
        self.state_server = req.new_state
        self.state = self.state_server
        self.loop(self.state)
        rospy.sleep(2)
        gripperservice.current_state = "reached"
        return gripperservice.current_state

    def loop(self, state = "open"):
        parser = argparse.ArgumentParser(description='A test program.')
        parser.add_argument("-p", "--usb_port", help="USB port.", default="/dev/ttyACM0")
        args = parser.parse_args()
        arduino = serial.Serial(args.usb_port, 9600)  # Adjust the baud rate accordingly
        # The different gripper states
        if state == "open":
            angleset = 180
        if state == "close":
            angleset = 57
        if state  == "imu":
            angleset = 110
        if state == "secretLid":
            angleset = 90

        angle = int(angleset)
        if 57 <= angle <= 180:
            arduino.write((str(angle)).encode('utf-8'))
    

def main():
    try:
        GripperController()
        rospy.spin()

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()