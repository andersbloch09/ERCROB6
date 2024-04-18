#!/usr/bin/env python3


import rospy
from std_msgs.msg import Int16
from gripper.srv import gripperservice 

class GripperController:
    def __init__(self):
        rospy.init_node('gripper_control', anonymous=True)

        self.state_server= rospy.Service('gripper_state', gripperservice, self.state_receiver)

        self.pub_gripper_state = rospy.Publisher('gripper_ino', Int16, queue_size=3)
        # self.rate = rospy.Rate(10)  # 10hz
        self.gripper_state = Int16()
        self.gripper_state.data = 180
        self.pub_gripper_state.publish(self.gripper_state)
        print("Start control")

    def state_receiver(self, req):
        self.state_server = req.new_state
        self.state = self.state_server
        self.loop()
        rospy.sleep(1)
        gripperservice.current_state = "reached"
        return gripperservice.current_state

    def loop(self):

        if self.state == "open":
            angle = 180
        if self.state== "close":
            angle = 60
        if self.state  == "imu":
            angle = 115
        if self.state == "secretLid":
            angle = 90
        else:
            print("No state selected")
        print(self.state)
        self.gripper_state.data = angle
        self.pub_gripper_state.publish(self.gripper_state)
            
         


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