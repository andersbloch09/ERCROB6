#!/usr/bin/env python3
   
import rospy
from gripper.srv import gripperservice

def add_two_ints_client(new_state):
      rospy.wait_for_service('gripper_state')
      try:
          state = rospy.ServiceProxy('gripper_state', gripperservice)
          resp = state(new_state)
          return resp
      except rospy.ServiceException as e:
          print("Service call failed: %s"%e)


if __name__ == "__main__":
    state_update = "secretLid"
    print("%s, %s"%(state_update, add_two_ints_client(state_update)))