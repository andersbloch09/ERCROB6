#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import tf2_ros
import tf_conversions
from math import pi, dist, fabs, cos
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction




from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import pandas as pd
import os




def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class MoveGroupPythonInterface(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonInterface, self).__init__()

        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface", anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        self.robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # Clear all objects from the planning scene
        self.scene.remove_attached_object()
        self.scene.remove_world_object()
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
       

        # Initialize action client for trajectory execution
        self.action_client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
        #self.action_client.wait_for_server()
        rospy.sleep(1)
        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the UR robot, so we set the group's name to "ur_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        self.group_name = "manipulator"
        
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        #self.move_group.set_planner_id("BiEST")
    

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        # Positions (x, y, z)
        self.positions = np.array([
            [0.2, 0.12831087, 0.20145045],
            [0.2, 0.05482361, 0.20215079],
            [0.2, 0.07592431, 0.23591905],
            [0.2, 0.075645, 0.21088705],
            [0.2, -0.18118105, 0.2405908],
            [0.2, -0.11541513, 0.2247965],
            [0.2, -0.18239338, 0.20134693],
            [0.2, 0.06883533, 0.20443079],
            [0.2, -0.11047896, 0.24702723],
            [0.2, -0.06652349, 0.23466539],
            [0.2, 0.10791653, 0.23280931],
            [0.2, 0.05766854, 0.20644384],
            [0.2, -0.14262432, 0.21810992],
            [0.2, -0.09986896, 0.24106908],
            [0.2, 0.10544007, 0.22452494],
            [0.2, 0.16342951, 0.20665921],
            [0.2, -0.14836896, 0.22661779],
            [0.2, 0.02076992, 0.2035994],
            [0.2, -0.00295457, 0.24702693],
            [0.2, -0.14076836, 0.20613295]
])

        # Orientations (roll, pitch ,yaw)
        self.orientations = np.array([
            [0, -0.36569745, -1.42682508],
            [0, 0.10977711, -2.71329267],
            [0, -0.24492438, 1.24881798],
            [0, -0.38851459, 0.18266727],
            [0, -0.83310836, -2.19191059],
            [0, 0.28072423, 2.14849698],
            [0, 0.07775577, -0.19498023],
            [0, -1.48981568, 0.32317866],
            [0, 0.5795462, 2.85472382],
            [0, 0.24614277, 2.87564342],
            [0, -0.57570346, 2.81717969],
            [0, -0.01264057, -0.38687345],
            [0, 0.61634918, -2.87887768],
            [0, 0.38882319, 0.98474804],
            [0, -0.9394612, -1.01721447],
            [0, 1.17582133, -0.82069193],
            [0, -1.02823537, -0.25230007],
            [0, -0.97712549, 1.46067229],
            [0, 0.34920779, -0.86823543],
            [0, 1.05451274, -3.11324579]
        ])

    def go_to_pose_goal(self, pose, wait=True):
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        
        pos_quaternion = tf_conversions.transformations.quaternion_from_euler(pose[3], pose[4], pose[5])

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = pos_quaternion[0]
        pose_goal.orientation.y = pos_quaternion[1]
        pose_goal.orientation.z = pos_quaternion[2]
        pose_goal.orientation.w = pos_quaternion[3]
    
        pose_goal.position.x = pose[0]
        pose_goal.position.y = pose[1]
        pose_goal.position.z = pose[2]
        print(pose[0], pose[1], pose[2])
        self.move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = self.move_group.go(wait)
        tcp_poses = []
        # Check if the move has finished
        start_time = rospy.get_time()
        while not all_close(pose_goal, self.move_group.get_current_pose().pose, 0.01):
            # Save the TCP pose constantly
            current_pose = self.move_group.get_current_pose().pose
            tcp_poses.append([current_pose.position.x, current_pose.position.y, current_pose.position.z])
            check_time = rospy.get_time()
            if check_time - start_time > 20:
                tcp_poses = []
                break
                
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        self.move_group.clear_pose_targets()
        
        print(all_close(pose_goal, self.move_group.get_current_pose().pose, 0.01))

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01), tcp_poses

    def display_trajectory(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher

        ## Displaying a Trajectory
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
        ## group.plan() method does this automatically so this is not that useful
        ## here (it just displays the same trajectory again):
        ##
        ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        ## We populate the trajectory_start with our current robot state to copy over
        ## any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        display_trajectory_publisher.publish(display_trajectory)


    def wait_for_state_update(
        self, box_is_known=False, box_is_attached=False, timeout=4, box_name=""
    ):
        ## wait_for_scene_update
        ##
        ## Ensuring Collision Updates Are Received
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## If the Python node was just created (https://github.com/ros/ros_comm/issues/176),
        ## or dies before actually publishing the scene update message, the message
        ## could get lost and the box will not appear. To ensure that the updates are
        ## made, we wait until we see the changes reflected in the
        ## ``get_attached_objects()`` and ``get_known_object_names()`` lists.
        ## For the purpose of this tutorial, we call this function after adding,
        ## removing, attaching or detaching an object in the planning scene. We then wait
        ## until the updates have been made or ``timeout`` seconds have passed.
        ## To avoid waiting for scene updates like this at all, initialize the
        ## planning scene interface with  ``synchronous = True``.
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = self.scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = box_name in self.scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False


    def add_box(self, object_name, size=(0.8, 0.4, 0.01), position=[0.10, 0.10, 0.0], orientation=[0, 0, 0, 0], frame_id="base"):
        timeout=4
        ## Adding Objects to the Planning Scene
        ## First, we will create a box in the planning scene between the fingers:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = frame_id
        box_pose.pose.orientation.x = orientation[0]
        box_pose.pose.orientation.y = orientation[1]
        box_pose.pose.orientation.z = orientation[2]
        box_pose.pose.orientation.w = orientation[3]
        box_pose.pose.position.x = position[0]
        box_pose.pose.position.y = position[1]
        box_pose.pose.position.z = position[2]
        self.scene.add_box(object_name, box_pose, size=size)

        return self.wait_for_state_update(box_is_known=True, timeout=timeout, box_name=object_name)

    def attach_box(self, timeout=4, box_name="", eef_link="tool0"):
        ## Attaching Objects to the Robot
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## Next, we will attach the box to the Panda wrist. Manipulating objects requires the
        ## robot be able to touch them without the planning scene reporting the contact as a
        ## collision. By adding link names to the ``touch_links`` array, we are telling the
        ## planning scene to ignore collisions between those links and the box. For the UR
        ## robot, we set ``grasping_group = 'hand'``. If you are using a different robot,
        ## you should change this value to the name of your end effector group name.
        grasping_group = "manipulator"
        touch_links = self.robot.get_link_names(group=grasping_group)
        self.scene.attach_box(eef_link, box_name, touch_links=touch_links)

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_attached=True, box_is_known=False, timeout=timeout,
        )

    def gohome(self):
        homeJoints = [0, -np.deg2rad(90), 0, -np.deg2rad(90), 0, 0]    
        
        joint_goal = self.move_group.get_current_joint_values()
    
        joint_goal[0] = homeJoints[0]
        joint_goal[1] = homeJoints[1]
        joint_goal[2] = homeJoints[2]
        joint_goal[3] = homeJoints[3]
        joint_goal[4] = homeJoints[4]
        joint_goal[5] = homeJoints[5]

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        success = self.move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()


        # For testing:
        current_joints = self.move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01), success
    
    def move_to_waypoints(self):
        df = pd.DataFrame(columns=['Time', 'Position X', 'Position Y', 'Position Z', 'Roll', 'Pitch', 'Yaw', 'Distance'])
        self.add_box(object_name="gripper_box", size=(0.05, 0.05, 0.01), position=[0.0, 0.0, 0.01], orientation=[0, 0, 0, 0], frame_id="tool0")
        self.attach_box(box_name="gripper_box")
        # Tiden det tager at lave movement 
        # End-effector position under hele bevægelsen euclisk afstand 
        # Sidste end-effector position 
        # Go home
        table_quat = tf_conversions.transformations.quaternion_from_euler(0, 0, np.deg2rad(45))
        self.add_box(object_name="table", orientation=[table_quat[0], table_quat[1], table_quat[2], table_quat[3]])
        self.gohome()
        
        time_list = []
        pose_list = []
        distance_sum = 0  # Initialize the sum of euclidean distances
        
        # Get the initial pose
        initial_pose = self.move_group.get_current_pose().pose
        initial_position = [initial_pose.position.x, initial_pose.position.y, initial_pose.position.z]
        distance = 0
        # Iterate over each waypoint
        for i in range(len(self.positions)):
            print("Number of waypoints: ", i+1, "out of ", len(self.positions), "waypoints.")
            pose = self.positions[i]
            orientation = self.orientations[i]
            pose_with_orientation = [pose[0], pose[1], pose[2], orientation[0], orientation[1], orientation[2]]
            
            # Measure the start time
            start_time = rospy.get_time()
            
            # Move to the pose goal without waiting
            allClose, tcp_poses = self.go_to_pose_goal(pose_with_orientation, wait=False)
            
            # Measure the end time and calculate elapsed time
            end_time = rospy.get_time()
            elapsed_time = end_time - start_time
            
            # Append elapsed time to the time list
            time_list.append(elapsed_time)
            
            # Get the current pose and convert orientation to Euler angles
            current_pose = self.move_group.get_current_pose().pose
            euler_angles = tf_conversions.transformations.euler_from_quaternion([
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            ])
            euler_angles = [np.rad2deg(angle) for angle in euler_angles]
            
            # Append current pose (position and Euler angles) to the pose list
            pose_list.append([
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z,
                euler_angles[0],
                euler_angles[1],
                euler_angles[2]
            ])
            distance_sum = 0
            for i in range(1, len(tcp_poses)):
                distance = np.linalg.norm(np.array([tcp_poses[i][0], tcp_poses[i][1], tcp_poses[i][2]]) - np.array([tcp_poses[i-1][0], tcp_poses[i-1][1], tcp_poses[i-1][2]]))
            
                distance_sum += distance
            
            rospy.sleep(1)
            # Add the distance to the DataFrame
            df.loc[i, 'Distance'] = distance_sum
        
        # Get the current directory
        current_dir = os.getcwd()
        # Change the current directory to the directory of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        # Iterate over each waypoint
        for i in range(len(time_list)):
            df.iloc[i] = [time_list[i], pose_list[i][0], pose_list[i][1], pose_list[i][2], pose_list[i][3], pose_list[i][4], pose_list[i][5], df['Distance'].iloc[i]]

        # Save the DataFrame to an Excel file
        df.to_excel('chomp.xlsx', index=False)
        # Change the current directory back to the original directory
        os.chdir(current_dir)


def main():
    try:
        move_node = MoveGroupPythonInterface()

        move_node.move_to_waypoints()

        print("Program is done")
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        exit()
        return

if __name__ == "__main__":
    main()
