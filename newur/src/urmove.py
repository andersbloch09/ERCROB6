#!/usr/bin/env python3

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import transforms3d.quaternions as quaternions
import transforms3d.euler as euler
import tf2_ros
import tf_conversions
import random
from visualization_msgs.msg import Marker
from tf2_geometry_msgs import PoseStamped
from math import pi, dist, fabs, cos
from scannode.msg import aruco
from gripper.srv import gripperservice
from moveit_commander.conversions import pose_to_list


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

class arucoObject():
    """
    Represents an ArUco object.

    Args:
        ids (list): List of ArUco marker IDs.
        pos (list): List of position coordinates [x, y, z].
        quat (list): List of quaternion coordinates [x, y, z, w].
        tf_buffer (tf2_ros.Buffer): Buffer for storing and retrieving transforms.
        tf_listener (tf2_ros.TransformListener): Listener for retrieving transforms.
        boardNumber (int, optional): Board number. Defaults to 0.
    """

    def __init__(self, ids, pos, quat, tf_buffer, tf_listener, boardNumber=0):
        self.ids = ids
        self.pos = pos
        self.quat = quat
        self.boardNumber = boardNumber
        self.button = []
        self.anchorUsed = 0
        self.source_point = PoseStamped()
        self.target_point = PoseStamped()
        self.target_frame = "button_frame"
        self.tf_buffer = tf_buffer
        self.tf_listener = tf_listener

        self.button_locations()

    def button_locations(self):
        self.source_point.header.frame_id = "base_link"
        self.source_point.pose.position.x = self.pos[0]
        self.source_point.pose.position.y = self.pos[1]
        self.source_point.pose.position.z = self.pos[2]

        self.source_point.pose.orientation.x = self.quat[0]
        self.source_point.pose.orientation.y = self.quat[1]
        self.source_point.pose.orientation.z = self.quat[2]
        self.source_point.pose.orientation.w = self.quat[3]

        transform = self.tf_buffer.lookup_transform(
            self.target_frame, "base_link", rospy.Time(), rospy.Duration(1.0)
        )
        self.target_point = self.tf_buffer.transform(self.source_point, self.target_frame)

class MoveGroupPythonInterface(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonInterface, self).__init__()

        # Initialize the moveit_commander and a ROS node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface", anonymous=True)

        # Create a RobotCommander object to access robot's kinematic model and joint states
        self.robot = moveit_commander.RobotCommander()

        # Create a PlanningSceneInterface object to interact with the environment
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # Remove all objects from the planning scene
        self.scene.remove_attached_object()
        self.scene.remove_world_object()
        
        # Setup the TF2 ROS buffer and listener for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
       
        # Setup the MoveGroupCommander for controlling the robot's arm
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        
        # Initialize lists to save detected ArUco markers
        self.large_list_saved = []
        self.small_list_saved = []
        self.board_buttons = []
        self.button_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.largearuco = aruco()
        self.smallaruco = aruco()
        self.search_state = 1
        
        # Variables for competition settings
        self.button_string = "2193"
        self.imu_angle = 60

        # Publisher for visualization markers in Rviz
        self.marker_change = rospy.Publisher("visualization_marker", Marker, queue_size=10)
        
        # Move robot to home position
        self.gohome()
        
        # Delete all markers in Rviz within a range
        for i in range(-15, 15):
            rospy.sleep(0.2)
            marker = Marker()
            marker.id = i
            marker.header.frame_id = "base_link"
            marker.action = Marker.DELETE
            self.marker_change.publish(marker)

        # Publish a fixed frame transformation
        self.publish_fixed_frame(frame_name="button_frame", target_frame="end_effector_link")
        
        # Subscribe to ArUco marker data
        rospy.Subscriber("/aruco_data", aruco, self.aruco_callback)

        # Publisher for displaying planned trajectories in Rviz
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )
        
        # Print the current state of the robot for debugging
        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")

        # Define end effector frame
        self.end_effector_frame = "end_effector_link"

        # Define dimensions of various boards used in the setup
        self.imu_board_height = 0.36
        self.imu_board_width = 0.25
        self.secret_board_height = 0.18
        self.secret_board_width = 0.25
        self.secret_box_depth = 0.10 
        self.secret_box_height = 0.07
        self.secret_box_width = 0.13

        # Define different gripper states
        self.gripperOpen = "open"
        self.gripperClosed = "close"
        self.gripperImuBox = "imu"
        self.gripperSecretLid = "secretLid"
        
        # Paths to mesh files for the gripper in different states
        self.mesh_path_open = "package://ur3e_moveit_config/meshes/ur3e/collision/full_assembly_part_open.stl"
        self.mesh_path_closed = "package://ur3e_moveit_config/meshes/ur3e/collision/full_assembly_closed.stl"


    def aruco_callback(self, msg):
        # This function will be called whenever a new largescan or smallscan message is received
        # The functiom checks if the id is already known and if so it will not create a new object for the given ide. 
        # It will only add it to the global class variable
        large_checker = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14]
        small_checker = [10, 13]
        if msg.ids in large_checker and msg.aruco_type == "Large":
            self.largearuco = msg
            if self.large_list_saved == []:
                if self.largearuco.ids > 0: 
                    self.large_list_saved.append(arucoObject(self.largearuco.ids, self.largearuco.position, self.largearuco.quaternion, self.tf_buffer, self.tf_listener))
            elif all(obj.ids != self.largearuco.ids for obj in self.large_list_saved):
                self.large_list_saved.append(arucoObject(self.largearuco.ids, self.largearuco.position, self.largearuco.quaternion, self.tf_buffer, self.tf_listener))

            
        if msg.ids in small_checker and msg.aruco_type == "Small":
            self.smallaruco = msg
            if self.small_list_saved == []:
                if self.smallaruco.ids > 0:
                  self.small_list_saved.append(arucoObject(self.smallaruco.ids, self.smallaruco.position, self.smallaruco.quaternion, self.tf_buffer, self.tf_listener))
            elif all(obj.ids != self.smallaruco.ids for obj in self.small_list_saved):
                self.small_list_saved.append(arucoObject(self.smallaruco.ids, self.smallaruco.position, self.smallaruco.quaternion, self.tf_buffer, self.tf_listener))

        
    def gripper_client(self, new_state):
        # The function calls the gripper service to open or close the gripper
        rospy.wait_for_service('gripper_state')
        try:
            state = rospy.ServiceProxy('gripper_state', gripperservice)
            resp = state(new_state)
            return resp
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)


    def go_to_joint_state(self, joints):
        ## Planning to a Joint Goal
        # We get the joint values from the group and change some of the values:
        joint_goal = self.move_group.get_current_joint_values()
        
        joint_goal[0] = joints[0]
        joint_goal[1] = joints[1]
        joint_goal[2] = joints[2]
        joint_goal[3] = joints[3]
        joint_goal[4] = joints[4]
        joint_goal[5] = joints[5]

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        success = self.move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()


        # For testing:
        current_joints = self.move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01), success


    def plan_cartesian_path(self, frame_name, pos, avoidCollision = True):
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through. If executing  interactively in a
        ## Python shell, set scale = 1.0.
        ##
        waypoints = []
    
        # Create a PoseStamped message to hold the waypoints
        wpose = PoseStamped()
        wpose.header.frame_id = frame_name
        wpose.pose.position.x = pos[0]
        wpose.pose.position.y = pos[1]
        wpose.pose.position.z = pos[2]

        pos_quaternion = tf_conversions.transformations.quaternion_from_euler(pos[3], pos[4], pos[5])

        wpose.pose.orientation.x = pos_quaternion[0]
        wpose.pose.orientation.y = pos_quaternion[1]
        wpose.pose.orientation.z = pos_quaternion[2]
        wpose.pose.orientation.w = pos_quaternion[3]

        # Transform the pose to the end-effector frame
        transformed_pose_stamped = self.tf_buffer.transform(wpose, "base_link")
        waypoints.append(transformed_pose_stamped.pose)

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space.
        run_fraction = 0
        # This checks if the has hit a fraction of 0, if so it will try to plan again
        while run_fraction == 0: 
            plan_list = []
            fraction_list = []
            for i in range(3):
                (plan, fraction) = self.move_group.compute_cartesian_path(
                    waypoints, 0.01, 0.0, avoid_collisions = avoidCollision  # waypoints to follow  # eef_step
                )  # jump_threshold
                plan_list.append(plan)
                fraction_list.append(fraction)
                print("Fraction  ", fraction)

            max_index = fraction_list.index(max(fraction_list))
            plan = plan_list[max_index]
            run_fraction = max(fraction_list)
            if run_fraction == 0: 
                rospy.sleep(2)

        print("Used fraction   ", run_fraction)
        
        # Note: We are just planning, not asking move_group to actually move the robot yet:
        self.move_group.execute(plan, wait=True)


    def display_trajectory(self, plan):
        ## Displaying a Trajectory
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
        ## group.plan() method does this automatically so this is not that useful
        ## here (it just displays the same trajectory again):
        ##
        ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        ## We populate the trajectory_start with our current robot state to copy over
        ## any AttachedCollisionObjects and add our plan to the trajectory.
        self.display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        self.display_trajectory.trajectory_start = self.robot.get_current_state()
        self.display_trajectory.trajectory.append(plan)
        # Publish
        self.display_trajectory_publisher.publish(self.display_trajectory)


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
        # Frame id will describe in which the frame the object will be made relative to
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


    def attach_box(self, timeout=4, box_name="", eef_link="end_effector_link"):
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


    def detach_box(self, timeout=4, box_name = "", eef_link="end_effector_link"):
        ## Detaching Objects from the Robot
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can also detach and remove the object from the planning scene:
        self.scene.remove_attached_object(eef_link, name=box_name)

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_known=True, box_is_attached=False, timeout=timeout
        )


    def remove_box(self, timeout=4, box_name = ""):
        ## Removing Objects from the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can remove the box from the world.
        self.scene.remove_world_object(box_name)

        ## **Note:** The object must be detached before we can remove it from the world

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_attached=False, box_is_known=False, timeout=timeout
        )


    def get_transform(self, target_frame):
        ## Getting the Transform from the base_link Frame to the Target Frame
        try:
            transform = self.tf_buffer.lookup_transform("base_link", target_frame, rospy.Time(0), rospy.Duration(1.0))
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to lookup transform: %s", str(e))
            return None


    def publish_fixed_frame(self, frame_name, target_frame, pos=[0, 0, 0], quat = [0, 0, 0, 0]):
        # This function creates a fixed frame in the rviz environment
        tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        transform = self.get_transform(target_frame)
        if quat[3] != 0: 
            transform.transform.rotation.x = quat[0]
            transform.transform.rotation.y = quat[1]
            transform.transform.rotation.z = quat[2]
            transform.transform.rotation.w = quat[3]

            
        # Define the transform from the fixed reference frame to the base frame
        transform.child_frame_id = frame_name  # Fixed reference frame
        transform.transform.translation.x += pos[0]
        transform.transform.translation.y += pos[1]
        transform.transform.translation.z += pos[2]
        transform.header.stamp = rospy.Time.now()
        tf_broadcaster.sendTransform(transform)
        rospy.sleep(1)


    def move_relative_to_frame(self, frame_id, pos=[0, 0, 0,
             np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]):
        # This function moves the robot relative to a given frame
        # Usefull when you want to move the robot relative to a aruco

        pos_quaternion = tf_conversions.transformations.quaternion_from_euler(pos[3], pos[4], pos[5])
        
        pose_goal = geometry_msgs.msg.PoseStamped()
        pose_goal.header.frame_id = frame_id
        pose_goal.pose.orientation.x = pos_quaternion[0]
        pose_goal.pose.orientation.y = pos_quaternion[1]
        pose_goal.pose.orientation.z = pos_quaternion[2]
        pose_goal.pose.orientation.w = pos_quaternion[3]
    
        pose_goal.pose.position.x = pos[0]
        pose_goal.pose.position.y = pos[1]
        pose_goal.pose.position.z = pos[2]
        print(pos[0], pos[1], pos[2])
        
        transformed_point = self.tf_buffer.transform(pose_goal, "base_link")
        

        self.move_group.set_pose_target(transformed_point)
        
        print(pose_goal.header.frame_id)
        ## Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
         
        success = self.move_group.go(wait=True)

        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        self.move_group.clear_pose_targets()

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose()
        return all_close(pose_goal, current_pose, 0.01), success


    def gohome(self, gripper_state = "open"):
        self.gripper_client(gripper_state)
        homeJoints = [1.6631979942321777, -1.1095922750285645, -2.049259662628174,
                  3.189222975368164, -0.6959036032306116, -3.1415]    
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

    def frame_exists(self, frame_id):
        # checks if a frame exists
        try: 
            self.tf_buffer.lookup_transform("base_link", frame_id, rospy.Time())
            return True 
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return False
            

    def match_aruco(self, id, aruco_size, frame_name, pos=[0, 0, -0.05, 0, 0, 0]):
        # This function tries to match a given aruco id with the saved aruco ids
        # It will move relativ to a fixed frame to a given position
        scanlist = []
        scan_num = 0
        while len(scanlist) < 3:#Scanner flere gange fra forskellige positioner 
            if aruco_size == "small":
                aruco_list = self.small_list_saved
                aruco = self.smallaruco
            elif aruco_size == "large":
                aruco_list = self.large_list_saved
                aruco = self.largearuco
            
            if aruco_list[-1].ids == id: 
                scanlist.append(scan_num)
                anchor = arucoObject(aruco.ids, aruco.position, aruco.quaternion, self.tf_buffer, self.tf_listener)
                euler_anchor = tf_conversions.transformations.euler_from_quaternion(anchor.quat)
                anchor.quat = tf_conversions.transformations.quaternion_from_euler(euler_anchor[0] + np.deg2rad(180), euler_anchor[1], euler_anchor[2])
                self.publish_fixed_frame(frame_name, "base_link",  anchor.pos, anchor.quat)
                rospy.sleep(1)
                self.move_relative_to_frame(frame_name, pos)
                print("I AM MATCHING")
            rospy.sleep(1)
              

    def search_aruco(self, id, aruco_size, frame_name, direction, pos=[0, 0, 0, 0, 0, 0]):
        # This function searches for a given aruco id 
        # by moving the robot in a given direction which is given as a list
        x = direction[0]
        y = direction[1]
        z = direction[2]
        rx = direction[3]
        ry = direction[4]
        rz = direction[5]

        if aruco_size == "small":
            aruco_list = self.small_list_saved
        elif aruco_size == "large":
            aruco_list = self.large_list_saved

        pos = pos
        while all(obj.ids != id for obj in aruco_list):#find board(can be done from buttonboard or scan)
            pos[0] = pos[0] + x
            pos[1] = pos[1] + y
            pos[2] = pos[2] + z
            pos[3] = pos[3] + np.deg2rad(rx)
            pos[4] = pos[4] + np.deg2rad(ry)
            pos[5] = pos[5] + np.deg2rad(rz)
            self.move_relative_to_frame(frame_name, pos)
            rospy.sleep(0.2)
            
            #Stops the code from going rampant if stopped from some reason, mostly here for convinience 
            if any(abs(x) > 0.4 for x in pos[:3]):
                exit()

 
    def define_board(self): 
        # Search for aruco 
        # when found save id and position as object in object in the lists 
        # search again 
        x = -0.02 
        y = -0.04
        pos = [0, 0, 0, 0, 0, 0]
        # we add exits to break the loop so the program stops to be a background issue
        while len(self.large_list_saved) < 2:
            pos[1] = pos[1] + y
            self.plan_cartesian_path("button_frame", pos)
            if pos[1] < -0.30:
                exit()
            
        while len(self.large_list_saved) < 3: 
            pos[0] = pos[0] + x
            self.plan_cartesian_path("button_frame", pos)
            if pos[0] < -0.20: 
                exit()

        # calculate the distance between the arucos to define the board
        yLength = abs(self.large_list_saved[1].target_point.pose.position.y) - abs(self.large_list_saved[0].target_point.pose.position.y)
        xLength = abs(self.large_list_saved[2].target_point.pose.position.x) - abs(self.large_list_saved[1].target_point.pose.position.x)
        
        # The lengths added are the know outer distance from centers of arucos
        button_board_height = ((yLength * 2) + (0.04 + 0.095))
        button_board_width = ((xLength * 2)  + 0.08)
        button_board_displacement = (0.095-0.04) / 2
        
        # Now the boards are defined in rviz for collision checking
        euler_button_board = tf_conversions.transformations.euler_from_quaternion([self.large_list_saved[0].target_point.pose.orientation.x, self.large_list_saved[0].target_point.pose.orientation.y, self.large_list_saved[0].target_point.pose.orientation.z, self.large_list_saved[0].target_point.pose.orientation.w])
        button_board_quat = tf_conversions.transformations.quaternion_from_euler(euler_button_board[0], euler_button_board[1], euler_button_board[2] - np.deg2rad(90))
        
        
        # Add the button board 
        self.add_box(object_name="button_board_plane", 
                     size=(button_board_height, button_board_width, 0.04)
                    , position = (self.large_list_saved[0].target_point.pose.position.x, 
                     self.large_list_saved[0].target_point.pose.position.y + button_board_displacement, 
                     self.large_list_saved[0].target_point.pose.position.z) 
                    , orientation = (button_board_quat[0], button_board_quat[1], button_board_quat[2], button_board_quat[3])
                    , frame_id="button_frame")
        
        # The angle between the boards are 18 degrees and the lengths of the boards are known
        # The boards are calculated based on trigonometry
        imu_board_x = (button_board_width/2) + ((self.imu_board_width/2) * np.cos(np.deg2rad(18)))
        imu_board_y = (yLength + 0.04) - self.imu_board_height/2
        imu_board_z = (self.imu_board_width/2) * np.sin(np.deg2rad(18))
        imu_board_quat = tf_conversions.transformations.quaternion_from_euler(euler_button_board[0] - np.deg2rad(18), euler_button_board[1], euler_button_board[2] + np.deg2rad(90))
        
        # Add the imu_board 
        self.add_box(object_name="imu_board_plane", 
                     size=(self.imu_board_height, self.imu_board_width, 0.01)
                    , position = (self.large_list_saved[0].target_point.pose.position.x - imu_board_x, 
                     self.large_list_saved[0].target_point.pose.position.y - imu_board_y, 
                     self.large_list_saved[0].target_point.pose.position.z - imu_board_z) 
                    , orientation = (imu_board_quat[0], imu_board_quat[1], imu_board_quat[2], imu_board_quat[3])
                    , frame_id="button_frame")

        # we know the angle between the boards are 18 degrees
        secret_board_x = (button_board_width/2) + ((self.secret_board_width/2) * np.cos(np.deg2rad(18)))
        secret_board_y = (yLength + 0.095) - self.secret_board_height/2
        secret_board_z = (self.secret_board_width/2) * np.sin(np.deg2rad(18))
        secret_board_quat = tf_conversions.transformations.quaternion_from_euler(euler_button_board[0] + np.deg2rad(18), euler_button_board[1], euler_button_board[2] + np.deg2rad(90))
        
        # Add the secret_board 
        self.add_box(object_name="secret_board_plane", 
                     size=(self.secret_board_height, self.secret_board_width, 0.01)
                    , position = (self.large_list_saved[0].target_point.pose.position.x + secret_board_x, 
                     self.large_list_saved[0].target_point.pose.position.y + secret_board_y, 
                     self.large_list_saved[0].target_point.pose.position.z - secret_board_z) 
                    , orientation = (secret_board_quat[0], secret_board_quat[1], secret_board_quat[2], secret_board_quat[3])
                    , frame_id="button_frame")

        secret_box_x = (button_board_width/2) + ((np.sqrt(((self.secret_box_depth/2+0.01)**2)+((self.secret_board_width/2)**2))) * np.cos(((np.sin((self.secret_box_depth/2+0.01)/(self.secret_board_width/2)) + np.deg2rad(18)))))
        secret_box_y = (yLength + 0.095) - self.secret_board_height/2 - ((self.secret_board_height - self.secret_box_height)/2)
        secret_box_z = (np.sin((np.sin((self.secret_box_depth/2+0.01)/(self.secret_board_width/2)) + np.deg2rad(18)))) * (np.sqrt(((self.secret_box_depth/2+0.01)**2)+((self.secret_board_width/2)**2)))
        secret_box_quat = tf_conversions.transformations.quaternion_from_euler(euler_button_board[0] + np.deg2rad(18), euler_button_board[1], euler_button_board[2] + np.deg2rad(90))
        

        # Add the secret box 
        self.add_box(object_name="secret_box_plane", 
                     size=(self.secret_box_height, self.secret_box_width, self.secret_box_depth)
                    , position = (self.large_list_saved[0].target_point.pose.position.x + secret_box_x, 
                     self.large_list_saved[0].target_point.pose.position.y + secret_box_y, 
                     self.large_list_saved[0].target_point.pose.position.z - secret_box_z) 
                    , orientation = (secret_box_quat[0], secret_box_quat[1], secret_box_quat[2], secret_box_quat[3])
                    , frame_id="button_frame")
    

        boardNumber = 0
        # This iterates through the board buttons and creates a button object for each button
        for i in range(3):
            for j in range(3):
                x = j * xLength
                y = i * yLength
                print(x, y)

                boardNumber += 1

                pos = [-xLength + x, -yLength + y, 0, 0, 0, 0]
                self.plan_cartesian_path("button_frame", pos)
                if self.board_buttons == []:
                    if self.largearuco.ids > 0: 
                            self.board_buttons.append(arucoObject(self.largearuco.ids, self.largearuco.position, self.largearuco.quaternion, self.tf_buffer, self.tf_listener, boardNumber=boardNumber))
                elif all(obj.ids != self.largearuco.ids for obj in self.board_buttons):
                    self.board_buttons.append(arucoObject(self.largearuco.ids, self.largearuco.position, self.largearuco.quaternion, self.tf_buffer, self.tf_listener, boardNumber=boardNumber))
        print(len(self.board_buttons))


    def clickButton(self, bstring):
        self.gohome()
        # The functions clicks the specific button based on its object position
        # If the number is found in the given string
        for j in range(len(bstring)):
            current_id_list = []
            depth_list = []
            currentTarget = bstring[j]
            currentTarget = int(currentTarget)
            self.gripper_client(self.gripperOpen)
            for i in self.board_buttons:
                # If the first target matches the ArUco ID in the array
                # then go to the location of the ArUco ID
                # with 5 cm distance on the Z-axis
                if currentTarget == i.ids:
                    pos = [i.target_point.pose.position.x, i.target_point.pose.position.y, i.target_point.pose.position.z - 0.04, 0, 0, 0]
                    
                    self.plan_cartesian_path("button_frame", pos)
                
                    while len(current_id_list) < 5:
                        if self.largearuco.ids != 0:
                            current_id_list.append(arucoObject(self.largearuco.ids, self.largearuco.position, self.largearuco.quaternion, self.tf_buffer, self.tf_listener))
                            z = current_id_list[len(current_id_list)-1].target_point.pose.position.z
                            depth_list.append(z)
                        rospy.sleep(0.5)
                    depth_median = np.median(depth_list)
                    print(depth_list)
                    print(depth_median)
                    # The distance to the button before clicking
                    pos[1] = pos[1] + 0.05
                    self.plan_cartesian_path("button_frame", pos)
                    self.gripper_client(self.gripperClosed)
                    # Here the distance to click the  button is changed a bit depending on where on the board 
                    # because the board was not as stiff as it should be 
                    # in the competition this should be changed back as the boards are metal
                    if i.boardNumber > 5:
                        pos[2] = depth_median - 0.027
                    else: 
                        pos[2] = depth_median - 0.025
                    self.plan_cartesian_path("button_frame", pos)
                    pos[2] = depth_median - 0.06
                    self.plan_cartesian_path("button_frame", pos)
                    self.gohome()

    def buttonTask(self):
        # The button task is seperated into two parts
        self.define_board()
        self.clickButton(self.button_string)


    def imuTask(self):
        self.gripper_client(self.gripperOpen)

        # Searching for the IMU board.
        self.search_aruco(11, "large", "button_frame", direction=[-0.04, -0.02, 0, 0, -5, 0])
        
        for objects in self.large_list_saved:#Creates and anchor with id 11
            if objects.ids == 11:
                anchor = objects
        # if  found publish frame
        euler_anchor = tf_conversions.transformations.euler_from_quaternion(anchor.quat)
        anchor.quat = tf_conversions.transformations.quaternion_from_euler(euler_anchor[0] + np.deg2rad(180), euler_anchor[1], euler_anchor[2])
        
        self.publish_fixed_frame("IMU_board_frame", "base_link",  anchor.pos, anchor.quat)
        

        # Matching with the IMU_board and creating the frame IMU after each match
        self.match_aruco(11, "large", "IMU_board_frame", pos=[0, 0, -0.1, 0, 0, 0])

        # Searching for the IMU
        scan_table_pose = [0, 0.05, -0.22, -np.deg2rad(89), 0, 0]
        self.search_aruco(10, "small", "button_frame", pos=scan_table_pose, direction=[-0.03, 0, 0, 0, 0, 0])

        # Matching with the IMU
        self.match_aruco(10, "small", "IMU", pos=[0.0, 0.0, -0.05, 0, 0, -np.deg2rad(89)])

        #Move down to and grab the IMU
        pickup_pos = [0, 0, 0.04, 0, 0, -np.deg2rad(89)]
        self.plan_cartesian_path("IMU", pickup_pos)
        self.gripper_client(self.gripperImuBox)


        # we add the imu as an object to avoid collisions and for visuals
        box_orientation = tf_conversions.transformations.quaternion_from_euler(0 , 0, np.deg2rad(90))
        self.add_box("imu_box", size=(0.1, 0.05, 0.05), position=[0, 0, -0.01], orientation=box_orientation, frame_id="end_effector_link")
        self.attach_box(box_name="imu_box")

        # Pick up the IMU
        pos = [0.0, 0.0, -0.1, 0, 0, -np.deg2rad(89)]
        self.plan_cartesian_path("IMU", pos, avoidCollision=False)

        # Asures that the robot moves to home before moving on
        success = False 
        while success == False: 
            allClose, success = self.gohome(gripper_state=self.gripperImuBox)

        # Go in front of the velcro board
        board_place_pos = [0.09, 0.17, -0.08, 0, 0, 0]
        self.move_relative_to_frame("IMU_board_frame", board_place_pos)

        # Place the imu on the velcro 
        board_place_pos = [0.09, 0.17, -0.08, 0, 0, np.deg2rad(self.imu_angle)]
        self.plan_cartesian_path("IMU_board_frame", board_place_pos)

        # The -0.01 is the distance the tcp changes when holding the imu relative to closed
        board_place_pos = [0.09, 0.17, -0.01, 0, 0, np.deg2rad(self.imu_angle)]
        self.plan_cartesian_path("IMU_board_frame", board_place_pos, avoidCollision = False)
        
        # Release the IMU both physicly and sim
        self.gripper_client(self.gripperOpen)
        self.detach_box(box_name="imu_box")
        self.add_box("imu_box", size=(0.1, 0.05, 0.05), position=[0, 0, -0.01], orientation=box_orientation, frame_id="end_effector_link")
        

        # Move back and go home
        board_place_pos = [0.09, 0.17, -0.1, 0, 0, np.deg2rad(self.imu_angle)]
        self.plan_cartesian_path("IMU_board_frame", board_place_pos)

        self.gohome()


    def secretBoxTask(self):

        # Local variables, for easy changes and tweaks
        angle_to_box = 15
        box_lid_angle = 35
        table_lid_angle = 55
        secret_aruco_id = 0

        for objects in self.large_list_saved:
            if objects.ids == 12:
                anchor1 = objects
                # The frame for the table is made if id 12 is found
                euler_anchor = tf_conversions.transformations.euler_from_quaternion(anchor1.quat)
                anchor1.quat = tf_conversions.transformations.quaternion_from_euler(euler_anchor[0] + np.deg2rad(180), euler_anchor[1], euler_anchor[2])
                self.publish_fixed_frame("secret_box_frame", "base_link",  anchor1.pos, anchor1.quat)


        # Start by seaching for the Secret box
        scan_box_pose = [0, 0, 0, -np.deg2rad(45), np.deg2rad(angle_to_box), 0]
        self.search_aruco(12, "large", "button_frame", pos=scan_box_pose, direction=[0.04, 0, -0.02, 0, 0, 0])

        pos = [0.0, 0.0, -0.04, 0, 0, 0]
        scanlist = []
        # match the id 3 times as a good frame is preffered for the secret box
        while len(scanlist) < 3: 
            if self.largearuco.ids == 12: 
                scanlist.append(self.largearuco)
                anchor = arucoObject(self.largearuco.ids, self.largearuco.position, self.largearuco.quaternion, self.tf_buffer, self.tf_listener)
                euler_anchor = tf_conversions.transformations.euler_from_quaternion(anchor.quat)
                anchor.quat = tf_conversions.transformations.quaternion_from_euler(euler_anchor[0] + np.deg2rad(180), euler_anchor[1], euler_anchor[2])
                self.publish_fixed_frame("secret_box_frame", "base_link",  anchor.pos, anchor.quat)
                self.move_relative_to_frame("secret_box_frame", pos)
                rospy.sleep(1)

            # If the object is not detected above this will catch it and make the frame
            elif any(obj.ids == 12 for obj in self.large_list_saved) and self.largearuco.ids != 12:
                if self.frame_exists("secret_box_frame"):
                     self.move_relative_to_frame("secret_box_frame", pos)
                     print("FRAME IS TRUE")
                else: 
                    for objects in self.large_list_saved:
                        if objects.ids == 12:
                            anchor2 = objects
                            euler_anchor = tf_conversions.transformations.euler_from_quaternion(anchor2.quat)
                            anchor2.quat = tf_conversions.transformations.quaternion_from_euler(euler_anchor[0] + np.deg2rad(180), euler_anchor[1], euler_anchor[2])
                            self.publish_fixed_frame("secret_box_frame", "base_link",  anchor2.pos, anchor2.quat)
                            self.move_relative_to_frame("secret_box_frame", pos)
            


        # Now we find the aruco marker on the table
        scan_table_pose = [0, 0.0, -0.04, -np.deg2rad(80), 0, 0]
        self.search_aruco(14, "large", "secret_box_frame", pos=scan_table_pose, direction=[0, 0, -0.01, 0, 0, 0]) 
 
        # Make a frame when found
        for objects in self.large_list_saved:
            if objects.ids == 14:
                anchor = objects
                euler_anchor = tf_conversions.transformations.euler_from_quaternion(anchor.quat)
                anchor.quat = tf_conversions.transformations.quaternion_from_euler(euler_anchor[0] + np.deg2rad(180), euler_anchor[1], euler_anchor[2])
                # The frame for the table is made
                self.publish_fixed_frame("secret_table_frame", "base_link",  anchor.pos, anchor.quat)
        
        # Move to the table relative to button_frame as the position is very diffecult to reach with relative to table_frame
        pos = [anchor.target_point.pose.position.x, anchor.target_point.pose.position.y - 0.1, anchor.target_point.pose.position.z - 0.1, -np.deg2rad(box_lid_angle), 0, 0]
        success = False
        while success == False: 
            if self.largearuco.ids == 14: 
                anchor = arucoObject(self.largearuco.ids, self.largearuco.position, self.largearuco.quaternion, self.tf_buffer, self.tf_listener)
                euler_anchor = tf_conversions.transformations.euler_from_quaternion(anchor.quat)
                anchor.quat = tf_conversions.transformations.quaternion_from_euler(euler_anchor[0] + np.deg2rad(180), euler_anchor[1], euler_anchor[2])
                pos = [anchor.target_point.pose.position.x, anchor.target_point.pose.position.y - 0.1, anchor.target_point.pose.position.z - 0.1, -np.deg2rad(box_lid_angle), 0, 0]
                self.publish_fixed_frame("secret_table_frame", "base_link",  anchor.pos, anchor.quat)
                rospy.sleep(0.5)
            allClose, success = self.move_relative_to_frame("button_frame", pos)
            rospy.sleep(1)
            print(success)

        # We match with the secret table aruco
        match_table_pos = [0, 0.1, -0.2, np.deg2rad(40), 0, 0]
        self.match_aruco(14, "large", "secret_table_frame", pos=match_table_pos)

        # Tempoaroly finding id 12's position and orientation to move with respect to button_frame, as the planner had trouble with moving in relation to secret_box_frame
        for objects in self.large_list_saved:
                if objects.ids == 12:
                    anchor = objects

        # Now the lid will be grabed, this movement is required to succed
        pos = [anchor.target_point.pose.position.x + 0.015, anchor.target_point.pose.position.y - 0.15, anchor.target_point.pose.position.z + 0.05, -np.deg2rad(box_lid_angle), np.deg2rad(angle_to_box), 0]
        success = False
        while success == False: 
            allClose, success = self.move_relative_to_frame("button_frame", pos)
            print(success)

        # cartesian moves to grab the lid
        grab_lid_pose = [anchor.target_point.pose.position.x + 0.015, anchor.target_point.pose.position.y - 0.075, anchor.target_point.pose.position.z + 0.05, -np.deg2rad(box_lid_angle), np.deg2rad(angle_to_box), 0]
        self.plan_cartesian_path("button_frame", grab_lid_pose)

        self.gripper_client(self.gripperSecretLid)

        # We add the lid as an object to avoid collisions
        lid_orientation = tf_conversions.transformations.quaternion_from_euler(-np.deg2rad(table_lid_angle) , 0, 0)
        self.add_box("secret_lid", size=(0.15, 0.1, 0.005), position=[0, 0, 0.0], orientation=lid_orientation, frame_id="end_effector_link")
        self.attach_box(box_name="secret_lid")

        grab_lid_pose = [anchor.target_point.pose.position.x + 0.015, anchor.target_point.pose.position.y - 0.15, anchor.target_point.pose.position.z + 0.05, -np.deg2rad(box_lid_angle), np.deg2rad(angle_to_box), 0]
        self.plan_cartesian_path("button_frame", grab_lid_pose)

        # Save the joints for later use
        pick_up_joints = self.move_group.get_current_joint_values()

        # The lid is placed on the table
        place_lid_pos_above = [0.0, 0.13, -0.25, np.deg2rad(box_lid_angle), 0, np.deg2rad(89)]
        success = False
        while success == False:
            allClose, success = self.move_relative_to_frame("secret_table_frame", place_lid_pos_above)
            print(success)

        place_lid_pos_above_joints = self.move_group.get_current_joint_values()

        # Place the lid on the table
        success = False 
        place_lid_pos = [0, 0.13, -0.06, np.deg2rad(table_lid_angle), 0, np.deg2rad(89)]
        while success == False: 
            allClose, success = self.move_relative_to_frame("secret_table_frame", place_lid_pos)

        # Place it two cm above the table
        place_lid_pos[2] = place_lid_pos[2] + 0.02
        success = False
        while success == False: 
            allClose, success = self.move_relative_to_frame("secret_table_frame", place_lid_pos)
        
        # release and detach the lid
        self.gripper_client(self.gripperOpen)
        self.detach_box(box_name="secret_lid")
        
        # force back joints movements
        success = False
        while success == False:
            allClose, success = self.go_to_joint_state(place_lid_pos_above_joints)

        success = False
        while success == False:
            allClose, success = self.go_to_joint_state(pick_up_joints)

        # We look for the secret id 
        grab_lid_pose = [anchor.target_point.pose.position.x - 0.03, anchor.target_point.pose.position.y - 0.08, anchor.target_point.pose.position.z, -np.deg2rad(65), np.deg2rad(angle_to_box + 5), 0]
        
        self.plan_cartesian_path("button_frame", grab_lid_pose)  

        # Adds the secret id to a variable
        secret_aruco_id = self.largearuco.ids

        # Go back to pick up lid 
        success = False
        while success == False:
            allClose, success = self.go_to_joint_state(place_lid_pos_above_joints)

        success = False
        while success == False:
            allClose, success = self.move_relative_to_frame("secret_table_frame", place_lid_pos)
        
        self.gripper_client(self.gripperSecretLid)
        self.attach_box(box_name="secret_lid")

        success = False
        while success == False:
            allClose, success = self.move_relative_to_frame("secret_table_frame", place_lid_pos_above)

        self.go_to_joint_state(pick_up_joints)
        
        grab_lid_pose = [anchor.target_point.pose.position.x + 0.015, anchor.target_point.pose.position.y - 0.15, anchor.target_point.pose.position.z + 0.04, -np.deg2rad(box_lid_angle), np.deg2rad(angle_to_box), 0]
        self.plan_cartesian_path("button_frame", grab_lid_pose)

        grab_lid_pose = [anchor.target_point.pose.position.x + 0.015, anchor.target_point.pose.position.y - 0.06, anchor.target_point.pose.position.z + 0.04, -np.deg2rad(box_lid_angle), np.deg2rad(angle_to_box), 0]
        self.plan_cartesian_path("button_frame", grab_lid_pose)

        self.gripper_client(self.gripperOpen)
        self.detach_box(box_name="secret_lid")

        grab_lid_pose = [anchor.target_point.pose.position.x + 0.015, anchor.target_point.pose.position.y - 0.15, anchor.target_point.pose.position.z + 0.04, -np.deg2rad(box_lid_angle), np.deg2rad(angle_to_box), 0]
        self.plan_cartesian_path("button_frame", grab_lid_pose)

        self.gohome()
        
        # click the secret id
        self.clickButton(str(secret_aruco_id))

        self.gohome()

    def setupEnv(self):
        # define the table for no collision
        table_quat = tf_conversions.transformations.quaternion_from_euler(0, 0, np.deg2rad(45))
        self.add_box(object_name="table", orientation=[table_quat[0], table_quat[1], table_quat[2], table_quat[3]])
        rospy.sleep(1)
        # Right table 
        self.add_box(object_name="table_right", size=(0.7, 0.21, 0.01), position=[0.42, 0, 0.02], orientation=[table_quat[0], table_quat[1], table_quat[2], table_quat[3]])
        rospy.sleep(1)
        # Left side table
        self.add_box(object_name="table_left", size=(0.7, 0.21, 0.01), position=[0, 0.42, -0.05], orientation=[table_quat[0], table_quat[1], table_quat[2], table_quat[3]])
        rospy.sleep(1)       
        

def main():
    try:
        # Initialize the node
        move_node = MoveGroupPythonInterface()

        # Setup the environment and the tasks in the following order
        move_node.setupEnv()

        move_node.buttonTask()

        move_node.imuTask()

        move_node.secretBoxTask()
        

        print("Program is done")
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        exit()

if __name__ == "__main__":
    main()
