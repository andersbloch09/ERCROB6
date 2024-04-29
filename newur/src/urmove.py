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
from tf2_geometry_msgs import PoseStamped
from math import pi, dist, fabs, cos
from scannode.msg import aruco
from gripper.srv import gripperservice


from std_msgs.msg import String
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
    def __init__(self, ids, pos, quat, tf_buffer, tf_listener, boardNumber = 0):
        self.ids = ids
        self.pos = pos
        self.quat = quat
        self.boardNumber = boardNumber
        self.button= []
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

        transform = self.tf_buffer.lookup_transform(self.target_frame, "base_link", rospy.Time(), rospy.Duration(1.0))
        self.target_point = self.tf_buffer.transform(self.source_point, self.target_frame)


class MoveGroupPythonInterface(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonInterface, self).__init__()

        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface", anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        self.scene = moveit_commander.PlanningSceneInterface()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
       
        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the UR robot, so we set the group's name to "ur_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.move_group.set_planner_id("RRTstar")
        self.large_list_saved = []
        self.small_list_saved = []
        self.board_buttons = []
        self.button_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.largearuco = aruco()
        self.smallaruco = aruco()
        self.search_state = 1

        #Competition variables 
        self.button_string = "1985"
        self.imu_angle = 45

        self.gohome()
        self.publish_fixed_frame(frame_name="button_frame", target_frame="end_effector_link")
        
        # Receives the data from the large scan aruco
        rospy.Subscriber("/aruco_data", aruco, self.aruco_callback)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )
        
        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")


        self.end_effector_frame = "end_effector_link"


        # board sizes 
        self.imu_board_height = 0.36
        self.imu_board_width = 0.25
        self.secret_board_height = 0.18
        self.secret_board_width = 0.25
        self.secret_box_depth = 0.10 
        self.secret_box_height = 0.07
        self.secret_box_width = 0.15


        # Different gripper states
        self.gripperOpen = "open"
        self.gripperClosed = "close"
        self.gripperImuBox = "imu"
        self.gripperSecretLid = "secretLid"
        
        self.mesh_path_open = "package://ur3e_moveit_config/meshes/ur3e/collision/full_assembly_part_open.stl"
        self.mesh_path_closed = "package://ur3e_moveit_config/meshes/ur3e/collision/full_assembly_closed.stl"

    def aruco_callback(self, msg):
        #rospy.loginfo("Received ArUco data: x_distance={}, y_distance={}, z_distance={}, ids={}, rotation_matrix={}, aruco_size={}".format(msg.x_distance, msg.y_distance, msg.z_distance, msg.ids, msg.rotation_matrix, msg.aruco_type))
        # This function will be called whenever a new largescan or smallscan message is received
        # Process the rself.move_group = move_groupeceived message here
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
        rospy.wait_for_service('gripper_state')
        try:
            state = rospy.ServiceProxy('gripper_state', gripperservice)
            resp = state(new_state)
            return resp
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def go_to_joint_state(self, joints):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## Planning to a Joint Goal
        # We get the joint values from the group and change some of the values:
        joint_goal = move_group.get_current_joint_values()
        
        joint_goal[0] = joints[0]
        joint_goal[1] = joints[1]
        joint_goal[2] = joints[2]
        joint_goal[3] = joints[3]
        joint_goal[4] = joints[4]
        joint_goal[5] = joints[5]

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()


        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)


    def go_to_pose_goal(self, pose, wait=True):
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        
        pos_quaternion = tf_conversions.transformations.quaternion_from_euler(pose[3], pose[4], pose[5])
        #print("Current POSE!!!", self.move_group.get_current_pose())
        #print("QUATONIONS POSE!!", pos_quaternion)
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
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        self.move_group.clear_pose_targets()


        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)


    def plan_cartesian_path(self, frame_name, pos):
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
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )  # jump_threshold
        self.execute_plan(plan)
        # Note: We are just planning, not asking move_group to actually move the robot yet:
        


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


    def execute_plan(self, plan):
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        self.move_group.execute(plan, wait=True)

        ## **Note:** The robot's current joint state must be within some tolerance of the
        ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail


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


    def add_box(self, object_name, size=(0.8, 0.4, 0.01), position=[0.10, 0.10, 0.01], orientation=[0, 0, 0, 0], frame_id="base"):
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


    def attach_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        robot = self.robot
        scene = self.scene
        eef_link = self.eef_link

        ## Attaching Objects to the Robot
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## Next, we will attach the box to the Panda wrist. Manipulating objects requires the
        ## robot be able to touch them without the planning scene reporting the contact as a
        ## collision. By adding link names to the ``touch_links`` array, we are telling the
        ## planning scene to ignore collisions between those links and the box. For the Panda
        ## robot, we set ``grasping_group = 'hand'``. If you are using a different robot,
        ## you should change this value to the name of your end effector group name.
        grasping_group = "manipulator"
        touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_box(eef_link, box_name, touch_links=touch_links)

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_attached=True, box_is_known=False, timeout=timeout,
        )


    def detach_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene
        eef_link = self.eef_link

        ## Detaching Objects from the Robot
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can also detach and remove the object from the planning scene:
        scene.remove_attached_object(eef_link, name=box_name)

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_known=True, box_is_attached=False, timeout=timeout
        )


    def remove_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## Removing Objects from the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can remove the box from the world.
        scene.remove_world_object(box_name)

        ## **Note:** The object must be detached before we can remove it from the world

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_attached=False, box_is_known=False, timeout=timeout
        )


    def get_transform(self, target_frame):
        try:
            transform = self.tf_buffer.lookup_transform("base_link", target_frame, rospy.Time(0), rospy.Duration(1.0))
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to lookup transform: %s", str(e))
            return None
        
    def publish_fixed_frame(self, frame_name, target_frame, pos=[0, 0, 0], quat = [0, 0, 0, 0]):
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
        

        pos_quaternion = tf_conversions.transformations.quaternion_from_euler(pos[3], pos[4], pos[5])
        
        #print("Current POSE!!!", self.move_group.get_current_pose())
        #print("QUATONIONS POSE!!", pos_quaternion)
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
        return all_close(pose_goal, current_pose, 0.01)

    
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
        self.move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()


        # For testing:
        current_joints = self.move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

        
    def match_aruco(self):
        anchor = self.large_list_saved[-1]
        anchor.quat
        euler_anchor = tf_conversions.transformations.euler_from_quaternion(anchor.quat)
        anchor.quat = tf_conversions.transformations.quaternion_from_euler(euler_anchor[0] + np.deg2rad(180), euler_anchor[1], euler_anchor[2])
        
        self.publish_fixed_frame("anchor", "base_link",  anchor.pos, anchor.quat)
        
        pos = [0, 0, -0.05, 0, 0, 0]
        self.plan_cartesian_path("anchor", pos)
        #self.move_relative_to_frame("anchor", pos) 
        

        self.large_list_saved[-1].anchorUsed = 1
                    
        
    def define_board(self): 
        # Search for aruco 
        # when found save id and position as object in object in the lists 
        # search again 
        x = -0.02 
        y = -0.04
        pos = [0, 0, 0, 0, 0, 0]
        while len(self.large_list_saved) < 2:
            pos[1] = pos[1] + y
            self.plan_cartesian_path("button_frame", pos)
            
        while len(self.large_list_saved) < 3: 
            pos[0] = pos[0] + x
            self.plan_cartesian_path("button_frame", pos)

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
                     size=(button_board_height, button_board_width, 0.01)
                    , position = (self.large_list_saved[0].target_point.pose.position.x, 
                     self.large_list_saved[0].target_point.pose.position.y + button_board_displacement, 
                     self.large_list_saved[0].target_point.pose.position.z) 
                    , orientation = (button_board_quat[0], button_board_quat[1], button_board_quat[2], button_board_quat[3])
                    , frame_id="button_frame")
        
        # we know the angle between the boards are 18 degrees
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

        
        # we know the angle between the boards are 18 degrees

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
                    pos = [i.target_point.pose.position.x, i.target_point.pose.position.y, i.target_point.pose.position.z - 0.1, 0, 0, 0]
                    self.plan_cartesian_path("button_frame", pos)
                    while len(current_id_list) < 3:
                        if self.largearuco.ids != 0:
                            current_id_list.append(arucoObject(self.largearuco.ids, self.largearuco.position, self.largearuco.quaternion, self.tf_buffer, self.tf_listener))
                            z = current_id_list[len(current_id_list)-1].target_point.pose.position.z
                            depth_list.append(z)
                    depth_median = np.median(depth_list)
                    print(depth_list)
                    print(depth_median)
                    pos[1] = pos[1] + 0.05
                    self.plan_cartesian_path("button_frame", pos)
                    self.gripper_client(self.gripperClosed)
                    pos[2] = depth_median - 0.027
                    self.plan_cartesian_path("button_frame", pos)
                    pos[2] = depth_median - 0.1
                    self.plan_cartesian_path("button_frame", pos)

    def buttonTask(self):
        self.define_board()
        self.clickButton(self.button_string)
    
    def imuTask(self):
        self.gripper_client(self.gripperOpen)

        x = -0.04
        y = -0.02
        ry = -5
        pos = [0 + x, 0 + y, 0, 0, 0 + np.deg2rad(ry), 0]
        while all(obj.ids != 11 for obj in self.large_list_saved):#find board(can be done from buttonboard or scan)
            self.plan_cartesian_path("button_frame", pos)
            pos[0] = pos[0] + x
            pos[1] = pos[1] + y
            pos[4] = pos[4] + np.deg2rad(ry)
            rospy.sleep(0.2)
        

        for objects in self.large_list_saved:#Laver et anchor point ved id 11
            if objects.ids == 11:
                anchor = objects
        
        euler_anchor = tf_conversions.transformations.euler_from_quaternion(anchor.quat)
        anchor.quat = tf_conversions.transformations.quaternion_from_euler(euler_anchor[0] + np.deg2rad(180), euler_anchor[1], euler_anchor[2])
        
        self.publish_fixed_frame("anchor", "base_link",  anchor.pos, anchor.quat)
        
        pos = [0.0, 0.0, -0.1, 0, 0, 0]
        self.move_relative_to_frame("anchor", pos)
        scanlist = []
        while len(scanlist) < 2:#Scanner flere gange fra forskellige positioner 
            if self.large_list_saved[-1].ids == 11: 
                scanlist.append(self.largearuco)
                anchor = arucoObject(self.largearuco.ids, self.largearuco.position, self.largearuco.quaternion, self.tf_buffer, self.tf_listener)
                euler_anchor = tf_conversions.transformations.euler_from_quaternion(anchor.quat)
                anchor.quat = tf_conversions.transformations.quaternion_from_euler(euler_anchor[0] + np.deg2rad(180), euler_anchor[1], euler_anchor[2])
                self.publish_fixed_frame("anchor", "base_link",  anchor.pos, anchor.quat)
                self.move_relative_to_frame("anchor", pos)
                rospy.sleep(0.2)
            
        print("DONE MATCHING!")

        x = -0.03
        scan_table_pose = [0, 0.05, -0.22, -np.deg2rad(89), 0, 0]
        while all(obj.ids != 10 for obj in self.small_list_saved):#find board(can be done from buttonboard or scan)
            scan_table_pose[0] = scan_table_pose[0] + x
            self.move_relative_to_frame("button_frame", scan_table_pose)
            rospy.sleep(0.2)

        pos = [0.0, 0.0, -0.04, 0, 0, -np.deg2rad(89)]
        scanlist = []
        while len(scanlist) < 3:#Scanner flere gange fra forskellige positioner 
            if self.small_list_saved[-1].ids == 10: 
                scanlist.append(self.smallaruco)
                anchor = arucoObject(self.smallaruco.ids, self.smallaruco.position, self.smallaruco.quaternion, self.tf_buffer, self.tf_listener)
                euler_anchor = tf_conversions.transformations.euler_from_quaternion(anchor.quat)
                anchor.quat = tf_conversions.transformations.quaternion_from_euler(euler_anchor[0] + np.deg2rad(180), euler_anchor[1], euler_anchor[2])
                self.publish_fixed_frame("IMU", "base_link",  anchor.pos, anchor.quat)
                self.plan_cartesian_path("IMU", pos)
                rospy.sleep(1)
        
        pickup_pos = [0, 0, 0.05, 0, 0, -np.deg2rad(89)]
        self.plan_cartesian_path("IMU", pickup_pos)

        self.gripper_client(self.gripperImuBox)

        pos = [0.0, 0.0, -0.1, 0, 0, -np.deg2rad(89)]
        self.plan_cartesian_path("IMU", pos)

        self.gohome(gripper_state=self.gripperImuBox)

        board_place_pos = [0.09, 0.17, -0.05, 0, 0, 0]
        self.move_relative_to_frame("anchor", board_place_pos)

        # Plance the imu on the velcro 
        board_place_pos = [0.09, 0.17, -0.05, 0, 0, np.rad2deg(self.imu_angle)]
        self.plan_cartesian_path("anchor", board_place_pos)

        # The -0.025 is the distance the tcp changes when holding the imu relative to closed
        board_place_pos = [0.09, 0.17, 0.0, 0, 0, np.rad2deg(self.imu_angle)]
        self.plan_cartesian_path("anchor", board_place_pos)

        self.gripper_client(self.gripperOpen)

        # Plance the imu on the velcro 
        board_place_pos = [0.09, 0.17, -0.1, 0, 0, np.rad2deg(self.imu_angle)]
        self.plan_cartesian_path("anchor", board_place_pos)

        self.gohome()

        ##go to imu scan position(this postion can be predetermined and can be multiple)
        #    #find imu match with id number
        #        #match imu
        #        #scan imu agian to get better position
        #        #match best frame
#
        ##pickup imu by matching tcp with best imu frame and displace it a bit(use a cartietian move)
            #lift the imu the first couple of cm using cartetian move

        #place in orientation
            #go to a pose orthangonal to the generated board plane
            #rotate the Imu to correct orientation
            #push the Imu into the board using a cartetian move
            #release Imu, use cartetian move to backup
            #pass

    def secretBoxTask(self):
        pass


    def setupEnv(self):
        # define the table for no collision
        table_quat = tf_conversions.transformations.quaternion_from_euler(0, 0, np.deg2rad(45))
        self.add_box(object_name="table", orientation=[table_quat[0], table_quat[1], table_quat[2], table_quat[3]])
        rospy.sleep(2)
        # Generate and go to random start pose relative to the board        


def main():
    try:
        move_node = MoveGroupPythonInterface()

        move_node.setupEnv()

        move_node.buttonTask()

        move_node.imuTask()

        move_node.secretBoxTask()
        

        print("Program is done")
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        exit()
        return

if __name__ == "__main__":
    main()

## .. _moveit_commander:
##    http://docs.ros.org/noetic/api/moveit_commander/html/namespacemoveit__commander.html
##
## .. _MoveGroupCommander:
##    http://docs.ros.org/noetic/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html
##
## .. _RobotCommander:
##    http://docs.ros.org/noetic/api/moveit_commander/html/classmoveit__commander_1_1robot_1_1RobotCommander.html
##
## .. _PlanningSceneInterface:
##    http://docs.ros.org/noetic/api/moveit_commander/html/classmoveit__commander_1_1planning__scene__interface_1_1PlanningSceneInterface.html
##
## .. _DisplayTrajectory:
##    http://docs.ros.org/noetic/api/moveit_msgs/html/msg/DisplayTrajectory.html
##
## .. _RobotTrajectory:
##    http://docs.ros.org/noetic/api/moveit_msgs/html/msg/RobotTrajectory.html
##
## .. _rospy:
##    http://docs.ros.org/noetic/api/rospy/html/
## imports
## setup
## basic_info
## plan_to_joint_state
## plan_to_pose
## plan_cartesian_path
## display_trajectory
## execute_plan
## add_box
## wait_for_scene_update
## attach_object
## detach_object
## remove_object