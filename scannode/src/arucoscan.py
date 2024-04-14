#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
from scannode.msg import aruco
import tf2_ros
from geometry_msgs.msg import Point
from tf2_geometry_msgs import PoseStamped
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation

class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('aruco_detector_node', anonymous=True)

        # ArUco dictionary and parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                            cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters =  cv2.aruco.DetectorParameters()

        # Initialize the ArUco ID and translation vector publisher
        self.aruco_data_pub = rospy.Publisher('/aruco_data', aruco, queue_size=10)
        self.rate = rospy.Rate(500)
        
        #This is the sub for the frames to find save the point in relation to the global coordinate system
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Pub for rviz visualization
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
 
        # Camera matrix and distortion coefficients
        self.mtx = np.array([[1.44003309e+03, 0.00000000e+00, 6.86010223e+02],
                             [0.00000000e+00, 1.43870157e+03, 4.31888568e+02],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        self.dist = np.array([[7.29890166e-02, -7.14335748e-01, 1.44494297e-02,
                    9.08325864e-04, 7.15318943e+00]])
        
        # Open a connection to the camera
        # (adjust the index as needed, typically 0 or 1)
        self.cap = cv2.VideoCapture(0)
    
        ## Check if the camera opened successfully
        #if not self.cap.isOpened():
        #    print("Error: Unable to open camera")
        #    exit()

        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # Set the video capture object to use the maximum resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


        # Known size of the ArUco marker in real-world units (e.g., in meters)
        # Adjust this based on the actual size of your ArUco marker
        self.aruco_marker_size_small = 0.04
        self.aruco_marker_size_large = 0.05
        self.startTime = time.time()
        self.aruco_type = ""
        self.aruco_size = self.aruco_marker_size_small
        self.point_date = []
        self.id_list = []

    def scan_aruco(self):
        try:
            while True:
                data = aruco()
                cap = self.cap
                # Read a frame from the camera
                ret, frame = cap.read()

                # Convert the image to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                # Detect ArUco markers
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.parameters)

                checkTime = time.time()
                if checkTime - self.startTime >= 0.5:  # Check if 200 milliseconds have elapsed
                    if self.aruco_size == self.aruco_marker_size_small:
                        self.aruco_size = self.aruco_marker_size_large
                        self.aruco_type = "Large"
                    else:
                        self.aruco_size = self.aruco_marker_size_small
                        self.aruco_type = "Small"
                        
                    self.startTime = time.time()  # Update last toggle time



                # Publish ArUco IDs and translation vectors
                if ids is not None:
                    # Estimate the distance to each detected marker
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    for i in range(len(ids)):
                        # Use estimatePoseSingleMarkers
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners[i], self.aruco_size, self.mtx, self.dist)
                        # Extract the translation vector (distance) along the z-axis
                        z_distance = tvec[0, 0, 2]
                        y_distance = tvec[0, 0, 1]
                        x_distance = tvec[0, 0, 0]

                        # Display the distance for each marker
                        cv2.putText(frame, f"Marker {ids[i][0]} Distance: {z_distance: .2f} meters", (10, 30 + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        #print("Distance", round(100*z_distance), "cm")
                        #print("x coordinate", x_distance)
                        #print("y coordinate", y_distance)

                        print("ID", ids[i][0])

                        cv2.drawFrameAxes(frame, self.mtx, self.dist, rvec[0], tvec[0], 0.1)
                        
                        # This finds the rotation matrix used to calculate
                        # the euler values for the rotation
                        rotation_matrix = cv2.Rodrigues(rvec)
                        rotation_matrix = rotation_matrix[0]

                        r = Rotation.from_matrix(rotation_matrix)
                        quaternion = r.as_quat()

                        print("quaternion", quaternion)
                        self.point_data = [x_distance, y_distance, z_distance]


                        # Flatten the 2D array into a 1D array
                        #print("Rotation matrix", rotation_matrix)
                        # If the tvec contains any values it is returned
                        if tvec.any():
                            transformed_point_data = self.transform_point_to_global(self.point_data)
                            #print("Point relative to base" ,transformed_point_data)
                            if int(ids[i][0]) not in self.id_list and int(ids[i][0]) != 0:
                                marker_id = int(ids[i][0])
                                self.display_marker(transformed_point_data, marker_id)
                                self.id_list.append(int(ids[i][0]))
                            data.x_distance = x_distance
                            data.y_distance = y_distance
                            data.z_distance = z_distance
                            data.ids = int(ids[i][0])
                            data.rotation_matrix = rotation_matrix
                            data.aruco_type = self.aruco_type
                            self.aruco_data_pub.publish(data)

                # return 0 values of the arucos are not found
                else:
                    x_distance, y_distance, z_distance, ids = 0, 0, 0, 0
                    rotation_matrix = [0.0,0.0,0.0,
                                       0.0,0.0,0.0,
                                       0.0,0.0,0.0]
                    data.x_distance = x_distance
                    data.y_distance = y_distance
                    data.z_distance = z_distance
                    data.ids = 0
                    data.rotation_matrix = rotation_matrix
                    data.aruco_type = self.aruco_type
                    self.aruco_data_pub.publish(data)
                if ret:
                        # Display the frame
                        cv2.imshow('Webcam Input', frame)
                        # Wait for a key press and check if 'q' was pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
    
        except Exception as e:
            rospy.logerr('Error processing image: {}'.format(str(e)))
    
    def transform_point_to_global(self, point_data):
        point_in_frame = PoseStamped()
        point_in_frame.header.frame_id = "camera_frame"  # Specify the frame ID
        point_in_frame.pose.position.x = point_data[0]
        point_in_frame.pose.position.y = point_data[1]
        point_in_frame.pose.position.z = point_data[2]
        

        try: 
            # Get the transformation between base and end_effector
            transform = self.tf_buffer.lookup_transform("base_link", "camera_frame", rospy.Time(0))

            # Transform the point to the target frame
            transformed_point = self.tf_buffer.transform(point_in_frame, "base_link")

            return transformed_point.pose.position
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to transform point: %s" % str(e))
            return None

    def display_marker(self, point_data, marker_id):
        # Create a Marker message
        marker = Marker()
        marker.id = marker_id
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # Point size
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 1.0  # Red color
        marker.color.a = 1.0  # Fully opaque

        # Set the point position
        
        marker.pose.position.x = point_data.x
        marker.pose.position.y = point_data.y
        marker.pose.position.z = point_data.z

        # Publish the Marker message
        self.marker_pub.publish(marker)

        # Create a Marker message for the text label
        text_marker = Marker()
        text_marker.id = -marker_id
        text_marker.header.frame_id = "base_link"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = point_data.x
        text_marker.pose.position.y = point_data.y
        text_marker.pose.position.z = point_data.z + 0.05  # Offset above the main marker
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.05  # Text size
        text_marker.color.g = 1.0  # Red color
        text_marker.color.a = 1.0  # Fully opaque

        # Set the text label
        text_marker.text = str(marker.id)

        # Publish the text Marker message
        self.marker_pub.publish(text_marker)

if __name__ == '__main__':
    try:
        aruco_detector_node = ArucoDetectorNode()
        aruco_detector_node.scan_aruco()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass