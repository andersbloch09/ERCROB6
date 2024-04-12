#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
from scannode.msg import aruco

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
                        values = cv2.Rodrigues(rvec)
                        rotation_matrix = values[0]
                        rotation_matrix
                        # Flatten the 2D array into a 1D array
                        rotation_matrix = [element for row in rotation_matrix for element in row]
                        print("Rotation matrix", rotation_matrix)
                        # If the tvec contains any values it is returned
                        if tvec.any():
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

if __name__ == '__main__':
    try:
        aruco_detector_node = ArucoDetectorNode()
        aruco_detector_node.scan_aruco()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass