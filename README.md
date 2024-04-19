# Maintenance Task
## Dependencies 
- Numpy 
- Scipy
- Opencv-contrib-python 
- tf2_ros 
- transforms3d
- Moveit, ROS Noetic 
- UR_drivers, ROS Noetic

## How to launch the scripts
To launch go into src folder of workspace and the following line remember to change the variables respectively:
```bash
./runit.sh
```
```bash
rosrun newur urmove.py
```

If for testing without robot in pure Rviz use the following commands: 
```bash 
./demo_runit.sh
```
```bash 
rosrun newur urmove.py 
```
The demo and the execution.launch will be using the same settings for Rviz. 