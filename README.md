# IITGN Robotics
# HPM's Introduction to Robotics - ME 639
# Team Slasher
[Pradeep Saini | Yash Meshram | Nikhil Yadav]

-------------------------------------------------------------------------------------------------------------

# Compile codes
* 'Robotics_project.ipynb' notebook contains all the code of following:
    * Forward kinematics
    * Inverse kinematics velocity
    * Inverse kinematics
    * Robot Dynamics
    * Independent joint control (PI control)

# Python simulation
* Check the requirements in 'setup.py' file
* Open 'simulation.py' and give the desired inputs and run the file

# ROS simulation
* Save 'moveitlatest' and 'moveit_utorials' in the src folder of the catkin workspace.
* build the packages in the catkin workspace:
```
$ cd ~/catkin_ws
$ catkin_make
```
* To add the workspace to your ROS environment, source the generated setup file:
```
$ . ~/catkin_ws/devel/setup.bash
```
* Launch the rviz file for visualization:
```
$ roslaunch moveitlatest demo.launch
```
* Run the following python file for simulating the robot in rviz:
```
$ rosrun moveit_tutorials move_group_python_interface_tutorial.py
```

# References:
* For python simulation: https://github.com/petercorke/robotics-toolbox-python
* Adapted for our robot: https://github.com/ros-planning/moveit_tutorials/blob/master/doc/move_group_python_interface/scripts/move_group_python_interface_tutorial.py
