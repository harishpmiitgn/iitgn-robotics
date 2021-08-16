Videh Patel (19110192) : videh.p@iitgn.ac.in
The submission of MiniProject

RRM_videh.py contains code for dynamics simulation and animation of the RRManipulator(RRM)
Points to note:
- RRM is initialised as arko
- RRM takes in only torque values as an input from controller, rest all it calculated by the dynamics equations of RRM

RRMC_videh.py contains code for the controller that controls RRM and makes it perform complex tasks
- RRMC is initialised as arkoCon
- RRM calculates the torques to be applied on RRM to make it perform desired tasks

HOW TO TEST TASKS:
#Requirements
- python
- matplotlib (python lib)
- numpy (python lib)

#Starting Simulation
- Download all the files in a folder
- Run command prompt/terminal from that folder
- Enter python interactive mode: Type 'py (or) python (or) python3' in the terminal
- In python interactive mode, run command : from RRMC_videh import*
- A window will appear with RRM on it

#Task1
- Run: arkoCon.Task1()
- Will follow either a line or semicircle, based on the input
- Once, will follow the path in 1 sec and another time in 0.25 sec
- Will plot out the torques applied both times to compare, in a new window

#Task2
-Run arkoCon.Task2()
- Will follow a straight line to a wall shown on the screen and apply a force of 10 newtons of force on wall: y = 0.5x - 7 for 5 seconds
- If it shows an error : Specified path contains points out of workspace too; Restart the simulator, and run the command

#Task3
-Run arkoCon.Task3()
-Will apply a force that an ideal spring would apply with eq_position = (3,4) and stiffness = 5 N/m
NOTE:: Static conditions are established by applying forces at the end effector. The forces shown in the window and comamnd line are the externally applied forces which are balanced by the torques applied

#Task4
-Run arkoCon.Task4()
-Will plot points with specified intervals to plot out the workspace when q1,q2 are both constrained withn [35 deg, 145 deg]

----THANK YOU---------------
