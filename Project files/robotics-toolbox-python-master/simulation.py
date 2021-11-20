from typing import KeysView
from matplotlib.pyplot import savefig
from numpy.core.getlimits import _KNOWN_TYPES
import roboticstoolbox as rtb
robot = rtb.models.DH.Panda()
print(robot)

T = robot.fkine(robot.qz)  # forward kinematics
print(T)

# A = list(map(int,input("\nEnter the x,y,z co-ordinate (space sperated) : ").strip().split()))[:3]
A = [-96.6, 76.6, 112]

from spatialmath import SE3
T = SE3(A[0], A[1], A[2]) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ikine_LM(T)         # solve IK
print(sol)

q_pickup = sol.q
print(robot.fkine(q_pickup))    # FK shows that desired end-effector pose was achieved

qt = rtb.jtraj(robot.qz, q_pickup, 100)
robot.plot(qt.q)