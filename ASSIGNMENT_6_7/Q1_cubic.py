import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

## PUMA MANIPULATOR

link_lengths = [0.25, 0.25, 0.25] # order : d1, a1, a2 replace d1 with link_lengths[0], a3 with link_lengths[1]. # a3 with link_lengths[2]
init_point = [0.4,0.06,0.1]
final_point = [0.4, 0.01,.1]
t0 = 0
tf = 5
v0 = np.array([0,0,0])
vf = np.array([0,0,0]) 
t = 0
def puma_inverse(point,link_lengths):
    solutions = []
    theta1 = np.arctan2(point[1],point[0])
    D = (point[0]**2 + point[1]**2 + (point[2]-link_lengths[0])**2 -link_lengths[1]**2 - link_lengths[2]**2)/(2*link_lengths[1]*link_lengths[2])
    if abs(D)<=1:
        theta3 = np.arctan2(np.sqrt(1-D**2),D)
        theta2 = np.arctan2(point[2]-link_lengths[0],np.sqrt(point[0]**2 + point[1]**2)) - np.arctan2(link_lengths[2]*np.sin(theta3),link_lengths[1] + link_lengths[2]*np.cos(theta3))

        solutions.append([theta1,theta2,theta3])

        theta3 = np.arctan2(-np.sqrt(1-D**2),D)
        theta2 = np.arctan2(point[2]-link_lengths[0],np.sqrt(point[0]**2 + point[1]**2)) - np.arctan2(link_lengths[2]*np.sin(theta3),link_lengths[1] + link_lengths[2]*np.cos(theta3))

        theta1 = theta1%(2*np.pi)
        theta2 = theta2%(2*np.pi)
        theta3 = theta3%(2*np.pi)
        solutions.append([theta1,theta2,theta3])
    else:
        print("Error. The given inputs are out of bounds of workspace")
    
    return solutions

def puma_forward(q1,q2,q3):
    xc = link_lengths[1]*np.cos(q2)*np.cos(q1) + link_lengths[2]*np.cos(q2+q3)*np.cos(q1)
    yc = link_lengths[1]*np.cos(q2)*np.sin(q1) + link_lengths[2]*np.cos(q2+q3)*np.sin(q1)
    zc = link_lengths[0] + link_lengths[1]*np.sin(q2) + link_lengths[2]*np.sin(q2+q3)
    end_effector = [xc,yc,zc]
    return end_effector

q_ini = puma_inverse(init_point,link_lengths) # the first three arguements are xc, yc & zc respectively

q_fin = puma_inverse(final_point,link_lengths)
q0 = np.array(q_ini[0])
qf = np.array(q_fin[0])

M = np.array([[1, t0, t0**2, t0**3],
             [0, 1, 2*t0, 3*t0**2],
             [1, tf, tf**2, tf**3],
             [0, 1, 2*tf, 3*tf**2]])

RHS = np.vstack((q0,v0,qf,vf))

C = np.matmul(np.linalg.inv(M),RHS)
print(C)
print(RHS)
q_ = []
q_dot_ = []
q_ddot_ = []
t_linspace = np.linspace(t0,tf,100)

for i in range(len(t_linspace)):
    
    q_.append(C[0] + C[1]*t_linspace[i] + C[2]*t_linspace[i]**2 + C[3]*t_linspace[i]**3)
    q_dot_.append(C[1] + 2*C[2]*t_linspace[i] + 2*C[3]*t_linspace[i]**2)
    q_ddot_.append(2*C[2] + 6*C[3]*t_linspace[i])
q = np.array(q_)
q_dot = np.array(q_dot_)
q_ddot = np.array(q_ddot_)

plt.plot(t_linspace,q[:,0])
plt.plot(t_linspace,q[:,1])
plt.plot(t_linspace,q[:,2])
plt.title('Theta plots')
# plt.legend(handles =[Theta 1,Theta 2,Theta 3])
plt.show()
