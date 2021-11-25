import numpy as np
from numpy.core.function_base import linspace
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from Q1_kaula import linear_traj

##################____________MANDATORY CODE BLOCKS______________
def wrapping(angle):
    if abs(angle%(2*np.pi) - 2*np.pi)<10^(-5):
        return 0
    return angle%(2*np.pi)

def scara_inverse(xc,yc,zc, link_lengths):
    if np.sqrt(xc**2+yc**2)>link_lengths[2] + link_lengths[1]:
        print("No Solution can be Found!")
        return []
    else: 
        def inv_func(x):
            return [
                    - xc + link_lengths[1]*np.cos(x[0]) + link_lengths[2]*np.cos(x[0]+x[1]),
                    - yc + link_lengths[1]*np.sin(x[0]) + link_lengths[2]*np.sin(x[0]+x[1]),
                    - zc + link_lengths[0] - x[2]
                    ]
        root = fsolve(inv_func,[1,1,1])

        q1,q2,d = root

        return [wrapping(q1),wrapping(q2),d]

def scara_forward(q1,q2,d):

    xc = link_lengths[1]*np.cos(q1) + link_lengths[2]*np.cos(q1+q2)
    yc = link_lengths[1]*np.sin(q1) + link_lengths[2]*np.sin(q1+q2)
    zc = link_lengths[0] - d

    return xc,yc,zc
##########################_____________________________________________________


def joint_traj(link_lengths,init_point,final_point, t0,tf,v0,vf,a0,af):
    eep = linear_traj(link_lengths,init_point,final_point, t0,tf,v0,vf,a0,af)
    q_ = []
    for point in eep:
        q_.append(scara_inverse(point[0],point[1],point[2],link_lengths))
        # print(point)
    q_des = np.array(q_)
    # print(q_des)
    vel_des = []
    acdes = []
    # steps = 20
    t_lin = linspace(t0,tf,200)
    dt = (t_lin[2] - t_lin[1])
    for i in range(len(t_lin)-1):
    
        vel_des.append((q_des[i+1] - q_des[i])/dt)
        # print(vel_des)
        dv = (((q_des[i+1] - q_des[i])/dt) - (q_des[i] - q_des[i-1])/dt)
        acdes.append(dv/dt)

    v_des = np.array(vel_des)
    a_des = np.array(acdes)
    # print(q_des)
    return q_des, v_des, a_des



if __name__=='__main__':
    link_lengths = [0.25, 0.25, 0.25] # order : d1, a1, a2 replace d1 with link_lengths[0], a2 with link_lengths[1]. # a3 with link_lengths[2]
    init_point = [0.4,0.06,0.1]
    final_point = [0.4, 0.01,.1]
    t0 = 0
    tf = 5
    v0 = np.array([0,0,0])
    vf = np.array([0,0,0]) 
    a0 = np.array([0,0,0])
    af = np.array([0,0,0])
    joint_traj(link_lengths,init_point,final_point, t0,tf,v0,vf,a0,af)