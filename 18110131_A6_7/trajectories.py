#Solution to question 1 and 2
#Going forward with SCARA manipulator

import numpy as np
import DH_functions as dh
from matplotlib import pyplot as plt
import sympy as sym
import os

#Scara dimensions
l0=0.2
L=[0.3,0.2,0.5]

def Q1(ini_pos,final_pos,ini_vel,final_vel,time):
    #End effector trajectory plot
    [trajectories,t]=dh.endEffectorTrajectory(ini_pos,final_pos,ini_vel,final_vel,time[0],time[-1])
    size=len(time)
    coord=[]
    coord_dot=[]
    coord_ddot=[]
    for k in range(3):
        X=[]
        X_dot=[]
        X_ddot=[]
        for i in range(size):
            tr=trajectories[k]
            d_tr=sym.diff(tr,t)
            dd_tr=sym.diff(d_tr,t)

            X.append(tr.subs({t:time[i]}))
            X_dot.append(d_tr.subs({t:time[i]}))
            X_ddot.append(dd_tr.subs({t:time[i]}))
        coord.append(X)
        coord_dot.append(X_dot)
        coord_ddot.append(X_ddot)

    Title=['X','Y','Z']
    fig=plt.figure()

    for i in range(3):
        plt.subplot(2,2,i+1)
        plt.title(Title[i]+' vs time')
        plt.plot(time,coord[i],'r')

    ax=fig.add_subplot(2,2,4,projection='3d')
    
    ax.set_title('Complete')
    ax.plot3D(coord[0],coord[1],coord[2],'r')

    fig1=plt.figure()

    for i in range(3):
        plt.subplot(2,2,i+1)
        plt.title('Velocity '+Title[i]+' vs time')
        plt.plot(time,coord_dot[i],'r')

    ax=fig1.add_subplot(2,2,4,projection='3d')
    
    ax.set_title('Complete')
    ax.plot3D(coord_dot[0],coord_dot[1],coord_dot[2],'r')

    fig2=plt.figure()

    for i in range(3):
        plt.subplot(2,2,i+1)
        plt.title('Acceleration '+Title[i]+' vs time')
        plt.plot(time,coord_ddot[i],'r')

    ax=fig2.add_subplot(2,2,4,projection='3d')
    
    ax.set_title('Complete')
    ax.plot3D(coord_ddot[0],coord_ddot[1],coord_ddot[2],'r')
    plt.show()


def SCARA_DH_gen(Q):
    q1=Q[0]
    q2=Q[1]
    d3=Q[2]

    DH_param=np.zeros([3,5],dtype=object)# First column link types, second for a, third for alpha , fourth for d and fifth for theta
    DH_param[0,0]='R'
    DH_param[0:1,1:5]=np.array([L[0],0,l0,q1])
    DH_param[1,0]='R'
    DH_param[1:2,1:5]=np.array([L[1],np.pi,0,q2])
    DH_param[2,0]='P'
    DH_param[2:3,1:5]=np.array([0,0,d3,0])

    return DH_param

def inverseKinematics(Y):
    x=Y[0]
    y=Y[1]
    z=Y[2]
    #Inverse kinematics
    q2=np.arccos((x**2+y**2-L[0]**2-L[1]**2)/(2*L[1]*L[0]))
    q1=np.arctan2(y,x)-np.arctan2((L[1]*np.sin(q2)),(L[0]+L[1]*np.cos(q2)))
    d3=l0-z   
    return[q1,q2,d3]

def SCARA_Jacobian_dot(Q,Q_dot):
    q1=Q[0]
    q2=Q[1]
    d3=Q[2]
    q1_dot=Q_dot[0]
    q2_dot=Q_dot[1]
    d3_dot=Q_dot[2]

    J_dot=np.array([[-(L[1]*(np.cos(q1+q2))*(q1_dot+q2_dot) +L[0]*(np.cos(q1))*q1_dot)  ,-L[1]*(np.cos(q1+q2))*(q1_dot+q2_dot)  ,0],
                [-(L[1]*(np.sin(q1+q2))*(q1_dot+q2_dot) +L[0]*(np.sin(q1))*q1_dot)      ,-L[1]*(np.sin(q1+q2))*(q1_dot+q2_dot)  ,0],
                [0                                                                      ,0                                      ,0],
                [0                                                                      ,0                                      ,0],
                [0                                                                      ,0                                      ,0],
                [0                                                                      ,0                                      ,0]])
                
    return J_dot

def Q2(ini_pos,final_pos,ini_vel,final_vel,time):
    #joint trajectory plots for SCARA

    [trajectories,t]=dh.endEffectorTrajectory(ini_pos,final_pos,ini_vel,final_vel,time[0],time[-1])
    size=len(time)
    joint=[]
    joint_dot=[]
    joint_ddot=[]

    for i in range(size):
        x=[]
        x_dot=np.array([[0],[0],[0],[0],[0],[0]],dtype=float)
        x_ddot=np.array([[0],[0],[0],[0],[0],[0]],dtype=float)
        for k in range(3):
            tr=trajectories[k]
            d_tr=sym.diff(tr,t)
            dd_tr=sym.diff(d_tr,t)

            x.append(float(tr.subs({t:time[i]})))
            x_dot[k:k+1,0:1]=float(d_tr.subs({t:time[i]}))
            x_ddot[k:k+1,0:1]=float(dd_tr.subs({t:time[i]}))

        q=inverseKinematics(x)
        DH_param=SCARA_DH_gen(q)
        P=np.array([[0],[0],[0],[1]])
        [J,P0]=dh.Jacobian_EndPoint_DH(DH_param,P)
        J_inv= np.linalg.pinv(J)
        q_dot=J_inv@x_dot

        J_dot=SCARA_Jacobian_dot(q,q_dot)
        q_ddot=J_inv@(x_ddot-J_dot@q_dot)
        joint.append(q)
        joint_dot.append([s for s in q_dot[:,0]])
        joint_ddot.append([s for s in q_ddot[:,0]])


    Title=['X','Y','Z']
    fig=plt.figure()

    for i in range(3):
        plt.subplot(2,2,i+1)
        plt.title(Title[i]+' vs time')
        plt.plot(time,[s[i] for s in joint],'r')

    fig1=plt.figure()

    for i in range(3):
        plt.subplot(2,2,i+1)
        plt.title('Velocity '+Title[i]+' vs time')
        plt.plot(time,[s[i] for s in joint_dot],'r')

    fig2=plt.figure()

    for i in range(3):
        plt.subplot(2,2,i+1)
        plt.title('Acceleration '+Title[i]+' vs time')
        plt.plot(time,[s[i] for s in joint_ddot],'r')

    plt.show()


#Main function calling above 3 parts
if __name__=="__main__":
    userInput=0
    ini_pos=[0.4,0.06,0.1]
    ini_vel=[0,0,0]
    final_pos=[0.4,0.01,0.1]
    final_vel=[0,0,0]
    time=np.linspace(0,10,100)

    while not(userInput == 3):
        print("Choose : \n1. Answer to Q1\n2. Answer to Q2\n3. Exit\nEnter index for respective result: ")
        userInput=int(str(input()))
        if userInput==1:
            Q1(ini_pos,final_pos,ini_vel,final_vel,time)           
        elif userInput==2:
            Q2(ini_pos,final_pos,ini_vel,final_vel,time)
        elif userInput==3:
            os._exit(0)
        else:
            print("Invalid Input. Enter again\n")
    os._exit(0)