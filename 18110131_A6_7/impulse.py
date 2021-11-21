
#SCARA 
#Imports

import numpy as np
import scipy.integrate
import DH_functions as dh
from matplotlib import animation,pyplot as plt
import sympy as sym
import os

#constants
M=[1,1.5,2]
r=0.05 #radius of all rods
l0=0.2  #Initial height of SCARA from ground
L=[0.3,0.2,0.5]#3rd one is to set limit on Prismatic length
# I=[(1/12)*M[0]*L[0]**2 ,(1/12)*M[1]*L[1]**2 ,0.5*M[2]*r**2]
I=[np.array([[0.5*M[0]*r**2,0,0],[0,(1/12)*M[0]*L[0]**2,0],[0,0,(1/12)*M[0]*L[0]**2]]),
        np.array([[0.5*M[1]*r**2,0,0],[0,(1/12)*M[1]*L[1]**2,0],[0,0,(1/12)*M[1]*L[1]**2]]),
        np.array([[(1/12)*M[2]*L[2]**2,0,0],[0,(1/12)*M[2]*L[2]**2,0],[0,0,0.5*M[2]*r**2]])]
g=9.81

#Motor side dynamics:
#From example on Controls tutorial By Michigan
#All motors are assumed to be same
Jm=np.array([[0.01, 0, 0],
            [0  , 0.01, 0],
            [0 , 0,  0.01]])
B_eff=np.array([[0.1 +1, 0, 0],
            [0  , 0.1+1, 0],
            [0 , 0,  0.1+1]])


#Set PD controller gains:
Kp=[36,36,300] #zeta = 1
Kd=[12,12,100]


#Straight trajectory start and end points
start=[0.4,0.06,0.1]
start_vel=[0,0,0]
end=[0.4,0.01,0.1]
end_vel=[0,0,0]


steps=200
time=np.linspace(0,steps/10,steps)
    
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

def jointCalculations(trajectories,T,t):
    x=[]
    x_dot=np.array([[0],[0],[0],[0],[0],[0]],dtype=float)
    x_ddot=np.array([[0],[0],[0],[0],[0],[0]],dtype=float)
    for k in range(3):
        tr=trajectories[k]
        d_tr=sym.diff(tr,t)
        dd_tr=sym.diff(d_tr,t)

        x.append(float(tr.subs({t:T})))
        x_dot[k:k+1,0:1]=float(d_tr.subs({t:T}))
        x_ddot[k:k+1,0:1]=float(dd_tr.subs({t:T}))

    q=inverseKinematics(x)
    DH_param=SCARA_DH_gen(q)
    P=np.array([[0],[0],[0],[1]])
    [J,P0]=dh.Jacobian_EndPoint_DH(DH_param,P)
    J_inv= np.linalg.pinv(J)
    q_dot=J_inv@x_dot

    J_dot=SCARA_Jacobian_dot(q,q_dot)
    q_ddot=J_inv@(x_ddot-J_dot@q_dot)
    return[q,[s for s in q_dot[:,0]],[s for s in q_ddot[:,0]]]

def PD(Kp,Kd,e,e_dot, Jm,B_eff,D,C,G,Q_d_dot,Q_d_ddot):
    # wn=25
    # zeta=1
    # Kp=[(Jm[0,0]+D[0,0])*wn**2,(Jm[1,1]+D[1,1])*wn**2,(Jm[2,2]+D[2,2])*wn**2]
    # Kd=[(Jm[0,0]+D[0,0])*2*wn*zeta - B_eff[0,0],(Jm[1,1]+D[1,1])*2*wn*zeta-B_eff[1,1],(Jm[2,2]+D[2,2])*2*wn*zeta -B_eff[2,2]]
    E=np.array(e)
    E_dot=np.array(e_dot)
    U=np.diag(Kp)@E + np.diag(Kd)@E_dot 
    return U

def PD_FF(Kp,Kd,e,e_dot, Jm,B_eff,D,C,G,Q_d_dot,Q_d_ddot):
    E=np.array(e)
    E_dot=np.array(e_dot)
    D_diag=np.diag([D[0,0],D[1,1],D[2,2]])
    Q_d_dot=np.array(Q_d_dot)
    Q_d_ddot=np.array(Q_d_ddot)

    U=np.diag(Kp)@E + np.diag(Kd)@E_dot +(Jm+D_diag)*Q_d_ddot +B_eff*Q_d_dot
    return U

def PD_FF_disturbance(Kp,Kd,e,e_dot,Jm,B_eff,D,C,G,Q_d_dot,Q_d_ddot):
    E=np.array(e)
    E_dot=np.array(e_dot)
    Q_d_dot=np.array(Q_d_dot)
    Q_d_ddot=np.array(Q_d_ddot)
    
    U=np.diag(Kp)@E + np.diag(Kd)@E_dot +(Jm+D)*Q_d_ddot +(B_eff+C)*Q_d_dot +G
    return U

def multiVariable(Kp,Kd,e,e_dot,Jm,B_eff,D,C,G,Q_d_dot,Q_d_ddot):
    E=np.array(e)
    E_dot=np.array(e_dot) 
    v=np.diag(Kp)@E + np.diag(Kd)@E_dot  
    U=(Jm+D)*(Q_d_ddot+v) +(B_eff+C)*Q_d_dot +G
    return U

def dynamicSystem(T,y,D,C,Jm,B_eff,Q,Q_dot,trajectories,controller,Kp,Kd):
    q1=y[0]
    q2=y[1]
    d3=y[2]
    q_dot1=y[3]
    q_dot2=y[4]
    d_dot3=y[5]
    
    print(T)

    [Q_d, Q_d_dot, Q_d_ddot]= jointCalculations(trajectories,T,t)
    Dd=D
    Cd=C
    dict={}
    for i in range(3):
        dict.update({Q[i]:Q_d[i], Q_dot[i]:Q_d_dot[i]})
    Dd=np.vectorize(lambda x:x.subs(dict))(Dd).astype(np.float64)
    Cd=np.vectorize(lambda x:x.subs(dict))(Cd).astype(np.float64)

    e=[(Q_d[0]-q1),(Q_d[1]-q2),(Q_d[2]-d3)]
    e_dot=[(Q_d_dot[0]-q_dot1) ,(Q_d_dot[1]-q_dot2) ,(Q_d_dot[2]-d_dot3)]

    G=np.array([[0],[0],[-M[2]*g]])
    Gd=G

    #Impulsive disturbance from T=4 to T=5 s
    if T>4 and T<5 :
        im=10
    else:
        im=0

    U=controller(Kp,Kd,e,e_dot, Jm,B_eff,Dd,Cd,Gd,Q_d_dot,Q_d_ddot)
    dict={}
    for i in range(3):
        dict.update({Q[i]:y[i], Q_dot[i]:y[i+3]})
    D=np.vectorize(lambda x:x.subs(dict))(D).astype(np.float64)
    C=np.vectorize(lambda x:x.subs(dict))(C).astype(np.float64)  
    Q_dot=np.array([[q_dot1],[q_dot2],[d_dot3]])


    Q_ddot=np.linalg.inv(Jm +D)@((U+im) - G - (C+B_eff)@Q_dot)
    q_ddot1=Q_ddot[0,0]
    q_ddot2=Q_ddot[1,0]
    d_ddot3=Q_ddot[2,0]
    dydt=[q_dot1,q_dot2,d_dot3,q_ddot1,q_ddot2,d_ddot3] #q1_dot, q1_ddot, q2_dot, q2_ddot
    return dydt

#Plotting function

def updatePlot(i,Q1,Q2,D3,start,end,ax,ax2):
    q1=Q1[i]
    q2=Q2[i]
    d3=D3[i]
    DH_param=SCARA_DH_gen([q1,q2,d3])
    pos=dh.manipulatorPos(DH_param)
    P0=pos[0]
    P1=pos[1]
    P2=pos[2]
    P3=pos[3]
  
    ax.clear() # clear figure before each plot

    # set axis limits. Without this the limits will be autoadjusted which will make it difficult to understand.
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 0.5])
    ax.set_zlim([0, 0.5])
    ax.set_title('With Manipulator')
    ax.plot3D([P0[0],P0[0]],[P0[1],P0[1]],[P0[2],l0],'b-o')
    ax.plot3D([P0[0],P1[0]],[P0[1],P1[1]],[l0,P1[2]],'b-o')
    ax.plot3D([P1[0],P2[0]],[P1[1],P2[1]],[P1[2],P2[2]],'b-o')
    ax.plot3D([P2[0],P3[0]],[P2[1],P3[1]],[P2[2],P3[2]],'y-o')
    ax.plot3D([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],'k')

    ax2.clear() # clear figure before each plot

    # set axis limits. Without this the limits will be autoadjusted which will make it difficult to understand.
    ax2.set_xlim([0, 0.5])
    ax2.set_ylim([0, 0.5])
    ax2.set_zlim([0, 0.5])
    ax2.set_title('Without Manipulator')

    ax2.plot3D(P3[0],P3[1],P3[2],'y*')
    ax2.plot3D([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],'k')

def plotSCARA(X,Xd,time,mainTitle,Q1,Q2,D3):
    fig=plt.figure()
    Title=['X','Y','Z']
    
    for i in range(3):
        ax=fig.add_subplot(2,2,i+1)
        ax.set_title(Title[i])
        min_lim=min(Xd[0][i],Xd[-1][i]) - 0.02
        max_lim=max(Xd[0][i],Xd[-1][i]) + 0.02
        ax.set_ylim([min_lim, max_lim])
        ax.plot(time,[s[i] for s in X ],'r')
        ax.plot(time,[s[i] for s in Xd ],'b')
        ax.legend(["Actual","Desired"])
    
    min_limx=min(Xd[0][0],Xd[-1][0]) - 0.02
    max_limx=max(Xd[0][0],Xd[-1][0]) + 0.02
    min_limy=min(Xd[0][1],Xd[-1][1]) - 0.02
    max_limy=max(Xd[0][1],Xd[-1][1]) + 0.02
    ax=fig.add_subplot(2,2,4)
    ax.set_title(" Y vs X plot")
    ax.set_xlim([min_limx,max_limx])
    ax.set_ylim([min_limy,max_limy])
    ax.plot([s[0] for s in X ],[s[1] for s in X ],'r')
    ax.plot([s[0] for s in Xd ],[s[1] for s in Xd ],'b')
    ax.legend(["Actual","Desired"])

    plt.suptitle(mainTitle)
    plt.show()

    fig=plt.figure()
    ax=fig.add_subplot(1,2,1,projection='3d')
    ax2=fig.add_subplot(1,2,2,projection='3d')
    
    anim=animation.FuncAnimation(fig,updatePlot,frames=steps,interval=60,fargs=[Q1,Q2,D3,Xd[0],Xd[-1],ax,ax2])

    anim.save('18110131_A6_SCARA_'+mainTitle+'.mp4')
    

def forward_kinematics(q1):
    DH_param=SCARA_DH_gen(q1)
    P=np.array([[0],[0],[0],[1]])

    P0=dh.End_Position(DH_param,P)
    return P0

if __name__=="__main__":
    
    Q= sym.symbols(['q1','q2','q3'])
    DH_param=SCARA_DH_gen(Q)
    [D,C,Q,Q_dot,N]=dh.dynamics_equation_generator(DH_param,M,I,L)
    print(sym.simplify(N)) # this shows that N matrix is skew symetric hence the code is working perfectly and equations are correct.

    [trajectories,t]=dh.endEffectorTrajectory(start,end,start_vel,end_vel,time[0],time[-1])
    
    Q_d0= inverseKinematics(start) 
    size=len(time)
    coord=[]
    for i in range(size):
        x=[]
        for k in range(3):
            tr=trajectories[k]
            x.append(float(tr.subs({t:time[i]})))
        coord.append(x)
           
    userInput=0
    maintitle=" "

    while not(userInput == 5):
        print("Choose Controller : \n1. PD\n2. PD + feedforward\n3. PD + Feedforward+ Disturbance \n4. Multi variable \n5. Exit\nEnter index for respective result: ")
        userInput=int(str(input()))
        if userInput==1:
            solution=scipy.integrate.solve_ivp(dynamicSystem ,[0, time[-1]],[Q_d0[0],Q_d0[1],Q_d0[2],0,0,0],t_eval=time, args=[D,C,Jm,B_eff,Q,Q_dot,trajectories, PD,Kp,Kd])           
            maintitle="PD controller"
        elif userInput==2:
            solution=scipy.integrate.solve_ivp(dynamicSystem ,[0, time[-1]],[Q_d0[0],Q_d0[1],Q_d0[2],0,0,0],t_eval=time, args=[D,C,Jm,B_eff,Q,Q_dot,trajectories, PD_FF,Kp,Kd])
            maintitle="PD controller + Feed Forward"
        elif userInput==3:
            solution=scipy.integrate.solve_ivp(dynamicSystem ,[0, time[-1]],[Q_d0[0],Q_d0[1],Q_d0[2],0,0,0],t_eval=time, args=[D,C,Jm,B_eff,Q,Q_dot,trajectories, PD_FF_disturbance,Kp,Kd])
            maintitle="PD controller + Feed Forward + Disturbance Control"
        elif userInput==4:
            solution=scipy.integrate.solve_ivp(dynamicSystem ,[0, time[-1]],[Q_d0[0],Q_d0[1],Q_d0[2],0,0,0],t_eval=time, args=[D,C,Jm,B_eff,Q,Q_dot,trajectories, multiVariable,Kp,Kd])
            maintitle="Multi Variable control"
        elif userInput==5:
            os._exit(0)
        else:
            print("Invalid Input. Enter again\n")
            continue

        Q1=solution.y[0]
        Q2=solution.y[1]
        D3=solution.y[2]

        X=[]
        for i in range(len(Q1)):
            P=forward_kinematics([Q1[i],Q2[i],D3[i]])
            X.append([P[0,0],P[1,0],P[2,0]])
        plotSCARA(X,coord,time,maintitle,Q1,Q2,D3)
    
    os._exit(0)





    
