#PUMA

#Imports

import numpy as np
import scipy.integrate
import DH_functions as dh
from matplotlib import animation,pyplot as plt
import sympy as sym

#constants
M=[1,1.5,2]
r=0.05 #radius of all rods

L=[1,1.5,2] #3rd one is to set limit on Prismatic length
# I=[(1/12)*M[0]*L[0]**2 ,(1/12)*M[1]*L[1]**2 ,0.5*M[2]*r**2]
I=[np.array([[(1/12)*M[0]*L[0]**2,0,0],[0,(1/12)*M[0]*L[0]**2,0],[0,0,0.5*M[0]*r**2]]),
        np.array([[0.5*M[1]*r**2,0,0],[0,(1/12)*M[1]*L[1]**2,0],[0,0,(1/12)*M[1]*L[1]**2]]),
        np.array([[0.5*M[2]*r**2,0,0],[0,(1/12)*M[2]*L[2]**2,0],[0,0,(1/12)*M[2]*L[2]**2]])]
g=9.81


#Straight trajectory start and end points
start=[1,0,1]
end=[2,1,-1]

#Error storage
E1=[]
E2=[]
E3=[]
T=[]


steps=250
time=np.linspace(0,steps/10,steps)
    
def PUMA_DH_gen(Q):
    q1=Q[0]
    q2=Q[1]
    q3=Q[2]

    DH_param=np.zeros([3,5],dtype=object)# First column link types, second for a, third for alpha , fourth for d and fifth for theta
    DH_param[0,0]='R'
    DH_param[0:1,1:5]=np.array([0,np.pi/2,L[0],q1])
    DH_param[1,0]='R'
    DH_param[1:2,1:5]=np.array([L[1],0,0,q2])
    DH_param[2,0]='R'
    DH_param[2:3,1:5]=np.array([L[2],0,0,q3])

    return DH_param

def inverseKinematics(Y):
    x=Y[0]
    y=Y[1]
    z=Y[2]
    #Inverse kinematics
    q1=np.arctan2(y,x)
    q3=-np.arccos((x**2+y**2+(z-L[0])**2-L[1]**2-L[2]**2)/(2*L[2]*L[1]))
    q2=np.arctan2(z-L[0],np.sqrt(x**2+y**2))-np.arctan2((L[2]*np.sin(q3)),(L[1]+L[2]*np.cos(q3)))
            
    return[q1,q2,q3]

def integral_sum(E,T):
    sum=0
    if(len(E)>41):
        for i in range(41):
            sum=sum+ (E[-(1+i)])*(T[-(1+i)]-T[-(2+i)]) 
    
    return sum

def dynamicSystem(t,y,D,C,Q,Q_dot,start,end):
    q1=y[0]
    q2=y[1]
    q3=y[2]
    q_dot1=y[3]
    q_dot2=y[4]
    q_dot3=y[5]
    
    X_des=[0,0,0]
    for i in range(3):
        X_des[i]=start[i] + (end[i]-start[i])*t/(steps/10)

    Q_d= inverseKinematics(X_des)

    dict={}
    for i in range(3):
        dict.update({Q[i]:y[i], Q_dot[i]:y[i+3]})
    D=np.vectorize(lambda x:x.subs(dict))(D).astype(np.float64)
    C=np.vectorize(lambda x:x.subs(dict))(C).astype(np.float64)

    e1=(Q_d[0]-q1)
    e2=(Q_d[1]-q2)
    e3=(Q_d[2]-q3)
    E1.append(e1)
    E2.append(e2)
    E3.append(e3)
    T.append(t)
    print(t)

    u1=9.5*e1+0.5*integral_sum(E1,T)
    u2=10.0*e2  +0.5*integral_sum(E2,T)
    u3=10.0*e3 +0.5*integral_sum(E3,T)
    
    U=np.array([[u1],[u2],[u3]])
    G=np.array([[0],[M[1]*g*(L[1]/2)*np.cos(q2)+M[2]*g*L[1]*np.cos(q1)],[M[2]*g*(L[2]/2)*np.cos(q3)]])
    Q_dot=np.array([[q_dot1],[q_dot2],[q_dot3]])
    Q_ddot=np.linalg.inv(D)@(U - G - C@Q_dot)

    q_ddot1=Q_ddot[0,0]
    q_ddot2=Q_ddot[1,0]
    q_ddot3=Q_ddot[2,0]
    dydt=[q_dot1,q_dot2,q_dot3,q_ddot1,q_ddot2,q_ddot3] #q1_dot, q1_ddot, q2_dot, q2_ddot
    return dydt

def updatePlot(i,Q1,Q2,Q3,start,end,ax,ax2):
    q1=Q1[i]
    q2=Q2[i]
    q3=Q3[i]
    DH_param=PUMA_DH_gen([q1,q2,q3])
    pos=dh.manipulatorPos(DH_param)
    P0=pos[0]
    P1=pos[1]
    P2=pos[2]
    P3=pos[3]
  
    ax.clear() # clear figure before each plot

    # set axis limits. Without this the limits will be autoadjusted which will make it difficult to understand.
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.set_zlim([-1, 2])
    ax.set_title('With Manipulator')
    ax.plot3D([P0[0],P1[0]],[P0[1],P1[1]],[P0[2],P1[2]],'b-o')
    ax.plot3D([P1[0],P2[0]],[P1[1],P2[1]],[P1[2],P2[2]],'b-o')
    ax.plot3D([P2[0],P3[0]],[P2[1],P3[1]],[P2[2],P3[2]],'y-o')
    ax.plot3D([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],'k')

    ax2.clear() # clear figure before each plot

    # set axis limits. Without this the limits will be autoadjusted which will make it difficult to understand.
    ax2.set_xlim([-1, 2])
    ax2.set_ylim([-1, 2])
    ax2.set_zlim([-1, 2])
    ax2.set_title('Without Manipulator')

    ax2.plot3D(P3[0],P3[1],P3[2],'y*')
    ax2.plot3D([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],'k')

def forward_kinematics(q1):
    DH_param=PUMA_DH_gen(q1)
    P=np.array([[0],[0],[0],[1]])

    P0=dh.End_Position(DH_param,P)
    return P0

if __name__=="__main__":
    
    Q= sym.symbols(['q1','q2','q3'])
    
    DH_param=PUMA_DH_gen(Q)

    [D,C,Q,Q_dot]=dh.dynamics_equation_generator(DH_param,M,I,L)
    Q_d0= inverseKinematics(start)
    solution=scipy.integrate.solve_ivp(dynamicSystem ,[0, time[-1]],[Q_d0[0],Q_d0[1],Q_d0[2],0,0,0],t_eval=time, args=[D,C,Q,Q_dot,start,end,])
    Q1=solution.y[0]
    Q2=solution.y[1]
    Q3=solution.y[2]
    # print(Q1,Q2,Q3)

    Y=[]
    for i in range(len(Q1)):
        P=forward_kinematics([Q1[i],Q2[i],Q3[i]])
        Y.append([P[0,0],P[1,0],P[2,0]])
    Title=['X','Y','Z']


    fig=plt.figure()
    ax=fig.add_subplot(1,2,1,projection='3d')
    ax2=fig.add_subplot(1,2,2,projection='3d')
    
    anim=animation.FuncAnimation(fig,updatePlot,frames=steps,interval=60,fargs=[Q1,Q2,Q3,start,end,ax,ax2])

    anim.save('18110131_A5_PUMA.mp4')

    ax.clear()
    plt.clf()
    fig=plt.figure()

    for i in range(3):
        plt.subplot(2,2,i+1)
        plt.title(Title[i])
        plt.plot(time,[s[i] for s in Y ],'r')
        plt.plot([0,time[-1]],[start[i],end[i]],'b')

    ax=fig.add_subplot(2,2,4,projection='3d')
    
    ax.set_title('Complete')
    ax.plot3D([s[0] for s in Y ],[s[1] for s in Y ],[s[2] for s in Y ],'r')
    ax.plot3D([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],'b')
    plt.show()
    

    