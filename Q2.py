import math
import scipy
import sympy as sp
from sympy.core.evalf import N
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 

def Jacobian(q):
    DH=np.array([[q[0],0.25,0,-np.pi/2],
                [q[1],0,0.25,0],
                [q[2],0,0.25,0]])
    A=[0]*3
    R=[0]*3
    for i in range(3):
        Theta_i=DH[i][0]
        di=DH[i][1]
        ai=DH[i][2]
        alpha_i=DH[i][3]
        A[i]=np.array([[np.cos(Theta_i),-(np.sin(Theta_i)*np.cos(alpha_i)),(np.sin(Theta_i)*np.sin(alpha_i)),ai*np.cos(Theta_i)],
        [np.sin(Theta_i),np.cos(Theta_i)*np.cos(alpha_i),-(np.cos(Theta_i)*np.sin(alpha_i)),ai*np.sin(Theta_i)],
        [0,np.sin(alpha_i),np.cos(alpha_i),di],
        [0,0,0,1]])
        R[i]=np.array([[np.cos(Theta_i),-(np.sin(Theta_i)*np.cos(alpha_i)),(np.sin(Theta_i)*np.sin(alpha_i))],
        [np.sin(Theta_i),np.cos(Theta_i)*np.cos(alpha_i),-(np.cos(Theta_i)*np.sin(alpha_i))],
        [0,np.sin(alpha_i),np.cos(alpha_i)]])
    R_0_n = []
    B = np.identity(3)
    for i in range(3):
      R_0_n.append(B)
      B = np.matmul(B, R[i])
    R_0_n.append(B)
    R_0_n = np.array(R_0_n)

    H_0_n = []
    C = np.identity(4)
    for i in range(3):
      H_0_n.append(C)
      C = np.matmul(C, A[i])
    H_0_n.append(C)
    H_0_n = np.array(H_0_n)

    z = []
    k = np.array([[0],[0],[1]])
    for i in range(4):
      z.append(np.matmul(R_0_n[i],k))
    z = np.array(z)

    d = np.array([[0],[0],[0],[1]])
    O = np.matmul(H_0_n,d)[:4]
    On = np.delete(O,(3),axis=1)
    
    J = []
    for i in range(3):  
        X = On[-1] - On[i]
        J_v = np.cross(z[i], X,axis = 0)
        J_v = np.vstack((J_v, z[i]))
        J.append(J_v)
    J = np.array(J)
    return J

def Joint_accel(a,q,Joint_velo):
    l1=0.25
    l2=0.25
    l3=0.25
    Jv=np.array([[-(l1*np.sin(q[0])+l2*np.sin(q[0]+q[1])+l3*np.sin(q[0]+q[1]+q[2])),-(l2*np.sin(q[0]+q[1])+l3*np.sin(q[0]+q[1]+q[2])),-(l3*np.sin(q[0]+q[1]+q[2]))],
                 [(l1*np.cos(q[0])+l2*np.cos(q[0]+q[1])+l3*np.cos(q[0]+q[1]+q[2])),(l2*np.cos(q[0]+q[1])+l3*np.cos(q[0]+q[1]+q[2])),(l3*np.cos(q[0]+q[1]+q[2]))],
                 [0,0,0]])
    a_des=np.array([[a[0][0]+(l1*np.cos(q[0])*(Joint_velo[0][0][0]**2))+(l2*np.cos(q[0]+q[1])*(Joint_velo[0][0][0]+Joint_velo[1][0][0])*Joint_velo[0][0][0])+(l3*np.cos(q[0]+q[1]+q[2])*(Joint_velo[0][0][0]+Joint_velo[1][0][0]+Joint_velo[2][0][0])*Joint_velo[0][0][0])+(l2*np.cos(q[0]+q[1])*(Joint_velo[0][0][0]+Joint_velo[1][0][0])*Joint_velo[1][0][0])+(l3*np.cos(q[0]+q[1]+q[2])*(Joint_velo[0][0][0]+Joint_velo[1][0][0]+Joint_velo[2][0][0])*Joint_velo[1][0][0])+(l3*np.cos(q[0]+q[1]+q[2])*(Joint_velo[0][0][0]+Joint_velo[1][0][0]+Joint_velo[2][0][0])*Joint_velo[2][0][0])],
                    [a[1][0]+(l1*np.sin(q[0])*(Joint_velo[0][0][0]**2))+(l2*np.sin(q[0]+q[1])*(Joint_velo[0][0][0]+Joint_velo[1][0][0])*Joint_velo[0][0][0])+(l3*np.sin(q[0]+q[1]+q[2])*(Joint_velo[0][0][0]+Joint_velo[1][0][0]+Joint_velo[2][0][0])*Joint_velo[0][0][0])+(l2*np.sin(q[0]+q[1])*(Joint_velo[0][0][0]+Joint_velo[1][0][0])*Joint_velo[1][0][0])+(l3*np.sin(q[0]+q[1]+q[2])*(Joint_velo[0][0][0]+Joint_velo[1][0][0]+Joint_velo[2][0][0])*Joint_velo[1][0][0])+(l3*np.sin(q[0]+q[1]+q[2])*(Joint_velo[0][0][0]+Joint_velo[1][0][0]+Joint_velo[2][0][0])*Joint_velo[2][0][0])],
                    [a[2][0]]])
    PJv=np.linalg.pinv(Jv)
    q_dot_dot=PJv@a_des
    return q_dot_dot

def Joint_angles(d):
    l1=0.25
    l2=0.25
    l3=0.25
    q1=np.arctan2(d[1,0],d[0,0])
    dx=math.sqrt((d[1,0]**2)+(d[0,0]**2))
    dy=-0.05
    m3=(((dx**2)+(dy**2)-(l1**2)-(l2**2))/(2*l1*l2))
    q3=np.arccos(m3)
    q2=np.arctan2(dy,dx)-np.arctan2(np.sin(q3),(1+np.cos(q3)))
    q=np.array([q1,q2,q3])
    return q

end_effec0=[0.4,0.06,0.1]
end_effec1=[0.4,0.01,0.1]

y0=end_effec0[1]
y1=end_effec1[1]

Values=list(map(float, input("Enter the initial and fianl Values=[v0,v1,t0,tf,a0,a1]: ").split()))
v0=Values[0]
v1=Values[1]
t0=int(Values[2])
tf=int(Values[3])
a0=Values[4]
a1=Values[5]

t=[]
for i in range(int(((tf-t0)*100))):
    t.append((int(t0)+i)/100)
# print(t)

M=np.array([[1,t0,t0**2,t0**3,t0**4,t0**5],
            [0,1,2*t0,3*t0**2,4*t0**3,5*t0**4],
            [0,0,2,6*t0,12*t0**2,20*t0**3],
            [1,tf,tf**2,tf**3,tf**4,tf**5],
            [0,1,2*tf,3*tf**2,4*tf**3,5*tf**4],
            [0,0,2,6*tf,12*tf**2,20*tf**3]])

b=np.array([[y0],
            [v0],
            [a0],
            [y1],
            [v1],
            [a1]])
a=(np.linalg.inv(M)@b).T
# d=np.array([[0.4],
#    [0.06],
#    [0.1]])
Q1=[]
Q2=[]
Q3=[]
Q1_dot=[]
Q2_dot=[]
Q3_dot=[]
Q1_dot_dot=[]
Q2_dot_dot=[]
Q3_dot_dot=[]
for j in t:
    yd=a[0][0]+a[0][1]*j+a[0][2]*(j**2)+a[0][3]*(j**3)+a[0][4]*(j**4)+a[0][5]*(j**5)
    vx=0
    vd=a[0][1]+2*a[0][2]*j+3*a[0][3]*(j**2)+4*a[0][4]*(j**3)+5*a[0][5]*(j**4)
    vz=0
    ax=0
    ad=2*a[0][2]+6*a[0][3]*j+12*a[0][4]*(j**2)+20*a[0][5]*(j**3)
    az=0
    d=np.array([[0.4],
                [yd],
                [0.1]])
    v=np.array([[vx],
                [vd],
                [vz]])
    acceleration=np.array([[ax],
                [ad],
                [az]])
    q=Joint_angles(d)
    Q1.append(q[0])
    Q2.append(q[1])   
    Q3.append(q[2])
    Jacob=Jacobian(q)
    Jacob_inv=np.linalg.pinv(Jacob)
    v=np.array([[0],
                [vd],
                [0],
                [0],
                [0],
                [0]])
    Joint_velocity=np.matmul(Jacob_inv, v)
    # print(Joint_velocity[0][0][0])
    Q1_dot.append(Joint_velocity[0][0][0])
    Q2_dot.append(Joint_velocity[1][0][0])
    Q3_dot.append(Joint_velocity[2][0][0])
    Joint_acceleration=Joint_accel(acceleration,q,Joint_velocity)
    Q1_dot_dot.append(Joint_acceleration[0][0])
    Q2_dot_dot.append(Joint_acceleration[1][0])
    Q3_dot_dot.append(Joint_acceleration[2][0])
    # print(Joint_acceleration)
fig = plt.figure()
# plt1 = fig.add_subplot(221)
# plt2 = fig.add_subplot(222)
# plt3 = fig.add_subplot(223)

# x=np.linspace(t0,(tf-t0),(tf-t0)*100)
# plt.plot(x,Q1, label='q1')
# plt.plot(x,Q2, label='q2')
# plt.plot(x,Q3,label='q3')
# plt.title('$Position-Time Graph$')

# x=np.linspace(t0,(tf-t0),(tf-t0)*100)
# plt.plot(x,Q1_dot,label='q1"')
# plt.plot(x,Q2_dot,label='q2"')
# plt.plot(x,Q3_dot,label='q3"')
# plt.title('$JointVelocity-Time Graph$')

x=np.linspace(t0,(tf-t0),(tf-t0)*100)
plt.plot(x,Q1_dot_dot,label='q1""')
plt.plot(x,Q2_dot_dot,label='q2""')
plt.plot(x,Q3_dot_dot,label='q3""')
plt.title('$JointAcceleration-Time Graph$')
# fig.subplots_adjust(hspace=.5,wspace=0.5)
plt.legend()
plt.show()
