import math
import scipy
import sympy as sp
from sympy.core.evalf import N
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 
from scipy.integrate import solve_ivp, ode
from scipy.optimize import fsolve

def func(t,y):
    q1=y[0]
    q1_dot=y[1]
    q2=y[2]
    q2_dot=y[3]
    q3=y[4]
    q3_dot=y[5]
    q_dot=[q1_dot, q2_dot, q3_dot]
    q=[q1,q2,q3]

    temp=eqn.subs([('g',9.8),('l1',0.25),('l2',0.25),('l3',0.25),('m1',1),('m2',1),('m3',1),('q1_dot',q_dot[0]),('q2_dot',q_dot[1]),('q3_dot',q_dot[2]), ('q1',q[0]),('q2',q[1]),('q3',q[2])])
    # temp=fsolve(equation_solve, (0, 0, 0), ([q1,q2,q3], [q1_dot,q2_dot,q3_dot]),xtol=1)
    
    # print(temp.shape)
    a1=temp[0][0].coeff('q1_dot_dot')
    b1=temp[0][0].coeff('q2_dot_dot')
    c1=temp[0][0].coeff('q3_dot_dot')
    # d1=temp[0][0].coeff('1')
    a2=temp[1][0].coeff('q1_dot_dot')
    b2=temp[1][0].coeff('q2_dot_dot')
    c2=temp[1][0].coeff('q3_dot_dot')
    # d2=temp[1][0].coeff('1')
    a3=temp[2][0].coeff('q1_dot_dot')
    b3=temp[2][0].coeff('q2_dot_dot')
    c3=temp[2][0].coeff('q3_dot_dot')
    # d3=temp[2][0].coeff('1')

    d1=temp[0][0].as_coefficients_dict()[1]+dist1
    d2=temp[1][0].as_coefficients_dict()[1]+dist2
    d3=temp[2][0].as_coefficients_dict()[1]+dist3
    
    # print(d1,d2,d3)
    M=np.array([[a1, b1, c1],
                [a2, b2, c2],
                [a3, b3, c3]],dtype="float")
    T=np.array([[v1-d1],
                [v2-d2],
                [v3-d3]])
    temp=np.linalg.inv(M)@T
    # print(M)
    # print(M@temp-T)
    # print(temp)
    # print(t1,t2,f3)
    # temp=[t1,t2,f3]
    q1_dot_dot=temp[0][0]
    q2_dot_dot=temp[1][0]
    q3_dot_dot=temp[2][0]
    # print([q1_dot,q1_dot_dot[0], q2_dot,q2_dot_dot[0], q3_dot,q3_dot_dot[0]])
    return [q1_dot,q1_dot_dot, q2_dot,q2_dot_dot, q3_dot,q3_dot_dot]

def D_Calc():
    m1=sp.Symbol('m1')
    m2=sp.Symbol('m2')
    m3=sp.Symbol('m3')
    q1 = sp.Symbol('q1')
    q2 = sp.Symbol('q2')
    q3 = sp.Symbol('q3')
    Jv1=np.array([[0,0,0],
        [0,0,0],
        [0,0,0]])
    Jv2=np.array([[-1.0*0.25*sp.sin(q1)*sp.cos(q2)/2, -1.0*0.25*sp.sin(q2)*sp.cos(q1)/2,0],
        [1.0*0.25*sp.cos(q1)*sp.cos(q2)/2, -1.0*0.25*sp.sin(q1)*sp.sin(q2)/2,0],
        [0, -1.0*0.25*sp.cos(q2)/2,0]])
    Jv3=np.array([[-1.0*0.25*sp.sin(q1)*sp.cos(q2) - 1.0*0.25*sp.sin(q1)*sp.cos(q3 + q2)/2, -1.0*0.25*sp.sin(q2)*sp.cos(q1) - 1.0*0.25*sp.sin(q3 + q2)*sp.cos(q1)/2, -1.0*0.25*sp.sin(q3 + q2)*sp.cos(q1)/2],
        [1.0*0.25*sp.cos(q1)*sp.cos(q2) + 1.0*0.25*sp.cos(q1)*sp.cos(q3 + q2)/2, -1.0*0.25*sp.sin(q1)*sp.sin(q2) - 1.0*0.25*sp.sin(q1)*sp.sin(q3 + q2)/2, -1.0*0.25*sp.sin(q1)*sp.sin(q3 + q2)/2],
        [0, -1.0*0.25*sp.cos(q2) - 1.0*0.25*sp.cos(q3 + q2)/2, -1.0*0.25*sp.cos(q3 + q2)/2]])
    Jw1=np.array([[0,0,0],
        [0,0,0],
        [1,0,0]])
    Jw2=np.array([[0,0,0],
        [0,0,0],
        [0,1,0]])
    Jw3=np.array([[0,0,0],
        [0,0,0],
        [0,0,1]])
    R_0_1=np.array([[1.0*sp.cos(q1), 0, -1.0*sp.sin(q1)],
            [1.0*sp.sin(q1), 0 ,1.0*sp.cos(q1)],
            [0 ,-1.0, 0]])
    R_0_2=np.array([[1.0*sp.cos(q1)*sp.cos(q2), - 1.0*sp.sin(q2)*sp.cos(q1),-1.0*sp.sin(q1)],
            [1.0*sp.sin(q1)*sp.cos(q2),-1.0*sp.sin(q1)*sp.sin(q2),1.0*sp.cos(q1)],
            [-1.0*sp.sin(q2),-1.0*sp.cos(q2) ,0]])
    R_0_3=np.array([[1.0*sp.cos(q1)*sp.cos(q2+q3), -1.0*sp.cos(q1)*sp.sin(q2+q3),-1.0*sp.sin(q1)],
            [1.0*sp.sin(q1)*sp.cos(q2+q3), -1.0*sp.sin(q1)*sp.sin(q2+q3),1.0*sp.cos(q1)],        
            [1.0*sp.sin(q2+q3),-1.0*sp.cos(q2+q3),0]])
    I1=np.array([[m[0]*0.0625/12,0,0],
        [0,m[0]*0.0625/12,0],
        [0,0,0]])
    I2=np.array([[m[1]*0.0625/12,0,0],
        [0,m[1]*0.0625/12,0],
        [0,0,0]])
    I3=np.array([[m[2]*0.0625/12,0,0],
        [0,m[2]*0.0625/12,0],
        [0,0,0]])    
    # D=m[0]*Jv1.T@Jv1 + m[1]*Jv2.T@Jv2 + m[2]*Jv3.T@Jv3  
    D=m1*Jv1.T@Jv1 + m2*Jv2.T@Jv2 + m3*Jv3.T@Jv3+Jw1.T@R_0_1@I1@R_0_1.T@Jw1 + Jw2.T@R_0_2@I2@R_0_2.T@Jw2 + Jw3.T@R_0_3@I3@R_0_3.T@Jw3
    # print(D)
    # print(D.shape)
    return D

def ComputeEOM(D,n):
    m1=sp.Symbol('m1')
    m2=sp.Symbol('m2')
    m3=sp.Symbol('m3')
    l1=sp.Symbol('l1')
    l2=sp.Symbol('l2')
    l3=sp.Symbol('l3')
    g=sp.Symbol('g')
    q1 = sp.Symbol('q1')
    q1_dot = sp.Symbol('q1_dot')
    q1_dot_dot = sp.Symbol('q1_dot_dot')
    q2 = sp.Symbol('q2')
    q2_dot = sp.Symbol('q2_dot')
    q2_dot_dot = sp.Symbol('q2_dot_dot')
    q3 = sp.Symbol('q3')
    q3_dot = sp.Symbol('q3_dot')
    q3_dot_dot = sp.Symbol('q3_dot_dot')
    
    phi=[0]*n
    c = [[[0] * n] * n] * n
    Tau=[0]*n
    d=0
    ct=0

    q=np.array([[q1],
                [q2],
                [q3]])
    q_dot=np.array([[q1_dot],
                    [q2_dot],
                    [q3_dot]])
    q_dot_dot=np.array([[q1_dot_dot],
                        [q2_dot_dot],
                        [q3_dot_dot]])
    V=m1*g*l1/2 + m2*g*(l1+l2/2*sp.sin(q[1][0])) + m3*g*(l1+l2*sp.sin(q[1][0])+l3/2*sp.sin(q[1][0]+q[2][0]))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[k][j][i]=0.5*(sp.diff(D[i][j], q[k][0]) + sp.diff(D[i][k], q[j][0]) - sp.diff(D[k][j], q[i][0]))
    for i in range(n):
        phi[i]=sp.diff(V,q[i][0])
        for j in range(n):
            d += D[i][j]*q_dot_dot[j][0]
            for k in range(n):
                ct+= c[k][j][i]*q_dot[k][0]*q_dot[j][0]
        Tau[i]=d+ct+phi[i]
    Tau_final=np.array([[Tau[0]],
                        [Tau[1]],
                        [Tau[2]]])
    motor_dynamics=np.array([[Jm[0]/r[0]*q1_dot_dot + (Bm[0]+kb[0]*km[0]/R[0])/r[0]*q1_dot],
                             [Jm[1]/r[1]*q2_dot_dot + (Bm[1]+kb[1]*km[1]/R[1])/r[1]*q2_dot],
                             [Jm[2]/r[2]*q3_dot_dot + (Bm[2]+kb[2]*km[2]/R[2])/r[2]*q3_dot]])
    Tau_final=Tau_final+motor_dynamics
    for i in range(len(Tau_final)):
        Tau_final[i]=Tau_final[i]*R[i]/km[i]
    eqn=sp.Array(Tau_final)
    return eqn


def simulate(q1,q2,q3,v1,v2,v3,dt):
    # print(ode_eqn.t)
    newstate=ode_eqn.integrate(ode_eqn.t+dt)
    q1=newstate[0]
    q1_dot=newstate[1]
    q2=newstate[2]
    q2_dot=newstate[3]
    q3=newstate[4]
    q3_dot=newstate[5]
    # print(q1)
    return q1,q2,q3, q1_dot,q2_dot,q3_dot

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
    q3=np.arccos(((d[2,0]-l1)**2+(d[0,0]/np.cos(q1))**2-l2**2-l3**2)/(2*l2*l3))
    q2=np.arctan2(d[2,0]-l1,d[0,0]/np.cos(q1))-np.arctan2(l3*np.sin(q3),l2+l3*np.cos(q3))
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
q_cont=Joint_angles(d=np.array([[0.4],
                                [0.06],
                                [0.1]]))
q1_cont,q2_cont,q3_cont=q_cont[0],q_cont[1],q_cont[2]
Jacob_cont=Jacobian(q_cont)
Jacob_cont_inv=np.linalg.pinv(Jacob_cont)
v_cont=np.array([[0],[v0],[0],[0],[0],[0]])
q_dot_cont=np.matmul(Jacob_cont_inv,v_cont)
q1_dot_cont,q2_dot_cont,q3_dot_cont=q_dot_cont[0][0][0],q_dot_cont[1][0][0],q_dot_cont[2][0][0]
ainit=np.array([[0],[a0],[0]])
q_dot_dot_cont=Joint_accel(ainit,q_cont,q_dot_cont)
q1_dot_dot_cont,q2_dot_dot_cont,q3_dot_dot_cont=q_dot_dot_cont[0][0],q_dot_dot_cont[1][0],q_dot_dot_cont[2][0]
r=np.array([1, 1, 1]) #gear ratio
Jm=np.array([1, 1, 1])
Bm=np.array([1, 1, 1])
kb=np.array([1, 1, 1])
km=np.array([1, 1, 1])
R=np.array([1, 1, 1])
m=np.array([[1],[1],[1]])

D=sp.simplify(D_Calc())
# print(D)
eq1=ComputeEOM(D,3)
# print(eq1)
# print(eq1.shape)
eqn=sp.simplify(eq1)
# print(eqn)
ode_eqn=ode(func).set_integrator('vode', nsteps=20, method='bdf')
state = [q_cont[0],0, q_cont[1],0, q_cont[2],0]
ode_eqn.set_initial_value(state,0)

kp=100
kd=19

ko=np.array([[100,0,0],
            [0,100,0],
            [0,0,100]])
k1=np.array([[20,0,0],
            [0,20,0],
            [0,0,20]])

xs_cont,ys_cont = [],[]
xs_d, ys_d = [],[]
times=[]
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
    q1_d,q2_d,q3_d=q[0],q[1],q[2]
    x_d=(0.25*np.cos(q2_d)+0.25*np.cos(q2_d+q3_d))*np.cos(q1_d)
    y_d=(0.25*np.cos(q2_d)+0.25*np.cos(q2_d+q3_d))*np.sin(q1_d)
    xs_d.append(x_d)
    ys_d.append(y_d)
    Jacob=Jacobian(q)
    Jacob_inv=np.linalg.pinv(Jacob)
    v=np.array([[0],
                [vd],
                [0],
                [0],
                [0],
                [0]])
    Joint_velocity=np.matmul(Jacob_inv, v)
    Q1_dot.append(Joint_velocity[0][0][0])
    Q2_dot.append(Joint_velocity[1][0][0])
    Q3_dot.append(Joint_velocity[2][0][0])
    q1_dot_d,q2_dot_d,q3_dot_d=Joint_velocity[0][0][0],Joint_velocity[1][0][0],Joint_velocity[2][0][0]

    Joint_acceleration=Joint_accel(acceleration,q,Joint_velocity)
    Q1_dot_dot.append(Joint_acceleration[0][0])
    Q2_dot_dot.append(Joint_acceleration[1][0])
    Q3_dot_dot.append(Joint_acceleration[2][0])
    q1_dot_dot_d,q2_dot_dot_d,q3_dot_dot_d=Joint_acceleration[0][0],Joint_acceleration[1][0],Joint_acceleration[2][0]

    rt1=q1_dot_dot_d + ko[0][0]*q1_d + k1[0][0]*q1_dot_d
    rt2=q2_dot_dot_d + ko[1][1]*q2_d + k1[1][1]*q2_dot_d
    rt3=q3_dot_dot_d + ko[2][2]*q3_d + k1[2][2]*q3_dot_d
    dist1=np.random.normal(0,0.1)*R[0]/km[0]
    dist2=np.random.normal(0,0.1)*R[1]/km[1]
    dist3=np.random.normal(0,0.1)*R[2]/km[2]
    v1=rt1+ kp*(q1_d-q1_cont)+kd*(q1_dot_d-q1_dot_cont)
    v2=rt2+kp*(q2_d-q2_cont)+kd*(q2_dot_d-q2_dot_cont)
    v3=rt3+kp*(q3_d-q3_cont)+kd*(q3_dot_d-q3_dot_cont)

    u=eqn.subs([('g',9.8),('l1',0.25),('l2',0.25),('l3',0.25),('m1',1),('m2',1),('m3',1),('q1_dot_dot',v1),('q2_dot_dot',v2),('q3_dot_dot',v3),('q1_dot',q1_dot_d),('q2_dot',q2_dot_d),('q3_dot',q3_dot_d),('q1',q1_d),('q2',q2_d),('q3',q3_d)])
    dist1=np.random.normal(0,0.1)*R[0]/km[0]
    dist2=np.random.normal(0,0.1)*R[1]/km[1]
    dist3=np.random.normal(0,0.1)*R[2]/km[2]
    V1=u[0][0]
    V2=u[1][0]
    V3=u[2][0]

    q1_cont,q2_cont,q3_cont, q1_dot_cont,q2_dot_cont,q3_dot_cont = simulate(q1_cont,q2_cont,q3_cont,V1,V2,V3,0.1)
    x_cont=(0.25*np.cos(q2_cont)+0.25*np.cos(q2_cont+q3_cont))*np.cos(q1_cont)
    y_cont=(0.25*np.cos(q2_cont)+0.25*np.cos(q2_cont+q3_cont))*np.sin(q1_cont)
    xs_cont.append(x_cont)
    ys_cont.append(y_cont)

    times.append(j)
    # plt.plot(times,ys_d,"blue", label="Desired Y coordinates")
    # plt.plot(times,ys_cont,"orange", label="PD Controlled ")

    # plt.pause(0.01)
print("Xs_d")
print()
print(xs_d)
print("Ys_d")
print()
print(ys_d)
print("Xs_cont")
print()
print(xs_cont)
print("Ys_cont")
print()
print(ys_cont)

# plt.show()
    
    



