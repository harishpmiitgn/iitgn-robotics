import cv2
import numpy as np
from numpy.lib.function_base import append
from scipy.integrate import solve_ivp, ode
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import sympy as sym 
from time import time
import pickle

def SCARA_invkin(p):
    x,y,z=p
    D=(x**2+y**2-l2**2-l3**2)/(2*l2*l3)
    q2=np.arctan2(np.sqrt(1-D**2),D)
    q1=np.arctan2(y,x)-np.arctan2(l3*np.sin(q2),l2+l3*np.cos(q2))
    
    d3=z-l1
    
    jp=np.array([np.rad2deg(q1),np.rad2deg(q2),d3])
    return q1,q2,d3

def SCARA_fkin(q1,q2,d3,k):
    n=3
    dh=np.array([[q1, l1, l2, 0],
                 [q2, 0, l3, 0],
                 [0, d3, 0, 0]])

    A=[0]*n
    for i in range(n):
        theta, d, a, alpha = dh[i]
        A[i]=np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                       [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                       [0, np.sin(alpha), np.cos(alpha), d],
                       [0, 0, 0, 1]])
    ans=np.identity(4)

    for i in range(k):
        ans=ans@A[i]
    pos=ans[0:3,3]
    return pos

def D_calculator():
    q1 = sym.Symbol('q1')
    q1_dot = sym.Symbol('q1_dot')
    q1_dot_dot = sym.Symbol('q1_dot_dot')
    q2 = sym.Symbol('q2')
    q2_dot = sym.Symbol('q2_dot')
    q2_dot_dot = sym.Symbol('q2_dot_dot')
    q3 = sym.Symbol('q3')
    q3_dot = sym.Symbol('q3_dot')
    q3_dot_dot = sym.Symbol('q3_dot_dot')

    jv1=np.array([[-l2/2*sym.sin(q1), 0, 0],
                  [l2/2*sym.cos(q1), 0, 0],
                  [0, 0, 0]])
    jv2=np.array([[-l2*sym.sin(q1)-l3/2*sym.sin(q1+q2), -l3/2*sym.sin(q1+q2), 0],
                  [l2*sym.cos(q1)+l3/2*sym.cos(q1+q2), l3/2*sym.cos(q1+q2), 0],
                  [0, 0, 0]])
    jv3=np.array([[-l2*sym.sin(q1)-l3*sym.sin(q1+q2), -l3*sym.sin(q1+q2), 0],
                  [l2*sym.cos(q1)+l3*sym.cos(q1+q2), l3*sym.cos(q1+q2), 0],
                  [0, 0, 1/2]])
    D1=m1*jv1.T@jv1+m2*jv2.T@jv2+m3*jv3.T@jv3
    D2_1=np.array([[I1, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    D2_2=np.array([[I2, I2, 0],
                   [I2, I2, 0],
                   [0, 0, 0]])
    D2=D2_1+D2_2
    D=D1+D2
    return D

def dynamical_equation(D):
    n=3
    q1 = sym.Symbol('q1')
    q1_dot = sym.Symbol('q1_dot')
    q1_dot_dot = sym.Symbol('q1_dot_dot')
    q2 = sym.Symbol('q2')
    q2_dot = sym.Symbol('q2_dot')
    q2_dot_dot = sym.Symbol('q2_dot_dot')
    q3 = sym.Symbol('q3')
    q3_dot = sym.Symbol('q3_dot')
    q3_dot_dot = sym.Symbol('q3_dot_dot')

    # D=np.array([[m1*l2**2/3+m2*l2**2, m2*l1*l2/2*sym.cos(q2-q1)],
    #             [m2*l1*l2/2*sym.cos(q2-q1), m2*l2**2/3]])
    # V=m1*g*l1/2*sym.sin(q1)+m2*g*(l1*sym.sin(q1)+l2/2*sym.sin(q2))
    V = g*(m1*l1+m2*l1+m3*(l1+q3/2))

    phi=np.array([[sym.diff(V, q1)],
                  [sym.diff(V, q2)],
                  [sym.diff(V, q3)]])
    q=np.array([[q1],
                [q2],
                [q3]])
    q_dot=np.array([[q1_dot],
                    [q2_dot],
                    [q3_dot]])
    q_dot_dot=np.array([[q1_dot_dot],
                        [q2_dot_dot],
                        [q3_dot_dot]])
    c=[0]*n
    for k in range(n):
        for i in range(n):
            for j in range(n):
                sum=sym.diff(D[k][j], "q"+str(i+1))+sym.diff(D[k][i], "q"+str(j+1))+sym.diff(D[i][j], "q"+str(k+1))
                c[k]+=0.5*(sum)*sym.Symbol("q"+str(i+1))*sym.Symbol("q"+str(j+1))

    motor_dynemics=np.array([[Jm[0]/r[0]*q1_dot_dot + (Bm[0]+kb[0]*km[0]/R[0])/r[0]*q1_dot],
                             [Jm[1]/r[1]*q2_dot_dot + (Bm[1]+kb[1]*km[1]/R[1])/r[1]*q2_dot],
                             [Jm[2]/r[2]*q3_dot_dot + (Bm[2]+kb[2]*km[2]/R[2])/r[2]*q3_dot]])
    final=D@q_dot_dot+phi+np.transpose([c]) + motor_dynemics
    for i in range(len(final)):
        final[i]=final[i]*R[i]/km[i]
    eqn=sym.Array(final)
    return eqn

def func(t,y):
    q1=y[0]
    q1_dot=y[1]
    q2=y[2]
    q2_dot=y[3]
    q3=y[4]
    q3_dot=y[5]
    q_dot=[q1_dot, q2_dot, q3_dot]
    q=[q1,q2,q3]

    temp=eqn.subs([('q1_dot',q_dot[0]),('q2_dot',q_dot[1]),('q3_dot',q_dot[2]), ('q1',q[0]),('q2',q[1]),('q3',q[2])])
    # temp=fsolve(equation_solve, (0, 0, 0), ([q1,q2,q3], [q1_dot,q2_dot,q3_dot]),xtol=1)
    # print(temp)
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

    d1=temp[0][0].as_coefficients_dict()[1]
    d2=temp[1][0].as_coefficients_dict()[1]
    d3=temp[2][0].as_coefficients_dict()[1]
    
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

    return [q1_dot,q1_dot_dot, q2_dot,q2_dot_dot, q3_dot,q3_dot_dot]

def simulate(q1,q2,d3,v1,v2,v3,dt):
    # q1+=0.1*t1+t2/100
    # q2+=0.1*t2+f3/10
    # d3+=0.1*f3+(t1+t2)/100
    newstate=ode_eqn.integrate(ode_eqn.t+dt)
    q1=newstate[0]
    q1_dot=newstate[1]
    q2=newstate[2]
    q2_dot=newstate[3]
    d3=newstate[4]
    d3_dot=newstate[5]
    # print(q1)
    return q1,q2,d3, q1_dot,q2_dot,d3_dot

def get_state(a,t):
    if(t>5):
        t=5

    a=a.T[0]
    x = 0.4
    y = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
    z = 0.1

    x_ = 0
    y_ = a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4
    z_ = 0

    x__ = 0
    y__ = 2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3
    z__ = 0

    pos=[x,y,z]
    pos_=[x_,y_,z_]
    pos__=[x__,y__,z__]
    return pos,pos_,pos__

def joit_params(q1,q2,d3,vx,vy,vz,ax,ay,az):
    Jv=np.array([[-l3*np.sin(q1+q2)-l2*np.sin(q1), -l3*np.sin(q1+q2), 0],
                 [l3*np.cos(q1+q2)+l2*np.cos(q1), l3*np.cos(q1+q2), 0],
                 [0, 0, 1]])
    v=np.array([[vx],
                [vy],
                [vz]])
    qs=np.linalg.inv(Jv)@v
    q_dot=qs.T[0]
    
    q1_dot,q2_dot,d3_dot = q_dot
    a=np.array([[ax + l3*np.cos(q1+q2)*(q1_dot+q2_dot)**2 + l2*np.cos(q1)*q1_dot**2],
                [ay + l3*np.sin(q1+q2)*(q1_dot+q2_dot)**2 + l2*np.sin(q1)*q1_dot**2],
                [az]])
    qds=np.linalg.inv(Jv)@a
    q_dot_dot=qds.T[0]
    q1_dot_dot,q2_dot_dot,d3_dot_dot = q_dot_dot
    return q1_dot,q2_dot,d3_dot, q1_dot_dot,q2_dot_dot,d3_dot_dot



y0=0.06
y_0=0
y__0=0
y1=0.01
y_1=0
y__1=0

t0=0
t1=5

b=np.array([[y0],
            [y_0],
            [y__0],
            [y1],
            [y_1],
            [y__1]])
M=np.array([[1, t0, t0**2, t0**3, t0**4, t0**5],
            [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
            [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
            [1, t1, t1**2, t1**3, t1**4, t1**5],
            [0, 1, 2*t1, 3*t1**2, 4*t1**3, 5*t1**4],
            [0, 0, 2, 6*t1, 12*t1**2, 20*t1**3]])
a=np.linalg.inv(M)@b

l1=0.25
l2=0.25
l3=0.25
m1=0.8
m2=0.8
m3=0.8
I1=0.005
I2=0.005
I3=0.8
g=9.81


pos,pos_,pos__ = get_state(a,0)
q1,q2,d3=SCARA_invkin(pos)
q1_dot,q2_dot,d3_dot, q1_dot_dot,q2_dot_dot,d3_dot_dot = joit_params(q1,q2,d3, pos_[0],pos_[1],pos_[2], pos__[0],pos__[1],pos__[2])

r=np.array([1, 1, 1]) #gear ratio
Jm=np.array([1, 1, 1])
Bm=np.array([1, 1, 1])
kb=np.array([1, 1, 1])
km=np.array([1, 1, 1])
R=np.array([1, 1, 1])

D=sym.simplify(D_calculator())
eqn1=dynamical_equation(D)
eqn=sym.simplify(eqn1)
print("dynamic equation with motor dynamics:")
print(eqn)
print()
ode_eqn=ode(func).set_integrator('vode', nsteps=20, method='bdf')
state = [q1,0, q2,0, d3,0]
ode_eqn.set_initial_value(state,0)

kp=100
kd=19

xs,ys,time = [],[],[]
xs_d, ys_d = [],[]
fig = plt.figure()
while(ode_eqn.t<6):
    pos,pos_,pos__ = get_state(a,ode_eqn.t)
    time.append(ode_eqn.t)
    q1_d,q2_d,d3_d=SCARA_invkin(pos)
    q1_dot_d,q2_dot_d,d3_dot_d, q1_dot_dot_d,q2_dot_dot_d,d3_dot_dot_d = joit_params(q1,q2,d3, pos_[0],pos_[1],pos_[2], pos__[0],pos__[1],pos__[2])

    v1 = kp*(q1_d-q1) + kd*(q1_dot_d-q1_dot)
    v2 = kp*(q2_d-q2) + kd*(q2_dot_d-q2_dot)
    v3 = kp*(d3_d-d3) + kd*(d3_dot_d-d3_dot)

    q1,q2,d3, q1_dot,q2_dot,d3_dot = simulate(q1,q2,d3,v1,v2,v3,0.1)
    x,y,z=SCARA_fkin(q1,q2,d3,3)

    xs.append(x)
    ys.append(y)
    xs_d.append(pos[0])
    ys_d.append(pos[1])
    plt.plot(time,ys_d,"black")
    plt.plot(time,ys,"red")

    if plt.waitforbuttonpress(0.01):
        plt.title('x vs time')
        plt.xlabel('time')
        plt.ylabel('x') 
        plt.legend(['y desired','y'])
        break

plt.show()

with open('data\\3a3.pkl', 'wb') as file:
    pickle.dump([xs_d,ys_d,xs,ys,time], file)

# myvar=0
# with open('data\scara7.pkl', 'rb') as file:
#     myvar = pickle.load(file)
# xs_d,ys_d,xs,ys,time=myvar