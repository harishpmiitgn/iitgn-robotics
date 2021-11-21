import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
from numpy import sin, cos, pi, outer, ones, size, linspace

# Parameters
# For Arm1
l1 = 0.3   # m, Length of arm1
r1 = l1/2
R1 = 0.02   # m, Radius of Arm 1
D1 = 400    # kg/m^3, Density of Arm 1
m1 = (math.pi * R1 * R1 * l1) * D1      # kg, Mass of Arm 1
J1 = (1 / 3) * m1 * l1 * l1     # Inertia about end point for Arm1

# For Arm2
l2 = 0.2     # m,Length of Arm2
r2 = l2/2
R2 = 0.02       # m, Radius of Arm 2
D2 = 1000       # kg/m^3, Density of Arm 2
m2 = (math.pi * R2 * R2 * l2) * D2     # kg, Mass of Arm 2
J2 = (1 / 3) * m2 * l2 * l2             # Inertia about end point for Arm2]

# For Arm3
l3 = 0.1      # m,Length of Arm2
r3 = l3/2
R3 = 0.02       # m, Radius of Arm 2
D3 = 1000       # kg/m^3, Density of Arm 2
m3 = (math.pi * R3 * R3 * l3) * D3     # kg, Mass of Arm 2
J3 = (1 / 3) * m3 * l3 * l3            # Inertia about end point for Arm2]

# Motor dynamics

J = np.array([[0.01, 0, 0], [0.01, 0, 0], [0.01, 0, 0]])
B = np.array([[0.001, 0, 0], [0.001, 0, 0], [0.001, 0, 0]])

E1 = []
E2 = []
E3 = []
T = []
g = 9.81


def inversek(x, y, z, l1, l2, l3):
    theta2 = np.arccos((x*x + y*y - l1*l1 - l2*l2)/(2*l1*l2))
    theta1 = np.arctan(y/x) - np.arctan((l2*np.sin(theta2)/(l1 + l2*np.cos(theta2))))

    d = 0.3 - l3 - z

    return theta1, theta2, d


def inversevelocity(theta1, theta2, yd, l1, l2, l3):
    theta1d = (np.sin(theta1+theta2)/(l1*np.sin(theta2)))*yd
    theta2d = ((l2*np.sin(theta1 + theta2) + l1*np.sin(theta1)) / (l1 * l2* np.sin(theta2))) * yd
    return theta1d, theta2d


def inverseacceleation(theta1, theta2, theta1d, theta2d, yd, ydd, l1, l2, l3):
    theta1dd = ydd*(np.sin(theta1+theta2)/(l1*np.sin(theta2))) + \
               ((np.sin(theta2)*cos(theta1+theta2)*(theta1d + theta2d) - np.sin(theta1+theta2)*np.cos(theta2)*theta2d)*yd)/(l1*np.sin(theta2)*np.sin(theta2))

    theta2dd = -theta1dd - (ydd*np.sin(theta1))/(l2*np.sin(theta2)) + (yd*np.sin(theta1 - theta2))/(l2*np.sin(theta2)*np.sin(theta2))

    return theta1dd, theta2dd


# function that returns dy/dt
def modelD(y, t, m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb):
    theta1 = y[0]
    theta1d = y[1]
    theta2 = y[2]
    theta2d = y[3]
    d = y[4]
    dd = y[5]

    y_des = 0.06 - (0.15 * t * t) / (tf * tf) + (0.1 * t * t * t) / (tf * tf * tf)
    yd_des = -(0.3 * t) / (tf * tf) + (0.3 * t * t) / (tf * tf * tf)
    ydd_des = -0.3 / (tf * tf) + (0.6 * t) / (tf * tf * tf)

    Q = inversek(0.4, y_des, 0.1, l1, l2, l3)
    d_theta1 = Q[0]
    d_theta2 = Q[1]
    d_d = Q[2]

    Qd = inversevelocity(d_theta1, d_theta2, yd_des, l1, l2, l3)
    d_theta1d = Qd[0]
    d_theta2d = Qd[1]

    Qdd = inverseacceleation(d_theta1, d_theta2, d_theta1d, d_theta2d, yd_des, ydd_des, l1, l2, l3)
    d_theta1dd = Qdd[0]
    d_theta2dd = Qdd[1]

    alpha = J1 + r1 * r1 * m1 + l1 * l1 * m2 + l1 * l1 * m3
    beta = J2 + J3 + l2 * l2 * m3 + m2 * r2 * r2
    gamma = l1 * l2 * m3 + l1 * r2 * m2

    e1 = d_theta1 - theta1
    E1.append(e1)
    e1_d = d_theta1d - theta1d

    e2 = d_theta2 - theta2
    e2_d = d_theta2d - theta2d
    E2.append(e2)

    e3 = d_d - d
    E3.append(e3)
    e3_d = 0 - dd

    if (t <= 5.005) and (t > 4.995):
        impulse1 = -1
        print(impulse1)
    else:
        impulse1 = 0

    if (t <= 15.005) and (t > 14.995):
        impulse2 = -1
        print(impulse2)
    else:
        impulse2 = 0

    Kp = np.array([[625, 0, 0],
                   [0, 625, 0],
                   [0, 0, 0.15]])

    Kd = np.array([[50, 0, 0],
                   [0, 50, 0],
                   [0, 0, 20]])

    # Dynamics equations from: Dynamic modeling of SCARA robot based on Udwadia–Kalaba theory by Yaru Xu and Rong Liu

    MM = np.array([[alpha + beta + 2*gamma*np.cos(theta2), beta + 2*gamma*np.cos(theta2), 0],
                   [beta + 2*gamma*np.cos(theta2), beta, 0],
                   [0, 0, m3]]) + J
    C = np.array([[-gamma*np.sin(theta2)*theta2d, -gamma*np.sin(theta2)*(theta2d + theta1d), 0],
                  [gamma*np.sin(theta2)*theta1d, 0, 0],
                  [0, 0, 0]]) + B
    G = np.transpose(np.array([0, 0, m3*g]))

    aq = np.transpose([d_theta1dd, d_theta2dd, 0]) - np.matmul(Kd, np.transpose([-e1_d, -e2_d, -e3_d])) - np.matmul(Kp, np.transpose([-e1, -e2, -e3]))

    U = np.matmul(MM, aq) + np.matmul(C, np.transpose([theta1d, theta2d, dd])) + G + np.transpose([impulse1, impulse2, 0])

    W = (U - np.matmul(C, np.transpose([theta1d, theta2d, dd])) - G)

    dydt = np.matmul(np.linalg.inv(MM), W)

    yd = np.array([theta1d, dydt[0], theta2d, dydt[1], dd, dydt[2]])
    return yd


def modelC(y, t, m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb):
    theta1 = y[0]
    theta1d = y[1]
    theta2 = y[2]
    theta2d = y[3]
    d = y[4]
    dd = y[5]

    y_des = 0.06 - (0.15 * t * t) / (tf * tf) + (0.1 * t * t * t) / (tf * tf * tf)
    yd_des = -(0.3 * t) / (tf * tf) + (0.3 * t * t) / (tf * tf * tf)
    ydd_des = -0.3 / (tf * tf) + (0.6 * t) / (tf * tf * tf)

    Q = inversek(0.4, y_des, 0.1, l1, l2, l3)
    d_theta1 = Q[0]
    d_theta2 = Q[1]
    d_d = Q[2]

    Qd = inversevelocity(d_theta1, d_theta2, yd_des, l1, l2, l3)
    d_theta1d = Qd[0]
    d_theta2d = Qd[1]

    Qdd = inverseacceleation(d_theta1, d_theta2, d_theta1d, d_theta2d, yd_des, ydd_des, l1, l2, l3)
    d_theta1dd = Qdd[0]
    d_theta2dd = Qdd[1]

    alpha = J1 + r1 * r1 * m1 + l1 * l1 * m2 + l1 * l1 * m3
    beta = J2 + J3 + l2 * l2 * m3 + m2 * r2 * r2
    gamma = l1 * l2 * m3 + l1 * r2 * m2

    e1 = d_theta1 - theta1
    E1.append(e1)
    e1_d = d_theta1d - theta1d

    e2 = d_theta2 - theta2
    e2_d = d_theta2d - theta2d
    E2.append(e2)

    e3 = d_d - d
    E3.append(e3)
    e3_d = 0 - dd

    if (t <= 5.005) and (t > 4.995):
        impulse1 = -1
        print(impulse1)
    else:
        impulse1 = 0

    if (t <= 15.005) and (t > 14.995):
        impulse2 = -1
        print(impulse2)
    else:
        impulse2 = 0


    # Computed torque calculation:

    MM_Des = np.array([[0, beta + 2*gamma*np.cos(d_theta2), 0],
                   [beta + 2*gamma*np.cos(d_theta2), 0, 0],
                   [0, 0, 0]])

    C_des = np.array([[-gamma * np.sin(d_theta2) * d_theta2d, -gamma * np.sin(d_theta2) * (d_theta2d + d_theta1d), 0],
                  [gamma * np.sin(d_theta2) * d_theta1d, 0, 0],
                  [0, 0, 0]]) + B

    G_des = np.transpose(np.array([0, 0, m3 * g]))

    # Dynamics equations from: Dynamic modeling of SCARA robot based on Udwadia–Kalaba theory by Yaru Xu and Rong Liu

    u1 = 2 * e1 + 1 * e1_d + (alpha + beta + 2 * gamma * np.cos(d_theta2) + 0.01) * d_theta1dd + np.matmul(MM_Des[0], np.transpose([d_theta1dd, d_theta2dd, 0])) + \
         np.matmul(C_des[0], np.transpose([d_theta1d, d_theta2d, 0])) + impulse1
    u2 = 2 * e2 + 1 * e2_d + (beta + 0.01)*d_theta2dd + np.matmul(MM_Des[1], np.transpose([d_theta1dd, d_theta2dd, 0])) + \
         np.matmul(C_des[1], np.transpose([d_theta1d, d_theta2d, 0])) + impulse2
    u3 = 50 * e3 + 0.001 * e3_d + G_des[2]

    MM = np.array([[alpha + beta + 2*gamma*np.cos(theta2), beta + 2*gamma*np.cos(theta2), 0],
                   [beta + 2*gamma*np.cos(theta2), beta, 0],
                   [0, 0, m3]]) + J
    C = np.array([[-gamma*np.sin(theta2)*theta2d, -gamma*np.sin(theta2)*(theta2d + theta1d), 0],
                  [gamma*np.sin(theta2)*theta1d, 0, 0],
                  [0, 0, 0]]) + B
    G = np.transpose(np.array([0, 0, m3*g]))

    U = np.transpose(np.array([u1, u2, u3]))

    W = (U - np.matmul(C, np.transpose([theta1d, theta2d, dd])) - G)

    dydt = np.matmul(np.linalg.inv(MM), W)

    yd = np.array([theta1d, dydt[0], theta2d, dydt[1], dd, dydt[2]])
    return yd


# function that returns dy/dt
def modelB(y, t, m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb):
    theta1 = y[0]
    theta1d = y[1]
    theta2 = y[2]
    theta2d = y[3]
    d = y[4]
    dd = y[5]

    y_des = 0.06 - (0.15 * t * t) / (tf * tf) + (0.1 * t * t * t) / (tf * tf * tf)
    yd_des = -(0.3 * t) / (tf * tf) + (0.3 * t * t) / (tf * tf * tf)
    ydd_des = -0.3 / (tf * tf) + (0.6 * t) / (tf * tf * tf)

    Q = inversek(0.4, y_des, 0.1, l1, l2, l3)
    d_theta1 = Q[0]
    d_theta2 = Q[1]
    d_d = Q[2]

    Qd = inversevelocity(d_theta1, d_theta2, yd_des, l1, l2, l3)
    d_theta1d = Qd[0]
    d_theta2d = Qd[1]

    Qdd = inverseacceleation(d_theta1, d_theta2, d_theta1d, d_theta2d, yd_des, ydd_des, l1, l2, l3)
    d_theta1dd = Qdd[0]
    d_theta2dd = Qdd[1]

    alpha = J1 + r1 * r1 * m1 + l1 * l1 * m2 + l1 * l1 * m3
    beta = J2 + J3 + l2 * l2 * m3 + m2 * r2 * r2
    gamma = l1 * l2 * m3 + l1 * r2 * m2

    if (t <= 5.005) and (t > 4.995):
        impulse1 = -1
        print(impulse1)
    else:
        impulse1 = 0

    if (t <= 15.005) and (t > 14.995):
        impulse2 = -1
        print(impulse2)
    else:
        impulse2 = 0


    e1 = d_theta1 - theta1
    E1.append(e1)
    e1_d = d_theta1d - theta1d

    e2 = d_theta2 - theta2
    e2_d = d_theta2d - theta2d
    E2.append(e2)

    e3 = d_d - d
    E3.append(e3)
    e3_d = 0 - dd

    # Dynamics equations from: Dynamic modeling of SCARA robot based on Udwadia–Kalaba theory by Yaru Xu and Rong Liu

    u1 = 2 * e1 + 1 * e1_d + (alpha + beta + 2*gamma*np.cos(d_theta2) + 0.01)*d_theta1dd + impulse1
    u2 = 2 * e2 + 1 * e2_d + + (beta + 0.01)*d_theta2dd + impulse2
    u3 = 50*e3 + 0.001*e3_d + m3*g + m3*0

    MM = np.array([[alpha + beta + 2*gamma*np.cos(theta2), beta + 2*gamma*np.cos(theta2), 0],
                   [beta + 2*gamma*np.cos(theta2), beta, 0],
                   [0, 0, m3]]) + J
    C = np.array([[-gamma*np.sin(theta2)*theta2d, -gamma*np.sin(theta2)*(theta2d + theta1d), 0],
                  [gamma*np.sin(theta2)*theta1d, 0, 0],
                  [0, 0, 0]]) + B
    G = np.transpose(np.array([0, 0, m3*g]))

    U = np.transpose(np.array([u1, u2, u3]))

    W = (U - np.matmul(C, np.transpose([theta1d, theta2d, dd])) - G)

    dydt = np.matmul(np.linalg.inv(MM), W)

    yd = np.array([theta1d, dydt[0], theta2d, dydt[1], dd, dydt[2]])
    return yd


# function that returns dy/dt
def modelA(y, t, m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb):
    theta1 = y[0]
    theta1d = y[1]
    theta2 = y[2]
    theta2d = y[3]
    d = y[4]
    dd = y[5]

    y_des = 0.06 - (0.15 * t * t) / (tf * tf) + (0.1 * t * t * t) / (tf * tf * tf)
    yd_des = -(0.3 * t) / (tf * tf) + (0.3 * t * t) / (tf * tf * tf)

    Q = inversek(0.4, y_des, 0.1, l1, l2, l3)
    d_theta1 = Q[0]
    d_theta2 = Q[1]
    d_d = Q[2]

    Qd = inversevelocity(d_theta1, d_theta2, yd_des, l1, l2, l3)
    d_theta1d = Qd[0]
    d_theta2d = Qd[1]

    alpha = J1 + r1 * r1 * m1 + l1 * l1 * m2 + l1 * l1 * m3
    beta = J2 + J3 + l2 * l2 * m3 + m2 * r2 * r2
    gamma = l1 * l2 * m3 + l1 * r2 * m2

    if (t <= 5.005) and (t > 4.995):
        impulse1 = -1
        print(impulse1)
    else:
        impulse1 = 0

    if (t <= 15.005) and (t > 14.995):
        impulse2 = -1
        print(impulse2)
    else:
        impulse2 = 0

    e1 = d_theta1 - theta1
    E1.append(e1)
    e1_d = d_theta1d - theta1d

    e2 = d_theta2 - theta2
    e2_d = d_theta2d - theta2d
    E2.append(e2)

    e3 = d_d - d
    E3.append(e3)
    e3_d = 0 - dd

    # Dynamics equations from: Dynamic modeling of SCARA robot based on Udwadia–Kalaba theory by Yaru Xu and Rong Liu

    u1 = 2 * e1 + 1 * e1_d + impulse1
    u2 = 2 * e2 + 1 * e2_d + impulse2
    u3 = 50 * e3 + 0.001 * e3_d + m3 * g

    MM = np.array([[alpha + beta + 2*gamma*np.cos(theta2), beta + 2*gamma*np.cos(theta2), 0],
                   [beta + 2*gamma*np.cos(theta2), beta, 0],
                   [0, 0, m3]])
    C = np.array([[-gamma*np.sin(theta2)*theta2d, -gamma*np.sin(theta2)*(theta2d + theta1d), 0],
                  [gamma*np.sin(theta2)*theta1d, 0, 0],
                  [0, 0, 0]])
    G = np.transpose(np.array([0, 0, m3*g]))

    U = np.transpose(np.array([u1, u2, u3]))

    W = (U - np.matmul(C, np.transpose([theta1d, theta2d, dd])) - G)

    dydt = np.matmul(np.linalg.inv(MM), W)

    yd = np.array([theta1d, dydt[0], theta2d, dydt[1], dd, dydt[2]])
    return yd


# initial condition
A = [0.4, 0.06, 0.1]
B = [0.4, 0.01, 0.1]

yda = inversek(A[0], A[1], A[2], l1, l2, l3)
ydb = inversek(B[0], B[1], B[2], l1, l2, 3)

y0 = np.array([yda[0], 0, yda[1], 0, yda[2], 0])

tf = 20


# time points
t = np.linspace(0, tf, num=1000)

# solve ODE
yA = odeint(modelA, y0, t, args=(m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb), hmax=1e-3)
yB = odeint(modelB, y0, t, args=(m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb), hmax=1e-3)
yC = odeint(modelC, y0, t, args=(m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb), hmax=1e-3)
yD = odeint(modelD, y0, t, args=(m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb), hmax=1e-3)


theta1D = yD[:, 0]
theta2D = yD[:, 2]
dD = yD[:, 4]

theta1C = yC[:, 0]
theta2C = yC[:, 2]
dC = yC[:, 4]

theta1B = yB[:, 0]
theta2B = yB[:, 2]
dB = yB[:, 4]

theta1A = yA[:, 0]
theta2A = yA[:, 2]
dA = yA[:, 4]


def forwardKinematicsx(theta1, theta2, l1, l2, l3):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    return x


def forwardKinematicsy(theta1, theta2, l1, l2, l3):
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return y


d_theta1 = np.zeros(1000)
d_theta2 = np.zeros(1000)
d_d = np.zeros(1000)
Ay = np.zeros(1000)
By = np.zeros(1000)
Cy = np.zeros(1000)
Dy = np.zeros(1000)
Ax = np.zeros(1000)
Bx = np.zeros(1000)
Cx = np.zeros(1000)
Dx = np.zeros(1000)
yd = np.zeros(1000)
xd = np.ones(1000)*0.4

for i in range(1000):
    yd[i] = 0.06 - (0.15*t[i]*t[i])/(tf*tf) + (0.1*t[i]*t[i]*t[i])/(tf*tf*tf)

    Q = inversek(0.4, yd[i], 0.1, l1, l2, l3)
    d_theta1[i] = Q[0]
    d_theta2[i] = Q[1]
    d_d[i] = Q[2]

    Ax[i] = forwardKinematicsx(theta1A[i], theta2A[i], l1, l2, l3)
    Bx[i] = forwardKinematicsx(theta1B[i], theta2B[i], l1, l2, l3)
    Cx[i] = forwardKinematicsx(theta1C[i], theta2C[i], l1, l2, l3)
    Dx[i] = forwardKinematicsx(theta1D[i], theta2D[i], l1, l2, l3)
    Ay[i] = forwardKinematicsy(theta1A[i], theta2A[i], l1, l2, l3)
    By[i] = forwardKinematicsy(theta1B[i], theta2B[i], l1, l2, l3)
    Cy[i] = forwardKinematicsy(theta1C[i], theta2C[i], l1, l2, l3)
    Dy[i] = forwardKinematicsy(theta1D[i], theta2D[i], l1, l2, l3)

fig1 = plt.figure()
plt.plot(t, Ay, label='theta1')
plt.plot(t, By, label='theta1')
plt.plot(t, Cy, label='theta1')
plt.plot(t, Dy, label='theta1')
plt.plot(t, yd, color='y', label='theta1', alpha=0.3, linewidth=4)
plt.xlabel('t (sec)')
plt.ylabel('y (m)')
plt.legend(['Ay', 'By', 'Cy', 'Dy', 'desired-y-Trajectory'])

fig1.suptitle('Problem6: End Effector motion from A SCARA Robot using VARIOUS Controller, with -1 Nm impulse at Joint '
             '1 @5 sec and -1 Nm impulse at Joint 2 @15 sec')

plt.grid()
plt.show()

fig2 = plt.figure()
plt.plot(t, Ax, label='theta1')
plt.plot(t, Bx, label='theta1')
plt.plot(t, Cx, label='theta1')
plt.plot(t, Dx, label='theta1')
plt.plot(t, xd, color='y', label='theta1', alpha=0.3, linewidth=4)
plt.legend(['xA', 'xB', 'xC', 'xD', 'desired-x'])
plt.xlabel('t (sec)')
plt.ylabel('x (m)')
fig2.suptitle('Problem6: End Effector motion from A SCARA Robot using VARIOUS Controller, with -1 Nm impulse at Joint '
             '1 @5 sec and -1 Nm impulse at Joint 2 @15 sec')

plt.grid()
plt.show()


fig3 = plt.figure()
plt.plot(Ax, Ay, label='theta1')
plt.plot(Bx, By, label='theta1')
plt.plot(Cx, Cy, label='theta1')
plt.plot(Dx, Dy, label='theta1')
plt.plot(xd, yd, color='y', label='theta1', alpha=0.3, linewidth=4)
plt.legend(['A', 'B', 'C', 'D', 'desired-Trajectory'])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
fig3.suptitle('Problem6: End Effector motion from A SCARA Robot using VARIOUS Controller, with -1 Nm impulse at Joint '
             '1 @5 sec and -1 Nm impulse at Joint 2 @15 sec')

plt.grid()
plt.show()

fig4 = plt.figure()
plt.plot(t, theta1A, label='theta1')
plt.plot(t, theta1B, label='theta1')
plt.plot(t, theta1C, label='theta1')
plt.plot(t, theta1D, label='theta1')
plt.plot(t, d_theta1, color='y', label='theta1', alpha=0.3, linewidth=4)
plt.legend(['theta1A', 'theta1B', 'theta1C', 'theta1D', 'desired-theta1'])
plt.xlabel('t (sec)')
plt.ylabel('theta1 (rad)')
fig4.suptitle('Problem6: End Effector motion from A SCARA Robot using VARIOUS Controller, with -1 Nm impulse at Joint '
             '1 @5 sec and -1 Nm impulse at Joint 2 @15 sec')

plt.grid()
plt.show()

fig5 = plt.figure()
plt.plot(t, theta2A, label='theta1')
plt.plot(t, theta2B, label='theta1')
plt.plot(t, theta2C, label='theta1')
plt.plot(t, theta2D, label='theta1')
plt.plot(t, d_theta2, color='y', label='theta1', alpha=0.3, linewidth=4)
plt.legend(['theta2A', 'theta2B', 'theta2C', 'theta2D', 'desired-theta2'])
plt.xlabel('t (sec)')
plt.ylabel('theta2 (rad)')


fig5.suptitle('Problem6: End Effector motion from A SCARA Robot using VARIOUS Controller, with -1 Nm impulse at Joint '
             '1 @5 sec and -1 Nm impulse at Joint 2 @15 sec')

plt.grid()
plt.show()