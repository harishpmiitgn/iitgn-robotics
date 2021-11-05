import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

# Parameters
# For Arm1
l1 = 1   # m, Length of arm1
r1 = l1/2
R1 = 0.02   # m, Radius of Arm 1
D1 = 400    # kg/m^3, Density of Arm 1
m1 = (math.pi * R1 * R1 * l1) * D1      # kg, Mass of Arm 1
J1 = (1 / 3) * m1 * l1 * l1     # Inertia about end point for Arm1

# For Arm2
l2 = 1      # m,Length of Arm2
r2 = l2/2
R2 = 0.02       # m, Radius of Arm 2
D2 = 1000       # kg/m^3, Density of Arm 2
m2 = (math.pi * R2 * R2 * l2) * D2     # kg, Mass of Arm 2
J2 = (1 / 3) * m2 * l2 * l2             # Inertia about end point for Arm2]

# For Arm3
l3 = 1      # m,Length of Arm2
r3 = l3/2
R3 = 0.02       # m, Radius of Arm 2
D3 = 1000       # kg/m^3, Density of Arm 2
m3 = (math.pi * R3 * R3 * l3) * D3     # kg, Mass of Arm 2
J3 = (1 / 3) * m3 * l3 * l3             # Inertia about end point for Arm2]
E1 = []
E2 = []
E3 = []
T = []
g = 10
I = 0

tprev = 0


def ikpuma(x, y, z, d1, d2, d3):
    # using formulae from the textbook
    theta1 = math.atan(y/x)

    theta3 = math.acos((x**2 + y**2 + (z-d1)**2 - d2**2 - d3**2)/(2*d2*d3))
    theta2 = math.atan(z/((x**2 + y**2)**0.5) - math.atan((d3*math.sin(theta3))/(d2 + d3*math.cos(theta3))))

    return theta1, theta2, theta3


def integrator(E, T):
    n = len(E)
    I = 0
    if n > 50:
        for i in range(n-50, n, 1):
            I = I + E[i]*(T[i] - T[i-1])
    return I


# function that returns dy/dt
def model(y, t, m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb, I):
    theta1 = y[0]
    theta1d = y[1]
    theta2 = y[2]
    theta2d = y[3]
    theta3 = y[4]
    theta3d = y[5]

    d_theta1 = yda[0] + ((ydb[0] - yda[0])*t)/50
    d_theta2 = yda[1] + ((ydb[1] - yda[1]) * t) / 50
    d_theta3 = yda[2] + ((ydb[2] - yda[2]) * t) / 50

    a1 = m2*r2*r2 + m3*l2*l2
    a2 = m3*r3*r3
    a3 = m3*r3*l2
    b1 = (m2*r2 + m3*l2)*g
    b2 = m3*r3*g

    e1 = d_theta1 - theta1
    E1.append(e1)

    e2 = d_theta2 - theta2
    E2.append(e2)

    e3 = d_theta3 - theta3
    E3.append(e3)

    T.append(t)

    I1 = integrator(E1, T)
    I2 = integrator(E2, T)
    I3 = integrator(E3, T)
    print(e1)

    m11 = a1*np.cos(theta2)**2 + a2*np.cos(theta3+theta3)**2 + 2*a3*np.cos(theta2)*np.cos(theta2+theta3)+J1
    m22 = a1 + a2 + 2*a3*np.cos(theta3) + J2
    m33 = a2 + J3
    m23 = a2 + a3*np.cos(theta3)
    m32 = a2 + a3 * np.cos(theta3)

    b11 = (-1/2)*a1*theta1d*np.sin(2*theta2) - (1/2)*a2*(theta2d + theta3d)*np.sin(2*theta2 + 2*theta3) \
          - a3*theta2d*np.sin(2*theta2 + theta3) - a3*theta3d*np.cos(theta2)*np.sin(theta2 + theta3)

    b12 = (-1/2)*a1*theta1d*np.sin(2*theta2) - (1/2)*a2*theta1d*np.sin(2*theta2 + 2*theta3) \
          - a3*theta1d*np.sin(2*theta2 + theta3)
    b13 = (-1/2)*a2*theta1d*np.sin(2*theta2 + 2*theta3) - a3*theta1d*np.sin(theta2 + theta3)
    b21 = -b12
    b22 = -a3*theta3d*np.sin(theta3)
    b23 = -a3*(theta2d + theta3d)*np.sin(theta3)
    b31 = -b13
    b32 = a3*theta2d*np.sin(theta3)
    b33 = 0

    u1 = 10*e1 + 1*I1
    u2 = b1*np.cos(theta2) + b2*np.cos(theta2 + theta3) + 10*e2 + 0.01*I2
    u3 = b2*np.cos(theta2 + theta3) + 100*e3 + 0.010*I3

    MM = np.array([[m11, 0, 0],
                   [0, m22, m23],
                   [0, m32, m33]])
    C = np.array([[b11, b12, b13],
                  [b21, b22, b23],
                  [b31, b32, b33]])
    G = np.transpose(np.array([0, b1*np.cos(theta2) + b2*np.cos(theta2 + theta3), b2*np.cos(theta2 + theta3)]))

    U = np.transpose(np.array([u1, u2, u3]))

    W = (U - np.matmul(C, np.transpose([theta1d, theta2d, theta3d])) - G)

    dydt = np.matmul(np.linalg.inv(MM), W)

    yd = np.array([theta1d, dydt[0], theta2d, dydt[1], theta3d, dydt[2]])

    return yd


# initial condition
A = [1, 0, 0]
B = [1, 1, 1.1]

yda = ikpuma(A[0], A[1], A[2], l1, l2, 1)
ydb = ikpuma(B[0], B[1], B[2], l1, l2, 1)

y0 = np.array([yda[0], 0, yda[1], 0, yda[2], 0])

# time points
t = np.linspace(0, 50, num=1000)

# solve ODE
y = odeint(model, y0, t, args=(m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb, I))
print(yda, ydb)


theta1 = y[:, 0]
theta2 = y[:, 2]
theta3 = y[:, 4]

fig = plt.figure()
plt.plot(t, theta1, color='r', label='Phi')
fig.suptitle('End Effector motion from A = [1, 0, 0]:(0.0, -0.665, 1.570) , B = [1, 1, 1.1]:(0.785, -0.005, 1.565) for PUMA Robot using PI Controller')
plt.xlabel('time')
plt.ylabel('theta1')

plt.grid()
plt.show()
print(yda, ydb)





