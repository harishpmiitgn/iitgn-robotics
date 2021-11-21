import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
from numpy import sin, cos, pi, outer, ones, size, linspace

# Parameters
# For Arm1
l1 = 0.3    # m, Length of arm1

# For Arm2
l2 = 0.2    # m,Length of Arm2

# For Arm3
l3 = 0.1      # m,Length of Arm2

tf = 20

t = np.linspace(0, tf, num= 1000)

y = np.zeros(1000)
yd = np.zeros(1000)
ydd = np.zeros(1000)
theta1 = np.zeros(1000)
theta2 = np.zeros(1000)
theta1d = np.zeros(1000)
theta2d = np.zeros(1000)
theta1dd = np.zeros(1000)
theta2dd = np.zeros(1000)
d = np.zeros(1000)
for i in range(1000):
    y[i] = 0.06 - (0.15*t[i]*t[i])/(tf*tf) + (0.1*t[i]*t[i]*t[i])/(tf*tf*tf)
    yd[i] = -(0.3*t[i])/(tf*tf) + (0.3*t[i]*t[i])/(tf*tf*tf)
    ydd[i] = -0.3/(tf*tf) + (0.6*t[i])/(tf*tf*tf)


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


for i in range(1000):
    y[i] = 0.06 - (0.15*t[i]*t[i])/(tf*tf) + (0.1*t[i]*t[i]*t[i])/(tf*tf*tf)
    yd[i] = -(0.3*t[i])/(tf*tf) + (0.3*t[i]*t[i])/(tf*tf*tf)
    ydd[i] = -0.3/(tf*tf) + (0.6*t[i])/(tf*tf*tf)

    Q = inversek(0.4, y[i], 0.1, l1, l2, l3)
    Qd = inversevelocity(Q[0], Q[1], yd[i], l1, l2, l3)
    Qdd = inverseacceleation(Q[0], Q[1], Qd[0], Qd[1], yd[i], ydd[i], l1, l2, l3)
    theta1[i] = Q[0]
    theta2[i] = Q[1]
    d[i] = Q[2]
    theta1d[i] = Qd[0]
    theta2d[i] = Qd[1]
    theta1dd[i] = Qdd[0]
    theta2dd[i] = Qdd[1]

print(theta2dd)
fig = plt.figure()
plt.plot(t, theta2dd, color='r', label='theta1')

fig.suptitle('Problem2: theta2 acceleration motion for A to B motion of End Effector')
plt.xlabel('time (sec)')
plt.ylabel('theta2dd (rad/sec^2)')
plt.grid()
plt.show()