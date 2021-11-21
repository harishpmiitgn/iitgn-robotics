import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
from numpy import sin, cos, pi, outer, ones, size, linspace

# Parameters
# For Arm1
l1 = 1   # m, Length of arm1

# For Arm2
l2 = 1      # m,Length of Arm2

# For Arm3
l3 = 1      # m,Length of Arm2

tf = 20

t = np.linspace(0, tf, num= 1000)

y = np.zeros(1000)
yd = np.zeros(1000)
ydd = np.zeros(1000)

for i in range(1000):
    y[i] = 0.06 - (0.15*t[i]*t[i])/(tf*tf) + (0.1*t[i]*t[i]*t[i])/(tf*tf*tf)
    yd[i] = -(0.3*t[i])/(tf*tf) + (0.3*t[i]*t[i])/(tf*tf*tf)
    ydd[i] = -0.3/(tf*tf) + (0.6*t[i])/(tf*tf*tf)

print(y)
fig = plt.figure()
plt.plot(t, ydd, color='r', label='theta1')

fig.suptitle('Problem1: End Effector Y axis motion from 0.06 to 0.01, acceleration')
plt.xlabel('time')
plt.ylabel('y acceleration')
plt.grid()
plt.show()