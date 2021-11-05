import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
from numpy import sin, cos, pi, outer, ones, size, linspace

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
J3 = (1 / 3) * m3 * l3 * l3            # Inertia about end point for Arm2]
E1 = []
E2 = []
E3 = []
T = []
g = 10

I = 0


def ikscara(x, y, z, d1, d2, d3):
    # using formulae from the textbook
    r = abs((x ** 2 + y ** 2 - d1 ** 2 - d2 ** 2) / (2 * d1 * d2))
    theta2 = np.arctan(np.sqrt(abs(1 - r ** 2)) / r)
    theta1 = np.arctan(y / x) - np.arctan(
        (d2 * np.sin(theta2)) / (d1 + d2 * np.cos(theta2)))
    d = d3 - z
    return theta1, theta2, d


def integrator(E, T):
    n = len(E)
    I = 0
    if n > 30:
        for i in range(n-30, n, 1):
            I = I + E[i]*(T[i] - T[i-1])
    return I


# function that returns dy/dt
def model(y, t, m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb):
    theta1 = y[0]
    theta1d = y[1]
    theta2 = y[2]
    theta2d = y[3]
    d = y[4]
    dd = y[5]

    d_theta1 = yda[0] + ((ydb[0] - yda[0])*t)/50
    d_theta2 = yda[1] + ((ydb[1] - yda[1]) * t) / 50
    d_d = yda[2] + ((ydb[2] - yda[2]) * t) / 50

    alpha = J1 + r1 * r1 * m1 + l1 * l1 * m2 + l1 * l1 * m3
    beta = J2 + J3 + l2 * l2 * m3 + m2 * r2 * r2
    gamma = l1 * l2 * m3 + l1 * r2 * m2

    e1 = d_theta1 - theta1
    E1.append(e1)

    e2 = d_theta2 - theta2
    E2.append(e2)

    e3 = d_d - d
    E3.append(e3)

    T.append(t)

    I1 = integrator(E1, T)
    I2 = integrator(E2, T)
    I3 = integrator(E3, T)
    # Dynamics equations from: Dynamic modeling of SCARA robot based on Udwadia–Kalaba theory by Yaru Xu and Rong Liu

    u1 = -gamma*np.sin(theta2)*theta2d*theta1d - gamma*np.sin(theta2)*(theta2d + theta1d)*theta2d + 10*e1 - 0.7*I1
    u2 = gamma*np.sin(theta2)*theta1d*theta1d - 20*e2 - 0.2*I2
    u3 = m3*g + 10*e3 - 0.1*I3

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
A = [1, 0, 5]
B = [2, 2, 3]

yda = ikscara(A[0], A[1], A[2], l1, l2, 5)
ydb = ikscara(B[0], B[1], B[2], l1, l2, 5)

y0 = np.array([yda[0], 0, yda[1], 0, yda[2], 0])

# time points
t = np.linspace(0, 50, num=100)

# solve ODE
y = odeint(model, y0, t, args=(m1, m2, m3, l1, l2, l3, r1, r2, r3, J1, J2, J3, g, yda, ydb))
print(yda, ydb)

theta1 = y[:, 0]
theta2 = y[:, 2]
d = y[:, 4]

# plt.plot(t, d, color='r', label='Phi')
#
# plt.grid()
# plt.show()
# print(len(T))


def atransformation(LinkParameters):
    a = LinkParameters[0]
    d = LinkParameters[2]
    alpha = LinkParameters[1]
    theta = LinkParameters[3]
    A = np.array([[[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                   [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                   [0, np.sin(alpha), np.cos(alpha), d],
                   [0, 0, 0, 1]]])
    return (A)


def forward_kinematics_homogenous_matrix(DH):
    A = np.identity(4)
    n = np.shape(DH)[0]

    for i in range(n):
        Anext = atransformation(DH[i])
        A = np.matmul(A,Anext)
    return A


def fk(t1, t2, d):
    LinkParameters = [[l1, 0, 0, t1], [l2, math.pi, 0, t2], [0, 0, d, 0]]
    T = forward_kinematics_homogenous_matrix(LinkParameters)
    coordinates = np.matmul(T, [[0], [0], [0], [1]])
    return coordinates


P = []
Q = []
R = []

for i in range(len(theta1)):
    w = fk(theta1[i], theta2[i], d[i])
    p = w[0, 0]
    P.append(p)
    q = w[0, 1]
    Q.append(q)
    r = w[0, 2] + 5
    R.append(r)

print(R[0][0])


# The amount of frames in the animation
frames = len(R)

# Generate each frame
for n in range(frames):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n):
        ax.scatter(P[i][0], Q[i][0], R[i][0], 'green')

    ax.set_title("End Effector motion from A = [1, 0, 5], B = [2, 2, 3] for SCARA Robot using PI Controller")
    ax.view_init(15, 75)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim3d([0.0, 3.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([0.0, 3.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([2.0, 6.0])
    ax.set_zlabel('Z')

    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_zticks([2, 3, 4, 5, 6])

    plt.savefig(f"{n}.png")

    plt.close()
    print(n)

# Use pillow to save all frames as an animation in a gif file
from PIL import Image

images = [Image.open(f"{n}.png") for n in range(frames)]

images[0].save('ball.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
