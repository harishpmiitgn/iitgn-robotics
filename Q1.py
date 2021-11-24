import math
import scipy
import sympy as sp
from sympy.core.evalf import N
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D



end_effec0=[0.4,0.06,0.1]
end_effec1=[0.4,0.01,0.1]

y0=end_effec0[1]
y1=end_effec1[1]

# Computing from code from midsem Q1(a)


# We apply the cubic polynomial trjaectory putting the vaules of q0 and q1 in input
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
yds=[]
vds=[]
ads=[]
for j in t:
    yd=a[0][0]+a[0][1]*j+a[0][2]*(j**2)+a[0][3]*(j**3)+a[0][4]*(j**4)+a[0][5]*(j**5)
    vd=a[0][1]+2*a[0][2]*j+3*a[0][3]*(j**2)+4*a[0][4]*(j**3)+5*a[0][5]*(j**4)
    ad=2*a[0][2]+6*a[0][3]*j+12*a[0][4]*(j**2)+20*a[0][5]*(j**3)
    yds.append(yd)
    vds.append(vd)
    ads.append(ad)

# fig,(ax1,ax2) = plt.subplots(1,2)
fig = plt.figure()


# x=np.linspace(t0,(tf-t0),(tf-t0)*100)
# x1=[0.4]*((tf-t0)*100)
# z1=[0.1]*((tf-t0)*100)
# plt.plot(x,yds,label='Y-coordinates')
# plt.plot(x,x1,label='X-coordinate')
# plt.plot(x,z1, label='Z-Coordinate')
# # plt1.xlabel('Time')
# # plt1.ylabel('Angle(q)')
# plt.title('$Position-Time Graph$')

# x=np.linspace(t0,(tf-t0),(tf-t0)*100)
# v=vds
# x2=[0]*((tf-t0)*100)
# z2=[0]*((tf-t0)*100)
# plt.plot(x,v,label='Velocity y"')
# plt.plot(x,x2,label='Velocity x"')
# plt.plot(x,z2,label='Velocity Z"')
# # plt2.xlabel('Time')
# # plt2.ylabel('Velocity of end-effector')
# plt.title('$Velocity-Time Graph$')

x=np.linspace(t0,(tf-t0),(tf-t0)*100)
ac=ads
x3=[0]*((tf-t0)*100)
z3=[0]*((tf-t0)*100)
plt.plot(x,ac,label='Acceleration Y""')
plt.plot(x,x3,label='Aceeleration X""')
plt.plot(x,z3,label='Acceleration z""')
# plt3.xlabel('Time')
# plt3.ylabel('Acceleration of end-effector')
plt.title('$Acceleration-Time Graph$')
# fig.subplots_adjust(hspace=.5,wspace=0.5)
plt.legend()
plt.show()

