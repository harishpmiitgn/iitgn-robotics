#Videh Patel : videh.p@iitgn.ac.in
#19110192

# DEV DEV           DEV DEV DEV     DEV            DEV
# DEV   DEV         DEV DEV DEV     DEV            DEV
# DEV       DEV     DEV              DEV          DEV
# DEV       DEV     DEV DEV           DEV        DEV
# DEV       DEV     DEV                 DEV     DEV
# DEV   DEV         DEV DEV DEV           DEV DEV
# DEV DEV           DEV DEV DEV             DEV


import numpy as np
import matplotlib.pyplot as plt

# Given starting and end points
d0 = np.matrix((0.40,0.06,0.1)).T
d1 = np.matrix((0.40,0.01,0.1)).T
# Other parameters
f = 0 # Parameter to parameterize the motion to make the end effector traverse a line
# d = d0 + f*(d1-d0) :: for 0 <= f <= 1

t0 = 0 # Initial time (sec) 
tf = 5 # Final time (sec)

# Thus we will try to find f(t):(f as a function of time) by cubic polynomial fitting as explained in the textbook
B = np.matrix([0, 0, 1, 0]).T
# [[1],
#  [0],
#  [0]
#  [0]]]

A = np.matrix([[1, t0, t0**2, t0**3],[0, 1, 2*t0, 3*t0**2], [1, tf, tf**2, tf**3],[0, 1, 2*tf, 3*tf**2]])
# [[1, t0, t0**2, t0**3],
#  [0, 1, 2*t0, 3*t0**2], 
#  [1, tf, tf**2, tf**3],
#  [0, 1, 2*tf, 3*tf**2]]

X = np.linalg.inv(A)@B

t = np.linspace(t0, tf, 50)
f_t = X[0,0] + X[1,0]*t + X[2,0]*t**2 + X[3,0]*t**3
df_dt = X[1,0] + 2*X[2,0]*t + 3*X[3,0]*t**2
df_dtdt = 2*X[2,0] + 6*X[3,0]*t

position = []
velocity = []
acceleration = []

for f_,df_dt_, df_dtdt_ in zip(f_t,df_dt,df_dtdt):
    position.append(d0 + f_*(d1-d0))
    velocity.append(df_dt_*(d1-d0))
    acceleration.append(df_dtdt_*(d1-d0))

if __name__ == '__main__':
    fig, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(3,3, sharex = True)
    ############X
    ax1.plot(t, [d_[0,0] for d_ in position])
    ax1.set_title('Position Curve(X)')
    ax1.set_ylabel('Position of\n the end effector')
    ax1.grid()

    ax2.plot(t, [df_dt_[0,0] for df_dt_ in velocity])
    ax2.set_title('Velocity Curve(X)')
    ax2.set_ylabel('Velocity of\n the end effector')
    ax2.grid()

    ax3.plot(t, [df_dtdt_[0,0] for df_dtdt_ in acceleration])
    ax3.set_title('Acceleration Curve(X)')
    ax3.set_ylabel('Acceleration of\n the end effector')
    ax3.grid()


    ############Y
    ax4.plot(t, [d_[1,0] for d_ in position])
    ax4.set_title('Position Curve(Y)')
    ax4.grid()

    ax5.plot(t, [df_dt_[1,0] for df_dt_ in velocity])
    ax5.set_title('Velocity Curve(Y)')
    ax5.grid()

    ax6.plot(t, [df_dtdt_[1,0] for df_dtdt_ in acceleration])
    ax6.set_title('Acceleration Curve(Y)')
    ax6.grid()

    ############Z
    ax7.plot(t, [d_[2,0] for d_ in position])
    ax7.set_title('Position Curve(Z)')
    ax7.grid()

    ax8.plot(t, [df_dt_[2,0] for df_dt_ in velocity])
    ax8.set_title('Velocity Curve(Z)')
    ax8.grid()

    ax9.plot(t, [df_dtdt_[2,0] for df_dtdt_ in acceleration])
    ax9.set_title('Acceleration Curve(Z)')
    ax9.grid()

    fig.suptitle('Task1:: (DEV)')

    plt.show()
