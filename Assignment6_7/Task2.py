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
import sympy as sp

import Task1 as T1
from SCARA.SCARA_raw import *

Manip = SCARAManipulator()

q = []  # Array to store all the values of joint variables for the given waypoints
q_dot = [] #Array to store all the values of joint velocities for the given waypoints
q_dot2 = [] #Array to store all the values of joint acceleration for the given waypoints

for i,pos in enumerate(T1.position):
    q.append(Manip.inv_kin(np.squeeze(np.asarray(pos))))


for q_,v,a in zip(q, T1.velocity, T1.acceleration):
    Manip.set_position(q_)
    q_dot_ = (Manip.Jacobian()[:3,:].pinv())@v
    q_dot.append(q_dot_)
    b = a - Manip.sym_djacob_dt.subs([(sp.symbols('q1'), q_[0]),
                        (sp.symbols('q2'), q_[1]),
                        (sp.symbols('q3'), q_[2]),
                        (sp.symbols('q1_dot'), q_dot_[0,0]),
                        (sp.symbols('q2_dot'), q_dot_[1,0]),
                        (sp.symbols('q3_dot'), q_dot_[2,0])]).evalf()[:3,:]@q_dot_
    Ja = Manip.sym_Jacobian.subs([(sp.symbols('q1'), q_[0]),
                        (sp.symbols('q2'), q_[1]),
                        (sp.symbols('q3'), q_[2])]).evalf()
    q_dot2.append(Ja[:3,:].pinv()@b)

if __name__ == '__main__':
    fig, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = plt.subplots(3,3, sharex = True)
    ############ X
    ax1.plot(T1.t, [d_[0] for d_ in q])
    ax1.set_title('Position Curve(q1)')
    ax1.set_ylabel('Positions')
    ax1.grid()

    ax2.plot(T1.t, [df_dt_[0,0] for df_dt_ in q_dot])
    ax2.set_title('Velocity Curve(q1)')
    ax2.set_ylabel('Velocities')
    ax2.grid()

    ax3.plot(T1.t, [df_dtdt_[0,0] for df_dtdt_ in q_dot2])
    ax3.set_title('Acceleration Curve(q1)')
    ax3.set_ylabel('Accelerations')
    ax3.grid()


    ############ Y
    ax4.plot(T1.t, [d_[1] for d_ in q])
    ax4.set_title('Position Curve(q2)')
    ax4.grid()

    ax5.plot(T1.t, [df_dt_[1,0] for df_dt_ in q_dot])
    ax5.set_title('Velocity Curve(q2)')
    ax5.grid()

    ax6.plot(T1.t, [df_dtdt_[1,0] for df_dtdt_ in q_dot2])
    ax6.set_title('Acceleration Curve(q2)')
    ax6.grid()

    ############ Z
    ax7.plot(T1.t, [d_[2] for d_ in q])
    ax7.set_title('Position Curve(d3)')
    ax7.grid()

    ax8.plot(T1.t, [df_dt_[2,0] for df_dt_ in q_dot])
    ax8.set_title('Velocity Curve(d3)')
    ax8.grid()

    ax9.plot(T1.t, [df_dtdt_[2,0] for df_dtdt_ in q_dot2])
    ax9.set_title('Acceleration Curve(d3)')
    ax9.grid()

    fig.suptitle('Task2:: (DEV)')

    plt.show()