import Task2 as T2
import matplotlib.pyplot as plt
import numpy as np
from SCARA.SCARA_with_noise import*


arko = SCARAManipulator()
print('MSG:: SCARAdManipulator initialised, named arko')
print('MSG:: link1 length = ' + str(arko.l1) + ' m; link2 length = ' + str(arko.l2) +' m')
print('MSG:: link1 mass = ' + str(arko.m1) + ' kg; link2 mass = ' + str(arko.m2) + ' kg')
print('MSG:: time delta = ' + str(arko.dt) + ' seconds')
arko.run()
print()
arko.fig.canvas.manager.set_window_title('Your own SCARA Manipulator: ARKO')

# PD Controller
from SCARA.SCARAPIDController import PID_Position_Controller
# Near the given coordinates
# For CRITICAL DAMPING
# For joint 1----
# Jeff = 58
# Beff = 2  ===> For omega_n = 5, Kp = 58*5**2 = 1450; Kd = 2*5*58 - 2 = 578
# For joint 2----
# Jeff = 17.25
# Beff = 2 ===> For omega_n = 5, Kp = 17.25*5**2 = 431; Kd = 2*5*17.25 - 2 = 170
# For joint 3----
# Jeff = 11
# Beff = 2 ===> For Kp = 1e4 ===> omega_n = 30; Kd = 2*30*11 - 2 = 658
print('STARTING PD CONTROLLER')
arkoConPD = PID_Position_Controller(arko, [1450, 431, 1e4], [0,0,0], [578, 170, 658])
ee_pos_mat_pd = []
arko.ax.plot((0.40,0.40),(0.06,0.01), (-arko.l1+0.1,-arko.l1+0.1),color = 'g', alpha=0.5)

arkoConPD.follow_trajectory(T2.q, T2.q_dot, 5, ee_pos_mat_pd)


# FeedForward Controller
from SCARA.SCARAFeedForwardController import FeedForward_Position_Controller
# Near the given coordinates
# For CRITICAL DAMPING
# For joint 1----
# Jeff = 58
# Beff = 2  ===> For omega_n = 5, Kp = 58*5**2 = 1450; Kd = 2*5*58 - 2 = 578
# For joint 2----
# Jeff = 17.25
# Beff = 2 ===> For omega_n = 5, Kp = 17.25*5**2 = 431; Kd = 2*5*17.25 - 2 = 170
# For joint 3----
# Jeff = 11
# Beff = 2 ===> For Kp = 1e4 ===> omega_n = 30; Kd = 2*30*11 - 2 = 658
print('STARTING Feed Forward CONTROLLER')
ee_pos_mat_ff = []
arkoConFF = FeedForward_Position_Controller(arko, [1450, 431, 1e4], [0,0,0], [578, 170, 658])
arko.ax.plot((0.40,0.40),(0.06,0.01), (-arko.l1+0.1,-arko.l1+0.1),color = 'g', alpha=0.5)

# FFc.arkoCon.Achieve_EE_Position((0.4,0.06,-0.4))

arkoConFF.follow_trajectory(T2.q, T2.q_dot, T2.q_dot2, 5, ee_pos_mat_ff)


# Computed Torques Controller
from SCARA.SCARAComputedTorqueFFController import ComputedTorque_FF_Position_Controller
# Near the given coordinates
# For CRITICAL DAMPING
# For joint 1----
# Jeff = 58
# Beff = 2  ===> For omega_n = 5, Kp = 58*5**2 = 1450; Kd = 2*5*58 - 2 = 578
# For joint 2----
# Jeff = 17.25
# Beff = 2 ===> For omega_n = 5, Kp = 17.25*5**2 = 431; Kd = 2*5*17.25 - 2 = 170
# For joint 3----
# Jeff = 11
# Beff = 2 ===> For Kp = 1e4 ===> omega_n = 30; Kd = 2*30*11 - 2 = 658
print('STARTING Computed Torques CONTROLLER')
arkoConCT = ComputedTorque_FF_Position_Controller(arko, [1450, 431, 1e4], [0,0,0], [578, 170, 658])
ee_pos_mat_ct = []
arko.ax.plot((0.40,0.40),(0.06,0.01), (-arko.l1+0.1,-arko.l1+0.1),color = 'g', alpha=0.5)

# CTc.arkoCon.Achieve_EE_Position((0.4,0.06,-0.4))
arkoConCT.follow_trajectory(T2.q, T2.q_dot, T2.q_dot2, 5, ee_pos_mat_ct)

# Multivariable Controller
from SCARA.SCARAMultivariableController import Multivariable_Position_Controller
print('STARTING Multivariable Controls CONTROLLER')
arkoConMC = Multivariable_Position_Controller(arko, 10, 6)
ee_pos_mat_mc = []
arko.ax.plot((0.40,0.40),(0.06,0.01), (-arko.l1+0.1,-arko.l1+0.1),color = 'g', alpha=0.5)

# Mc.arkoCon.Achieve_EE_Position((0.4,0.06,-0.4))
arkoConMC.follow_trajectory(T2.q, T2.q_dot, T2.q_dot2, 5, ee_pos_mat_mc)

plt.ioff()
fig, (ax1,ax2,ax3) = plt.subplots(3,1)

ax1.plot(T2.T1.t, [np.squeeze(np.asarray(pos))[0] for pos in T2.T1.position], label = 'Desired Path')
ax1.plot(np.linspace(0,5,532), [pos[0] for pos in ee_pos_mat_pd], label = 'PD')
ax1.plot(np.linspace(0,5,532), [pos[0] for pos in ee_pos_mat_ff], label = 'FeedForward')
ax1.plot(np.linspace(0,5,532), [pos[0] for pos in ee_pos_mat_ct], label = 'Computed Torques')
ax1.plot(np.linspace(0,5,532), [pos[0] for pos in ee_pos_mat_mc], label = 'Multivariable Control')
ax1.set_title('X coordinates')
ax1.legend()
ax1.grid()

ax2.plot(T2.T1.t, [np.squeeze(np.asarray(pos))[1] for pos in T2.T1.position], label = 'Desired Path')
ax2.plot(np.linspace(0,5,532), [pos[1] for pos in ee_pos_mat_pd], label = 'PD')
ax2.plot(np.linspace(0,5,532), [pos[1] for pos in ee_pos_mat_ff], label = 'FeedForward')
ax2.plot(np.linspace(0,5,532), [pos[1] for pos in ee_pos_mat_ct], label = 'Computed Torques')
ax2.plot(np.linspace(0,5,532), [pos[1] for pos in ee_pos_mat_mc], label = 'Multivariable Control')
ax2.set_title('Y coordinates')
ax2.legend()
ax2.grid()

ax3.plot(T2.T1.t, [np.squeeze(np.asarray(pos))[2] for pos in T2.T1.position], label = 'Desired Path')
ax3.plot(np.linspace(0,5,532), [0.5+pos[2] for pos in ee_pos_mat_pd], label = 'PD')
ax3.plot(np.linspace(0,5,532), [0.5+pos[2] for pos in ee_pos_mat_ff], label = 'FeedForward')
ax3.plot(np.linspace(0,5,532), [0.5+pos[2] for pos in ee_pos_mat_ct], label = 'Computed Torques')
ax3.plot(np.linspace(0,5,532), [0.5+pos[2] for pos in ee_pos_mat_mc], label = 'Multivariable Control')
ax3.set_title('Z coordinates')
ax3.legend()
ax3.grid()

plt.show()

fig,ax = plt.subplots()
ax.plot([np.squeeze(np.asarray(pos))[0] for pos in T2.T1.position], [np.squeeze(np.asarray(pos))[1] for pos in T2.T1.position], label = 'Desired Path')
ax.plot([pos[0] for pos in ee_pos_mat_pd], [pos[1] for pos in ee_pos_mat_pd], label = 'PD')
ax.plot([pos[0] for pos in ee_pos_mat_ff], [pos[1] for pos in ee_pos_mat_ff], label = 'FeedForward')
ax.plot([pos[0] for pos in ee_pos_mat_ct], [pos[1] for pos in ee_pos_mat_ct], label = 'Computed Torques')
ax.plot([pos[0] for pos in ee_pos_mat_mc], [pos[1] for pos in ee_pos_mat_mc], label = 'Multivariable Control')
ax.set_title('X-Y Plane View')
ax.legend()
ax.grid()
plt.show()
