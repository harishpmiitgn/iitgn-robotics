#Videh Patel : videh.p@iitgn.ac.in
#19110192

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#Crude Maniulator without any logic
#Takes only torque as input and calculates other properties using equations


class RRManipulator:
    def __init__(self,l1 = 4, l2 = 3, q1 = 0, q2 = 0, q1_dot = 0, q2_dot = 0, m1 = 1, m2 = 1):
        self.l1 = l1        #Link1 length (m)
        self.l2 = l2        #Link2 length (m)
        self.m1 = m1        #Mass1 length (kg)
        self.m2 = m2        #Mass2 length (kg)

        self.state = np.array([q1, q2, q1_dot, q2_dot]) #[(rad), (rad), (rad/s), (rad/s)]

        self.time = 0       #time elapsed since start (sec)

        self.dt = 1./100.

    #Checks if a point lies in workspace
    def LiesInWorkspace(self, point):
        return (self.l1 - self.l2)**2 <= (point[0]**2 + point[1]**2) <= (self.l1 + self.l2)**2

    def apply_torques(self, tor_tup, f = (0,0)):   #Updates the state of system and return it

        q1, q2, q1dot, q2dot = self.state

        if f != (0,0):
            print('MSG:: ACtive force applied at end effector: Value(newtons) : ' + str(f))

        torque1ap, torque2ap = tor_tup

        fx,fy = f

        tau1ex = self.l1*np.sin(q1)*fx - self.l1*np.cos(q1)*fy
        tau2ex = self.l2*np.sin(q2)*fx - self.l2*np.cos(q2)*fy

        torque1 = torque1ap - tau1ex
        torque2 = torque2ap - tau2ex

        g = 9.81

        A1 = 1./3.*self.m1*self.l1**2. + self.m2*self.l1**2.
        A2 = 1./2.*self.m2*self.l1*self.l2*np.cos(q2 - q1)
        B1 = 1./2.*self.m2*self.l1*self.l2*np.cos(q2-q1)
        B2 = 1./3.*self.m2*self.l2**2 #+ 1./4.*self.m2*self.l2**2
        C1 = 1./2.*self.m1*g*self.l1*np.cos(q1) + self.m2*g*self.l1*np.cos(q1) - 1./2.*self.m2*self.l1*self.l2*q2dot*q2dot*np.sin(q2-q1) - torque1
        C2 = 1./2.*self.m2*g*self.l2*np.cos(q2) + 1./2.*self.m2*self.l1*self.l2*q1dot*q1dot*np.sin(q2 - q1) - torque2

        q1dot2 = (B1*C2 - B2*C1)/(A1*B2 - A2*B1)
        q2dot2 = (C1*A2 - C2*A1)/(A1*B2 - A2*B1)

        if (q1dot2 > 1e14) or (q2dot2 > 1e14):
            print('WARNING:: 1he acceleration values too large, system may not respond as calculated')
            print('________________________________________________________________________________')
            return None

        q1dot_n = q1dot + q1dot2*self.dt
        q2dot_n = q2dot + q2dot2*self.dt

        q1_n = q1 + (q1dot)*self.dt
        q2_n = q2 + (q2dot)*self.dt


        self.state = np.array([q1_n, q2_n, q1dot_n, q2dot_n])
        self.time += self.dt

        self.torque_text1.set_text('torque1 (N-m): %.2f' % tor_tup[0])
        self.torque_text2.set_text('torque2 (N-m): %.2f' % tor_tup[1])
        self.torque_text2.set_text('torque2 (N-m): %.2f' % tor_tup[1])

        self.force_text.set_text('force on End Effector (N): (%.2f, %.2f)' %f)
        self.anim_update()
        return self.state

    def run(self):
        #Kick starts animation and dynamics
        plt.ion()
        self.fig, self.ax = plt.subplots()

        self.anim_init()

    def anim_init(self):
        #Initialises animation
        l1_end_position = self.l1*np.array((np.cos(self.state[0]), np.sin(self.state[0])))                        #End of link1
        l2_end_position = l1_end_position + self.l2*np.array((np.cos(self.state[1]), np.sin(self.state[1])))      #End of link2

        self.InnerCircle = plt.Circle((0,0), abs(self.l2-self.l1), ls = '--', color = 'b', fill = False, alpha = 0.5)
        self.OuterCircle = plt.Circle((0,0), abs(self.l2+self.l1), ls = '--', color = 'b', fill = False, alpha =0.5)

        self.ax.add_patch(self.InnerCircle)
        self.ax.add_patch(self.OuterCircle)

        self.l1_end_circle = plt.Circle(l1_end_position, radius=0.25, fc='#0077b6')
        self.l2_end_circle = plt.Circle(l2_end_position, radius=0.25, fc='g',  alpha = 0.5)
        self.l1_line = plt.Line2D((0, l1_end_position[0]), (0, l1_end_position[1]), color = "#0077b6" , lw=2.5)
        self.l2_line = plt.Line2D((l1_end_position[0], l2_end_position[0]), (l1_end_position[1], l2_end_position[1]), color = '#0077b6', lw=2.5)

        #self.force_arrow = plt.arrow(l2_end_position[0], l2_end_position[1], 0,0, color = 'y', alpha = 0.5)

        self.ax.axhline(-(self.l1 + self.l2), alpha = 0)
        self.ax.axhline((self.l1 + self.l2), alpha = 0)
        self.ax.axvline((self.l1 + self.l2), alpha = 0)
        self.ax.axvline(-(self.l1 + self.l2), alpha = 0)

        self.ax.add_line(self.l2_line)
        self.ax.add_line(self.l1_line)
        self.ax.add_patch(self.l1_end_circle)
        self.ax.add_patch(self.l2_end_circle)

        self.time_text = self.ax.text(0.02, 0.95, 'time (seconds): 0.00', transform=self.ax.transAxes)
        self.torque_text1 = self.ax.text(0.02, 0.9, 'torque1 (N-m): 0.00', transform=self.ax.transAxes)
        self.torque_text2 = self.ax.text(0.02, 0.85, 'torque2 (N-m): 0.00', transform=self.ax.transAxes)
        self.force_text = self.ax.text(0.02, 0.8, 'force on End Effector (N): (0.00, 0,00)', transform=self.ax.transAxes)

        self.angle1 = self.ax.text(0.02, 0.1, 'q1 (deg): 0.00', transform=self.ax.transAxes)
        self.angle2 = self.ax.text(0.02, 0.05, 'q2 (deg): 0.00', transform=self.ax.transAxes)

        plt.axis('scaled')

    def anim_update(self):
        #Updates the animation as the current state: Should be called wherever state changes

        q1, q2,*_ = self.state

        l1_end_position = self.l1*np.array((np.cos(q1), np.sin(q1)))
        l2_end_position = l1_end_position + self.l2*np.array((np.cos(q2), np.sin(q2)))

        self.l1_end_circle.set_center(l1_end_position)
        self.l2_end_circle.set_center(l2_end_position)

        self.l1_line.set_data((0, l1_end_position[0]), (0, l1_end_position[1]))
        self.l2_line.set_data((l1_end_position[0], l2_end_position[0]), (l1_end_position[1], l2_end_position[1]))

        self.time_text.set_text('time (seconds): %.2f' % self.time)
        self.angle1.set_text('q1 (deg): %.2f' % (q1*180/np.pi))
        self.angle2.set_text('q2 (deg): %.2f' % (q2*180/np.pi))

        plt.pause(self.dt)

    def apply_constant_torques(self, torque_tup, force_tup = (0,0),time = 2):
        for i in range(int(time/self.dt)):
            self.apply_torques(torque_tup, force_tup)

##Setting functions:: Position setting ignores dynamics
    def set_position(self, angle_tup):
        self.set_state(np.array([angle_tup[0],angle_tup[1],0,0]))

    def set_state(self, state_array):
        #anim_updating_func
        self.state = state_array
        self.anim_update()

    def get_state(self):
        return self.state

    def set_ee_position(self, position):
        if not self.LiesInWorkspace(position):
            print("ERROR:: Given position out of workspace")
        else:
            x,y = position
            cos = (x*x + y*y - self.l1**2 - self.l2**2)/(2*self.l1*self.l2)
            cos = 1 if cos>1 else cos
            cos = -1 if cos<-1 else cos

            theta = np.arccos(cos) #Calculates theta values at each given position
            q1 = (np.arctan2(y,x) - np.arctan2(self.l2*np.sin(theta),(self.l1 + self.l2*np.cos(theta))))
            q2 = q1 + theta

            self.set_position((q1,q2))

    def get_ee_position(self):
        return (self.l1*np.cos(self.state[0]) + self.l2*np.cos(self.state[1]),self.l1*np.sin(self.state[0]) + self.l2*np.sin(self.state[1]))

    def reset(self, position = (0,0)):
        #Resets the RRM at a given position
        self.time = 0
        self.set_position(position)

arko = RRManipulator()
print('MSG:: RRManipulator initialised, named arko')
print('MSG:: link1 length = ' + str(arko.l1) + ' m; link2 length = ' + str(arko.l2) +' m')
print('MSG:: link1 mass = ' + str(arko.m1) + ' kg; link2 mass = ' + str(arko.m2) + ' kg')
print('MSG:: time delta = ' + str(arko.dt) + ' seconds')
arko.run()
print('MSG:: Animation started')
print()
arko.fig.canvas.manager.set_window_title('Your own RRManipulator: ARKO')
