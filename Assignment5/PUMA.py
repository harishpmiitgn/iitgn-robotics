#Videh Patel : videh.p@iitgn.ac.in
#19110192

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import sympy as sp


from Calculate_Dynamic_Equations import *
#Crude Stanford Maniulator without any logic
#Takes only torque as input and calculates other properties using equations


class PUMAManipulator:
    def __init__(self,l1 = 5, l2 = 5, l3 = 5, q1 = 0, q2 = 0, q3 = 0, q1_dot = 0, q2_dot = 0, q3_dot = 0, m1 = 1, m2 = 1, m3 = 1):
        self.l1 = l1        #Link1 length (m)
        self.l2 = l2        #Link2 length (m)
        self.l3 = l3        #Link3 length (m)

        self.m1 = m1        #Mass1 length (kg)
        self.m2 = m2        #Mass2 length (kg)
        self.m3 = m3        #Mass3 length (kg)

        self.state = np.array([q1, q2, q3, q1_dot, q2_dot, q3_dot]) #[(rad), (rad), (m), (rad/s), (rad/s), (m/s)]

        self.time = 0       #time elapsed since start (sec)

        self.dt = 1./100.

        self.generate_dynamic_equations()       

    def generate_dynamic_equations(self):
        self.q_symbols = sp.Matrix(sp.symbols('q1 q2 q3'))
        self.q_dot_symbols = sp.Matrix(sp.symbols('q1_dot q2_dot q3_dot'))
        self.tau_symbols = sp.Matrix(sp.symbols('tau1 tau2 tau3'))

        I1 = sp.Matrix([[self.m1*self.l1**2/12, 0, 0],
                        [0 , 0, 0],
                        [0, 0, self.m1*self.l1**2/12]])
        
        I2 = sp.Matrix([[0, 0, 0],
                        [0 , self.m2*self.l2**2/12, 0],
                        [0, 0, self.m2*self.l2**2/12]])
        
        I3 = sp.Matrix([[0, 0, 0],
                        [0 , self.m3*self.l3**2/12, 0],
                        [0, 0, self.m3*self.l3**2/12]])

        Dhmi = sp.Matrix([[self.q_symbols[0], self.l1, 0, sp.pi/2, self.m1, I1],
                          [self.q_symbols[1], 0, self.l2, 0, self.m2, I2],
                          [self.q_symbols[2], 0, self.l3, 0, self.m3, I3]])

        
        print('Starting to calculate dynamic parameters for Stanford Manipulator')
        print('_________________________________________________________')
        self.D, self.C, self.g_ = Dynamic_Equations_from_Dhmi(Dhmi)

        print('D-----------------------')
        print(self.D)
        print('C----------------------')
        print(self.C)
        print('g_----------------------')
        print(self.g_)

        print('Calculating q_dot2_array')
        self.q_dot2_array = sp.simplify(sp.nsimplify(sp.simplify(sp.nsimplify(self.D.inv()@(self.tau_symbols - self.C@self.q_dot_symbols - self.g_), tolerance = 1e-10))))
        print('qdot2_array ---------------------------------------')
        print(self.q_dot2_array)
        print("Dynamic Parameters generated successfully")
        print('_________________________________________________________')


    #Checks if a point lies in workspace
    def LiesInWorkspace(self, point):
        return (self.l2-self.l3)**2 <= point[0]**2 + point[1]**2 + (point[2] - self.l1)**2 <= (self.l2+self.l3)**2

    def inv_kin(self, ee_position):
        if not self.LiesInWorkspace(ee_position):
            print("ERROR:: Given position out of workspace")
            return self.state[:3]
        
        x,y,z = ee_position

        cos = (x**2 + y**2 + (z - self.l1)**2 - self.l2**2 - self.l3**2)/(2*self.l2*self.l3)
        cos = 1 if cos>1 else cos
        cos = -1 if cos<-1 else cos

        q1 = np.arctan2(y,x)
        q3 = np.arccos(cos) 
        q2 = (np.arctan2(z-self.l1,np.sqrt(x**2+y**2)) - np.arctan2(self.l3*np.sin(q3),(self.l2 + self.l3*np.cos(q3))))

        return np.array((q1, q2, q3))

    def apply_torques(self, tor_tup):   #Updates the state of system and return it

        q1, q2, q3, q1dot, q2dot, q3dot = self.state
        q = sp.Matrix([q1, q2, q3])
        qdot = sp.Matrix([q1dot, q2dot, q3dot])

        q_dot2 = self.q_dot2_array.subs([(self.q_symbols[0], q1),
                        (self.q_symbols[1], q2),
                        (self.q_symbols[2], q3),
                        (self.q_dot_symbols[0], q1dot),
                        (self.q_dot_symbols[1], q2dot),
                        (self.q_dot_symbols[2], q3dot),
                        (self.tau_symbols[0], tor_tup[0]),
                        (self.tau_symbols[1], tor_tup[1]),
                        (self.tau_symbols[2], tor_tup[2]),
                        (sp.symbols('g'), 9.81)]).evalf()

        q_dot_n = qdot + q_dot2*self.dt
        q_n = q + qdot*self.dt

        self.time += self.dt

        self.set_state(np.array([float(q_n[0]), float(q_n[1]), float(q_n[2]), float(q_dot_n[0]), float(q_dot_n[1]), float(q_dot_n[2])]))
        return self.state

    def run(self):
        #Kick starts animation and dynamics
        plt.ion()

        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig, auto_add_to_figure = False)
        self.fig.add_axes(self.ax)

        self.anim_init()

    def anim_init(self):
        q1, q2, q3, *_ = self.state

        self.link1_line = self.ax.plot([0,0], [0,0], [0,self.l1], color = '#009dff')[0]
        self.link2_line = self.ax.plot(np.array([0, self.l2*np.cos(q2)*np.cos(q1)]),np.array([0, self.l2*np.cos(q2)*np.sin(q1)]),np.array([self.l1, self.l1+self.l2*np.sin(q2)]), color = '#009dff')[0]
        self.link3_line = self.ax.plot(np.array([self.l2*np.cos(q2)*np.cos(q1),self.l2*np.cos(q2)*np.cos(q1) + self.l3*np.cos(q2+q3)*np.cos(q1)]), np.array([self.l2*np.cos(q2)*np.sin(q1), self.l2*np.cos(q2)*np.sin(q1)+self.l2*np.cos(q2+q3)*np.sin(q1)]), np.array([self.l1+self.l2*np.sin(q2),self.l1+self.l2*np.sin(q2)+self.l3*np.sin(q2+q3)]), color = '#009dff')[0]

        # boundary lines
        self.ax.set_xlim3d([-(self.l2+self.l3), (self.l2+self.l3)])
        self.ax.set_xlabel('X')

        self.ax.set_ylim3d([-(self.l2+self.l3), (self.l2+self.l3)])
        self.ax.set_ylabel('Y')

        self.ax.set_zlim3d([0, (self.l1+2*(self.l2+self.l3))])
        self.ax.set_zlabel('Z')
        self.title  = self.fig.suptitle('PUMA Manipulator\n' + 
                                        'time(sec):     0.00\n' +
                                        'q1(rad):       0.00\n' +
                                        'q2(rad):       0.00\n' +
                                        'q3(m):         0.00\n' +
                                        'q1dot(rad/s):  0.00\n' +
                                        'q2dot(rad/s):  0.00\n' +
                                        'q3dot(m/s):  0.00')

    def anim_update(self):
        #Updates the animation as the current state: Should be called wherever state changes
        q1, q2, q3, *_ = self.state

        self.link2_line.set_data_3d(np.array([0, self.l2*np.cos(q2)*np.cos(q1)]),np.array([0, self.l2*np.cos(q2)*np.sin(q1)]),np.array([self.l1, self.l1+self.l2*np.sin(q2)]))
        self.link3_line.set_data_3d(np.array([self.l2*np.cos(q2)*np.cos(q1),self.l2*np.cos(q2)*np.cos(q1) + self.l3*np.cos(q2+q3)*np.cos(q1)]), np.array([self.l2*np.cos(q2)*np.sin(q1), self.l2*np.cos(q2)*np.sin(q1)+self.l2*np.cos(q2+q3)*np.sin(q1)]), np.array([self.l1+self.l2*np.sin(q2),self.l1+self.l2*np.sin(q2)+self.l3*np.sin(q2+q3)]))

        self.title.set_text('''PUMA Manipulator
time(sec):     {:.2f}
q1(rad):       {:.2f} 
q2(rad):       {:.2f} 
q3(m):         {:.2f} 
q1dot(rad/s):  {:.2f} 
q2dot(rad/s):  {:.2f} 
q3dot(m/s):    {:.2f}'''.format(float(self.time), float(self.state[0]), float(self.state[1]), float(self.state[2]), float(self.state[3]), float(self.state[4]), float(self.state[5])))
        plt.pause(self.dt)

    def apply_constant_torques(self, torque_tup,time = 2):
        for i in range(int(time/self.dt)):
            self.apply_torques(torque_tup)

##Setting functions:: Position setting ignores dynamics
    def set_position(self, q_):
        self.set_state(np.array([q_[0],q_[1],q_[2],0,0,0]))

    def set_state(self, state_array):
        #anim_updating_func
        self.state = state_array
        self.anim_update()

    def get_state(self):
        return self.state

    def set_ee_position(self, position):
        self.set_position(self.inv_kin(position))

    def get_ee_position(self):
        return np.array([arr[1] for arr in self.extension_line.get_data_3d()])

    def reset(self, position = (0,0,1)):
        #Resets the RRM at a given position
        self.time = 0
        self.set_position(position)

    def apply_constant_torques(self, torque_tup,time = 2):
        for i in range(int(time/self.dt)):
            self.apply_torques(torque_tup)


arko = PUMAManipulator()
print('MSG:: StanfordManipulator initialised, named arko')
print('MSG:: link1 length = ' + str(arko.l1) + ' m; link2 length = ' + str(arko.l2) +' m')
print('MSG:: link1 mass = ' + str(arko.m1) + ' kg; link2 mass = ' + str(arko.m2) + ' kg')
print('MSG:: time delta = ' + str(arko.dt) + ' seconds')
arko.run()
print('MSG:: Animation started')
print()
arko.fig.canvas.manager.set_window_title('Your own PUMAManipulator: ARKO')
