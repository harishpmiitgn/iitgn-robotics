#Videh Patel : videh.p@iitgn.ac.in
#19110192

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import sympy as sp


from .Calculate_Dynamic_Equations import *
#Crude SCARA Maniulator Wrapper: Applies Torque to another system and provides inverse kinematic functions, etc. to Controller that calculates the voltage values according to the wrapper values 
#Takes only torque as input and calculates other properties using equations
# Editted to include motor dynamics
# All motors have same dynamics 


class SCARAWrapper:
    def __init__(self,RRModel, l1 = 0.5, l2 = 0.25, l3 = 0.25*1.2, q1 = 0, q2 = 0, d3 = 0.3, q1_dot = 0, q2_dot = 0, d3_dot = 0, m1 = 100, m2 = 1000, m3=1000, m4=10, d3_max = 0.5):
        self.Real_Manipulator = RRModel
        self.l1 = l1        #Link1 length (m)
        self.l2 = l2        #Link2 length (m)
        self.l3 = l3        #Link3 length (m)

        self.m1 = m1        #Mass1 mass (kg)
        self.m2 = m2        #Mass2 mass (kg)
        self.m3 = m3        #Mass3 mass (kg)
        self.m4 = m4        #Extension Link mass (kg)

        self.d3_max = d3_max    #Max prismatic extension (m)

        self.state = np.array([q1, q2, d3, q1_dot, q2_dot, d3_dot]) #[(rad), (rad), (m), (rad/s), (rad/s), (m/s)]
        self.last_joint_accelerations = np.array([0, 0, 0])

        self.time = 0       #time elapsed since start (sec)

        self.dt = 1./100.

        self.generate_dynamic_equations()

        self.Jm = 1 # Moment of inertia of shaft (kgm^2)
        self.Bm = 1 # Damping Coefficient of shaft
        self.Kb = 1 # Volatage to Angular velocity constant
        self.Km = 1 # Torque to current constant
        self.R = 1 # Resistance of motor coil (ohm) 
        self.r = 1 # Gear ratio of motors 

    def apply_voltages(self, voltages):
        # voltages is a 3x1 sympy matrix
        self.Real_Manipulator.apply_voltages(voltages)

    def tau_from_voltage(self, voltages):
        # voltages is a 3x1 sympy matrix
        qdot2 = sp.Matrix(self.last_joint_accelerations)*self.r
        qdot = sp.Matrix(self.state[3:])*self.r
        return (self.Km/self.R*voltages - self.Jm*qdot2 - (self.Bm + self.Kb*self.Km/self.R)*qdot)/self.r

    def generate_dynamic_equations(self):
        self.q_symbols = sp.Matrix(sp.symbols('q1 q2 q3'))
        self.q_dot_symbols = sp.Matrix(sp.symbols('q1_dot q2_dot q3_dot'))
        self.tau_symbols = sp.Matrix(sp.symbols('tau1 tau2 tau3'))

        I1 = sp.Matrix([[0, 0, 0],
                        [0 , self.m2*self.l2**2/12, 0],
                        [0, 0, self.m2*self.l2**2/12]])
        
        I2 = sp.Matrix([[self.m3*(self.l3)**2/12, 0, 0],
                        [0 , self.m3*(self.l3)**2/12, 0],
                        [0, 0, 0]])
        
        I3 = sp.Matrix([[self.m4*(self.q_symbols[2])**2/12, 0, 0],
                        [0 , self.m4*(self.q_symbols[2])**2/12, 0],
                        [0, 0, 0]])

        Dhmi = sp.Matrix([[self.q_symbols[0], 0, self.l2, 0, self.m2, I1],
                          [self.q_symbols[1], 0, self.l3, sp.pi, self.m3, I2],
                          [0, self.q_symbols[2], 0, 0, self.m4, I3]])

        
        print('Starting to calculate dynamic parameters for SCARA Manipulator')
        print('_________________________________________________________')
        self.D, self.C, self.g_ = Dynamic_Equations_from_Dhmi(Dhmi, prismatic=[3])

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
        return ((self.l2-self.l3)**2 <= (point[0]**2 + point[1]**2) <= (self.l2+self.l3)**2) and (-self.d3_max <= point[2] <= 0) 

    def apply_torques(self, tor_tup):   #Updates the state of system and return it
        self.state = self.Real_Manipulator.apply_torques()

    def run(self):
        #Kick starts animation and dynamics
        return self.Real_Manipulator.run()

    def apply_constant_torques(self, torque_tup,time = 2):
        for i in range(int(time/self.dt)):
            self.apply_torques(torque_tup)

##Setting functions:: Position setting ignores dynamics
    def set_position(self, q_):
        self.set_state(np.array([q_[0],q_[1],q_[2],0,0,0]))

    def set_state(self, state_array):
        #anim_updating_func
        if (state_array[2] > self.d3_max) or (state_array[2] < 0):
            print("Clipped the extension of prismatic joint")
            state_array[2] = np.float64(max(min(self.d3_max, state_array[2]), 0))
            state_array[5] = (state_array[2] - self.state[2])/self.dt
        if (abs(state_array[5]) > 5):
            print("Clipped the speed of prismatic joint")
            state_array[5] = np.float64(max(min(5, state_array[5]), -5))

        self.Real_Manipulator.state = state_array
        self.state = state_array
        self.Real_Manipulator.anim_update()

    def get_state(self):
        return self.state


    def inv_kin(self, ee_position):
        if not self.LiesInWorkspace(ee_position):
            print("ERROR:: Given position out of workspace")
            return self.state[:3]
        
        x,y,z = ee_position 

        cos = (x*x + y*y - self.l2**2 - self.l3**2)/(2*self.l2*self.l3)
        cos = 1 if cos>1 else cos
        cos = -1 if cos<-1 else cos

        q2 = np.arccos(cos)
        q1 = (np.arctan2(y,x) - np.arctan2(self.l3*np.sin(q2),(self.l2 + self.l3*np.cos(q2))))
        d3 = -z
 
        return np.array((q1, q2, d3))

    def set_ee_position(self, position):
        self.set_position(self.inv_kin(position))

    def get_ee_position(self):
        return np.array([arr[1] for arr in self.Real_Manipulator.extension_line.get_data_3d()])

    def reset(self, position = (0,0,1)):
        #Resets the RRM at a given position
        self.time = 0
        self.set_position(position)

    def apply_constant_torques(self, torque_tup,time = 2):
        for i in range(int(time/self.dt)):
            self.apply_torques(torque_tup)

