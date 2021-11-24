#Videh Patel : videh.p@iitgn.ac.in
#19110192

# DEV DEV           DEV DEV DEV     DEV            DEV
# DEV   DEV         DEV DEV DEV     DEV            DEV
# DEV       DEV     DEV              DEV          DEV
# DEV       DEV     DEV DEV           DEV        DEV
# DEV       DEV     DEV                 DEV     DEV
# DEV   DEV         DEV DEV DEV           DEV DEV
# DEV DEV           DEV DEV DEV             DEV

# Same code as the SCARA manipulator class in Assignment 5
# Edited to not run animation

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from DHMatrix2Homo_and_Jacob import DHMatrix2Homo_and_Jacob as Dh2HJ

class SCARAManipulator:
    def __init__(self,l1 = 0.5, l2 = 0.25, l3 = 0.25, q1 = 0, q2 = 0, d3 = 0.3, q1_dot = 0, q2_dot = 0, d3_dot = 0, m1 = 100, m2 = 1000, m3=1000, m4=10, d3_max = 0.5):
        self.l1 = l1        #Link1 length (m)
        self.l2 = l2        #Link2 length (m)
        self.l3 = l3        #Link3 length (m)

        self.m1 = m1        #Mass1 mass (kg)
        self.m2 = m2        #Mass2 mass (kg)
        self.m3 = m3        #Mass3 mass (kg)
        self.m4 = m4        #Extension Link mass (kg)

        self.d3_max = d3_max    #Max prismatic extension (m)

        self.state = np.array([q1, q2, d3, q1_dot, q2_dot, d3_dot]) #[(rad), (rad), (m), (rad/s), (rad/s), (m/s)]

        self.time = 0       #time elapsed since start (sec)

        self.dt = 1./100.

        #self.generate_dynamic_equations()    
        self.calculate_symbolic_homo_n_jacobian()   

    def calculate_symbolic_homo_n_jacobian(self):
        self.q_symbols = sp.Matrix(sp.symbols('q1 q2 q3'))
        self.q_dot_symbols = sp.Matrix(sp.symbols('q1_dot q2_dot q3_dot'))
        Dh = np.matrix([[self.q_symbols[0], self.l1, self.l2, 0],
                          [self.q_symbols[1], 0, self.l3, sp.pi],
                          [0, self.q_symbols[2], 0, 0]])
                          
        self.sym_homogeneous,self.sym_Jacobian = Dh2HJ(Dh, prismatic = [3])
        self.sym_djacob_dt = np.zeros(sp.shape(self.sym_Jacobian))
        for i in range(len(Dh)):
            self.sym_djacob_dt = self.sym_djacob_dt + self.sym_Jacobian.diff(self.q_symbols[i])*self.q_dot_symbols[i]

    #Checks if a point lies in workspace
    def LiesInWorkspace(self, point):
        return ((self.l2-self.l3)**2 <= (point[0]**2 + point[1]**2) <= (self.l2+self.l3)**2) and (self.l1-self.d3_max <= point[2] <= self.l1) 

    def get_state(self):
        return self.state

    def Jacobian(self):
        q1, q2, d3,*_ = self.state
        
        return self.sym_Jacobian.subs([(self.q_symbols[0], q1),
                        (self.q_symbols[1], q2),
                        (self.q_symbols[2], d3)]).evalf()

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
        d3 = self.l1-z
 
        return np.array((q1, q2, d3))
    
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

        self.state = state_array

    def set_ee_position(self, position):
        self.set_position(self.inv_kin(position))

    def get_ee_position(self):
        q1,q2,d3,*_ = self.state
        return np.array([self.l2*np.cos(q1) + self.l3*np.cos(q1+q2), self.l2*np.sin(q1) + self.l3*np.sin(q1+q2), self.l1-d3])

    def reset(self, position = (0,0,1)):
        #Resets the RRM at a given position
        self.time = 0
        self.set_position(position)

