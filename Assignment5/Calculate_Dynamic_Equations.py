# Calculates dynamic equations
# @author Videh Patel : videh.p@iitgn.ac.in : 19110192

from sympy.simplify.simplify import simplify
from Dynamic_Equations_from_DV import DynamicEqns_from_DV as Eqn_DV
from DHMatrix2Homo_and_Jacob import DHMatrix2Homo_and_Jacob as Dh2HJ

import numpy as np
import sympy as sp

# Dhmi matrix
# [[theta1,      d1,     l1,     alpha1,    m1,     I1],
#  [theta2,      d2,     l2,     alpha2,    m2,     I2]]

# I1, I2 are 3x3 inertia tensors wrt the position of next link frame 

def DV_Calculator(Dhmi, g_dir ='-z', prismatic = []):
    #Assumes the centres of mass of links to lie at the middle
    #A link is assumed with given mass connecting each frame
    #Also, assumes the links to have moment of inertia of (m*l**2)/12
    #mass_arr is symbolic matrix
    #Dhmi is a matrix with manipulator lengths, link masses passed as constants and joint variables passed as symbolic constants
    #g_dir is used to calculate the V, specifies the direction of g; Default is taken as -z
    #Can also pass everything as symbolic constants
    n = sp.shape(Dhmi)[0]
    D = np.matrix(np.zeros((n,n)))
    V = 0

    dir_check_index = 0
    if g_dir[0] == '-':
        dir_m = 1
        dir_check_index = 1
    else:
        dir_m = -1

    if g_dir[dir_check_index] == 'z':
        dir_i = 2
    elif g_dir[dir_check_index] == 'y':
        dir_i = 1
    elif g_dir[dir_check_index] == 'x':
        dir_i = 0

    print('Starting to calculate Inertia matrix(D) and potential function(V)')
    for i in range(1, n+1):
        print('... Iterating for D & V:: loop: ' + str(i))
        Dh_ = np.concatenate((Dhmi[:i-1,:4], np.matrix([Dhmi[i-1,0], Dhmi[i-1,1]/2, Dhmi[i-1,2]/2, Dhmi[i-1,3]])))
        H, J = Dh2HJ(Dh_, [x for x in prismatic if x<=i])

        Jv = sp.nsimplify(sp.Matrix(J[:3,:]).row_join(sp.zeros(3,n-i)), constants = [sp.pi], tolerance = 1e-5, rational = False)
        Jw = sp.nsimplify(sp.Matrix(J[3:,:]).row_join(sp.zeros(3,n-i)), constants = [sp.pi],tolerance = 1e-5, rational = False)
        R = sp.simplify(H[:3,:3])

        I = Dhmi[i-1,5]
        m = Dhmi[i-1,4]
        g = sp.symbols('g')

        D = D + sp.simplify(m*Jv.T@Jv + Jw.T@R@I@R.T@Jw)
        V = V + sp.simplify(dir_m*m*g*H[dir_i,3])
    
    return sp.simplify(sp.nsimplify(D, tolerance = 1e-10)), sp.simplify(sp.nsimplify(V, tolerance = 1e-10))

def Dynamic_Equations_from_Dhmi(Dhmi, g_dir ='-z', prismatic = []):
    #Calculates the dynamic equations using the Dhmni and prismatic matrix passed
    
    Dexp, Vexp = DV_Calculator(Dhmi, g_dir = g_dir, prismatic = prismatic)
    
    C, g_ = Eqn_DV(Dexp,Vexp)
    print('Calculated C and g_')
    return Dexp, sp.simplify(sp.nsimplify(C, tolerance = 1e-10)), sp.simplify(sp.nsimplify(g_, tolerance = 1e-10))

def q_dot2_array(Dhmi, tau, q_dot, g_dir ='-z', prismatic = []):
    # Returns array of q_dot2 for the given symbolic matrix of Dhmi tau and q_dot
    Dexp, C, g_ = Dynamic_Equations_from_Dhmi(Dhmi, g_dir = g_dir, prismatic = prismatic)

    return Dexp.inv()@(tau - C@q_dot - g_)

if __name__ == "__main__":
    from sympy import*

    m1, m2 = symbols('m1 m2')
    q_symbols = sp.Matrix(sp.symbols('q1 q2 q3'))
    q_dot_symbols = sp.Matrix(sp.symbols('q1_dot q2_dot q3_dot'))
    l1, l2 = sp.symbols('l1 l2')

    I_zero = sp.zeros(3)
    I1 = sp.Matrix([[m1*l1**2/12, 0, 0],
                    [0 , 0, 0],
                    [0, 0, m1*l1**2/12]])
    
    I2 = sp.Matrix([[m2*l2**2/12, 0, 0],
                    [0 , m2*l2**2/12, 0],
                    [0, 0, 0]])

    Dhmi = sp.Matrix([[q_symbols[0], l1, 0, np.pi/2, m1, I1],
                        [np.pi/2 + q_symbols[1], 0, 0, -np.pi/2, 0, I_zero],
                        [0, l2+ q_symbols[2], 0, 0, m2, I2]])
    
    D, C, g_ = Dynamic_Equations_from_Dhmi(Dhmi, prismatic=[3])