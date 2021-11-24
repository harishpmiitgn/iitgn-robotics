import numpy as np
import sympy as sp
# @author Videh Patel :: videh.p@iitgn.ac.in :: 19110192
#Did minor changes since assignment3 --> To implement frames correctly 

# EDITTED FOR SYMBOLIC MANIPULATION
# DH Parameter Matrix [nx4 Matrix]
# Number of links automatically calculated from the size of matrix
# theta (rotation about z), d (translation about z), a(translation about x), alpha(rotation about x)
# First columns of params are frame 0-->1 and last columns are frame (n-1)-->n :: n is the end effector frame

def DHMatrix2Homo_and_Jacob(Hmat, prismatic=[]):
    #Hmat is the DH parameter matrix
    #prismatic is an array of the joints that are prismatic: Joint corresponding to qi is ith joint
    #All revolute and prismatic axes are assumed to be aligned with the respective z axis

    def d_(k):
        #Returns position vector of ee wrt k-frame in o-frame basis
        return (T[k][:3,:3])@((T[k]).inv()@T[n]@(sp.Matrix([[0],[0],[0],[1]])))[:3,0]
        
    n = len(Hmat)
    T = [sp.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]  #Array of all absolute Homogeneous Transformations : Has T0_0 at the start (Identity matrix)
    J = []  #Manipulator Jacobian
    for params in Hmat:
        theta, d, a, alpha = np.array(params)[0]
        Ti = sp.Matrix([ [ sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],\
                         [ sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],\
                         [ 0            , sp.sin(alpha)               , sp.cos(alpha)              , d              ],\
                         [ 0            , 0                           , 0                          ,1              ] ])
        

        T.append(T[-1]@Ti)

    for i in range(n):
        if (i+1) in prismatic:
            Jv = T[i][:3,:3]@sp.Matrix([0,0,1])
            Jw = sp.Matrix([0,0,0])
        else: #Revolute
            Jw = T[i][:3,:3]@sp.Matrix([0,0,1])
            Jv = sp.Matrix(np.cross(Jw.T,d_(i).T).T)
        
        Ji = np.concatenate((sp.simplify(sp.nsimplify(sp.Matrix(Jv), [sp.pi], tolerance = 1e-10, rational= False)), sp.simplify(sp.nsimplify(sp.Matrix(Jw), [sp.pi], tolerance = 1e-10, rational= False))))
        
        if len(J) == 0:
            J = Ji
        else:
            J = np.concatenate((J,Ji), axis = 1)

        H = T[n]

    return H, J


if __name__ == "__main__":
    H = sp.Matrix([ [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 10, 0] ])
    q_dot = sp.Matrix([5, 5, 5, 5]).T #rad/s, rad/s, rad/s, m/s



    Homo, Jacob = DHMatrix2Homo_and_Jacob(H,[4]) 
    position = (Homo)[:3,3]
    velocity_mat = Jacob@q_dot
    linear_velo = velocity_mat[:3,0]
    angular_velo = velocity_mat[4:,0]

    print('Homogeneous Transformation Matrix ::')
    print(Homo)
    print()
    print("Jacobian::")
    print(Jacob)
    print()
    print("Position for given orientation::")
    print(position)
    print()
    print("Linear Velocity::")
    print(linear_velo)
    print()
    print("Angular Velocity::")
    print(angular_velo)
    print()