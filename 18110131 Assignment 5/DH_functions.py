#Calculating jacobian for arbitrary manipulators using DH parameters.
#Import this file in your manipulator specific python file. 
#For using jacobian calculator function, the DH_parameters has to be an array of 5 elements with dtype=object
# First column of DH parameters link types, second for a, third for alpha , fourth for d and fifth for theta


#Imports:
from os import link
import numpy as np
import sympy as sym
from sympy.simplify.simplify import simplify #Using sympy so that symbols can also be used. #Use sym.N() to convert answers to float

#to make skew symmetric from vector
def skew(z):
    skew_z=np.zeros([3,3],dtype=object)
    skew_z[0,1]=-z[2,0]
    skew_z[1,0]=z[2,0]
    skew_z[0,2]=z[1,0]
    skew_z[2,0]=-z[1,0]
    skew_z[1,2]=-z[0,0]
    skew_z[2,1]=z[0,0]
    return skew_z

def DH_Transform(a,alpha,d,theta):
    #Transforming 1 DH frame to next
    #Theta is rotation about z axis,
    #d is displacement about z axis
    #a is displacement in x axis
    #alpha is rotation about z axis

    H=np.array([[sym.cos(theta)   ,-sym.sin(theta)*sym.cos(alpha)   , sym.sin(theta)*np.sin(alpha)   ,a*sym.cos(theta)],
                [sym.sin(theta)  , sym.cos(theta)*sym.cos(alpha)   , -sym.cos(theta)*np.sin(alpha)  ,a*sym.sin(theta)],
                [0              , sym.sin(alpha)                 , sym.cos(alpha)                 ,d              ],
                [0              ,0                              ,0                              ,1              ]])
    
    return H

def Transformation_matrices (DH_param):
    #Returns a list of homogeneour transformation matrices: 0to0, 0to1, 0 to2, 0to 3rd frame, etc.
    H_list=[np.eye(4)] #ith frame's H transform w.r.t. 0 frame is stored at ith position. list length is linkCount+1
    linkCount=DH_param.shape[0]

    for i in range(linkCount):
        H=DH_Transform(DH_param[i,1],DH_param[i,2],DH_param[i,3],DH_param[i,4])
        H_list.append(H_list[i]@H)
    return H_list

def End_Position(DH_param,P):
    #Calculating end point w.r.t ground frame
    linkCount=DH_param.shape[0]
    H_list=Transformation_matrices(DH_param)

    P0=H_list[linkCount]@P #End point calculator
    return P0

def Jacobian_EndPoint_DH(DH_param,P):
    #To calculate Jacobian and end point w.r.t 0 frame 
    linkCount=DH_param.shape[0]
    H_list=Transformation_matrices(DH_param)
    O=np.array([[0],[0],[0],[1]]) #Represents origin in a frame. Used to convert origin of one particular frame to 0 frame perspective
    z=np.array([[0],[0],[1]]) # Z axis in a frame

    P0=H_list[linkCount]@P #End point calculator

    J0_P=np.zeros([6,linkCount],dtype=object) #Jacobian matrix

    for i in range(linkCount):
        H=H_list[i]
        if(DH_param[i,0]=='P'):
            #Prismatic joint

            zi_1=H[0:3,0:3]@z   #Multiplying Z with rotation matrix to get z axis w.r.t 0 frame
            J0_P[0:3,i:i+1]=zi_1 #Linear velocity component due to revolute
        else:
            #Revolute Joint

            Oi_1=H@O
            zi_1=H[0:3,0:3]@z   #Multiplying Z of i-1 frame with rotation matrix to get z axis w.r.t 0 frame
            
            skew_zi_1=skew(zi_1) #Skew symmetric matrix for matrix multiplication
            #J0_P[0:3,i:i+1]=np.matmul(skew_zi_1,np.subtract(P0,Oi_1)[0:3]) #Linear Velocity component due to revolute
            P_O_i_1=(P0-Oi_1)[0:3]        
            J0_P[0:3,i:i+1]=sym.simplify(skew_zi_1@P_O_i_1)
            J0_P[3:6,i:i+1]=sym.simplify(zi_1) #ANgular Velocity Component due to revolute

    return [J0_P,P0[0:3]]

def Jacobian_COM_i_link(DH_param, P_COM, link_index):
    #Calculate jacobian for i th link for its centre of mass, w.r.t ground(0 frame)

    linkCount=DH_param.shape[0]
    H_list=Transformation_matrices(DH_param)
    O=np.array([[0],[0],[0],[1]]) #Represents origin in a frame. Used to convert origin of one particular frame to 0 frame perspective
    z=np.array([[0],[0],[1]]) # Z axis in a frame

    P0=H_list[link_index]@P_COM #End point calculator

    Jvci=np.zeros([3,linkCount],dtype=object) #Jacobian matrix
    Jwi=np.zeros([3,linkCount],dtype=object) #Jacobian matrix 
    Ri=H_list[link_index-1][0:3,0:3] #Rotation matrix

    for i in range(link_index):
        H=H_list[i]
        if(DH_param[i,0]=='P'):
            #Prismatic joint
            zi_1=H[0:3,0:3]@z   #Multiplying Z with rotation matrix to get z axis w.r.t 0 frame
            Jvci[0:3,i:i+1]=zi_1 #Linear velocity component due to revolute

        else:
            #Revolute Joint

            Oi_1=H@O
            zi_1=H[0:3,0:3]@z   #Multiplying Z of i-1 frame with rotation matrix to get z axis w.r.t 0 frame
            
            skew_zi_1=skew(zi_1) #Skew symmetric matrix for matrix multiplication
            
            Jvci[0:3,i:i+1]=sym.simplify(np.matmul(skew_zi_1,np.subtract(P0,Oi_1)[0:3])) #Linear Velocity component due to revolute
            Jwi[0:3,i:i+1]=zi_1 #ANgular Velocity Component due to revolute

    return [P0,Jvci,Jwi,Ri]

def dynamics_equation_generator(DH_param,M,I,L):
    linkCount=DH_param.shape[0]
    #Declaring q and q dot symbolic arrays
    Q=[]
    Q_dot=[]
    for i in range(linkCount):
        Q.append(sym.Symbol('q'+str(i+1)))
        Q_dot.append(sym.Symbol('q_dot'+str(i+1)))   
    
    D=np.zeros([linkCount,linkCount])

    for link_i in range(1,linkCount+1):
        if(DH_param[link_i-1,0]=='P'):
            P_COM=np.array([[0],[0],[-L[link_i-1]/2],[1]])
        else:
            #Assuming link is parallel to x axis of next frame(link_i th frame)
            P_COM=np.array([[-L[link_i-1]/2],[0],[0],[1]])

        [P0,Jvci,Jwi,Ri]=Jacobian_COM_i_link(DH_param, P_COM, link_i)
        D=D+ M[link_i-1]*(np.transpose(Jvci) @ Jvci) + (np.transpose(Jwi)@Ri@ I[link_i-1]@np.transpose(Ri) @Jwi)
    # D=sym.simplify(D)
    
    C=np.zeros([linkCount,linkCount],dtype=object)

    for k in range(linkCount):
        for i in range(linkCount):
            #Summation over j. to get k th row and i th column
            for j in range(linkCount):
                C[k,i]=C[k,i]+(1/2)*Q_dot[j]*(sym.diff(D[k,i],Q[j])+sym.diff(D[k,j],Q[i])-sym.diff(D[j,i],Q[k]))

    return[D,C,Q,Q_dot]

def manipulatorPos(DH_param):
    O=np.array([[0],[0],[0],[1]])
    pos=[]
    H_list=Transformation_matrices(DH_param)
    for H in H_list:
        p=H@O
        pos.append(p[0:3,0])
    
    return pos




if __name__=="__main__":
    L= sym.symbols(['l1','l2','l3'])
    Q= sym.symbols(['q1','q2','q3'])
    #Test For above code.
    linkCount=2
    DH_param=np.zeros([linkCount,5],dtype=object)# First column link types, second for a, third for alpha , fourth for d and fifth for theta
    DH_param[0,0]='R'
    DH_param[1,0]='R'
    DH_param[0:1,1:5]=np.array([L[0],0,0,Q[0]])
    DH_param[1:2,1:5]=np.array([L[1],0,0,Q[1]])
    #q=pi/4,pi/4
    P=np.array([[0],[0],[0],[1]])

    J0_P,P0=Jacobian_EndPoint_DH(DH_param,P)
    print(J0_P)
    print(P0)

    q_dot=np.array([[1],[1]]) #each 1 rads per second

    #Velocity calculations
    v=J0_P[0:3,:]@q_dot
    w=J0_P[3:6,:]@q_dot

    print(v)
    print(w)

    M= sym.symbols(['m1','m2','m3'])
    I_xx= sym.symbols(['I1xx','I1yy','I1zz','I2xx','I2yy','I2zz','I3xx','I3yy','I3zz'])
    I=[np.array([[I_xx[0],0,0],[0,I_xx[1],0],[0,0,I_xx[2]]]),
        np.array([[I_xx[3],0,0],[0,I_xx[4],0],[0,0,I_xx[5]]]),
        np.array([[I_xx[6],0,0],[0,I_xx[7],0],[0,0,I_xx[8]]])]
    [D,C,Q,Q_dot]=dynamics_equation_generator(DH_param,M,I,L)
    print(sym.simplify(D))
    print(sym.simplify(C))
