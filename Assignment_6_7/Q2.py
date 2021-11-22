

import numpy as np
import matplotlib.pyplot as plt

def angle_wrap(angle):
    coeffs = angle%(2*np.pi)
    if abs(coeffs - 2*np.pi)<1e-3:
        coeffs = 0
    return (coeffs)

class PUMA():
    def __init__(self,d1=0.25,a2=0.25,a3=0.25):
        self.d1 = d1
        self.a2 = a2
        self.a3 = a3

        self.q = np.array([0,0,0])
        self.q_dot = np.array([0,0,0])
        self.q_ddot = np.array([0,0,0])


        self.update_endEffectorPosition()
    
    def update_endEffectorPosition(self,):
        self.x_EF,self.y_EF,self.z_EF = self.forward_kinematics(self.q)

    def inverse_kinematics(self,point):
        
        q1 = np.arctan2(point[1],point[0])
        D = (point[0]**2 + point[1]**2 + (point[2]-self.d1)**2 -self.a2**2 - self.a3**2)/(2*self.a2*self.a3)
        if abs(D)<=1:
            q3 = np.arctan2(np.sqrt(1-D**2),D)
            q2 = np.arctan2(point[2]-self.d1,np.sqrt(point[0]**2 + point[1]**2)) - np.arctan2(self.a2*np.sin(q3),self.a2 + self.a3*np.cos(q3))

            return True, [angle_wrap(q1),angle_wrap(q2),angle_wrap(q3)]

        else:
            print("Error. The given inputs are out of bounds of workspace")
            return False, [None,None,None]
            
            
    def forward_kinematics(self,q):
        xc = self.a2*np.cos(q[1])*np.cos(q[0]) + self.a3*np.cos(q[1]+q[2])*np.cos(q[0])
        yc = self.a2*np.cos(q[1])*np.sin(q[0]) + self.a3*np.cos(q[1]+q[2])*np.sin(q[0])
        zc = self.d1 + self.a2*np.sin(q[1]) + self.a3*np.sin(q[1]+q[2])

        return xc,yc,zc

    def plotter(self,q):
        plt.plot(q[0])
        plt.plot(q[1])
        plt.plot(q[2])
        plt.grid()

    def find_P2P_trajectory(self,points,vdots,tf,type = 'cubic'):
        intervals = 10
        if type == 'cubic':
            # Cubic Trajectory 
            init_point = np.array(points[0])
            # Checking and calculating inverse kinematics of given initial point
            ret,qi = self.inverse_kinematics(init_point)
            if ret:
                self.q = np.array(qi)
                self.q_dot = np.array(vdots[0])
                self.update_endEffectorPosition()

            # Checking and calculating inverse kinematics of given final point
            final_point = np.array(points[1])
            ret, qf = self.inverse_kinematics(final_point)

            if ret:
                qf = np.array(qf)
                final_vdots = np.array(vdots[1])
                B = np.vstack((self.q,
                            self.q_dot,
                            qf,
                            final_vdots))
                
                A = np.array([[1,0,     0,      0],
                              [0,1,     0,      0],
                              [1,tf,tf**2,  tf**3],
                              [0,1,  2*tf,3*tf**2]])

                coeffs = np.matmul(np.linalg.inv(A),B)

                t = np.linspace(0,tf,intervals).reshape(1,-1)

                c0 = coeffs[0].reshape(1,-1).transpose()
                c1 = coeffs[1].reshape(1,-1).transpose()
                c2 = coeffs[2].reshape(1,-1).transpose()
                c3 = coeffs[3].reshape(1,-1).transpose()

                q = c0 + c1@t + c2@t**2 + c3@t**3
                q_dot = c1 + 2*c2@t + 3*c3@t**2
                q_ddot = 2*c2 + 6*c3@t
                
                

                end_effector = []
                for i in range(intervals):
                    qs = q[:,i]
                    xe,ye,ze = self.forward_kinematics(qs)
                    end_effector.append([xe,ye,ze])
                end_effector =  np.array(end_effector)
       
        if type=='straight':
            init_point = np.array(points[0])
            final_point = np.array(points[1])
            ret,qi = self.inverse_kinematics(init_point)
            ret,qf = self.inverse_kinematics(final_point)

            end_effector = np.linspace(init_point,final_point,intervals)
            t = np.linspace(0,tf,intervals)
            dt = tf/intervals

            q = []
            q_dot = []
            q_ddot = []

            prev_q     = qi
            prev_qdot  = self.q_dot
            prev_qddot = self.q_ddot


            for i in range(intervals):
                ret, angles = self.inverse_kinematics(end_effector[i])
                if ret:
                    q.append(np.array(angles))
                    q_dot.append((np.array(angles)- prev_q)/dt)
                    temp = (np.array(angles)- prev_qdot)/dt 
                    q_ddot.append((temp- prev_qddot)/dt)


                    prev_q = angles
                    prev_qdot = temp
                    prev_qddot = (temp- prev_qddot)/dt

                else:
                    print('Point cannot be reached')
                    exit()
            
            q = np.array(q).T

            q_dot = np.array(q_dot).T
            q_ddot = np.array(q_ddot).T
        
        plt.figure(1)
        self.plotter(q)
        plt.figure(2)
        self.plotter(q_dot)
        plt.figure(3)
        self.plotter(q_ddot)
        plt.figure(4)
        ax = plt.axes(projection='3d')
        ax.plot3D(end_effector[:,0], end_effector[:,1], end_effector[:,2], 'gray')
    
        plt.show() 




            
p = PUMA()
p.find_P2P_trajectory([[0.4,0.06,0.1],
                       [0.4,0.01,0.1]],
                       
                       [[0,0,0],
                        [0,0,0]], 10,'cubic')