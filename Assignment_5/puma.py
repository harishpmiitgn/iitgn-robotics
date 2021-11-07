
import numpy as np
from scipy.optimize import fsolve
from controller import PI_Controller
import matplotlib.pyplot as plt

def angle_wrap(angle):

    a = angle%(2*np.pi)
    # print(a)
    if abs(a - 2*np.pi)<1e-3:
        a = 0
    return (a)

class PUMA():
    def __init__(self,d1=1,a2=1,a3=1, gains  = [[0,0,0],
                                                [0,0,0],
                                                [0,0,0]]):
        
        self.m1 = 1
        self.m2 = 1
        self.m3 = 1

        self.d1 = d1
        self.a2 = a2
        self.a3 = a3

        self.x = np.zeros(3)
        self.x_dot = np.zeros(3)
        self.x_d_dot = np.zeros(3)

        self.tau_1 = 0
        self.tau_2 = 0
        self.tau_3 = 0

        self.q1 = 0
        self.q2 = 0
        self.q3 = 0

        self.compute_inertias()
        self.update_endEffectorPosition()

        self.controller = PI_Controller(gains)


    def compute_inertias(self,):
        self.J1 = (1/3)*self.m1*self.d1*2
        self.J2 = (1/3)*self.m2*self.a2*2
        self.J3 = (1/3)*self.m3*self.a3*2

    def update_endEffectorPosition(self,):
        self.forward_kinematics([self.q1,self.q2,self.q3])
    
    def inverse_kinematics(self,x):
        xc,yc,zc = x
        # Workspace condition
        D = (xc**2 + yc**2 + (zc-self.d1)**2 - self.a2**2 - self.a3**2)/(2*self.a2*self.a3)
        if abs(D)<=1:
            def inv_func(x):
                return [
                        -xc + self.a2*np.cos(x[1])*np.cos(x[0]) + self.a3*np.cos(x[1]+x[2])*np.cos(x[0]),
                        -yc + self.a2*np.cos(x[1])*np.sin(x[0]) + self.a3*np.cos(x[1]+x[2])*np.sin(x[0]),
                        -zc + self.d1 + self.a2*np.sin(x[1]) + self.a3*np.sin(x[1]+x[2])
                        ]
            root = fsolve(inv_func,[0,0,0])
            q1,q2,q3 = root
            
            self.q1,self.q2,self.q3 = angle_wrap(q1),angle_wrap(q2),angle_wrap(q3)
            # Returns True if solution exists
            return True, [self.q1,self.q2,self.q3]
        else:
            # Returns True if solution exists
            print("Angles provided not in workspace")
            return False, [None, None, None]
    
    def forward_kinematics(self,x):
        q1,q2,q3 = x
        xc = self.a2*np.cos(q2)*np.cos(q1) + self.a3*np.cos(q2+q3)*np.cos(q1)
        yc = self.a2*np.cos(q2)*np.sin(q1) + self.a3*np.cos(q2+q3)*np.sin(q1)
        zc = self.d1 + self.a2*np.sin(q2) + self.a3*np.sin(q2+q3)

        self.xE,self.yE,self.zE = xc,yc,zc

        return xc,yc,zc
    
    def dynamics_solver(self,dt,x_des):
        g = 9.8

        e = x_des - self.x
        self.tau_1,self.tau_2,self.tau_3 = self.controller.track_angles(e)

        A1 = (self.m2/4 + self.m3)*self.d1**2
        A2 = self.m3*self.a3**2/4
        A3 = self.m3*self.a3*self.a2/2

        B1= (self.m2/2+self.m3)*self.a2*g
        B2 = self.m3*self.a3*g/2

        m11 = A1*np.cos(self.x[1]) + A2*np.cos(self.x[1]+self.x[2])**2 + 2*A3*np.cos(self.x[1])*np.cos(self.x[1]+self.x[2]) + self.J1
        m22 = A1 + A2 + 2*A3*np.cos(self.x[2]) + self.J2
        m33 = A2 + self.J3
        m23 = m32 = A2 + A3*np.cos(self.x[2])
        m12 = m21 = m13 = m31 = 0

        MM = np.array([[m11, m12, m13],
                       [m21, m22, m23],
                       [m31, m32, m33]])
        


        
        b11 = -1/2*A1*self.x_dot[1]*np.sin(2*self.x[1]) - 1/2*A2*(self.x_dot[1]+self.x_dot[2])*np.sin(2*(self.x[1]+self.x[2])) - A3*self.x_dot[1]*np.sin(2*self.x_dot[1]+self.x_dot[2]) - A3*self.x_dot[2]*np.cos(self.x[1]+self.x[2])
        b12 = -1/2*A1*self.x_dot[0]*np.sin(2*self.x[1]) - 1/2*A2*self.x_dot[0]*np.sin(2*(self.x[1]+self.x[2])) - A3*self.x_dot[0]*np.sin(2*self.x_dot[1]+self.x_dot[2]) 
        b13 = -1/2*A2*self.x_dot[0]*np.sin(2*(self.x[1]+self.x[2])) - A3*self.x_dot[0]*np.cos(self.x[1])*np.sin(self.x[1]+self.x[2])
        b21 = -b12
        b22 = -A3*self.x_dot[2]*np.sin(self.x[2])
        b23 = -A3*(self.x_dot[1]+self.x_dot[2])*np.sin(self.x[2])
        b31 = -b13
        b32 =  A3*self.x_dot[1]*np.sin(self.x[2])
        b33 =  0

        B_qdot = np.array([[b11, b12, b13],
                           [b21, b22, b23],
                           [b31, b32, b33]])

        f = np.array([0,
                      B1*np.cos(self.x[1]) + B2*np.cos(self.x[1]+self.x[2]),
                      B2*np.cos(self.x[1]+self.x[2])])
        
        Tau_matrix = np.array([self.tau_1,self.tau_2,self.tau_3]).transpose()

        q_d_dot = np.matmul(np.linalg.inv(MM),(Tau_matrix - f - np.matmul(B_qdot,self.x_dot.transpose())))

        self.x_dot = self.x_dot + q_d_dot*dt
        self.x = self.x_dot*dt + 1/2*q_d_dot*dt**2
        
        for i in range(3):
            self.x[i] = angle_wrap(self.x[i])
        
        self.update_endEffectorPosition()

    def point_tracking(self,trk_points):

        xt,yt,zt = trk_points

        ret, angles = self.inverse_kinematics([xt,yt,zt])

        max_time = 100
        num_steps = 3000
        t = np.linspace(0,max_time,num=num_steps )
        y = []
        y_inv = []
        y.append(self.x)
        y_inv.append([self.xE,self.yE,self.zE])

        for i in range(len(t)):
            self.dynamics_solver(max_time/num_steps,angles)
            y.append(self.x)
            y_inv.append([self.xE,self.yE,self.zE])
            

        return np.array(y),np.array(y_inv)

if __name__ == "__main__":
    s = PUMA(gains  = [[0,0.0,0.0],
                       [0,0.0,0.0],
                       [0,0.0,0]])
    x_t = 1
    y_t = 1
    z_t = 1

    y,y_inv = s.point_tracking([x_t,y_t,z_t])
    plt.figure(1)
    plt.plot(y_inv[:,2])
    plt.plot(np.ones_like(y_inv[:,2])*z_t)
    plt.plot(y_inv[:,0])
    plt.plot(np.ones_like(y_inv[:,2])*x_t)
    plt.plot(y_inv[:,1])
    plt.plot(np.ones_like(y_inv[:,2])*y_t)

    plt.show()