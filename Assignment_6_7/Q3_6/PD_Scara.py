import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from controller import PD_Controller,PD_FeedForw_Controller, PD_FF_CTM_Controller,MultiVariable_Controller
from traj import oneD_traj

def angle_wrap(angle):
    coeffs = angle%(2*np.pi)
    if abs(coeffs - 2*np.pi)<1e-3:
        coeffs = 0
    return (coeffs)

class SCARA():
    def __init__(self,d1=0.25,a1 = 0.25, a2=0.25):

        '''
        Initializing the geometric setup for SCARA.

                                     Z  Y
        o--------o--------o          | /
        |  m2,a1  m3,a2              |/___ X
        |  
        | m1,d1
        |
        =

        '''
        self.d1 = d1    # Initial elevation
        self.a1 = a1    # Link Length 2
        self.a2 = a2    # Link Length 3

        self.m1 = 1     # Mass of Link 1
        self.m2 = 1    
        self.m3 = 1

        self.I1 = 0
        self.I2 = self.m2*self.a1**2/3
        self.I3 = self.m3*self.a2**2/3
        
        '''
        Initializing the states for each angle.
        q -> angle position. For scara, the last term indicates the 
             position of prismatic joint
        q_dot -> derivative of the positions (velocity)
        q_ddot -> accelerations.
        '''

        self.q =      np.array([0,0,0])
        self.q_dot =  np.array([0,0,0])
        self.q_ddot = np.array([0,0,0])

        '''
        Use any of the 4 controllers
         - The first matrix of the input of the controller denotes the individual joint
           angles.
         - For the controllers with Feed Forward Term, the second term deontes the gains 
           for the motor dynamics
         - For Multivariable controllers, the gains are to be put as 
           [omega_1,omega_2,omega_3 .... omega_n]. Further calculations of the proportional distribution
           and derivative terms will be done in the controller block. 
           It is to be noted that Multivariable controller works a little differently than other controllers.
        '''

        # -------------------------------------------
        '''SIMPLE PD CONTROLLER'''
        # self.controller = PD_Controller([[2.0,0.5],
        #                                  [0.7,0.3],
        #                                  [100,0.5]])
        # self.cntrl_type = 'PD'
        # ----------------------------------------------
        '''PD CONTROLLER WITH FEED FORWARD'''
        # self.controller = PD_FeedForw_Controller([[2.05,0.5],
        #                                           [0.74,0.3],
        #                                           [100,0.5]], 1)
        # self.cntrl_type = 'PD_FF'

        # ----------------------------------------------
        '''PD CONTROLLER WITH FEED FORWARD and CTM'''
        #  
        # self.controller = PD_FF_CTM_Controller([[2.05,0.5],
        #                                           [0.75,0.3],
        #                                           [100,0.5]], 1)
        # self.cntrl_type = 'PD_FF_CTM'

        # ----------------------------------------------
        '''MULTIVARIABLE CONTROLLER'''
        # #
        self.controller = MultiVariable_Controller([0.2,0.1,0.2])
        self.cntrl_type = "multiVar_invDyna"



        # ----------------------------------------------

        # Adding disturbance to the system
        isdisturbance = False
        self.disturbance = 0
        if isdisturbance :
            self.disturbance = np.random.normal(0,0.1)
        
        # -----------------------------------------------

        # Adding uncertainity in the link lengths. This will be taken into account 
        # in the link lengths dynamics calculations. To add uncertainity, toggles
        # "isUncertain" to True.
        isUncertain = False
        self.uncert_Factor = 1
        if isUncertain :
            self.uncert_Factor = np.random.uniform(0.8,1.2)

        # ----------------------------------------------
        # Adding Impulse 
        self.impulse = 0.0
        self.impulse_flag = 1

        self.update_endEffectorPosition()

    def update_endEffectorPosition(self,):
        self.x_EF,self.y_EF,self.z_EF = self.forward_kinematics(self.q)

    def inverse_kinematics(self,points):
        xc,yc,zc = points
        # Workspace condition
        if np.sqrt(xc**2+yc**2)>self.a2 + self.a1:
            print("No Solution can be Found!")
            return False,[None, None, None]
        else: 
            def inv_func(x):
                return [
                        - xc + self.a1*np.cos(x[0]) + self.a2*np.cos(x[0]+x[1]),
                        - yc + self.a1*np.sin(x[0]) + self.a2*np.sin(x[0]+x[1]),
                        - zc + self.d1 - x[2]
                        ]
            root = fsolve(inv_func,[1,1,1])

            q1,q2,d = root

            return True, [angle_wrap(q1),angle_wrap(q2),d]
    
    def forward_kinematics(self,q):
        q1,q2,d = q
        xc = self.a1*np.cos(q1) + self.a2*np.cos(q1+q2)
        yc = self.a1*np.sin(q1) + self.a2*np.sin(q1+q2)
        zc = self.d1 - d

        return xc,yc,zc
    
    def calc_MM_Cor_Grav(self,q,q_dot):
        g  = 9.8
        
        a1= self.a1*self.uncert_Factor
        a2 = self.a2*self.uncert_Factor

        alpha = self.I1 + a1**2*(self.m1/4 + self.m2 + self.m3)
        beta = self.I2 + self.I3 + a2**2*(self.m2/4 +self.m3) 
        gamma = a1*a2*self.m3 + a1 * a2/2 * self.m2

        MM = np.around(np.array([[alpha + beta + 2*gamma*np.cos(q[1]), beta + 2*gamma*np.cos(self.q[1]), 0],
                                 [beta + 2*gamma*np.cos(q[1]), beta, 0],
                                 [0, 0, self.m3]]) , decimals= 4)
        
        
        C = np.around(np.array([[-gamma*np.sin(q[1])*q_dot[1], -gamma*np.sin(q[1])*(q_dot[1] + q_dot[0]), 0],
                                [gamma*np.sin(q[1])*q_dot[0], 0, 0],
                                [0, 0, 0]]), decimals = 4)

        G = np.around(np.transpose(np.array([0, 0, self.m3*g])), decimals = 4)
        
        return MM,C,G,gamma

    def dynamics_solver(self,dt,q_des, **kwargs):
        
        g = 9.8
        e = q_des - self.q

        MM,C,G ,gamma= self.calc_MM_Cor_Grav(self.q,self.q_dot)
        MM_d,C_d,G_d ,gamma= self.calc_MM_Cor_Grav(q_des,kwargs['qd_dot'])
        
        # Controller
        if self.cntrl_type!="multiVar_invDyna":

            self.tau_1,self.tau_2,self.F_3 = self.controller.track_angles(e,qd_dot = kwargs['qd_dot'],qd_ddot = kwargs['qd_ddot'],
                                                                            q_dot = self.q_dot,q_ddot = self.q_ddot,
                                                                            MIM_d = MM_d, Cor_d = C_d, Grav_d = G_d,
                                                                            MIM = MM, Cor = C, Grav = G,)
            

            #  Applying impulse to the dynamics
            impulse = 0
            if np.random.random() > 0.8 and self.impulse_flag:
                impulse = self.impulse
                self.impulse_flag = False

            # Applying disturbance to the dynamics
            self.tau_1 += self.disturbance + impulse
            self.tau_2 += self.disturbance + impulse
            self.F_3 += self.disturbance + impulse


            # Dynamics calculations
            u1 = self.tau_1
            u2 = self.tau_2
            u3 = self.F_3  + self.m3*g

            U = np.around(np.transpose(np.array([u1, u2,u3])), decimals = 4)

            K = np.around((U - np.matmul(C, np.transpose(self.q_dot))-G), decimals = 4)

            self.q_ddot = np.around(np.matmul(np.linalg.inv(MM), K), decimals = 4)
        
        else:
            self.q_ddot = self.controller.track_angles(e,qd_dot = kwargs['qd_dot'],qd_ddot = kwargs['qd_ddot'],
                                                                            q_dot = self.q_dot,q_ddot = self.q_ddot,
                                                                            MIM_d = MM_d, Cor_d = C_d, Grav_d = G_d,
                                                                            MIM = MM, Cor = C, Grav = G,)

        self.q_dot = self.q_dot + self.q_ddot*dt
        self.q = self.q + self.q_dot*dt + 1/2*self.q_ddot*dt**2


        for i in range(2):
            self.q[i] = angle_wrap(self.q[i])
        
        self.update_endEffectorPosition()

    def control(self,):
        points,t = oneD_traj([[0.4,0.06,0.1],
                               [0.4,0.01,0.1]],0,10)
        ef = [points[0]]
        des = [points[0]]


        ret, self.q = self.inverse_kinematics(points[0])
        self.q = np.array(self.q)
        self.update_endEffectorPosition()
        
        i = 1
        qd = self.q.copy()
        qd_dot = np.array([0,0,0])

        while True:

            if i<len(t)-1:
                dt = t[i]-t[i-1]
                ret,trk_q = self.inverse_kinematics(points[i])

                qd_dot_ = (trk_q - qd)/dt
                qd_ddot = (qd_dot_ - qd_dot)/dt
                qd_dot = qd_dot_
                qd = np.array(trk_q)


                self.dynamics_solver(dt,trk_q,qd_dot=qd_dot,qd_ddot=qd_ddot)

                ef.append([self.x_EF,self.y_EF,self.z_EF])
                des.append(points[i])
                # pass
            else:
                break
                pass
            
            i+=1
        ef = np.array(ef)
        des = np.array(des)

        # print(ef)

        plt.figure(0)
        plt.plot(ef[:,1])
        plt.plot(des[:,1])
        plt.title = "X"
        plt.ylim([-0.2,0.2])
        plt.grid()

        plt.figure(1)
        plt.title = "Y"

        plt.plot(ef[:,0])
        plt.plot(des[:,0])
        plt.ylim([-1,1])
        plt.grid()

        plt.figure(2)
        plt.title = "Z"

        plt.plot(ef[:,2])
        plt.plot(des[:,2])
        plt.ylim([-1,1])
        plt.grid()

        plt.show()





            
p = SCARA()
p.control()
# p.find_P2P_trajectory([[0.4,0.06,0.1],
#                        [0.4,0.01,0.1]],
                       
#                        [[0,0,0],
#                         [0,0,0]], 10,'cubic')