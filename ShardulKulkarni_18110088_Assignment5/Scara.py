import numpy as np 
import cv2
from render import Renderer
import scipy
from scipy.integrate import odeint

import matplotlib.pyplot as plt

class Robot(Renderer):
    def __init__(self,x0,y0,x,y,kp,ki,recordlocation=None):
        super().__init__(recordLocation=recordlocation)
        self.ode = scipy.integrate.ode(self.func).set_integrator('vode', nsteps = 500 , method = 'bdf')
        
        self.l1=300
        self.l2=200
        self.k=1

        #Destination
        ori=self.cordtoang(x0,y0)
        self.q10=ori[0]
        self.q20=ori[1]

        #Initial Point
        ini=self.cordtoang(x,y)
        self.q1=ini[0]
        self.q2=ini[1]
        self.q1dot=0
        self.q2dot=0

        #Controller Parameters
        self.P1=0
        self.I1=0

        self.P2=0
        self.I2=0

        self.preverr1=0
        self.preverr2=0

        self.kp=kp
        self.ki=ki

        #moments of inertia
        self.i1=30000
        self.i2=13333

    def cordtoang(self,x,y):
        #Inverse Kinematics
        theta = (x**2+y**2-self.l1**2-self.l2**2)/(2*self.l1*self.l2)
        theta = np.arccos(theta)

        q1 =np.arctan(y/x)-np.arctan(self.l2*np.sin(theta)/(self.l1+self.l2*np.cos(theta)))

        q2 = theta + q1

        return (q1,q2)

        

    def func(self,t,y):
        q1=y[0]
        q2=y[1]

        q1dot = y[2]
        q2dot = y[3]

        b=0.3 #Damping coefficient

        # PI controller Equations
        #1
        err1=self.q10-q1

        self.P1=self.kp*err1
        self.I1+=self.ki*(err1+self.preverr1)*0.0001

        self.preverr1=err1
        #2
        err2=self.q20-q2

        self.P2=self.kp*err2
        self.I2+=self.ki*(err2+self.preverr1)*0.0001

        self.preverr2=err2

        t1=(self.P1+self.I1)
        t2=(self.P2+self.I2)

        #Dynamics equations
        t=np.matrix([[t1],[t2]])

        c=np.matrix([[-self.l1*self.l2*np.sin(q2)*(q1dot+0.5*q2dot)*q2dot],[0.5*self.l1*self.l2*np.sin(q2)*q1dot*q1dot]])

        d11=self.i1+(self.l1)**2+0.25*((self.l2)**2)+self.i2+self.l1*self.l2*np.cos(q2)
        d12=0.25*(self.l2**2)+self.i2+0.5*self.l1*self.l2*np.cos(q2)
        d21=0.25*(self.l2**2)+self.i2+0.5*self.l1*self.l2*np.cos(q2)
        d22=0.25*(self.l2**2)+self.i2
        d=np.matrix([[d11,d12],[d21,d22]])

        #calculating qddot using t and dynamics equaions
        qddot = np.matmul(np.linalg.inv(d),t-c)



        dydt=[q1dot , q2dot , qddot[0][0] , qddot[1][0]]
        
        
        return dydt

    def step(self, dt):
        state = [self.q1, self.q2, self.q1dot , self.q2dot]

        self.ode.set_initial_value(state, 0)
        newstate = self.ode.integrate(dt)

        self.q1=newstate[0]
        self.q2=newstate[1]

        self.q1dot=newstate[2]
        self.q2dot=newstate[3]

        
    def getInfo(self):
        info = {
            "Position" : self.pos
        }
        return info

    def draw(self, image):
       

        j1 = (int(300+self.l1 * np.cos(-self.q1-np.pi/4)) , int(500+self.l1 * np.sin(-self.q1-np.pi/4)))

        j2 = (int(j1[0] + self.l2 * np.cos(-self.q2-np.pi/4)) , int(j1[1] + self.l2 * np.sin(-self.q2-np.pi/4)))

        self.pos=j2

        cv2.line(image,(300,500), j1, (0,255,0), 2)
        cv2.line(image,j1, j2, (0,0,255), 2)
        cv2.circle(image,j2,10,(255,0,0),-1)

        return image

    
# class PI():
#     def __init__(self,kp,ki,kd):
#         self.P=0
#         self.I=0
#         self.D=0

#         self.preverr=0

#         self.kp=kp
#         self.ki=ki
#         self.kd=kd

#     def update(self,setpoint,measurement):
#         err=setpoint-measurement

#         self.P=self.kp*err
#         self.I+=self.ki*(err+self.preverr)*0.1/2
#         self.D+=self.kd*(err-self.preverr)*2/0.1

#         self.preverr=err

robot = Robot(100,200,100,100,100,2,"Task3.mp4")


for i in range(500):
    robot.step(0.1)
    robot.render()
    



