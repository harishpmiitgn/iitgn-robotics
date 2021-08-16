#Videh Patel : videh.p@iitgn.ac.in
#19110192

from RRM_videh import*
from matplotlib.backend_bases import MouseButton


class RRMController:
    #Calculates and applies complex sets of torques to receive desired output from the RRManipulator
    def __init__(self, RRMmodel):
        self.Calliberate(RRMmodel)
        self.dt = RRMmodel.dt
        self.m1 = RRMmodel.m1
        self.m2 = RRMmodel.m2
        self.l1 = RRMmodel.l1
        self.l2 = RRMmodel.l2
        self.RRMAppliesTorques = RRMmodel.apply_torques
        self.fig = RRMmodel.fig
        self.ax = RRMmodel.ax
        self.RRMmodel = RRMmodel
        self.mchaji = 0
        self.spring_acting = False
        self.spring_k = None
        self.spring_pt = (None,None)

    def Applies_Torques(self, torq_tup, force_tup = (0,0)):
        self.state = self.RRMAppliesTorques(torq_tup, force_tup)
        self.time += self.dt

    def Calliberate(self, RRMmodel):
        self.state = RRMmodel.get_state()
        self.time = RRMmodel.time

    def LiesInWorkspace(self, point):
        return (self.l1 - self.l2)**2 <= (point[0]**2 + point[1]**2) <= (self.l1 + self.l2)**2

    def Achieve_EE_Position(self, position_tup, end_qdots = (0,0)):
        #Acheives the given ee position in 0.5 sec
        if position_tup == self.Get_EE_Position():
            return None
        if not self.LiesInWorkspace(position_tup):
            print("ERROR:: Given position out of workspace")
            return None
        else:
            x,y = position_tup
            cos = (x*x + y*y - self.l1**2 - self.l2**2)/(2*self.l1*self.l2)
            cos = 1 if cos>1 else cos
            cos = -1 if cos<-1 else cos

            theta = np.arccos(cos) #Calculates theta values at each given position
            q1 = (np.arctan2(y,x) - np.arctan2(self.l2*np.sin(theta),(self.l1 + self.l2*np.cos(theta))))
            q2 = q1 + theta

            q1_c, q2_c, *_ = self.state

            q1_follow = np.linspace(q1_c,q1, num = int(0.5/(2*self.dt)))
            q2_follow = np.linspace(q2_c,q2, num = int(0.5/(2*self.dt)))
            q1dot_follow = (np.concatenate((q1_follow[1:],q1_follow[-1:])) - q1_follow)/(2*self.dt) #Change with the function used below
            q2dot_follow = (np.concatenate((q2_follow[1:],q2_follow[-1:])) - q2_follow)/(2*self.dt)

            q1dot_follow[-1] = end_qdots[0] if (end_qdots[0] - q1dot_follow[-1])/self.dt < 1e14 else q1dot_follow[-1]
            q2dot_follow[-1] = end_qdots[1] if (end_qdots[1] - q2dot_follow[-1])/self.dt < 1e14 else q2dot_follow[-1]

            for state in zip(q1_follow, q2_follow, q1dot_follow, q2dot_follow):
                self.achieve_state_in2dt(np.array(state))

    def Free_Response(self, inter = 1): #Calculates free response for given time interval
        for i in range(int(inter/self.dt)):
            self.Applies_Torques((0,0))
        y_n = input('Continue?')
        if y_n in ['yes', 'y', 'Yes', 'YES']:
            self.Free_Response()
        elif y_n in ['no', 'n', 'No', 'NO']:
            pass
        else:
            print('Neither yes Nor no! -_-')

    def MouseControl(self):
        def on_click(event):
            if event.button is MouseButton.LEFT:
                if self.LiesInWorkspace(((event.xdata, event.ydata))):
                    print((event.xdata, event.ydata))
                    follow_list.append((event.xdata, event.ydata))
                    point_list.append(plt.Circle((event.xdata, event.ydata), radius=0.25, fc='y', alpha = 0.5))
                    self.ax.add_patch(point_list[-1])
                    if self.mchaji == 0:
                        self.mchaji = 1
                        inner_loop()
                else:
                    print('ERROR:: Chosen point out of workspace')
                    print('MSG:: Choose Appropriately')
            elif event.button is MouseButton.RIGHT:
                plt.disconnect(binding_id)
                print('MSG:: Turning off mouse control')
                return None
        def inner_loop():
            point = follow_list[0]
            circ = point_list[0]
            self.Achieve_EE_Position((point[0], point[1]))

            circ.remove()
            follow_list.pop(0)
            point_list.pop(0)

            if len(follow_list) != 0:
                inner_loop()
            else:
                self.mchaji = 0
        print('MSG:: Mouse control is ON')
        print('MSG:: The RRManipulator will follow the points in the order they were clicked')
        print('MSG:: Each point to point transversing will take 0.5 sec (in the simulation frame)')
        print('MSG:: Right click to turn off MouseControl')
        self.mchaji = 0
        follow_list = []
        point_list = []

        binding_id = plt.connect('button_press_event', on_click)

    def achieve_state_in2dt(self, final_state):
        #Achieves the specified state in next 2dt by applying calculated torques
        #Does not work but can work if applied twice: HENCE:: acheive_state_in4dt()
        g = 9.81

        q1, q2, q1dot, q2dot = self.state
        q1_nn, q2_nn, q1dot_nn, q2dot_nn = final_state

        q1dot_n = (q1_nn - q1)/self.dt - q1dot
        q2dot_n = (q2_nn - q2)/self.dt - q2dot

        q1dot2 = (q1dot_n - q1dot)/self.dt
        q2dot2 = (q2dot_n - q2dot)/self.dt

        q1_n = q1 + (q1dot + q1dot_n)/2.*self.dt
        q2_n = q2 + (q2dot + q2dot_n)/2.*self.dt

        q1dot2_n = (q1dot_nn - q1dot_n)/self.dt
        q2dot2_n = (q2dot_nn - q2dot_n)/self.dt

        A1 = 1./3.*self.m1*self.l1**2. + self.m2*self.l1**2.
        A2 = 1./2.*self.m2*self.l1*self.l2*np.cos(q2 - q1)
        B1 = 1./2.*self.m2*self.l1*self.l2*np.cos(q2-q1)
        B2 = 1./3.*self.m2*self.l2**2 #+ 1./4.*self.m2*self.l2**2

        torque1 = A1*q1dot2 + B1*q2dot2 + 1./2.*self.m1*g*self.l1*np.cos(q1) + self.m2*g*self.l1*np.cos(q1) - 1./2.*self.m2*self.l1*self.l2*q2dot*q2dot*np.sin(q2-q1)
        torque2 = A2*q1dot2 + B2*q2dot2 + 1./2.*self.m2*g*self.l2*np.cos(q2) + 1./2.*self.m2*self.l1*self.l2*q1dot*q1dot*np.sin(q2 - q1)

        A1_n = 1./3.*self.m1*self.l1**2. + self.m2*self.l1**2.
        A2_n = 1./2.*self.m2*self.l1*self.l2*np.cos(q2_n - q1_n)
        B1_n = 1./2.*self.m2*self.l1*self.l2*np.cos(q2_n-q1_n)
        B2_n = 1./3.*self.m2*self.l2**2 #+ 1./4.*self.m2*self.l2**2

        torque1_n = A1_n*q1dot2_n + B1_n*q2dot2_n +  1./2.*self.m1*g*self.l1*np.cos(q1_n) + self.m2*g*self.l1*np.cos(q1_n) - 1./2.*self.m2*self.l1*self.l2*q2dot*q2dot*np.sin(q2_n-q1_n)
        torque2_n = A2_n*q1dot2_n + B2_n*q2dot2_n + 1./2.*self.m2*g*self.l2*np.cos(q2_n) + 1./2.*self.m2*self.l1*self.l2*q1dot*q1dot*np.sin(q2_n - q1_n)

        self.Applies_Torques((torque1,torque2))
        #print('Calculated: ', [q1_n,q2_n,q1dot_n,q2dot_n])
        #print('arko: ', arko.state)
        #print('arkoCon', arkoCon.state)
        #print('__________________________________________')
        self.Applies_Torques((torque1_n,torque2_n))
        #print('arko: ', arko.state)
        #print('arkoCon', arkoCon.state)

        return (torque1,torque2,torque1_n,torque2_n)
        #return [q1_n,q2_n,q1dot_n,q2dot_n]

    def Get_EE_Position(self):
        return (self.l1*np.cos(self.state[0]) + self.l2*np.cos(self.state[1]),self.l1*np.sin(self.state[0]) + self.l2*np.sin(self.state[1]))

    def follow(self, x, y, return_torques = False):    #x and y are ndarrays
        if return_torques:
            torque_arr = []
        for i,j in zip(x,y):
            if not self.LiesInWorkspace((i,j)):
                path = plt.Line2D(x, y,ls='-.', lw=2.5, color = 'y', alpha = 0.5)
                print("ERROR:: Specified path contains points out of workspace too")
                return None

        cos_values = (x*x + y*y - self.l1**2 - self.l2**2)/(2*self.l1*self.l2)
        cos_values[cos_values > 1] = 1.
        cos_values[cos_values < -1] = -1.
        path = plt.Line2D(x, y,ls='-.', lw=2.5, color = 'y', alpha = 0.5)
        self.ax.add_line(path)


        theta_values = np.arccos(cos_values) #Calculates theta_values values at each given position
        q1_values = (np.arctan2(y,x) - np.arctan2(self.l2*np.sin(theta_values),(self.l1 + self.l2*cos_values)))
        q2_values = q1_values + theta_values

        q1dot_values = (np.concatenate((q1_values[1:],q1_values[-1:])) - q1_values)/(2*self.dt) #Change with the function used below
        q2dot_values = (np.concatenate((q2_values[1:],q2_values[-1:])) - q2_values)/(2*self.dt)

        for i,state in enumerate(zip(q1_values, q2_values, q1dot_values, q2dot_values)):
            if i == 0 :
                self.Achieve_EE_Position((x[0],y[0]), end_qdots = (q1_values[0], q2_values[0]))
            else:
                if return_torques:
                    torque_arr.append(self.achieve_state_in2dt(np.array(state)))
                else:
                    self.achieve_state_in2dt(np.array(state))
        path.remove()
        if return_torques:
            return torque_arr

    def follow_line(self,pt1,pt2,tym=2, return_torques = False):
        if tym == 0:
            print('ERROR:: Things dont happen instantaneously!')
            print('MSG:: Enter valid non-zero time value')
            return None
        x = np.linspace(pt1[0], pt2[0], num = int(tym/2/self.dt))
        y = np.linspace(pt1[1], pt2[1], num = int(tym/2/self.dt))

        return self.follow(x,y, return_torques = return_torques)

    def follow_semicircle(self, center, rad, tym = 2, return_torques = False):
        if tym == 0:
            print('ERROR:: Things dont happe instantaneously!')
            print('MSG:: Enter valid non-zero time value')
            return None
        r = self.l1 + self.l2
        #print(self.l1, self.l2, r, center)

        x_l = center[0] - rad
        x_u = center[0] + rad
        x = np.linspace(x_l, x_u, num = int(tym/2/self.dt))
        y = center[1] + np.sqrt(rad**2 - x**2)
        #y2 = (center[1] - np.sqrt(rad**2 - x**2))[-1::]
        return  self.follow(x,y, return_torques = return_torques)
        #self.follow(x,y2)###CANT BE USED RN

    def WallPlay(self, m, c, F):
        g = 9.81
        r = self.l1 + self.l2
        if m**2*(r)**2 <= c**2 - (r)**2:
            print('ERROR:: specified wall does not lie within workspace')
        else:

            x_l = (-1*m*c - np.sqrt((r**2)*(1+m**2) - c**2))/(1 + m**2)
            x_u = (-1*m*c + np.sqrt((r**2)*(1+m**2) - c**2))/(1 + m**2)

            x_o, y_o = self.Get_EE_Position()

            x_p = -1*c*m/(1 + m**2)
            y_p = c/(1 + m**2)

            y_l = m*x_l + c
            y_u = m*x_u + c

            print((x_l,y_l), (x_u, y_u))

            wall = plt.Line2D([x_u, x_l], [y_u, y_l], lw=2.5, color = 'black')
            self.ax.add_line(wall)

            self.follow_line((x_o, y_o),(x_p,y_p))

            q1,q2,*_ =  self.state

            torque1g = 1./2.*self.m1*g*self.l1*np.cos(q1) + self.m2*g*self.l1*np.cos(q1)
            torque2g = 1./2.*self.m2*g*self.l2*np.cos(q2)

            if c < 0:
                fx = F*(-m/(1+m**2)**0.5)
                fy = F*(1/(1+m**2)**0.5)
            else:
                fx = F*(m/(1+m**2)**0.5)
                fy = F*(-1/(1+m**2)**0.5)

            torque1f = self.l1*np.sin(q1)*fx - self.l1*np.cos(q1)*fy
            torque2f = self.l2*np.sin(q2)*fx - self.l2*np.cos(q2)*fy

            for i in range(int(2/self.dt)):
                self.Applies_Torques((torque1f + torque1g, torque2f + torque2g), (fx,fy))
            wall.remove()

    def SpringAct(self, k, point):
        def on_click(event):
            if event.button is MouseButton.LEFT:
                if self.LiesInWorkspace(((event.xdata, event.ydata))):
                    self.RRMmodel.set_ee_position((event.xdata, event.ydata))
                    self.Calliberate(self.RRMmodel)
                    self.spring_acting = True
                    while self.spring_acting:
                        self.spring_act_for1dt()
            elif event.button is MouseButton.RIGHT:
                plt.disconnect(binding_id)
                print('MSG:: Stopping to act like spring')
                self.spring_eq_point.remove()
                self.spring_acting = False

        binding_id = plt.connect('button_press_event', on_click)
        if not self.LiesInWorkspace(point):
            print('ERROR:: Equilibrium Point doesn\'t lie in workspace')
            return None
        else:
            print('MSG:: Starting to act like spring')
            print('MSG:: Could simulate only for a equilibrium state')
            print('MSG:: Thus, Equal and opposite active forces are applied at the end effector')
            print('MSG:: Click on any point to check the force applied by end effector')
            print('MSG:: Right Click on any point to EXIT the spring mode')
            self.spring_k = k
            self.spring_pt = point
            self.spring_eq_point = plt.Circle(point, radius=0.25, fc='r', alpha = 0.5)
            self.ax.add_patch(self.spring_eq_point)

    def spring_act_for1dt(self):
        g = 9.81
        q1, q2, q1dot, q2dot = self.state
        x_c,y_c = self.Get_EE_Position()

        x, y = x_c - self.spring_pt[0], y_c - self.spring_pt[1]
        fx, fy = -self.spring_k*x, -self.spring_k*y

        q1dot2 = 0
        q2dot2 = 0

        A1 = 1./3.*self.m1*self.l1**2. + self.m2*self.l1**2.
        A2 = 1./2.*self.m2*self.l1*self.l2*np.cos(q2 - q1)
        B1 = 1./2.*self.m2*self.l1*self.l2*np.cos(q2-q1)
        B2 = 1./3.*self.m2*self.l2**2 #+ 1./4.*self.m2*self.l2**2

        torque1g = A1*q1dot2 + B1*q2dot2 + 1./2.*self.m1*g*self.l1*np.cos(q1) + self.m2*g*self.l1*np.cos(q1) - 1./2.*self.m2*self.l1*self.l2*q2dot*q2dot*np.sin(q2-q1)
        torque2g = A2*q1dot2 + B2*q2dot2 + 1./2.*self.m2*g*self.l2*np.cos(q2) + 1./2.*self.m2*self.l1*self.l2*q1dot*q1dot*np.sin(q2 - q1)

        tau1ex = -self.l1*np.sin(q1)*fx + self.l1*np.cos(q1)*fy
        tau2ex = -self.l2*np.sin(q2)*fx + self.l2*np.cos(q2)*fy

        self.Applies_Torques((torque1g + tau1ex, torque2g + tau2ex), (-fx, -fy))

    def PlotWorkspace(self, l1_angles = (0,360), l2_angles = (0,360)):
        points_arr = []
        def on_click(event):
            if event.button is MouseButton.RIGHT:
                plt.disconnect(binding_id)
                print('MSG:: Removing plotted workspace')
                for point in points_arr:
                    point.remove()
        binding_id = plt.connect('button_press_event', on_click)

        for q1 in range(l1_angles[0], l1_angles[1], 5):
            q1 = q1*np.pi/180
            for q2 in range(l2_angles[0], l2_angles[1], 5):
                q2 = q2*np.pi/180
                point = plt.Circle((self.l1*np.cos(q1) + self.l2*np.cos(q2), self.l1*np.sin(q1) + self.l2*np.sin(q2)), radius=0.125, fc='b', alpha = 0.5)
                self.ax.add_patch(point)
                points_arr.append(point)
            plt.pause(self.dt)

    def Task1(self):
        print('MSG:: Starting Task 1')
        while True:
            path = input('Which path to follow? line/semicircle:')
            if path in ['line', 'semicircle']:
                break

        if path == 'line':
            slow_torques = self.follow_line((-3,4),(4,3),1, return_torques = True)
            print('Following the line in 1 sec')
            print('plotting slow_torques')
            self.torfig, self.torax = plt.subplots()
            self.torfig.suptitle('Torques comparision (fast and slow trajectories)')
            self.torax.plot(range(len(slow_torques)),[tup[0] for tup in slow_torques], color = 'g', label = 'tau1(slow)')
            self.torax.plot(range(len(slow_torques)),[tup[1] for tup in slow_torques], color = 'b', label = 'tau2(slow)')
            self.torax.legend()
            fast_torques = self.follow_line((-3,4),(4,3),0.25, return_torques = True)
            self.torax.plot(range(len(fast_torques)),[tup[0] for tup in fast_torques], color = 'r', label = 'tau1(fast)')
            self.torax.plot(range(len(fast_torques)),[tup[1] for tup in fast_torques], color = 'y', label = 'tau2(fast)')
            self.torax.legend()
            print('Following the line in 0.25 sec')
            print('plotting fast_torques')
        elif path == 'semicircle':
            slow_torques = self.follow_semicircle((0,4),1,1, return_torques = True)
            print('Following the semicircle in 1 sec')
            print('plotting slow_torques')
            self.torfig, self.torax = plt.subplots()
            self.torfig.suptitle('Torques comparision (fast and slow trajectories)')
            self.torax.plot(range(len(slow_torques)),[tup[0] for tup in slow_torques], color = 'g', label = 'tau1(slow)')
            self.torax.plot(range(len(slow_torques)),[tup[1] for tup in slow_torques], color = 'b', label = 'tau2(slow)')
            self.torax.legend()
            fast_torques = self.follow_semicircle((0,4),1,0.25, return_torques = True)
            self.torax.plot(range(len(fast_torques)),[tup[0] for tup in fast_torques], color = 'r', label = 'tau1(fast)')
            self.torax.plot(range(len(fast_torques)),[tup[1] for tup in fast_torques], color = 'y', label = 'tau2(fast)')
            self.torax.legend()
            print('Following the semicircle in 0.25 sec')
            print('plotting fast_torques')
            print()
            print('For more fun, try arkoCon.MouseControl() =D')

    def Task2(self):
        print('MSG:: Starting Task 2')
        print('Applying 10 newtons of force on wall: y = 0.5x - 7 for 5 seconds')
        self.WallPlay(0.5, -7, 10)

    def Task3(self):
        print('MSG:: Starting Task 3')
        print('Applying static response of a spring about (3,4) and stiffness = 5 N/m')
        print('The forces shown are applied at the end end effector. Thus, torques are applied such that they balance out the applied forces.')
        print('Might have to restart simulation after this task... sorry :\')')
        self.SpringAct(5, (3,4))

    def Task4(self):
        print('MSG:: Starting Task 4')
        print('Estimating workspace restricting min_angle = 35 deg and max_angle = 145 deg for q1 and q2')
        self.PlotWorkspace((35,145),(35,145))


arkoCon = RRMController(arko)
print('MSG:: RRManipulator Controller initialised, named arkoCon')
print('MSG:: Enter your command')
