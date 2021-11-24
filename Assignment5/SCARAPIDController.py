#Videh Patel : videh.p@iitgn.ac.in
#19110192

from SCARA import*

class PID_Position_Controller:
    # Simply applies individual joint PID control to achieve the target position
    def __init__(self, RRMmodel, Kp = (0,0,0), Ki = (0,0,0), Kd = (0,0,0)):
        self.Manipulator = RRMmodel
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki) 
        self.Kd = np.array(Kd)

        self.nDOF = int(len(self.Manipulator.state)/2)
        self.target_state = np.array(self.nDOF*[0.])

        self.integral_error = np.array(self.nDOF*[0.])
        self.previous_error = np.array(self.nDOF*[0.])
        self.error_matrix = np.zeros((3,3))

    def set_K(self, K_p, K_i, K_d):
        self.Kp = np.array(K_p)
        self.Ki = np.array(K_i)
        self.Kd = np.array(K_d)
        self.reset()

    def reset(self):
        curr_state = self.Manipulator.get_state()[:self.nDOF]

        self.integral_error = np.array(self.nDOF*[0.])
        self.previous_error = np.array(self.target_state - curr_state)

        self.error_matrix[0,:] = self.previous_error
        self.error_matrix[1,:] = self.previous_error
        self.error_matrix[2,:] = self.previous_error

    def set_target_state(self, state_tup):
        self.target_state = np.array(state_tup)
        self.reset()
    
    def calculate_output_torques(self):
        curr_state = self.Manipulator.get_state()[:self.nDOF]
        error = np.array(self.target_state - curr_state)

        diff_error = (error - self.previous_error)/self.Manipulator.dt
        self.integral_error = self.integral_error  + (error + self.previous_error)*self.Manipulator.dt/2

        self.previous_error = error

        self.error_matrix[0,:] = self.error_matrix[1,:]
        self.error_matrix[1,:] = self.error_matrix[2,:]
        self.error_matrix[2,:] = self.previous_error.T

        print('__________')
        print(error)

        t = ((self.Kp*error + self.Ki*self.integral_error + self.Kd*diff_error))
        print('t-----------------')
        print(t)
        return np.squeeze(np.asarray(t))

    def achieve_target_state(self):
        while np.linalg.norm(self.error_matrix) > 1e-2:
            self.Manipulator.apply_torques(self.calculate_output_torques())


    def Achieve_EE_Position(self, position_tup):
        # Acheives the given ee position
        if not self.Manipulator.LiesInWorkspace(position_tup):
            print("ERROR:: Given position out of workspace")
            return None
        self.graph_point = self.Manipulator.ax.scatter(position_tup[0], position_tup[1], position_tup[2])        
        q_mat = self.Manipulator.inv_kin(np.array(position_tup))
        print('qmat__________-----')
        print(q_mat)
        self.set_target_state(q_mat)
        self.achieve_target_state()

arkoCon = PID_Position_Controller(arko, [2000, 1000, 50], [0,0,0], [250, 400, 5])
print('MSG:: RRManipulator Controller initialised, named arkoCon')
print('MSG:: Enter your command')
