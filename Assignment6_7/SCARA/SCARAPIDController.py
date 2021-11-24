#Videh Patel : videh.p@iitgn.ac.in
#19110192
import numpy as np
import sympy as sp

class PID_Position_Controller:
    # Simply applies individual joint PID control to achieve the target position
    def __init__(self, RRMmodel, Kp = (0,0,0), Ki = (0,0,0), Kd = (0,0,0)):
        self.Manipulator = RRMmodel
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki) 
        self.Kd = np.array(Kd)

        self.nDOF = int(len(self.Manipulator.state)/2)
        self.target_state = np.array(2*self.nDOF*[0.])

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
        self.previous_error = np.array(self.target_state[:self.nDOF] - curr_state)

        self.error_matrix[0,:] = self.previous_error
        self.error_matrix[1,:] = self.previous_error
        self.error_matrix[2,:] = self.previous_error

    def set_target_state(self, state_tup):
        self.target_state = np.array(state_tup)
        self.reset()
    
    def calculate_output_voltages(self):
        curr_state = self.Manipulator.get_state()[:self.nDOF]
        error = np.array(self.target_state[:self.nDOF] - curr_state)

        diff_error = self.target_state[self.nDOF:] - self.Manipulator.get_state()[self.nDOF:]
        self.integral_error = self.integral_error  + (error + self.previous_error)*self.Manipulator.dt/2

        self.previous_error = error

        self.error_matrix[0,:] = self.error_matrix[1,:]
        self.error_matrix[1,:] = self.error_matrix[2,:]
        self.error_matrix[2,:] = self.previous_error.T

        # print('___________CURRENT_ERROR__________')
        # print(error)
        V = ((self.Kp*error + self.Ki*self.integral_error + self.Kd*diff_error))
        # print('Applied Voltages')
        # print(V)
        return sp.Matrix(V)

    def achieve_target_state(self):
        err_mat = self.error_matrix
        err_mat[:,2] = self.error_matrix[:,2]*0.01
        while (((np.linalg.norm(err_mat[0,:2]) + np.linalg.norm(err_mat[1,:2]) + np.linalg.norm(err_mat[2,:2]))/3) > 5e-2) or ((np.abs(err_mat[0,0]) + np.abs(err_mat[1,0]) + np.abs(err_mat[2,0])/3) > 8e-2) or np.linalg.norm(self.Manipulator.state[3:]) > 1e-2:
            self.Manipulator.apply_voltages(self.calculate_output_voltages())


    def follow_trajectory(self, q_mat, qdot_mat, t, ee_pos_values = None):
        # t: Total time for execution of trajectory
        # q_mat: array of q's
        # qdot_mat: array of qdot's
        if len(q_mat) != len(qdot_mat):
            print('q_mat and qdot_mat have different sizes')
            return None
        dt = t/len(qdot_mat)
        if dt < self.Manipulator.dt:
            print('Array passed has too refined path. Not all points will be achieved')
        self.Manipulator.reset()
        self.Manipulator.set_state(np.array(np.squeeze(np.asarray(q_mat[0])).tolist() + np.squeeze(np.asarray(qdot_mat[0])).tolist()))
        target_start_time = 0
        for i in range(len(q_mat)):
            print('Starting to follow next point: ' + str(i))
            self.set_target_state(np.array(np.squeeze(np.asarray(q_mat[i])).tolist() + np.squeeze(np.asarray(qdot_mat[i])).tolist()))
            target_start_time = self.Manipulator.time
            while (self.Manipulator.time - target_start_time) < dt:
                self.Manipulator.apply_voltages(self.calculate_output_voltages())
                if ee_pos_values is not None:
                    ee_pos_values.append(self.Manipulator.get_ee_position())


    def Achieve_EE_Position(self, position_tup):
        # Acheives the given ee position
        if not self.Manipulator.LiesInWorkspace(position_tup):
            print("ERROR:: Given position out of workspace")
            return None
        self.graph_point = self.Manipulator.ax.scatter(position_tup[0], position_tup[1], position_tup[2])        
        q_mat = self.Manipulator.inv_kin(np.array(position_tup))
        self.set_target_state(np.array(np.array(q_mat).tolist() + 3*[0]))
        self.achieve_target_state()