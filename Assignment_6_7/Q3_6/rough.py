import numpy as np

class findDH_manipulator():
    def __init__(self,dh_config=['R','R']):
        # configuration of the manipulator. User has 2 choices "R"->revolute. "P"->prismatic.
        # Default configuration is a 2R manipulator with all the angles at 0 degrees and lengths being 1 unit.
        self.config = dh_config 
        # User must input the dh parameters in matrix form i.e. "R"->revolute
        # [[a1 , alpha1 , d1, theta1]
        #  [a2 , alpha2 , d2, theta2]
        #  .
        #  .
        #  .
        #  [an , alphan , dn, thetan]]
        # n being the nth link of the manipulator.
        # self.dh=dh_params
    
    def calc_tranfMatrix(self, dh_params,i):
        # Calculating Trnasformation matrix
        a, alpha,d,theta = dh_params
        A = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                      [            0,                np.sin(alpha),                np.cos(alpha),               d],
                      [            0,                            0,                            0,               1]])
        return A 

    def forward(self,dh):
        # tr_01=self.calc_tranfMatrix(dh[0],0)
        tr = [[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]]
        Trs = [tr]
        # Calculating the individual transformation matrices. And appending to the T matrix in the following form.
        # A1
        # A1A2
        # A1A2A3 ... 
        for i in range(len(dh)):
            tr = np.matmul(tr,self.calc_tranfMatrix(dh[i],i))
            Trs.append(tr)

        Trs = np.array(Trs)
        # Calculating the jacobian matrix
        h = []
        for i in range(len(self.config)):
            temp2 = np.array([0,0,0])
            if self.config[i]=='R':
                temp2 = np.array([0,0,1])
            if  i ==0:
                temp = np.array(Trs[-1])

            else:
                temp = np.array(Trs[-1]) - np.array(Trs[i-1])
            
            h.append(np.cross(temp2,temp[:3,3:].transpose()).transpose())
        
        # Velocity jacobian
        # print(Trs)
        Js = []
        for k in range(len(dh)):
            dn = Trs[k+1][:3,3]     
            temp_Js = []       
            for i in range(len(dh)):
                if self.config[i]=='R':
                    R_prev = Trs[i][:3,2]
                    Jv = np.cross(R_prev,(dn-Trs[i][:3,3]))
                    Jw = Trs[i][:3,2]
                    temp_Js.append(np.hstack((Jv,Jw)))
            Js.append(np.array(temp_Js).T)
        Js = np.round(np.array(Js),3)
        
        # Overall Transformation matrix T06.
        Trs = np.around(np.array(Trs),3)
        # Manipulator jacobian

        return {'trans_mtrx':Trs[1:],'jacob':Js}


manipulator_config = ['R','R','R']
d1,d2,d3 = 1,1,1
dh_params = [[0,np.pi/2,    1,0],
             [1,      0,    0,0],
             [1,      0,    0,0]]

mani = findDH_manipulator(manipulator_config)
a = mani.forward(dh_params)
# print('transformation matrix')
print(np.round(a['jacob'].astype('float'),3))