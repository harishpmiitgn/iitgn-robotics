import numpy as np
import matplotlib.pyplot as plt


def oneD_traj(points,t0,tf):
    init_point = points[0][1]
    final_point = points[1][1]
    init_vel = [0]
    final_vel = [0]
    init_acc = [0]
    final_acc = [0]

    A = np.array([[1,t0, t0**2,t0**3,t0**4,t0**5],
                  [0,1,  2*t0,3*t0**2, 4*t0**3, 5*t0**4],
                  [0,0,2,6*t0,12*t0**2,20*t0**3],
                  [1,tf, tf**2,tf**3,tf**4,tf**5],
                  [0,1,  2*tf,3*tf**2, 4*tf**3, 5*tf**4],
                  [0,0,2,6*tf,12*tf**2,20*tf**3]])
    
    B = np.vstack((init_point, 
                   init_vel,
                   init_acc,
                   final_point,
                   final_vel,
                   final_acc))

    coeffs = np.matmul(np.linalg.inv(A),B)

    t = np.linspace(0,tf,100).reshape(1,-1)

    c0 = coeffs[0].reshape(1,-1).transpose()
    c1 = coeffs[1].reshape(1,-1).transpose()
    c2 = coeffs[2].reshape(1,-1).transpose()
    c3 = coeffs[3].reshape(1,-1).transpose()
    c4 = coeffs[4].reshape(1,-1).transpose()
    c5 = coeffs[5].reshape(1,-1).transpose()

    y = c0 + c1@t + c2@t**2 + c3@t**3 + c4@t**4 + c5@t**5 
    y_dot = c1 + 2*c2@t + 3*c3@t**2 + 4*c4@t**3  + 5*c5@t**4
    y_ddot = 2*c2 + 6*c3@t + 12*c4@t**2 + 20*c5@t**3

    print(y)
    plt.plot(y[0])
    plt.show()

oneD_traj([[0.4,0.06,0.1],
            [0.4,0.01,0.1]],0,10)

