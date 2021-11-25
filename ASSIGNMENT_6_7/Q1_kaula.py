from matplotlib import colors
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation

def linear_traj(link_lengths,init_point,final_point, t0,tf,v0,vf,a0,af):
    ## PUMA MANIPULATOR

    M = np.array([[1, t0, t0**2, t0**3, t0**4, t0**5],
                [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                [1, tf, tf**2, tf**3, tf**4, tf**5],
                [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])

    RHS = np.vstack((init_point, v0, a0, final_point, vf, af))

    C = np.matmul(np.linalg.inv(M), RHS)
    y_ = []
    y_dot_ = []
    y_ddot_ = []

    t_linspace = np.linspace(t0,tf,200)
    # print(C)

    for i in range(len(t_linspace)):
        y_.append(C[0] + C[1]*t_linspace[i] + C[2]*t_linspace[i]**2 + C[3]*t_linspace[i]**3 + C[4]*t_linspace[i]**4 + C[5]*t_linspace[i]**5)
        y_dot_.append(C[1] + 2*C[2]*t_linspace[i] + 3*C[3]*t_linspace[i]**2 + 4*C[4]*t_linspace[i]**3 + 5*C[5]*t_linspace[i]**4)
        y_ddot_.append(2*C[2] + 6*C[3]*t_linspace[i] + 12*C[4]*t_linspace[i]**2 + 20*C[5]*t_linspace[i]**3)

    end_eff = np.array(y_)
    end_eff_dot = np.array(y_dot_)
    end_eff_ddot = np.array(y_ddot_)
    # print(end_eff)
    isPLOT = False
    if isPLOT:


        #_______________________________________________________________
        ## PLOT A SCATTER PLOT W/O ANIMATION
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(end_eff[:,0],end_eff[:,1],end_eff[:,2], color = 'green')
        # ax.set_xlabel('X-axis', fontweight ='bold')
        # ax.set_ylabel('Y-axis', fontweight ='bold')
        # ax.set_zlabel('Z-axis', fontweight ='bold')
        # plt.title("Points traced out by End-effector")
        # plt.show()
        #_______________________________________________________________
        ## PLOT A SCATTER PLOT WITH ANIMATION
        df = pd.DataFrame(end_eff, columns=["x","y","z"])  #the matrix to be plotted goes here
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        sc = ax.scatter([],[],[], c='darkblue', alpha=0.5)
        def update(i):
            sc._offsets3d = (df.x.values[:i], df.y.values[:i], df.z.values[:i])
        ax.set_xlabel('X-axis', fontweight ='bold')
        ax.set_ylabel('Y-axis', fontweight ='bold')
        ax.set_zlabel('Z-axis', fontweight ='bold')
        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.1,0.1)
        ax.set_zlim(-0.5,0.5)
        ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(df), interval=70)
        plt.tight_layout()
        plt.show()
    
    return end_eff

if __name__=='__main__':
    link_lengths = [0.25, 0.25, 0.25] # order : d1, a1, a2 replace d1 with link_lengths[0], a2 with link_lengths[1]. # a3 with link_lengths[2]
    init_point = [0.4,0.06,0.1]
    final_point = [0.4, 0.01,.1]
    t0 = 0
    tf = 5
    v0 = np.array([0,0,0])
    vf = np.array([0,0,0]) 
    a0 = np.array([0,0,0])
    af = np.array([0,0,0])
    linear_traj(link_lengths,init_point,final_point, t0,tf,v0,vf,a0,af)