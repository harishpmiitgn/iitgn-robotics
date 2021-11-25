import numpy as np
from numpy.core.function_base import linspace
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from Q1_kaula import linear_traj
from Q2_kaula import joint_traj

def wrapping(angle):
    if abs(angle%(2*np.pi) - 2*np.pi)<10**(-3):
        return 0
    return angle%(2*np.pi)


link_lengths = [0.25, 0.25, 0.25] # d1 , a1, a2 these are the link lengths
init_point = [0.4,0.06,0.1]
final_point = [0.4, 0.01,.1]
t0 = 0
tf = 5
v0 = np.array([0,0,0])
vf = np.array([0,0,0]) 
a0 = np.array([0,0,0])
af = np.array([0,0,0])
q =np.array([0,0,0])
qd= np.array([0,0,0])
qdd= np.array([0,0,0])
tau1 = 0
tau2 = 0
F = 0
t_lin = linspace(t0,tf,200) #default steps generated are 50
dt = (t_lin[2] - t_lin[1])
g = 9.8
m1 = 0.5
m2 = 0.5
m3 = 0.5
I1 = 0
I2 = (m2*link_lengths[1]**2)/3
I3 = (m3*link_lengths[2]**2)/3

def scara_inverse(xc,yc,zc):
    if np.sqrt(xc**2+yc**2)>link_lengths[2] + link_lengths[1]:
        print("No Solution can be Found!")
        return []
    else: 
        def inv_func(x):
            return [
                    - xc + link_lengths[1]*np.cos(x[0]) + link_lengths[2]*np.cos(x[0]+x[1]),
                    - yc + link_lengths[1]*np.sin(x[0]) + link_lengths[2]*np.sin(x[0]+x[1]),
                    - zc + link_lengths[0] - x[2]
                    ]
        root = fsolve(inv_func,[1,1,1])

        q1,q2,d = root

        return [wrapping(q1),wrapping(q2),d]

def scara_forward(q1,q2,d):

    xc = link_lengths[1]*np.cos(q1) + link_lengths[2]*np.cos(q1+q2)
    yc = link_lengths[1]*np.sin(q1) + link_lengths[2]*np.sin(q1+q2)
    zc = link_lengths[0] - d

    return xc,yc,zc


q_d, q_dd,  q_ddd = joint_traj(link_lengths,init_point,final_point, t0,tf,v0,vf,a0,af)
# print(q_d)
end_eff_des = linear_traj(link_lengths,init_point,final_point, t0,tf,v0,vf,a0,af)

def PD_control(q,qd,qdd,q_d,q_dd,q_ddd,t0,tf,dt,prev_error):

    # print()
    gains = np.array([[5,1],
             [5,1],
             [5,1]])
    
    error = q_d - q
    e_d = (prev_error - error) / dt
    P = gains[:,0]*error
    D = gains[:,1]*e_d
    prev_error = error

    return (P+D), prev_error


def scara_dynamics(q,qd,qdd,q_d,q_dd,q_ddd,prev_error):

######### computing the matrices
    alpha = I1 + link_lengths[1]**2*(m1/4 + m2 + m3)
    beta = I2 + I3 + link_lengths[2]**2*(m2/4 +m3) 
    gamma = link_lengths[1]*link_lengths[2]*m3 + link_lengths[1] * link_lengths[2]/2 * m2

    MM = np.array([[alpha + beta + 2*gamma*np.cos(q[1]), beta + 2*gamma*np.cos(q[1]), 0],
                                [beta + 2*gamma*np.cos(q[1]), beta, 0],
                                [0, 0, m3]])
    
    C = np.array([[-gamma*np.sin(q[1])*qd[1], -gamma*np.sin(q[1])*(qd[1] + qd[0]), 0],
                            [gamma*np.sin(q[1])*qd[0], 0, 0],
                            [0, 0, 0]])

    G = np.transpose(np.array([0, 0, m3*g]))
    
    '''Enter the MM_Des & G_des etc matrix code block here'''

    cntrl_output,prev_error = PD_control(q,qd,qdd,q_d,q_dd,q_ddd,t0,tf,dt,prev_error)
    tau1, tau2,F = cntrl_output


################## computing the inputs
    u1 = -gamma*np.sin(q[1])*qd[1]*qd[0] - gamma*np.sin(q[1])*(qd[1] + qd[0])*qd[1] + tau1
    u2 = gamma*np.sin(q[1])*qd[0]**2 + tau2
    u3 = m3*g + F

    U = np.around(np.transpose(np.array([u1, u2,u3])), decimals = 2)

    K = np.around((U - np.matmul(C, np.transpose(qd))-G), decimals = 2)

    qdd = np.around(np.matmul(np.linalg.inv(MM), K), decimals = 2)

    qd = qd + qdd*dt
    q = q + qd*dt + (qdd*dt**2)/2
    print(q)
    for iter in range(len(q)-1):
        q[iter] = wrapping(q[iter])
    print(q)
    ef_actual = scara_forward(q[0],q[1],q[2])

    return ef_actual, prev_error

# end_eff_a = []

def implement_control(q,qd,qdd,q_d,q_dd,q_ddd):
    # print(end_eff_a)
    end_eff_a = []
    end_eff_a.append(scara_forward(q[0],q[1],q[2]))
    
    prev_error = 0
    i =1
    while True:
        # print(q_d)
        # print(q)
        if i<len(t_lin)-1:
            ef_actual,prev_error = scara_dynamics(q,qd,qdd,q_d[i],q_dd[i],q_ddd[i],prev_error)
            end_eff_a.append(ef_actual)
        else:
                break
                pass
        i += 1
    end_eff_a = np.array(end_eff_a)

    plt.figure(0)
    plt.plot(end_eff_a[:,1])
    plt.plot(end_eff_des[:,1])
    plt.grid()
    plt.show()

implement_control(q,qd,qdd,q_d,q_dd,q_ddd)




