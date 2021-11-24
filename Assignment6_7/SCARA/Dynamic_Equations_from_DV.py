from sympy import*
# @author Videh Patel : videh.p@iitgn.ac.in : 19110192
## D and V are given
#D(q)--> nxn matrix :: D should be in terms of q1,q2,q3... qn
#V(q)--> scalar potential

def DynamicEqns_from_DV(D,V):
    q = []
    q_dot = []
    q_dot2 = []
    T = []
    n = shape(D)[0]
    C = zeros(n) #Christoffel symbol matrix
    g_ = zeros(n,1) #Potential Field matrix
    for i in range(1,n+1):
        #Generating q matrix
        q_ = symbols('q' + str(i))
        q.append(q_)

        qdot_ = symbols('q' + str(i) + '_dot')
        q_dot.append(qdot_)

        qdot2_ = symbols('q' + str(i) + '_dot2')
        q_dot2.append(qdot2_)

        tau_ = symbols('tau' + str(i))
        T.append(tau_)



    for k in range(1,n+1):
        for j in range(1,n+1):
            c=0
            for i in range(1,n+1):
                c += 1/2*(diff(D[k-1,j-1], q[i-1]) + diff(D[k-1,i-1], q[j-1]) - diff(D[i-1,j-1], q[k-1]))*q_dot[i-1]
            C[k-1,j-1] = c

            print('...Calculating C(i,j): ' + str((k,j)))


    for i in range(1,n+1):
        g_[i-1,0] = diff(V, q[i-1])
        print('Calculating g_(i): ' + str(i))

    
    # for i,eqn in enumerate(D@Matrix(q_dot2) + C@Matrix(q_dot) + g_ - Matrix(T)):
    #     print('Equation' + str(i+1) + '_______________________________________________-')
    #     print(str(eqn) + " = 0")
    #     print()

    return C, g_,

if __name__ == '__main__':
    #Verifying for Planar RR Manipulator
    m1, m2 = symbols('m1 m2')
    l1, l2 = symbols('l1 l2')
    q1, q2 = symbols('q1 q2')
    g = symbols('g')

    D = Matrix([[m1*l1*l1/3+m2*l1*l1, m2*l1*l2/2*cos(q2-q1)], [m2*l1*l2/2*cos(q2-q1), m2*l2*l2/3]])
    V = m1*g*l1/2*sin(q1) + m2*g*(l1*sin(q1)+l2/2*sin(q2))

    D = Matrix([[m1*l1*l1/3+m2*l1*l1, m2*l1*l2/2*cos(q2)], [m2*l1*l2/2*cos(q2), m2*l2*l2/3]])
    DynamicEqns_from_DV(D,V)
