'''Y.-C. Lu and J. C. Chou, “Eight-space quaternion approach for robotic
hand-eye calibration,” IEEE International Conference on Systems, Man
and Cybernetics. Intelligent Systems for the 21st Century, pp. 3316–3321,
1995'''


import numpy as np
from numpy.linalg import inv, det, svd, eig
# from sim_data import *
from utility import *
import time
import analysis


def quaternion_2_R(q):
    R = np.array([[2*(q[0]*q[0] + q[1]*q[1]) - 1, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])], \
                   [2*(q[1]*q[2] + q[0]*q[3]), 2*(q[0]*q[0] + q[2]*q[2]) - 1, 2*(q[2]*q[3] - q[0]*q[1])], \
                   [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 2*(q[0]*q[0] + q[3]*q[3]) - 1]])

    return R

def R_2_quaternion(R):
    qw= 0.5*np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])
    qx = (R[2,1] - R[1,2])/( 4 *qw)
    qy = (R[0,2] - R[2,0])/( 4 *qw)
    qz = (R[1,0] - R[0,1])/( 4 *qw)

    return np.array([qw, qx, qy, qz])

def e_(Q,p):
    Q = Q.ravel()
    if Q.size == 3:
        q0 = 0
        q = Q
    else:
        q0 = Q[0]
        q = Q[1:]
    g1 = np.append(np.array([[q0]]), -q.reshape(1,3),axis=1)
    g2 = np.append(q.reshape(3,1), q0*np.eye(3) - np.sign(p)*skew(q),axis=1)
    G = np.append(g1,g2,axis=0)

    return G

def e(Q):
    q0 = Q[0]
    q = Q[1:]

    G = np.append(-q.reshape(3,1), q0*np.eye(3) + skew(q), axis=1)

    return G

def calibrate(A,B, sigmaA=(0,0), sigmaB=(0,0)):
    N = len(A)
    U_sum = np.zeros((4,4))
    V_sum = np.zeros((4,4))
    W_sum = np.zeros((4,4))

    S = None
    tic = time.perf_counter()
    for i in range(N):
        An = noise(A[i], sigmaB, sigmaA)
        Bn = B[i]
		
        RA = An[:3,:3]                               # relative rotation of camera between successive movement
        tA = An[:3,3].reshape(3,1)                   # relative translatioon of camera between successive movement
        RB = Bn[:3,:3]                               # relative rotation of robot between successive movement
        tB = Bn[:3,3].reshape(3,1)                   # relative translation of robot between successive movement

        

        qA = R_2_quaternion(RA)
        qB = R_2_quaternion(RB)

        a = np.dot(e_(qB,1), e_(tB,1) - e_(tA,-1))
        b = e_(qB,1) - e_(qA,-1)

        U_sum += np.dot(a.T,a) + np.dot(b.T,b)
        V_sum += np.dot(a.T,b)
        W_sum += np.dot(b.T,b)
    
    W_ = W_sum - np.dot(V_sum.T,np.dot(inv(U_sum),V_sum))

    phi = np.dot(inv(W_[:2,:2]-W_[2:,:2]),W_[2:,2:]-W_[:2,2:])
    alpha = -np.dot(inv(U_sum),V_sum).T
    omega = np.dot(phi.T,np.dot(alpha[:2,:2],phi)) + np.dot(alpha[2:,:2],phi) + np.dot(phi.T, alpha[:2,2:]) + alpha[2:,2:]
    beta = np.dot(alpha,alpha.T)
    Phi = np.dot(phi.T,np.dot(beta[:2,:2],phi)) + np.dot(beta[2:,:2],phi) + np.dot(phi.T, beta[:2,2:]) + beta[2:,2:]

    h11 = -((omega[0,1]+omega[1,0]) + np.sqrt((omega[0,1]+omega[1,0])**2 - 4*omega[0,0]*omega[1,1]))/(2*omega[0,0])
    h12 = -((omega[0,1]+omega[1,0]) - np.sqrt((omega[0,1]+omega[1,0])**2 - 4*omega[0,0]*omega[1,1]))/(2*omega[0,0])

    h21 = Phi[1,1] + (Phi[0,1]+Phi[1,0])*h11 + Phi[0,0]*h11*h11
    h22 = Phi[1,1] + (Phi[0,1]+Phi[1,0])*h12 + Phi[0,0]*h12*h12

    if(h21 > 0):
        r_ = np.array([[phi[0,0]*h11 + phi[0,1]], \
                    [phi[1,0]*h11 + phi[1,1]], \
                    [h11], \
                    [1]]) * np.sqrt(1/h21)
    else:
        r_ = np.array([[phi[0,0]*h12 + phi[0,1]], \
                    [phi[1,0]*h12 + phi[1,1]], \
                    [h12], \
                    [1]]) * np.sqrt(1/h22)

    qx = np.dot(alpha.T,r_)

    rx = np.dot(e(qx),r_)

    Rx = quaternion_2_R(qx).reshape(3,3)

    toc = time.perf_counter()

    tX = rx

    Hx = Pose(Rx,tX)

    return Rx, tX.reshape(3,1), Hx, toc-tic