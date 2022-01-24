'''Daniilidis, Konstantinos
Bayro-Corrochano, Eduardo'''



import numpy as np
from numpy.linalg import inv, det, svd, eig
# from sim_data import *
from utility import *
from scipy import optimize
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

J = None

def f(x):
    C = np.array([J[0][0]*x[0]**2 + J[0][1]*x[0]*x[1] + J[0][2]*x[1]**2, \
                  J[1][0]*x[0]**2 + J[1][1]*x[0]*x[1] + J[1][2]*x[1]**2])
    E = C - np.array([1,0])
    return E

def calibrate(A,B, sigmaA=(0,0), sigmaB=(0,0)):
    global J
    N = len(A)
    S = None
    RA_I = None
    TA = None
    TB = None
    
    tic = time.perf_counter()

    for i in range(N):
        An = noise(A[i], sigmaB, sigmaA)
        Bn = B[i]
        # An = noise(A[i], sigmaB, sigmaA)
        # An = A[i] if sigmaA == (0,0) else noiseX2(A[i],sigmaA)
        # B[i] = B[i] if sigmaB == (0,0) else noiseX(B[i],sigmaB)
        # An = noiseX2(A[i],sigmaA)
        # B[i] = noiseX(B[i],sigmaB)
        RA = An[:3,:3]                               # relative rotation of camera between successive movement
        tA = An[:3,3].reshape(3,1)                   # relative translatioon of camera between successive movement
        RB = Bn[:3,:3]                               # relative rotation of robot between successive movement
        tB = Bn[:3,3].reshape(3,1)                   # relative translation of robot between successive movement

        
        qA = R_2_quaternion(RA).reshape(-1,1)
        qB= R_2_quaternion(RB).reshape(-1,1)

        # qA_ = 0.5*np.dot(e_(tA,-1),qA)
        # qB_ = 0.5*np.dot(e_(tB,-1),qB)
        qA_ = 0.5*np.dot(e_(qA,1),np.append([[0]],tA,axis=0))
        qB_ = 0.5*np.dot(e_(qB,1),np.append([[0]],tB,axis=0))

        s10 = np.append(qA[1:]-qB[1:], skew(qA[1:]+qB[1:]),axis=1)
        s11 = np.append(np.zeros((3,1)),np.zeros((3,3)),axis=1)
        s1 = np.append(s10,s11,axis=1)

        s20 = np.append(qA_[1:]-qB_[1:], skew(qA_[1:]+qB_[1:]),axis=1)
        s21 = np.append(qA[1:]-qB[1:], skew(qA[1:]+qB[1:]),axis=1)
        s2 = np.append(s20,s21,axis=1)

        _S = np.append(s1,s2,axis=0)        
        S = np.append(S, _S, axis=0) if S is not None else _S         # populate system matrix

    _,_,VT = svd(S)
    UV = VT.T[:,-2:]
    u1 = UV[:4,0].reshape(-1,1)
    v1 = UV[4:,0].reshape(-1,1)
    u2 = UV[:4,1].reshape(-1,1)
    v2 = UV[4:,1].reshape(-1,1)

    a = np.dot(u1.T,u1)[0][0] 
    a_ = np.dot(u1.T,v1)[0][0]
    b = np.dot(u1.T,u2)[0][0] + np.dot(u2.T,u1)[0][0]
    b_ = np.dot(u1.T,v2)[0][0] + np.dot(u2.T,v1)[0][0]
    c = np.dot(u2.T,u2)[0][0] 
    c_ = np.dot(u2.T,v2)[0][0]

    J = [[a,b,c],[a_,b_,c_]]
    x0 = np.array([0.5,0.5])
    x = optimize.newton(f,x0,maxiter=100000,tol=1.48e-15)
    # x = optimize.fsolve(f,x0,xtol=1.48e-15)
    # x = optimize.

    qx = x[0]*u1 + x[1]*u2
    qx_ = x[0]*v1 + x[1]*v2

    Rx = quaternion_2_R(qx.ravel())

    tX = 2*np.dot(inv(e_(qx,1)),qx_)[1:]

    toc = time.perf_counter()

    Hx = Pose(Rx,tX)

    return Rx, tX.reshape(3,1), Hx, toc-tic


# _,_ = analysis.get_system_data()

# # tic = time.perf_counter()   # start timer

# Rx,tX,estPose  = calibrate(analysis.A, analysis.B)

# # toc = time.perf_counter()   # stop timer

# print("\nRotation Rx\n")
# print(np.matrix(Rx))

# print("\nTranslation tX\n")
# print(np.matrix(tX))

# print('\nEstimated pose\n')
# print(np.matrix(estPose))

# print('\nEstimated pose2\n')
# print(Pose2(estPose))

# print("\nGround Truth\n")
# print(np.matrix(groundTruth(Hx)))


# print("\nComputation time = {}".format(str(1000*(toc - tic ))) + "ms")

# res_norm = residual_norm(Rx, 100)

# print ("\nResisual norm (100): \n\n{}".format(res_norm))