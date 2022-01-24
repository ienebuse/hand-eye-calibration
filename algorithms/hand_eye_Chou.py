'''J.C. Chou, M. Kamel, Quaternions approach to solve the kinematic 
equation of rotation, AX = XB, of a sensor-mounted robotic manipulator, 
in IEEE International Conference on Robotics and Automation (IEEE, 1988), 
pp. 656–662'''



import numpy as np
from numpy.linalg import inv, det, svd, eig, pinv
# from sim_data import *
from utility import *
import time
import analysis

def R_2_angle_axis2(R):
    theta = np.arccos(( R[0,0] + R[1,1] + R[2,2] - 1)/2)
    _,v = eig(R)
    u = v[:,-1]

    return u.reshape(3,1), theta

def quaternion_2_R(q):
    R = np.array([[2*(q[0]*q[0] + q[1]*q[1]) - 1, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])], \
                   [2*(q[1]*q[2] + q[0]*q[3]), 2*(q[0]*q[0] + q[2]*q[2]) - 1, 2*(q[2]*q[3] - q[0]*q[1])], \
                   [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 2*(q[0]*q[0] + q[3]*q[3]) - 1]])

    return R


def get_Translation(R,RA_I,TA,TB):
    RxTB = np.dot(R,TB[:3,0]).reshape(3,1)
    for i in range(1,int((TB.shape[0])/3)):
        RxTB = np.append(RxTB,np.dot(R,TB[i*3:(i+1)*3,0].reshape(3,1)),axis=0)
    
    T = RxTB - TA
	
    tX = np.dot(pinv(RA_I),T)

    return tX


def calibrate(A,B, sigmaA=(0,0), sigmaB=(0,0)):
    N = len(A)
    G = None
    RA_I = None
    TA = None
    TB = None

    tic = time.perf_counter()
    
    for i in range(N):
        An = noise(A[i], sigmaB, sigmaA)
        Bn = B[i]
		
        RA = An[:3,:3]                               # relative rotation of camera between successive movement
        tA = An[:3,3].reshape(3,1)                   # relative translatioon of camera between successive movement
        RB = Bn[:3,:3]                               # relative rotation of robot between successive movement
        tB = Bn[:3,3].reshape(3,1)                   # relative translation of robot between successive movement

        # Rotation matrix to angle(w) - axis(u) convertion
        uA, wA = R_2_angle_axis(RA)
        uB, wB = R_2_angle_axis(RB)

        g1 = np.append(np.array([[0]]), -(uA-uB).T,axis=1)
        g2 = np.append(uA-uB, skew(uA)+skew(uB),axis=1)
        _G = np.append(g1,g2,axis=0)
        _RA_I = RA - np.eye(3)
        _TA = tA
        _TB = tB
        
        G = np.append(G, _G, axis=0) if G is not None else _G         # populate system matrix
        RA_I = np.append(RA_I,_RA_I,axis=0) if RA_I is not None else _RA_I
        TA = np.append(TA,_TA,axis=0) if TA is not None else _TA
        TB = np.append(TB,_TB,axis=0) if TB is not None else _TB

    x = solveSVD(G)

    Rx = quaternion_2_R(x.ravel())

    tX = get_Translation(Rx,RA_I,TA,TB)

    toc = time.perf_counter()

    Hx = Pose(Rx,tX)

    return Rx, tX.reshape(3,1), Hx, toc-tic