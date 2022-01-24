'''R. Y. Tsai, R. K. Lenz et al., “A new technique for fully autonomous
and efficient 3 d robotics hand/eye calibration,” IEEE Transactions on
Robotics and Automation, vol. 5, no. 3, pp. 345–358, 1989.'''



import numpy as np
from numpy.linalg import inv, det, svd, eig, norm,pinv
# from hand_eye_DualQuaternion import quaternion_2_R
# from sim_data import *
from utility import *
import time
from scipy import optimize, sparse
from scipy.sparse.linalg import svds, eigs
import analysis 
from scipy.spatial.transform import Rotation as Rot

def get_Translation(R,RA_I,TA,TB):
    RxTB = np.dot(R,TB[:3,0]).reshape(3,1)
    for i in range(1,int((TB.shape[0])/3)):
        RxTB = np.append(RxTB,np.dot(R,TB[i*3:(i+1)*3,0].reshape(3,1)),axis=0)
    
    T = RxTB - TA

    tX = np.dot(inv(np.dot(RA_I.T,RA_I)),np.dot(RA_I.T,T))
    tX = np.dot(pinv(RA_I),T)

    return tX

def exp_(w):
    w.reshape(-1,1)
    theta = norm(w)
    k = w/norm(w)
    R = np.cos(theta)*np.eye(3) + np.sin(theta)*skew(k) + (1-np.cos(theta))*np.outer(k,k)

    return R

def calibrate(A,B, sigmaA=(0,0), sigmaB=(0,0)):
    N = len(A)
    S = None
    RA_I = None
    T = None
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

        _S = skew(uA + uB)
        _T = uB - uA

        _RA_I = RA - np.eye(3)
        _TA = tA
        _TB = tB
        
        S = np.append(S, _S, axis=0) if S is not None else _S         # populate system matrix
        T = np.append(T,_T,axis=0) if T is not None else _T

        RA_I = np.append(RA_I,_RA_I,axis=0) if RA_I is not None else _RA_I
        TA = np.append(TA,_TA,axis=0) if TA is not None else _TA
        TB = np.append(TB,_TB,axis=0) if TB is not None else _TB

    ux = solveLS(S,T)

    # theta = 2*np.arctan(norm(ux))

    uX = 2*ux/(np.sqrt(1+norm(ux)**2))
    
    Rx = (1-norm(uX)**2/2)*np.eye(3) + 0.5*(uX*uX.T + np.sqrt(4-norm(uX)**2)*skew(uX))

    tX = get_Translation(Rx,RA_I,TA,TB)

    toc = time.perf_counter()

    Hx = Pose(Rx,tX)

    return Rx, tX.reshape(3,1), Hx, toc-tic