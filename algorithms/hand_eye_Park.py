'''F. C. Park and B. J. Martin, “Robot sensor calibration: solving AX
= XB on the Euclidean group,” IEEE Transactions on Robotics and
Automation, vol. 10, no. 5, pp. 717–721, 1994.'''

import numpy as np
from numpy import dot, eye, zeros, outer
from numpy.linalg import inv, pinv

from numpy.lib.function_base import append
# from sim_data import *
from utility import *
import time
import analysis


def log(R):
    # Rotation matrix logarithm
    theta = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1.0)/2.0)
    return np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))

def invsqrt(mat):
    u,s,v = np.linalg.svd(mat)
    return u.dot(np.diag(1.0/np.sqrt(s))).dot(v)

def calibrate(A,B, sigmaA=(0,0), sigmaB=(0,0)):

    tic = time.perf_counter()
    N = len(A)
    M = np.zeros((3,3))
    for i in range(N):
        A[i] = noise(A[i], sigmaB, sigmaA)

        Ra, Rb = A[i][0:3, 0:3], B[i][0:3, 0:3]
        M += outer(log(Rb), log(Ra))

    Rx = dot(invsqrt(dot(M.T, M)), M.T)

    C = zeros((3*N, 3))
    d = zeros((3*N, 1))
    for i in range(N):
        Ra,ta = A[i][0:3, 0:3], A[i][0:3, 3]
        Rb,tb = B[i][0:3, 0:3], B[i][0:3, 3]
        C[3*i:3*i+3, :] = eye(3) - Ra
        d[3*i:3*i+3, 0] = ta - dot(Rx, tb)

    tX = dot(inv(dot(C.T, C)), dot(C.T, d))

    toc = time.perf_counter()

    Hx = Pose(Rx,tX)

    return Rx, tX.reshape(3,1), Hx, toc-tic 