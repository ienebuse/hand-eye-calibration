'''N. Andreff, R. Horaud, and B. Espiau, “On-line hand-eye calibration,”
International Conference on 3-D Digital Imaging and Modeling, pp.
430–436, 1999.'''



import numpy as np
from numpy.lib.function_base import append
from numpy.linalg import inv, det, svd, eig
# from sim_data import *
from utility import *
import time
from scipy import sparse
import analysis


def vec(A):
    vecA = A.T.ravel().T.reshape(-1,1)
    return vecA

def get_RX_tX(X):
    _Rx = X[:9].reshape(3,3)
    _tX = X[9:]

    w = det(_Rx)
    w = np.sign(w)/(abs(w)**(1/3))

    Rx = w*_Rx
    tX = w*_tX

    return Rx.T,tX

def solveLS(A,B):
    '''Solves the equation Ax=B using Least Square'''
    x = np.dot(inv(np.dot(A.T,A)), np.dot(A.T,B))
    return x


def calibrate(A,B, sigmaA=(0,0), sigmaB=(0,0)):
    N = len(A)
    I = np.eye(3)
    I9 = np.eye(9)

    S = None
    T = None

    tic = time.perf_counter()

    for i in range(N):
        An = noise(A[i], sigmaB, sigmaA)
        Bn = B[i]
		
        RA = An[:3,:3]                               # relative rotation of camera between successive movement
        tA = An[:3,3].reshape(3,1)                   # relative translatioon of camera between successive movement
        tA_ = skew(tA)
        RB = Bn[:3,:3]                               # relative rotation of robot between successive movement
        tB = Bn[:3,3].reshape(3,1)                   # relative translation of robot between successive movement
        

        S1 = np.append(I9 - np.kron(RB,RA), np.zeros((9,3)), axis=1)
        S2 = np.append(np.kron(tB.T,tA_), np.dot(tA_,(I-RA)), axis=1)

        _S = np.append(S1, S2, axis=0)
        _T = np.append(np.zeros((9,1)),tA.reshape(3,1),axis=0)
        
        S = np.append(S, _S, axis=0) if S is not None else _S         # populate system matrix
        T = np.append(T, _T, axis=0) if T is not None else _T         # populate output matrix
        
    Rx_tX = solveSVD(S)

    Rx,tX = get_RX_tX(Rx_tX)

    toc = time.perf_counter()

    Rx = getRotation(Rx)

    Hx = Pose(Rx,tX)

    return Rx, tX.reshape(3,1), Hx, toc-tic