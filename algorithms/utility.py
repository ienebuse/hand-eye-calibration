import numpy as np
from numpy.lib.function_base import append
from numpy.linalg import inv, det, svd, eig, norm, pinv
# from data import *
# from data2 import *
from sim_data import *
# from sim_data3 import *
# from utility import *
from scipy import sparse
import time
import cv2 as cv

from scipy.spatial.transform import Rotation as Rot



def solveSVD(A):
    U,S,VT = svd(A)

    '''Solution using matrix kernel'''
    x = VT.T[:,-1]
    return x

def solveLS(A,B):

    u,s,v = svd(A)

    _s = inv(np.diag(s))
    _ss = np.zeros((3,u.shape[0]))
    _ss[:3,:3] = _s

    x = np.dot(np.dot(v.T,_ss),np.dot(u.T,B))
    
    # x = np.dot(inv(np.dot(A.T,A)),np.dot(A.T,B))
    return x

def groundTruth(HX):
    '''Returns the ground truth Hand-eye transform matrix'''
    return H_Pose(HX) if isinstance(HX,list) else HX

def skew(x):
    '''Transforms a vector to a skew symmetric matrix'''
    x = x.ravel()
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
                     
def rotR_2_6D(R):
    gR = R[:,:-1].T.flatten()
    return gR

def rot6D_2_R(gR):
    a1 = gR[:3].reshape(3,1)
    a2 = gR[3:].reshape(3,1)

    r1 = a1/norm(a1)
    r2 = (a2 - np.dot(a1.T,a2)*a1)/norm(a2 - np.dot(a1.T,a2)*a1)
    r3 = np.cross(r1.flatten(),r2.flatten()).reshape(3,1)
    fR = np.append(r1,np.append(r2,r3,axis=1),axis=1)
    return fR

def get_rad(deg):
    '''Converts angle in degree to radians'''
    theta = deg*np.pi/180
    return theta

def get_deg(rad):
    theta = rad*180/np.pi
    return theta

def getRotation(Rx):
    '''Get the rotation matrix that satisfy othorgonality'''
    u,s,v = svd(Rx)
    return np.dot(u,v)

def rot(x,y,z,rad=False):
    '''Computes the rotation matrix, given the independent rotations about x,y,z'''
    if(not rad):
        x = get_rad(x)
        y = get_rad(y)
        z = get_rad(z)

    R_x = np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    R_y = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    R_z = np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
    return np.dot(R_z,np.dot(R_y,R_x))


def rot_(r,rad=False):
    '''Computes the rotation matrix, given the independent rotations about x,y,z'''
    # r = r.ravel()
    x = r.ravel()[0]
    y = r.ravel()[1]
    z = r.ravel()[2]
    if(not rad):
        x = get_rad(x)
        y = get_rad(y)
        z = get_rad(z)

    R_x = np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    R_y = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    R_z = np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
    return np.dot(R_z,np.dot(R_y,R_x))

def rot2(z2,y,z1):
    '''Computes the rotation matrix, given the independent successive rotations about z,y,z'''
    R_z1 = np.array([[np.cos(z1),-np.sin(z1),0],[np.sin(z1),np.cos(z1),0],[0,0,1]])
    R_y = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    R_z2 = np.array([[np.cos(z2),-np.sin(z2),0],[np.sin(z2),np.cos(z2),0],[0,0,1]])
    return np.dot(R_z2,np.dot(R_y,R_z1))

def R_2_angle_axis(R):
    # U = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    # theta = np.arccos(np.round(0.5*(np.trace(R)-1),8))
    # # theta = np.arccos(0.5*(np.trace(R)-1))
    # if(np.isnan(theta)):
    #     theta = 0
    # u = (1/(2*np.sin(theta)))*U
    # return u.reshape(3,1), theta

    rotvec = Rot.from_matrix(R).as_rotvec()
    theta = norm(rotvec)
    u = rotvec/theta

    return u.reshape(3,1), theta

def angle_axis_2_R(a,w):
    # U = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    # theta = np.arccos(np.round(0.5*(np.trace(R)-1),8))
    # # theta = np.arccos(0.5*(np.trace(R)-1))
    # if(np.isnan(theta)):
    #     theta = 0
    # u = (1/(2*np.sin(theta)))*U
    # return u.reshape(3,1), theta
    rotvec = a*w
    R = Rot.from_rotvec(rotvec.flatten()).as_matrix()
    return R

def R_2_mrp(R):
    mrp = np.asarray(Rot.from_matrix(R).as_mrp())

    return mrp

def MRP_2_R(mrp):
    R = np.asarray(Rot.from_mrp(mrp.flatten()).as_matrix())

    return R


def quaternion_2_R(q):
    q = q.ravel()
    R = np.array([[2*(q[0]*q[0] + q[1]*q[1]) - 1, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])], \
                   [2*(q[1]*q[2] + q[0]*q[3]), 2*(q[0]*q[0] + q[2]*q[2]) - 1, 2*(q[2]*q[3] - q[0]*q[1])], \
                   [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 2*(q[0]*q[0] + q[3]*q[3]) - 1]])

    # R = Rot.from_quat(q.flatten()).as_matrix()

    return R

def R_2_quaternion(R):
    qw= 0.5*np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])
    qx = (R[2,1] - R[1,2])/( 4 *qw)
    qy = (R[0,2] - R[2,0])/( 4 *qw)
    qz = (R[1,0] - R[0,1])/( 4 *qw)
    # q = Rot.from_matrix(R).as_quat()
    return np.array([qw, qx, qy, qz]).reshape(-1,1)
    # return np.asarray(q)

def quaternion_2_angle_axis(q):
    q = q/norm(q)
    w_ = np.arccos(q[0])
    w = w_*2
    if(w_ != 0):
        x = q[1]/np.sin(w_) 
        y = q[2]/np.sin(w_) 
        z = q[3]/np.sin(w_)
        a = np.array([x,y,z]).reshape(-1,1)
    else:
        a = np.zeros((3,1))
    return a,w[0] 

def angle_axis_2_quaternion(a,w):
    q0 = np.cos(w/2)
    q_ = np.sin(w/2)*a.reshape(-1,1)
    q = np.append(np.array([q0]).reshape(-1,1),q_,axis=0)
    return q


def Pose(R,t):
    '''Compute the rotation matrix, given the rotation and translation'''
    t = t.reshape(3,-1)
    H = np.append(np.append(R,t,axis=1),[[0,0,0,1]], axis=0)
    return H

def H_Pose(Pose):
    '''Compute the pose given [x,y,z,a,b,c]'''
    R = rot(Pose[3], Pose[4], Pose[5])
    assert np.abs(det(R) - 1) < 1e-6, "det(R) is {}".format(det(R))
    t = np.array(Pose[:3]).reshape(3,-1)
    H = np.append(np.append(R,t,axis=1),[[0,0,0,1]], axis=0)
    return H

def get_camera_pose(img, cam_mtx, cam_dist, boardsize = (8, 7, 50)):
    CHECKERBOARD = boardsize[0:2]
    SQUARE_SIZE = boardsize[2]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)      # 3D object points
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)           # Get the corner points on the calibration pattern
    
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)       # Get the corner points on the calibration pattern with subpixel accuracy
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, cam_mtx, cam_dist)

        R = cv.Rodrigues(rvecs)[0]
        R_ = R.T
        t = np.dot(-R_, tvecs.reshape(3,1))

        return Pose(R,t)
    else:
        raise ValueError('COULD NOT GET CAMERA POSE FROM IMAGE')

def euler(Rot,rad=False):
    '''Comptutes euler angles (a,b,c) from rotation matrix'''
    rX = np.arctan2(Rot[2,1],Rot[2,2])*180/np.pi
    rY = np.arctan2(-Rot[2,0], np.sqrt(Rot[2,1]**2  +  Rot[2,2]**2))*180/np.pi
    rZ = np.arctan2(Rot[1,0], Rot[0,0])*180/np.pi

    # return[np.round(rX,3), np.round(rY,3), np.round(rZ,3)]
    if rad:
        return [rX*np.pi/180, rY*np.pi/180, rZ*np.pi/180]
    else:
        return [rX, rY, rZ]

def euler2(Rot,rad=False):
    '''Comptutes euler angles (a,b,c) from rotation matrix'''
    rX = np.arctan2(-Rot[2,1],Rot[2,2])*180/np.pi
    rY = np.arcsin(Rot[2,0])*180/np.pi
    rZ = np.arctan2(-Rot[1,0], Rot[0,0])*180/np.pi

    # return[np.round(rX,3), np.round(rY,3), np.round(rZ,3)]
    if rad:
        return [rX*np.pi/180, rY*np.pi/180, rZ*np.pi/180]
    else:
        return [rX, rY, rZ]

def Pose2(Pose1):
    R = euler(Pose1[:3,:3])
    P = [Pose1[0,3], Pose1[1,3], Pose1[2,3], R[0], R[1], R[2]]
    # return np.round(P,6).tolist()
    return P

def noise3(e=0.001,shape=(3,3)):
    _noise = np.random.normal(0,e,size=shape)
    return _noise.reshape(shape[0], shape[1])

def noise2(e=0.001,shape=(3,3)):
    _noise = np.random.rand(shape[0],shape[1])*e
    return _noise.reshape(shape[0], shape[1])

def noise1(P, n):
    Hcg = np.array([[-0.995625,     0.068554,    -0.063493,   -89.552823], \
            [-0.072742,    -0.995152,     0.066188,   -82.510240], \
            [-0.058647,     0.070517,     0.995785,    14.286884], \
            [0.000000,     0.000000,     0.000000,     1.000000]])
    Nce = np.dot(Hcg,np.dot(n,Hcg.T))
    return Nce

# def noise4(P, sigmaR = 0,sigmaT = 0, N=0):
#     Hcg = groundTruth(Hx)

#     tP = [np.random.rand()*sigmaT for t in range(3)] + [np.random.rand()*sigmaR for r in range(3)]
#     # tP = [np.random.normal(0,sigmaT) for t in range(3)] + [np.random.normal(0,sigmaR) for r in range(3)]
#     Pce = P
#     if (N is not 0):
#         Nce = np.dot(Hcg,np.dot(N,inv(Hcg)))    #   noisy movement in A caused by noisy movement in B
#         Pce = np.dot(P,Nce)                     #   total movement in A with noisy movement 
#         return Pce
    
#     else:
#         n = H_Pose(tP)
#         return P,n
#     # nP = np.dot(Pce,n)
#     # return nP
#     # return Pce
def randSign():
    return 1 if np.random.random() < 0.5 else -1
    # return 1


def noise(A, sigmaB=(0,0), sigmaA=(0,0),ret_noise=False):
    Hcg = groundTruth(Hx)

    A_n = A
    if(sigmaB != (0,0)):
        # tP = [np.random.rand()*sigmaB[1]/2 - sigmaB[1] for t in range(3)] + [np.random.rand()*sigmaB[0] - sigmaB[0]/2 for r in range(3)]
        # tP = [np.random.rand()*sigmaB[1] for t in range(3)] + [np.random.rand()*sigmaB[0] for r in range(3)]
        tP = [np.random.randn()*sigmaB[1] for t in range(3)] + [np.random.randn()*sigmaB[0] for r in range(3)]
        # tP = [np.random.normal(0,sigmaB[1]) for t in range(3)] + [np.random.normal(0,sigmaB[0]) for r in range(3)]
        # tP = [randSign()*np.random.beta(2,4)*sigmaB[1] for t in range(3)] + [randSign()*np.random.beta(2,4)*sigmaB[0] for r in range(3)]
        N = H_Pose(tP)
        Nce = np.dot(Hcg,np.dot(N,inv(Hcg)))    #   noisy movement in A caused by noisy movement in B
        # A_n = np.dot(A,Nce)                     #   total movement in A with noisy movement 
        A_n = np.dot(Nce,A)
        if(ret_noise):
            Re = R_2_angle_axis(N[:3,:3])[1]
            te = norm(N[:,3])

    if(sigmaA != (0,0)):
        ta = [np.random.rand()*sigmaA[1]/2 - sigmaA[1] for t in range(3)] + [np.random.rand()*sigmaA[0] - sigmaA[0]/2 for r in range(3)] 
        # ta = [np.random.rand()*sigmaA[1] for t in range(3)] + [np.random.rand()*sigmaA[0] for r in range(3)] 
        # ta = [np.random.randn()*sigmaA[1] for t in range(3)] + [np.random.randn()*sigmaA[0] for r in range(3)]
        # ta = [np.random.normal(0,sigmaA[1]) for t in range(3)] + [np.random.normal(0,sigmaA[0]) for r in range(3)]
        # ta = [randSign()*np.random.beta(2,4)*sigmaA[1] for t in range(3)] + [randSign()*np.random.beta(2,4)*sigmaA[0] for r in range(3)]
        aN = H_Pose(ta)
        # return np.dot(A_n,aN)
        return np.dot(aN,A_n)
    if ret_noise and (sigmaB != (0,0)):
        return A_n,[Re,te]
    else:
        return A_n

    

def random_pose(uniform_=True,sigma=(0,0)):

    def get_data(lim):
        x = np.round(np.random.rand()*(lim[1] - lim[0]) + lim[0],6)
        return x

    
    if uniform_:
        rlim = np.random.rand()*sigma[0]# - sigma[0]/2
        tlim = np.random.rand()*sigma[1]# - sigma[1]/2
    else:
        rlim = np.random.randn()*sigma[0]
        tlim = np.random.randn()*sigma[1]

    tlim = (-np.abs(tlim),np.abs(tlim))
    rlim = (-np.abs(rlim),np.abs(rlim))

    x = get_data(tlim)
    ylim = np.sqrt(tlim[1]**2 - x**2)
    y = get_data((-ylim,ylim))
    zlim = np.sqrt(tlim[1]**2 - (x**2 + y**2))
    z = get_data((0.75*zlim,zlim))
    a = get_data(rlim)
    blim = np.sqrt(rlim[1]**2 - a**2)
    b = get_data((-blim,blim))
    clim = np.sqrt(rlim[1]**2 - (a**2 + b**2))
    c = get_data((0.75*clim,clim))

    n = [x,y,z,a,b,c]

    return n


def noiseQ(A, sigmaB=(0,0), sigmaA=(0,0),ret_noise=False,uniform_=False):
    Hcg = groundTruth(Hx)

    A_n = A
    if(sigmaB != (0,0)):
        tP = [np.random.rand()*sigmaB[1]/2 - sigmaB[1] for t in range(3)] + [np.random.rand()*sigmaB[0] - sigmaB[0]/2 for r in range(3)]
        # tP = [np.random.rand()*sigmaB[1] for t in range(3)] + [np.random.rand()*sigmaB[0] for r in range(3)]
        # tP = [np.random.randn()*sigmaB[1] for t in range(3)] + [np.random.randn()*sigmaB[0] for r in range(3)]
        # tP = [np.random.normal(0,sigmaB[1]) for t in range(3)] + [np.random.normal(0,sigmaB[0]) for r in range(3)]
        # tP = [randSign()*np.random.beta(2,4)*sigmaB[1] for t in range(3)] + [randSign()*np.random.beta(2,4)*sigmaB[0] for r in range(3)]
        
        tP = random_pose(uniform_=uniform_,sigma=sigmaB)

        N = H_Pose(tP)
        Nce = np.dot(Hcg,np.dot(N,inv(Hcg)))    #   noisy movement in A caused by noisy movement in B
        # A_n = np.dot(A,Nce)                     #   total movement in A with noisy movement 
        A_n = np.dot(Nce,A)
        if(ret_noise):
            Re = R_2_angle_axis(N[:3,:3])[1]
            te = norm(N[:,3])

    if(sigmaA != (0,0)):
        ta = [np.random.rand()*sigmaA[1]/2 - sigmaA[1] for t in range(3)] + [np.random.rand()*sigmaA[0] - sigmaA[0]/2 for r in range(3)] 
        # ta = [np.random.rand()*sigmaA[1] for t in range(3)] + [np.random.rand()*sigmaA[0] for r in range(3)] 
        # ta = [np.random.randn()*sigmaA[1] for t in range(3)] + [np.random.randn()*sigmaA[0] for r in range(3)]
        # ta = [np.random.normal(0,sigmaA[1]) for t in range(3)] + [np.random.normal(0,sigmaA[0]) for r in range(3)]
        # ta = [randSign()*np.random.beta(2,4)*sigmaA[1] for t in range(3)] + [randSign()*np.random.beta(2,4)*sigmaA[0] for r in range(3)]
        aN = H_Pose(ta)
        # return np.dot(A_n,aN)
        return np.dot(aN,A_n)
    if ret_noise and (sigmaB != (0,0)):
        return A_n,[Re,te]
    else:
        return A_n

# def noiseX(A, sigma=(0,0)):
#     Hcg = groundTruth(Hx)

#     A_n = A
#     if(sigma != (0,0)):
#         tP = [np.random.rand()*sigma[1] - sigma[1]/2 for t in range(3)] + [np.random.rand()*sigma[0] - sigma[0]/2 for r in range(3)]
#         # tP = [randSign()*np.random.rand()*sigma[1] for t in range(3)] + [randSign()*np.random.rand()*sigma[0] for r in range(3)]
#         # tP = [np.random.randn()*sigma[1] for t in range(3)] + [np.random.randn()*sigma[0] for r in range(3)]
#         N = H_Pose(tP)
#         # A_n = np.dot(A,N)                     #   total movement in A with noisy movement 
#         A_n = np.dot(A,inv(N))                  # ORIGINAL

#     return A_n

def noiseX(A, sigma=(0,0)):
    Hcg = groundTruth(Hx)

    A_n = A
    if(sigma != (0,0)):
        tP = [np.random.rand()*sigma[1]/2 - sigma[1] for t in range(3)] + [np.random.rand()*sigma[0]/2 - sigma[0] for r in range(3)]
        # tP = [randSign()*np.random.rand()*sigma[1] for t in range(3)] + [randSign()*np.random.rand()*sigma[0] for r in range(3)]
        # tP = [np.random.randn()*sigma[1] for t in range(3)] + [np.random.randn()*sigma[0] for r in range(3)]
        N = H_Pose(tP)
        # A_n = np.dot(N,np.dot(A,N))                   #   total movement in A with noisy movement 
        A_n = np.dot(A,inv(N))                  # ORIGINAL
        # A_n = np.dot(A,N) 
        # A_n = np.dot(N,A) 

    return A_n

def noiseX2(A, sigma=(0,0)):
    Hcg = groundTruth(Hx)

    A_n = A
    if(sigma != (0,0)):
        tP = [np.random.rand()*sigma[1]/2 - sigma[1] for t in range(3)] + [np.random.rand()*sigma[0]/2 - sigma[0] for r in range(3)]
        # tP = [np.random.rand()*sigma[1] for t in range(3)] + [np.random.rand()*sigma[0] for r in range(3)]
        # tP = [np.random.randn()*sigma[1] for t in range(3)] + [np.random.randn()*sigma[0] for r in range(3)]
        N = H_Pose(tP)
        A_n = np.dot(A,N)                     #   total movement in A with noisy movement     ORIGINAL
        # A_n = np.dot(A,inv(N))
        # A_n = np.dot(N,A)

    return A_n

I = np.eye(3)