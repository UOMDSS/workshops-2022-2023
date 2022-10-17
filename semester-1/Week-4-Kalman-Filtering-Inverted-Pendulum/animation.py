import numpy as np
import matplotlib.pyplot as plt
import control
import math
import pendulum
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
#x,x dot,theta, theta dot
WINDOWDIMS = (1200, 600)
CARTDIMS = (50, 10)
PENDULUMDIMS = (6, 200)
GRAVITY = 0.13
REFRESHFREQ = 100
A_CART = 0.15
inv = pendulum.InvertedPendulumGame(WINDOWDIMS, CARTDIMS, PENDULUMDIMS, GRAVITY, A_CART, REFRESHFREQ)
A_=None
B_=None
u=None
filtered=[]
nonfiltered=[]

def Hj(x):
    return np.eye(5)
def H(x):
    return x

ekf=ExtendedKalmanFilter(dim_x=5,dim_z=5,dim_u=1)
ekf.R=0.1*np.eye(5)
ekf.Q=0.01*np.eye(5)
def kalman(x,u):
    ekf.predict_update(x,Hj,H,u=u)
def controller(x,x_true):
    global ekf,A_,B_,u
    if u is None:
        ekf.x=x
    else:
        kalman(x,u=np.array([u]))
    nonfiltered.append(np.mean((x-x_true)**2))
    filtered.append(np.mean((ekf.x-x_true)**2))
    A=np.array([0,1,0,0,0,0,0,0,0,0,0,np.cos(x[2])/float(inv.pendulum.PENDULUMLENGTH),0,1,0,0,0,0,0,inv.pendulum.GRAVITY*np.sin(x[2])/float(inv.pendulum.PENDULUMLENGTH),0,0,0,0,0]).reshape(5,5)
    B=np.array([0,1,0,0,0]).reshape(5,1)
    C=np.eye(5)
    D=np.zeros((5,1))
    system=control.StateSpace(A,B,C,D,dt=1)
    Q=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1000,0,0,0,0,0,10,0,0,0,0,0,0]).reshape(5,5)
    R=np.array([0])
    K,S,E=control.lqr(system,Q,R)
    u,ekf.F,ekf.B=-K[0].dot(ekf.x),A+np.eye(5),B
    return u
inv.game(controller)