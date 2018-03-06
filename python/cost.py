import math
import warnings
import numpy as np
from utils import calc_FourFit

def simple_cos_cost( params ):
    t0 = 0
    dt = .1
    tf = 120

    amp = params[0]
    per = params[1]

    t = np.arange(t0, tf, dt)
    angles = np.multiply(t, (2*math.pi)/per)
    x = np.multiply(np.cos(angles), amp)

    true_data = np.genfromtxt("cos.csv", delimiter=",");
    true_t = true_data[:,0]
    true_x = true_data[:,1]

    err = np.sum(np.square(np.subtract(x, true_x)))
    # print("err:", err)

    cost = err
    return cost

def multiple_cos_cost( params ):
    true_data = np.genfromtxt("mult_cos.csv", delimiter=",");
    true_t = true_data[:,0]
    true_x = true_data[:,1]
    
#     t0 = 0
#     dt = .1
#     tf = (true_t.shape[0]+t0)*dt

    x, t = calc_FourFit( true_t, params )

    true_data = np.genfromtxt("mult_cos.csv", delimiter=",");
    true_t = true_data[:,0]
    true_x = true_data[:,1]

    err = np.sum(np.square(np.subtract(x, true_x)))
    # print("err:", err)

    cost = err
    return cost
