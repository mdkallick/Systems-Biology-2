import math
import warnings
import numpy as np
from utils import calc_FourFit

def simple_cos_cost( params ):
    t0 = 0
    dt = .1
    tf = 120

    x, t = calc_FourFit( t0, dt, tf, params )

    true_data = np.genfromtxt("cos.csv", delimiter=",");
    true_t = true_data[:,0]
    true_x = true_data[:,1]

    err = np.sum(np.square(np.subtract(x, true_x)))
    # print("err:", err)

    cost = err
    return cost
