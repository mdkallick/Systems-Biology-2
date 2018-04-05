import os
import math
import copy
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
from gen_al_utils import GA
from cost import simple_cos_cost
from cost import multiple_cos_cost
from utils import calc_FourFit
from per_calc import find_pv

def run_multiple_cos(N, num_parents, num_children, num_generations, tourney_size, mutation):
    best_Ps = []
    best_Pcosts = []
    for i in range(N):
        lb = [i+1]+[.0001]*(2+(i+1)*2)
        ub = [i+1]+[1000]*(2+(i+1)*2)
        P, Pcost = GA(multiple_cos_cost, selection_func, lb, ub, num_parents,
                      num_children, num_generations, mutation, tourney_size=tourney_size)
        best_ind = np.argmin(Pcost)
        best_P = P[best_ind]
        best_Pcost = Pcost[best_ind]
        best_Ps.append(best_P)
        best_Pcosts.append(best_Pcost)
    best_ind = np.argmin(best_Pcosts)
    best_P = best_Ps[best_ind]
    best_Pcost = best_Pcosts[best_ind]
    return best_P, best_Pcost



for filename in ["021717_12h_starvation_Ca1a_Bmal1.csv", "021717_12h_starvation_10A_Bmal1.csv"]:
	find_pv(filename)
	break
# 
# true_data = np.genfromtxt("021717_12h_starvation_Ca1a_Bmal1.csv", delimiter=",", skip_header=3, skip_footer=1, missing_values=0);
# 
# true_t = true_data[:,4]
# true_x = true_data[:,5]
# 
# plt.plot( true_t, true_x, 'r')
# plt.show()
