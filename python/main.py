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
from per_calc import find_pv_full
from per_calc import find_pv_single

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

# max = (0,0)
# 
# for filename in ["021717_12h_starvation_Ca1a_Bmal1.csv", "021717_12h_starvation_10A_Bmal1.csv"]:
# 	tmpmax = find_pv_full("../data/"+filename, save_plot=False)
# 	print(tmpmax)
# 	if(tmpmax[0] > max[0]):
# 		max = (tmpmax[0], (tmpmax[1], filename))
# print(max)
# 
# true_data = np.genfromtxt("021717_12h_starvation_Ca1a_Bmal1.csv", delimiter=",", skip_header=3, skip_footer=1, missing_values=0);
# 
# true_t = true_data[:,4]
# true_x = true_data[:,5]
# 
# plt.plot( true_t, true_x, 'r')
# plt.show()

filename = '021717_12h_starvation_Ca1a_Bmal1.csv'
# filename = '021717_12h_starvation_10A_Bmal1.csv'

true_data = np.genfromtxt("../data/"+filename, delimiter=",", skip_header=3, skip_footer=1, missing_values=0)

col = 6
# true_t = true_data[:,col]
# true_x = true_data[:,col+1]

fit_x, inv_fit_x, true_t, true_x, fixed_x, pv_idx, new_idx = \
						find_pv_single(true_data, col, save_plot=False, show_plot=False)

per_t = np.diff(pv_idx[0])
print(per_t)

pow = 2
coeffs = np.polyfit(true_t[pv_idx[0][:-1]], per_t, pow)
fit_x = np.polyval(coeffs, true_t)

pred_x = [pv_idx[0][0]]
# pred_x = [0]
# for per in fit_x:
# 	pred_x.append(pred_x[-1]+per)


while True:
# 	print(pred_x[-1])
	pred_x.append(pred_x[-1]+fit_x[(int)(pred_x[-1])])
	if(true_t.shape[0] < pred_x[-1]):
		pred_x.pop()
# 		pred_x.pop(0) # remove the dummy first point
		break
	if(fit_x[(int)(pred_x[-1])] < 0):
		break
	
pred_x = np.array(pred_x).astype(int)
print(pred_x)

print(true_t.shape)
print(fixed_x.shape)

print(pv_idx[0])
plt.plot( true_t[pv_idx[0][:-1]], per_t, 'x')
plt.plot( true_t, fit_x, 'b--', label="polynomial fit (power "+str(pow)+")")
plt.show()
plt.clf()
plt.plot( true_t, true_x )
plt.scatter( true_t[pv_idx], true_x[pv_idx] )
plt.scatter( true_t[pred_x], true_x[pred_x])
plt.plot( true_t, fixed_x, 'm' )
plt.scatter( true_t[pv_idx], fixed_x[pv_idx] )
plt.scatter( true_t[pred_x], fixed_x[pred_x] )
plt.show()

