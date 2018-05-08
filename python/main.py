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

# best_P = np.array([  5.    ,     946.62732358,  69.80521613, 171.24921375,  19.04053311,
#  905.46152261, 230.99667462 ,930.91259142 ,513.27603457, 253.96031448,
#  141.89190855, 971.58915881 ,317.85239937])


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

print("../data/"+filename)
true_data = np.genfromtxt("../data/"+filename, delimiter=",", skip_header=3, skip_footer=1, missing_values=0)

for i in range((int)(true_data.shape[1]/2)):
	i=i*2
	true_t = true_data[:,i]
	true_x = true_data[:,i+1]
	day1 = np.argmin(np.absolute(np.subtract(true_t, 1.0)))
	true_t = true_data[:,i][day1:]
	true_x = true_data[:,i+1][day1:]
	plt.plot( true_t, true_x, linewidth=2 )
day1 = np.argmin(np.absolute(np.subtract(true_t, 1.0)))
plt.title("BMAL1 Oscillation Data")
plt.xlabel("Time (days)")
plt.ylabel("Counts per Second")
# plt.axvline(true_t[day1])
plt.show()

col = 0
# true_t = true_data[:,col]
# true_x = true_data[:,col+1]

fit_x, inv_fit_x, true_t, true_x, fixed_x, pv_idx, new_idx = \
						find_pv_single(true_data, col, save_plot=False, show_plot=False)

per_t = np.multiply(np.diff(pv_idx[0]),2) #convert from half-period to full period
print(per_t)

# convert the period from index change to time change (in hours)
dt = np.average(np.diff(true_t))
print(dt)
per_t = np.multiply(per_t, (dt*24))

pow = 3
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
# print(pred_x)

# print(true_t.shape)
# print(fixed_x.shape)

# generate plot that shows the inverted fit fixing the data
plt.plot( fit_x, true_t )
plt.plot( inv_fit_x, true_t )
plt.plot( fixed_x, true_t )
plt.plot( true_x, true_t )
plt.show()
plt.clf()

# print(pv_idx[0])
plt.plot( true_t[pv_idx[0][:-1]], per_t, 'rx', mew=2, markersize=8, label="measured period (from found peaks and valleys)")
plt.plot( true_t[:-100], fit_x[:-100], 'b--', linewidth=2, label="polynomial fit (power "+str(pow)+")")
plt.legend(loc="best")
plt.title("Oscillation Period over Time")
plt.xlabel("Time (days)")
plt.ylabel("Period (hours)")
plt.show()
plt.clf()
# plt.plot( true_t, true_x )
# plt.scatter( true_t[pv_idx], true_x[pv_idx] )
# plt.scatter( true_t[pred_x], true_x[pred_x])
# plt.plot( true_t, fixed_x, 'm' )
# plt.scatter( true_t[pv_idx], fixed_x[pv_idx] )
# plt.scatter( true_t[pred_x], fixed_x[pred_x] )
# plt.show()
