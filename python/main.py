import os
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
from gen_al_utils import GA
from cost import simple_cos_cost
from cost import multiple_cos_cost
from utils import calc_FourFit

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


"""
simple cos
"""
selection_func="tournament_select"
num_parents = 1000
num_children = 1000
num_generations = 20
tourney_size = 50
mutation = .1

#P, Pcost = GA(multiple_cos_cost, selection_func, lb, ub, num_parents,
#                num_children, num_generations, mutation, tourney_size=20)
#best_ind = np.argmin(Pcost)
#best_P = P[best_ind]
#print(best_P)


# best_P, best_Pcost = run_multiple_cos(5, num_parents, num_children, num_generations, tourney_size, mutation)
#
# print("best P:")
# print(best_P)
# print("best Pcost:")
# print(best_Pcost)

true_data = np.genfromtxt("021717_12h_starvation_Ca1a_Bmal1.csv", delimiter=",", skip_header=3, missing_values=0);
true_t = true_data[:,4]
true_x = true_data[:,5]

t0 = 0
dt = .1
tf = (true_t.shape[0]+t0)*dt

# best_P = np.array([  5.    ,     946.62732358,  69.80521613, 171.24921375,  19.04053311,
#  905.46152261, 230.99667462 ,930.91259142 ,513.27603457, 253.96031448,
#  141.89190855, 971.58915881 ,317.85239937])

# x,t = calc_FourFit( true_t, best_P )

# where t is x and x is y (confusing, I know)
coeffs = np.polyfit(true_t, true_x, 3)
fit_x = np.polyval(coeffs, true_t)

print(np.poly1d(coeffs))

inv_fit_x = np.max(fit_x) - fit_x

fixed_x = np.add(inv_fit_x, true_x)
avg_y = np.average(fixed_x)

fixed_x = sig.savgol_filter(fixed_x, 51, 3)

dx = sig.savgol_filter(np.diff(fixed_x), 51, 3)
shift_dx = dx[1:]
# dddx = sig.savgol_filter(np.diff(dx,2), 51, 3)

pv_idx = np.where(((dx[:-1] > 0) & (shift_dx < 0)) | ((dx[:-1] < 0) & (shift_dx > 0)))

pv_times = true_t[pv_idx]
pv_diff = np.diff(pv_times)
avg_time = np.average(pv_diff)
std_time = np.std(pv_diff)
print(pv_idx)
print(avg_time)
print(std_time)
print(pv_times)
print(pv_diff)

missing_idx = np.add(np.where(pv_diff > (avg_time + std_time)),1)

tmp_list = pv_idx[0]

for miss in missing_idx:
	miss = miss[0]
	print(miss)
	print(tmp_list)
	tmp_list = np.insert(tmp_list, miss, (pv_idx[0][miss-1]+pv_idx[0][miss])/2)
	print(tmp_list)

pv_idx = (np.array(tmp_list),)
# print(pv_idx)

cutoff=0

# plt.plot( t, x, 'b--' )
plt.plot( true_t[cutoff:], true_x[cutoff:] , 'r' )
plt.plot( true_t[cutoff:], fixed_x[cutoff:] , 'm' )
# plt.plot( true_t, fit_x, 'b--')
# plt.plot( true_t, inv_fit_x, '--')
plt.plot( true_t[cutoff:-1], np.add(np.multiply(dx,50),avg_y)[cutoff:], 'g--' )
plt.plot( true_t[pv_idx], true_x[pv_idx], 'bx' )
plt.plot( true_t[pv_idx], fixed_x[pv_idx], 'rx' )
plt.axhline( avg_y )
plt.show()
