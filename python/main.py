import os
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
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

true_data = np.genfromtxt("mult_cos.csv", delimiter=",");
true_t = true_data[:,0]
true_x = true_data[:,1]

t0 = 0
dt = .1
tf = (true_t.shape[0]+t0)*dt

# best_P = np.array([  5.    ,     946.62732358,  69.80521613, 171.24921375,  19.04053311,
#  905.46152261, 230.99667462 ,930.91259142 ,513.27603457, 253.96031448,
#  141.89190855, 971.58915881 ,317.85239937])

# x,t = calc_FourFit( true_t, best_P )

# where t is x and x is y (confusing, I know) 
coeffs = np.polyfit(true_t, true_x, 15)
fit_x = np.polyval(coeffs, true_t)

print(np.poly1d(coeffs))

# plt.plot( t, x, 'b--' )
plt.plot( true_t, true_x , 'r' )
plt.plot( true_t, fit_x, 'b--')
plt.show()
