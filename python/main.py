import os
import math
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
        lb = [i+1]+[0]*(2+(i+1)*2)
        ub = [i+1]+[100]*(2+(i+1)*2)
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
num_parents = 400
num_children = 400
num_generations = 10
tourney_size = 20
mutation = .1

#P, Pcost = GA(multiple_cos_cost, selection_func, lb, ub, num_parents,
#                num_children, num_generations, mutation, tourney_size=20)
#best_ind = np.argmin(Pcost)
#best_P = P[best_ind]
#print(best_P)

best_P, best_Pcost = run_multiple_cos(5, num_parents, num_children, num_generations, tourney_size, mutation)

print("best P:")
print(best_P)
print("best Pcost:")
print(best_Pcost)

t0 = 0
dt = .1
tf = 120

x,t = calc_FourFit( t0, dt, tf, best_P )

true_data = np.genfromtxt("mult_cos.csv", delimiter=",");
true_t = true_data[:,0]
true_x = true_data[:,1]

plt.plot( t, x, 'b--' )
plt.plot( true_t, true_x, 'r' )
plt.show()
