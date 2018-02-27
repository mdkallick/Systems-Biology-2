import os
import math
import numpy as np
import matplotlib.pyplot as plt
from gen_al_utils import GA
from cost import simple_cos_cost
from cost import multiple_cos_cost

def run_multiple_cos(N, num_parents, num_children, num_generations, tourney_size, mutation):
    best_Ps = []
    best_Pcosts = []
    for i in range(N):
        lb = [i+1]+[0]*(2+(i+1)*2)
        ub = [i+1]+[0]*(2+(i+1)*2)
        P, Pcost = GA(multiple_cos_cost, selection_func, lb, ub, num_parents,
                      num_children, num_generations, mutation, tourney_size=tourney_size)
        best_ind = np.argmin(Pcost)
        best_P = P[best_ind]
        best_Pcost = Pcost[best_ind]
        best_Ps.append(best_P)
        best_Pcosts.append(best_Pcost)
    print(best_Ps)
    print(best_Pcosts)
    best_ind = np.argmin(best_Pcosts)
    best_P = best_Ps[best_ind]
    best_Pcost = best_Pcosts[best_ind]
    return best_P, best_Pcost

        
"""
simple cos
"""
selection_func="tournament_select"
num_parents = 50
num_children = 50
num_generations = 9
tourney_size = 20
mutation = .1

#P, Pcost = GA(multiple_cos_cost, selection_func, lb, ub, num_parents,
#                num_children, num_generations, mutation, tourney_size=20)
#best_ind = np.argmin(Pcost)
#best_P = P[best_ind]
#print(best_P)

P, Pcost = run_multiple_cos(5, num_parents, num_children, num_generations, tourney_size, mutation)

t0 = 0
dt = .1
tf = 120

amp = best_P[0]
per = best_P[1]

t = np.arange(t0, tf, dt)
angles = np.multiply(t, (2*math.pi)/per)
x = np.multiply(np.cos(angles), amp)

true_data = np.genfromtxt("cos.csv", delimiter=",");
true_t = true_data[:,0]
true_x = true_data[:,1]

plt.plot( t, x, 'b--' )
plt.plot( true_t, true_x, 'r' )
plt.show()
