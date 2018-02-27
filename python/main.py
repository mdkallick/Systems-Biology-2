import os
import math
import numpy as np
import matplotlib.pyplot as plt
from gen_al_utils import GA
from cost import simple_cos_cost

"""
simple cos
"""
selection_func="tournament_select"
num_cos = 1
lb = [0] * (
ub = [100]
num_parents = 50
num_children = 50
num_generations = 9
mutation = .1

P, Pcost = GA(simple_cos_cost, selection_func, lb, ub, num_parents,
                num_children, num_generations, mutation, tourney_size=20)
best_ind = np.argmin(Pcost)
best_P = P[best_ind]
print(best_P)

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
