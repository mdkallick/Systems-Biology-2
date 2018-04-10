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

def find_pv(filename="021717_12h_starvation_Ca1a_Bmal1.csv", save_plot=True):
	true_data = np.genfromtxt(filename, delimiter=",", skip_header=3, skip_footer=1, missing_values=0);

	print(filename)
# 	print(true_data.shape[1]/2)

	max = (0,0)

	for i in range((int)(true_data.shape[1]/2)):
		true_t = true_data[:,2*i]
		true_x = true_data[:,(2*i)+1]
		
		cutoff = np.argmin(np.absolute(np.subtract(true_t, 1.0)))
# 		print(true_t.shape)
		true_t = true_t[cutoff:]
		true_x = true_x[cutoff:]

# 		print(true_t.shape)

# 		print(cutof	f)

		t0 = 0
		dt = .1
		tf = (true_t.shape[0]+t0)*dt
	
# 		plt.clf()
# 		plt.plot( true_t, true_x , 'r' )
# 		plt.show()
# 		
# 		true_x = true_x[~np.isnan(true_x)]
# 		true_t = true_t[~np.isnan(true_x)]

		# where t is x and x is y (confusing, I know)
		coeffs = np.polyfit(true_t, true_x, 3)
		fit_x = np.polyval(coeffs, true_t)

	# 	print(np.poly1d(coeffs))

		inv_fit_x = np.max(fit_x) - fit_x

		fixed_x = np.add(inv_fit_x, true_x)
		avg_y = np.average(fixed_x)

		fixed_x = sig.savgol_filter(fixed_x, 51, 3)

		dx = sig.savgol_filter(np.diff(fixed_x), 51, 3)
		shift_dx = dx[1:]

		pv_idx = np.where(((dx[:-1] > 0) & (shift_dx < 0)) | ((dx[:-1] < 0) & (shift_dx > 0)))

		pv_times = true_t[pv_idx]
		pv_diff = np.diff(pv_times)
		avg_time = np.average(pv_diff)
		std_time = np.std(pv_diff)
	# 	print(pv_idx)
	# 	print(avg_time)
	# 	print(std_time)
	# 	print(pv_times)
	# 	print(pv_diff)

		missing_idx = np.add(np.where(pv_diff > (avg_time + std_time)),1)

		tmp_list = []

		for miss in missing_idx:
# 			print(miss)
			if(len(miss) > 0):
				miss = miss[0]
		# 		print(miss)
		# 		print(tmp_list)
				tmp_list.append((int)((pv_idx[0][miss-1]+pv_idx[0][miss])/2))
		# 		print(tmp_list)

		new_idx = (np.array(tmp_list),)
# 		print(new_idx[0].shape[0])

		cutoff=0

		figsize=(20, 8)
		
		plt.clf()
		plt.figure(figsize=figsize)
		# plt.plot( t, x, 'b--' )
		plt.plot( true_t[cutoff:], true_x[cutoff:] , 'r', label="raw data" )
		plt.plot( true_t[cutoff:], fixed_x[cutoff:] , 'm', label="corrected data" )
		plt.plot( true_t, fit_x, 'b--', label="polynomial fit (power 3)")
		plt.plot( true_t, inv_fit_x, '--', label="inverted polynomial fit")
		plt.axhline( avg_y )
		plt.legend(loc="best",fontsize='xx-large')
		if(save_plot):
			plt.savefig(filename.rstrip(".csv")+str(i)+"_correct.png")
		# plt.show()

		plt.close()
		plt.figure(figsize=figsize)
		
		plt.plot( true_t[cutoff:], fixed_x[cutoff:] , 'b', label="corrected data" )
		plt.plot( true_t[cutoff:-1], np.add(np.multiply(dx,50),avg_y)[cutoff:], 'g--', label="first derivative" )
		# plt.plot( true_t[pv_idx], true_x[pv_idx], 'bx' )
		plt.plot( true_t[pv_idx], fixed_x[pv_idx], 'rx', mew=2, markersize=6, label="found peaks and valleys" )
		# plt.plot( true_t[new_idx], true_x[new_idx], 'gx' )
		if(new_idx[0].shape[0] != 0):
			plt.plot( true_t[new_idx], fixed_x[new_idx], 'gx', mew=2, markersize=8, label="predicted peaks and valleys" )
		plt.axhline( avg_y )
		plt.legend(loc="best",fontsize='xx-large')

		# print(true_x.shape)
		# window_len=11
		# plt.plot(true_t,smooth(true_x,window_len,"bartlett")[:-window_len+1])
	
		if(save_plot):
			plt.savefig(filename.rstrip(".csv")+str(i)+"_peaks.png")
# 		plt.show()

		if(new_idx[0].shape[0] > max[0]):
			max = (new_idx[0].shape[0], (2*i,(2*i)+1))
	return max
	
