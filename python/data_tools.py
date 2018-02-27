# Tools for finding qualities of the data
# Dakota Thompson

import numpy as np
import scipy as sp
import scipy.signal as sig



# offset corresponds to any shift of data input 
# ex: find_peaks(data[144:,1],144)
def find_peaks(dataset, offset = 0):

	# First derivative
	drv_1 = np.diff(dataset[:,1])/np.diff(dataset[:,0])
	drv_1 = sig.savgol_filter(drv_1, 51, 3)
	
	# Second Derivative
	drv_2 = np.diff(drv_1)/np.diff(dataset[:-1,0])
	drv_2 = sig.savgol_filter(drv_2, 51, 3)
	
	# Third Derivative
	drv_3 = np.diff(drv_2)/np.diff(dataset[:-2,0])
	drv_3 = sig.savgol_filter(drv_3, 51, 3)
	
	# Storage space for peak index
	peaks = []
	
	# Collect peak points using third derivative
	collect = True
	for i in range(1,len(drv_1)-1):
		if collect:
			if drv_1[i-1]>0 and drv_1[i+1]<0: # flip equalities for troughs
				peaks.append(i+offset)
				collect = False
		else:
			collect = True
	
	# Return results
	return peaks