import numpy as np
import warnings
import time
import math
import sys

# takes numpy arrays
def get_period(x,y):
	top = np.where(y>np.mean(y))
	jump = top[0][np.where(np.diff(top)[0]>1)]
	return np.mean(np.diff(x[jump]))

def get_amp_old(x,y):
	top = np.where(np.diff(y)>0)
	jump = np.where(np.diff(top)[0]>1)
	return abs(np.mean(y[top[0][jump]])-np.mean(y[top[0][np.add(jump,1)]]))

def get_amps(y):
	return (np.amax(y,axis=0) - np.amin(y,axis=0))

def calc_FourFit( t, params ):
	try:
		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("error", RuntimeWarning)
	
			num_cos = params[0].astype(int)
			C = params[1]
			tau = params[2]
			A = []
			phi = []
			for i in range(num_cos):
				A.append(params[3+i])
				phi.append(params[3+num_cos+i])

# 			t = np.arange(t0, tf, dt)
			x = np.zeros_like(t)
			for i in range(num_cos):
				tmp_angles = np.divide(np.multiply(2*math.pi, np.subtract(phi[i], t)), (tau/(i+1)))
				tmp_x = np.multiply(A[i], np.cos(tmp_angles))
				x = np.add(x, tmp_x)
			x = np.add(C, x)
	except:
		print("\nparams:\n"+str(params))
		raise RuntimeError
		
	return x, t

### TAKEN FROM: https://stackoverflow.com/a/15860757
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
	barLength = 10 # Modify this to change the length of the progress bar
	status = ""
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
		status = "error: progress var must be float\r\n"
	if progress < 0:
		progress = 0
		status = "Halt...\r\n"
	if progress >= 1:
		progress = 1
		status = "Done...\r\n"
	block = int(round(barLength*progress))
	text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100,6), status)
	sys.stdout.write(text)
	sys.stdout.flush()
