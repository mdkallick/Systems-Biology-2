import math
import numpy as np
import matplotlib.pyplot as plt

def generate_cos(amp, per, t0, dt, tf):
    t = np.arange(t0, tf, dt)
    angles = np.multiply(t, (2*math.pi)/per)
    x = np.multiply(np.cos(angles), amp)
    return x, t

def write_to_csv(x, t, filename):
    f = open(filename, 'w')
    for i in range(len(x)):
        f.write(str(t[i]) + "," + str(x[i]) + "\n")

if __name__ == '__main__':
    x, t = generate_cos(10, 24, 0, .1, 120)
    plt.plot( t, x )
    plt.show()
    write_to_csv(x, t, "cos.csv")
