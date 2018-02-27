import math
import numpy as np
import matplotlib.pyplot as plt

def generate_cos(amp, per, t0, dt, tf):
    t = np.arange(t0, tf, dt)
    angles = np.multiply(t, (2*math.pi)/per)
    x = np.multiply(np.cos(angles), amp)
    return x, t

def generate_multiple_cos(N, C, tau, amps, phis, t0, dt, tf):
    t = np.arange(t0, tf, dt)

    x = np.zeros_like(t)

    for i in range(N):
        tmp_angles = np.divide(np.multiply(2*math.pi, np.subtract(phis[i], t)), (tau/(i+1)))
        tmp_x = np.multiply(amps[i], np.cos(tmp_angles))
        x = np.add(x, tmp_x)
    x = np.add(C, x)
    return x, t

def write_to_csv(x, t, filename):
    f = open(filename, 'w')
    for i in range(len(x)):
        f.write(str(t[i]) + "," + str(x[i]) + "\n")

if __name__ == '__main__':
    N = 3
    C = 2
    tau = 24
    amps = [10, 5, 4]
    phis = [24, 20, 2]
    x, t = generate_multiple_cos(N, C, tau, amps, phis, 0, .1, 120)
    plt.plot( t, x )
    plt.show()
    write_to_csv(x, t, "mult_cos.csv")
