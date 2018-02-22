import math
import numpy as np

def becker_weimann( t, y, params ):
    v1b = params[0]
    k1b = params[1]
    k1i = params[2]
    c = params[3]
    p = params[4]
    k1d = params[5]
    k2b = params[6]
    q = params[7]
    k2d = params[8]
    k2t = params[9]
    k3t = params[10]
    k3d = params[11]
    v4b = params[12]
    k4b = params[13]
    r = params[14]
    k4d = params[15]
    k5b = params[16]
    k5d = params[17]
    k5t = params[18]
    k6t = params[19]
    k6d = params[20]
    k6a = params[21]
    k7a = params[22]
    k7d = params[23]

    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    y4 = y[3]
    y5 = y[4]
    y6 = y[5]
    y7 = y[6]

    dydt = np.zeros(len(y))

    dydt[0] = (v1b * (y7 + c))/((k1b * (1 + math.pow(y3/k1i,p))) + (y7 + c)) - (k1d * y1);
    dydt[1] = (k2b * math.pow(y1,q)) - (k2d * y2) - (k2t * y2) + (k3t * y3);
    dydt[2] = (k2t * y2) - (k3t * y3) - (k3d * y3);
    dydt[3] = ((v4b * math.pow(y3,r))/(math.pow(k4b,r) + math.pow(y3,r))) - (k4d * y4);
    dydt[4] = (k5b * y4) - (k5d * y5) - (k5t * y5) + (k6t * y6);
    dydt[5] = (k5t * y5) - (k6t * y6) - (k6d * y6) + (k7a * y7) - (k6a * y6);
    dydt[6] = (k6a *y6) - (k7a * y7) - (k7d * y7);

    return dydt
