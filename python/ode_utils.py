import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
import math

"""
Where the input is an array big enough for every timestep, which also can change at
every timestep.
"""

def ode15s(function, yinit, t0, dt, tf, params, inputs=None, rtol=1e-6, atol=1e-12):
    t = np.arange(t0, tf + (dt * 2), dt)

    if (inputs is not None):
        inputs = np.array([inputs])
        if (inputs.shape[0] != t.shape[0]):
            raise Exception("Size of input matrix does not match the number of time steps")
        params = np.array([params])
        params = np.repeat(params, inputs.shape[1], axis=0).T
        params = np.concatenate([params, inputs], axis=0)
        r = set_ode15s(function, yinit, t0, params[:, 0])

    if (inputs is None):
        r = set_ode15s(function, yinit, t0, params, rtol, atol)

    sol = np.zeros((t.shape[0], len(yinit)))
    sol[0] = yinit

    i = 0
    if (inputs is None):
        while r.successful() and r.t < tf:
            i += 1
            sol[i] = r.integrate(r.t + dt)

    elif (inputs is not None):
        while r.successful() and r.t < tf:
            i += 1
            sol[i] = r.integrate(r.t + dt)
            r = set_ode15s(function, r.y, r.t, params[:, i], rtol, atol)

    return t, sol


def set_ode15s(function, yinit, t0, params, rtol, atol):
    return ode(function).set_f_params(params) \
    .set_initial_value(yinit, t0).set_integrator('vode', method='bdf',
                                                order=5, atol=atol, rtol=rtol)

def ode_func(function, yinit, t0, dt, tf, params, method, inputs=None):
    if (inputs is not None):
        inputs = np.array([inputs])
        params = np.array([params])
        params = np.repeat(params, inputs.shape[1], axis=0).T
        params = np.concatenate([params, inputs], axis=0)

    if (method == 'forward_euler'):
        stepfunc = forward_euler
    elif (method == 'explicit_trapezoid'):
        stepfunc = explicit_trapezoid
    elif (method == 'RK4'):
        stepfunc = RK4
    else:
        raise Exception("Step Function requested does not match any implemented functions")

    t = np.arange(t0, tf + (dt), dt)

    sol = np.zeros((t.shape[0], len(yinit)))
    sol[0] = yinit

    i = 0
    for i in range(t.shape[0] - 1):
        if (inputs is None):
            p = params
        elif (inputs is not None):
            p = params[i + 1]

        sol[i + 1] = stepfunc(function, t, dt, sol[i], p)

    return t, sol


def forward_euler(function, t, dt, prev, params):
    return prev + (dt * np.array(function(t, prev, params)))


def explicit_trapezoid(function, t, dt, prev, params):
    pred_step = function(t, forward_euler(function, t, dt, prev, params), params)
    return prev + (.5 * dt * np.add(function(t, prev, params), pred_step))

def RK4(function, t, dt, prev, params):
    S0 = prev
    F1 = np.array(function(t, S0, params))
    S1 = prev + ((dt / 2) * F1)
    F2 = np.array(function(t + (dt / 2), S1, params))
    S2 = prev + ((dt / 2) * F2)
    F3 = np.array(function(t + (dt / 2), S2, params))
    S3 = prev + (dt * F3)
    F4 = np.array(function(t + dt, S3, params))
    cur = prev + ((dt / 6) * (F1 + (2 * F2) + (2 * F3) + F4))
    return cur

def pred_ode_func(function, yinit, t0, tf, EST, params,
                  dt_init=None, dt_out=None, inputs=None):
    if (dt_init == None):
        dt_init = float(tf - t0) / 100.
    if (inputs is not None):
        inputs = np.array([inputs])
        params = np.array([params])
        params = np.repeat(params, inputs.shape[1], axis=0).T
        params = np.concatenate([params, inputs], axis=0)

    t_cur = t0
    t = [t_cur]
    h = dt_init

    if (dt_out != None):
        # t_out = np.arange(t0, tf + (dt_out), dt_out)
        t_out = [t0]
        tmp_t = t_cur

    sol = np.array(yinit, ndmin=2)
    sol_out = np.array(sol, ndmin=2)

    i = 0
    if (dt_out != None):
        j = 0
    while (t_cur < tf):
        if (inputs is None):
            p = params
        elif (inputs is not None):
            p = params[i + 1]

        tmp_sol = forward_euler(function, t_cur, h, sol[i], p)
        tmp_pred_sol = explicit_trapezoid(function, t_cur, h, sol[i], p)

        err = np.linalg.norm(tmp_pred_sol - tmp_sol)

        h_new = h * math.pow((.9 * EST / err), .5)
        if (h_new > (2 * h)):
            h_new = (2 * h)

        if (err <= EST):
            sol = np.concatenate([sol, np.array(tmp_sol, ndmin=2)], axis=0)
            t_cur = t_cur + h
            if (t_cur + h_new > tf):
                h_new = tf - t_cur
            t.append(t_cur)
            i = i + 1

            if (dt_out != None):
                while (tmp_t < t_cur - dt_out):
                    slope = np.divide((np.array(sol[i + 1]) - np.array(sol[i])), h)
                    tmpnxt = np.array(sol_out[j] + (dt_out * slope), ndmin=2)
                    sol_out = np.concatenate([sol_out, tmpnxt], axis=0)
                    j = j + 1
                    tmp_t = tmp_t + dt_out
                    t_out.append(tmp_t)

        h = h_new

    if (dt_out != None):
        return np.array(t_out), np.array(sol_out)
    else:
        return np.array(t), np.array(sol)

def hill(top, bottom, n):
    return (math.pow(top, n) / (math.pow(top, n) + math.pow(bottom, n)))


# returns index of nearest value in the array
def find_nearest(array, value):
    return (np.abs(array - value)).argmin()
