# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import approx_fprime

from fcn_fiveholeprobe import fhp_sim
from pfqtoolbox import std_atm76_pressure_altitude, qcparatio_to_mach


def fhp_sim_state_vector(Xbar):
    """Returns simulated pressures on a 5-hole port.
    
    Parameters
    ----------
    Xbar : array-like
        State vector (instead of h, Mach, alpha, beta)
    """
    Pt, Pinf, alpha, beta = Xbar
    
    h = std_atm76_pressure_altitude(Pinf)
    mach = qcparatio_to_mach((Pt-Pinf)/Pinf)
    
    sim_results = fhp_sim(h, mach, alpha, beta)
    p_1, p_2, p_3, p_4, p_5, qc, q_inf, p_static, p_total = sim_results
    return [p_1, p_2, p_3, p_4, p_5]
    

def jacobian(f, x, *args, eps=None):
    """Computes the Jacobian of a vector-valued function f at x.
    
    Parameters
    ----------
    f : array-like of callables
        An m-dimensional vector function. f maps from R**n -> R**m.
        
    x : array-like
        A point in the n-dimensional function domain to evaluate the Jacobian.
        
    *args : args, optional
        Any other arguments to pass to f
        
    eps : scalar or array-like, default None
        The increment to determine the gradient. If a scalar, the same eps will
        be used for each dimension of x. If array-like, should match the
        dimensionality of x.
        
    Returns
    -------
    J : ndarray
        The m x n Jacobian matrix
    """
    if not eps: eps = np.sqrt(np.finfo(float).eps)
    n = len(x)
    grad = approx_fprime
    J = np.concatenate([grad(x, f_i, eps, *args).reshape((1,n)) for f_i in f],
                        axis=0)
    return J


def example():
    """Returns the jacobian of a sample 2-d vector function."""
    def f1(x):
        return np.sin(x[0]) + np.cos(x[1]) + 3*x[2]
    
    def f2(x):
        return np.cos(x[0]) + np.sin(x[1]) + 5*x[2]
    
    f = [f1, f2]
    x = np.array([1, 2, 3])
    return jacobian(f, x)    


if __name__ == '__main__':
    # Uncomment the below line to run example()
    # J = example()
    
    # MM example
    from pfqtoolbox import (std_atm76_pressure, mach_to_qcparatio, 
                            pres_mach_to_qbar)
        
    # Initial guess for air data state vector Xbar
    h = 10000   # Altitude [FT]
    mach = 0.5  # [ND]
    alpha = 2.
    beta= 0.
    
    delta, p_static = std_atm76_pressure(h)  # [PSF]
    
    # Compute pressures
    qcparatio = mach_to_qcparatio(mach)
    qc = qcparatio*p_static     #[PSF]
    p_total = qc + p_static    #[PSF]
    
    # Compute qbar
    q_inf = pres_mach_to_qbar(p_static, mach)
    
    # State vector initial guess
    Xbar0 = np.array([p_total, p_static, alpha, beta])
    
    f = fhp_sim_state_vector(Xbar0)
    H = jacobian(f, Xbar0)