def fhp_database():
    """
    Create 5-hole probe database over a range of
    altitudes, Mach numbers, angles of attack, and angles of sideslip. 
    
    This database will serve as an "empirical model" of a 5-hole probe to test
    the Shuttle Entry Air Data (SEADS) algorithm.
    
    Reference: AIAA 81-2455 Innovative air data system for the space shuttle orbiter: Data analysis methods
    
    """
    
    #===================================
    
    from fcn_fiveholeprobe import fhp_sim,fhp
    from numpy import array
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split 
    from sklearn.metrics import r2_score,mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from pfqtoolbox import qcparatio_to_mach,std_atm76_pressure_altitude
    import matplotlib.pyplot as plt
    
    # loop through h, mach, alpha, beta
    h_alt = arange(0.0,51000,5000)
    mach = arange(0.1,0.6,0.1)
    alpha = arange(0.0,16.,1)
    beta = arange(-3.,3.,1.)
    
    numberofrows = h_alt.size*mach.size*alpha.size*beta.size
    cols = ['alt','Mach','alpha','beta','p_1','p_2','p_3','p_4','p_5','qc','q_inf','p_static','p_total','k_mach','k_p_static','k_p_total','k_alpha','k_beta']
    
    df = pd.DataFrame(index=arange(0,numberofrows),columns=cols)
    
    idx = 0
    
    for h in h_alt:
        for m in mach:
            for a in alpha:
                for b in beta:
                    p_1,p_2,p_3,p_4,p_5,qc,q_inf,p_static,p_total=fhp_sim(h,m,a,b)
                    k_mach,k_p_static,k_p_total,k_alpha,k_beta=fhp(p_1,p_2,p_3,p_4,p_5)
                    df.loc[idx] = [h,m,a,b,p_1,p_2,p_3,p_4,p_5,qc,q_inf,p_static,p_total,k_mach,k_p_static,k_p_total,k_alpha,k_beta]
                    idx += 1
    return df




from fcn_fiveholeprobe import fhp_sim_state_vector
from numpy import array, sqrt, finfo, concatenate, identity, mat
from numpy.linalg import inv
from scipy.optimize import approx_fprime
from pfqtoolbox import std_atm76_pressure,mach_to_qcparatio

# simulated measured data
# flight conditions:
h = 10000
mach = 0.5
alpha = 2.
beta = 0.

delta,p_static = std_atm76_pressure(h) # [PSF]

# compute pressures
qcparatio = mach_to_qcparatio(mach)
qc = qcparatio * p_static  # [PSF]
p_total = qc + p_static  # [PSF]

# state vector Xbar
Xbar = array([p_total,p_static,alpha,beta])

# observation vector P
p_1,p_2,p_3,p_4,p_5 = fhp_sim_state_vector(Xbar)
P = array((p_1,p_2,p_3,p_4,p_5))


# air data state vector Xbar0 initial guess
p_total0 = 2200.
p_static0 = 2116.22
alpha0 = 3
beta0 = 1

Xbar0 = array([p_total0,p_static0,alpha0,beta0])

# equation 3.5 Jacobian matrix H
eps = sqrt(finfo(float).eps)
grad1 = approx_fprime(Xbar0,fhp_sim_state_vector,[eps,eps,eps,eps],'p_1')
grad2 = approx_fprime(Xbar0,fhp_sim_state_vector,[eps,eps,eps,eps],'p_2')
grad3 = approx_fprime(Xbar0,fhp_sim_state_vector,[eps,eps,eps,eps],'p_3')
grad4 = approx_fprime(Xbar0,fhp_sim_state_vector,[eps,eps,eps,eps],'p_4')
grad5 = approx_fprime(Xbar0,fhp_sim_state_vector,[eps,eps,eps,eps],'p_5')

grad1.shape = (1,4)
grad2.shape = (1,4)
grad3.shape = (1,4)
grad4.shape = (1,4)
grad5.shape = (1,4)

# assemble Jacobian
H = mat(concatenate((grad1,grad2,grad3,grad4,grad5),axis=0))

# equation 3.6 residual vector y
p_10,p_20,p_30,p_40,p_50 = fhp_sim_state_vector(Xbar0)
P0 = array((p_10,p_20,p_30,p_40,p_50))
y = mat(P - P0)
y.shape = (1,5)
# equation 3.10 observation error covariance matrix S
Sinv = mat(identity(5))

# equation 3.8 update vector deltaX
deltaX = inv(H.T * Sinv * H) * H.T * Sinv




