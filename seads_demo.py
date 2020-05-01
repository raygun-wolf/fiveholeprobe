"""
Use a simulated 5-hole probe to test the Shuttle Entry Air Data (SEADS) algorithm.
Reference: AIAA 81-2455 Innovative air data system for the space shuttle orbiter: Data analysis methods
"""
 
from fcn_fiveholeprobe import fhp_sim_state_vector
from numpy import array,sqrt,finfo,concatenate,identity,mat,any,asarray,squeeze,abs,arange,sin,empty,size
from numpy.linalg import inv
from scipy.optimize import approx_fprime
from pfqtoolbox import std_atm76_pressure,mach_to_qcparatio
import matplotlib.pyplot as plt
 
# simulated measured data
# flight conditions
time = arange(0,30,0.01)
h = 10000 + 1000/60*time
mach = 0.5 - 0.01*time
alpha = 2. + 2. * sin(0.1*time)
beta = 0. + 1. * sin(0.1*time)
 
delta,p_static = std_atm76_pressure(h) # [PSF]
 
# compute pressures
qcparatio = mach_to_qcparatio(mach)
qc = qcparatio * p_static  # [PSF]
p_total = qc + p_static  # [PSF]

# loop through each point in time history
i = 0
 
# air data state vector X0 initial guess
p_total0 = 2200.
p_static0 = 2116.22
alpha0 = 3
beta0 = 1
 
# initialize estimates
p_total_est = empty(size(time))
p_static_est = empty(size(time))
alpha_est = empty(size(time))
beta_est = empty(size(time))
 
for i,t in enumerate(time):
    
    # state vector X
    X = array([p_total[i],p_static[i],alpha[i],beta[i]])
    
    # observation vector P
    p_1,p_2,p_3,p_4,p_5 = fhp_sim_state_vector(X)
    P = array((p_1,p_2,p_3,p_4,p_5))
    P.shape = (5,1)
        
    X0 = array([p_total0,p_static0,alpha0,beta0])
    p_10,p_20,p_30,p_40,p_50 = fhp_sim_state_vector(X0)
    P0 = array((p_10,p_20,p_30,p_40,p_50))
    P0.shape = (5,1)
    
    # convergence tolerance
    tol = 0.0001
    y = mat(P - P0)
    
    while any(abs(y) > tol):
        
        # equation 3.5 Jacobian matrix H
        eps =  sqrt(finfo(float).eps)
        grad1 = approx_fprime(X0,fhp_sim_state_vector,[eps,eps,eps,eps],'p_1')
        grad2 = approx_fprime(X0,fhp_sim_state_vector,[eps,eps,eps,eps],'p_2')
        grad3 = approx_fprime(X0,fhp_sim_state_vector,[eps,eps,eps,eps],'p_3')
        grad4 = approx_fprime(X0,fhp_sim_state_vector,[eps,eps,eps,eps],'p_4')
        grad5 = approx_fprime(X0,fhp_sim_state_vector,[eps,eps,eps,eps],'p_5')
        
        grad1.shape = (1,4)
        grad2.shape = (1,4)
        grad3.shape = (1,4)
        grad4.shape = (1,4)
        grad5.shape = (1,4)
        
        # assemble Jacobian
        H = mat(concatenate((grad1,grad2,grad3,grad4,grad5),axis=0))
        
        # equation 3.6 residual vector y
        p_10,p_20,p_30,p_40,p_50 = fhp_sim_state_vector(X0)
        P0 = array((p_10,p_20,p_30,p_40,p_50))
        P0.shape = (5,1)
        y = mat(P - P0)
        y.shape = (5,1)
        
        # equation 3.10 observation error covariance matrix S
        Sinv = mat(identity(5))
        
        # equation 3.8 update vector deltaX
        deltaX = asarray(inv(H.T * Sinv * H) * H.T * Sinv * y)
        
        # refine initial state estimate X0
        X0.shape = (4,1)
        X0 = X0 + deltaX
        X0 = squeeze(X0)
    
    # Use previous frame's data as initial guess for next frame
    p_total_est[i] = X0[0]
    p_static_est[i] = X0[1]
    alpha_est[i] = X0[2]
    beta_est[i] = X0[3]
 
plt.figure()
plt.subplot(2,2,1)
plt.plot(time,p_total)
plt.title('P_total')
plt.subplot(2,2,2)
plt.plot(time,p_static)
plt.title('P_static')
plt.subplot(2,2,3)
plt.plot(time,alpha)
plt.title('alpha')
plt.subplot(2,2,4)
plt.plot(time,beta)
plt.title('beta')
plt.figure()
plt.subplot(2,2,1)
plt.plot(time,p_total - p_total_est)
plt.title('P_total - P_total_est')
plt.subplot(2,2,2)
plt.plot(time,p_static - p_static_est)
plt.title('P_static - P_static_est')
plt.subplot(2,2,3)
plt.plot(time,alpha - alpha_est)
plt.title('alpha - alpha_est')
plt.subplot(2,2,4)
plt.plot(time,beta - beta_est)
plt.title('beta - beta_est')