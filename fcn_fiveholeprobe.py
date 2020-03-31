def fhp_sim(h=0,mach=0,alpha=0,beta=0):
    """ 
    Function that simulates pressures measured by a 5-hole probe.
    Reference Reference JOURNAL OF AIRCRAFT Vol. 43, No. 3, May–June 2006, 
    New Method for Evaluating the Hemispheric Five-Hole Probes, 
    AIAA-16197-343
      1
    3 5 4
      2
    """
    from numpy import sin,cos,radians
    from pfqtoolbox import std_atm76_pressure,mach_to_qcparatio,pres_mach_to_qbar

    # altitude in [FT]
    delta,p_static = std_atm76_pressure(h) # [PSF]

    # Mach number [ND]
    
    # compute pressures
    qcparatio = mach_to_qcparatio(mach)
    qc = qcparatio * p_static  # [PSF]
    p_total = qc + p_static  # [PSF]

    # compute qbar: q_inf = (1/2)*rho_inf*V_inf**2
    q_inf = pres_mach_to_qbar(p_static,mach)  # [PSF]

   
    alpha_rad = radians(alpha)
    beta_rad = radians(beta)
    
    # Compute C_p for each port based on potential flow theory.
    # equations 10-14, pressure coefficients on a sphere with ports located 45-deg from center
    C_p_1 = -5/4 + 9/8 * (1-sin(2*alpha_rad))*cos(beta_rad)*cos(beta_rad)
    C_p_2 = -5/4 + 9/8 * (1+sin(2*alpha_rad))*cos(beta_rad)*cos(beta_rad)
    C_p_3 = -5/4 + 9/8 * (cos(alpha_rad)*cos(beta_rad)-sin(beta_rad))
    C_p_4 = -5/4 + 9/8 * (cos(alpha_rad)*cos(beta_rad)+sin(beta_rad))
    C_p_5 = -5/4 + 9/4 * (cos(alpha_rad)*cos(alpha_rad)*cos(beta_rad)*cos(beta_rad))
    
    # Convert C_p to pressures. 
    # C_p = (p_i - p_inf) / q_inf, where p_inf = p_static, q_inf = (1/2)rho_infV_inf**2
    p_1 = C_p_1 * q_inf + p_static
    p_2 = C_p_2 * q_inf + p_static
    p_3 = C_p_3 * q_inf + p_static
    p_4 = C_p_4 * q_inf + p_static
    p_5 = C_p_5 * q_inf + p_static
       
    return p_1,p_2,p_3,p_4,p_5,qc,q_inf,p_static,p_total
    
    
def fhp(p_1=2116.22,p_2=2116.22,p_3=2116.22,p_4=2116.22,p_5=2116.22):
    """
    Function takes pressures measured by a five-hole probe and calculates 
    calibration coefficients for altitude, Mach number, angle of attack, 
    and angle of sideslip.
    Reference 1:  JOURNAL OF AIRCRAFT Vol. 43, No. 3, May–June 2006, 
    New Method for Evaluating the Hemispheric Five-Hole Probes, 
    AIAA-16197-343
      1
    3 5 4
      2
    Reference 2: The Calibration and Application of Five-Hole Probes
    by A. L. Treaster and A. M. Yocum, TM 78-10, Jan 18, 1978, DTIC accession
    number a055870.
    """
    from numpy import sin,arctan,zeros

    # Angle of attack, Mach number, total pressure, and static pressure calibration coefficients:
    delta_p = p_5 - (p_1 + p_2)/2  # Ref 1 eq 4
    k_mach = delta_p / p_5  # Ref 1 eq 3
    k_p_static = (p_1+p_2+p_3+p_4)/4  
    k_p_total = (p_5)
    k_alpha = 0.5 * arctan(0.5*(p_2-p_1)/delta_p)  # Ref 1 eq 15
    
    # k_beta, Ref 1 eq 16
    shape = k_alpha.shape # remember shape of array for later
    idx = 0
    k_beta = zeros(k_alpha.ravel().shape) # flatten (ravel) array so it is one dimension, easier to loop on.
    for p1,p2,p3,p4,deltap,kalpha in zip(p_1.ravel(),p_2.ravel(),p_3.ravel(),p_4.ravel(),delta_p.ravel(),k_alpha.ravel()):
        if p1!= p2:
            kbeta= arctan(sin(kalpha)*(p3-p4)/(p2-p1))
        else: # p_1 == p_2
            kbeta = 0.25*(p3-p4)/deltap
        k_beta[idx] = kbeta
        idx += 1

    k_beta.shape = shape  # reshape array to same shape as k_alpha, etc.
    
    return k_mach,k_p_static,k_p_total,k_alpha,k_beta


    

