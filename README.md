# run_fhp_regression
This script uses fhp_sim to simulate pressures from a five hole probe over
a range of conditions. It also runs fhp to take those pressures and calculate
calibration coefficients. the regression then takes those calibration coefficients
or raw pressures and performs multiple linear regression to compute alpha, beta, 
p_static, and p_total. 

The second portion of the script generates neural networks to duplicate the 
linear regressions.


# fcn_fiveholeprobe
simulation of five hole probe based on potential flow theory
File contains two python functions:

fhp_sim(h=0,mach=0,alpha=0,beta=0)
    Function that simulates pressures measured by a 5-hole probe.
    Reference Reference JOURNAL OF AIRCRAFT Vol. 43, No. 3, May–June 2006, 
    New Method for Evaluating the Hemispheric Five-Hole Probes, 
    AIAA-16197-343
      1
    3 5 4
      2
      
fhp(p_1=2116.22,p_2=2116.22,p_3=2116.22,p_4=2116.22,p_5=2116.22)
    Function takes pressures measured by a five-hole probe and estimates the 
    corresponding altitude, Mach number, angle of attack, and angle of sideslip.
    Reference 1:  JOURNAL OF AIRCRAFT Vol. 43, No. 3, May–June 2006, 
    New Method for Evaluating the Hemispheric Five-Hole Probes, 
    AIAA-16197-343
      1
    3 5 4
      2
    Reference 2: The Calibration and Application of Five-Hole Probes
    by A. L. Treaster and A. M. Yocum, TM 78-10, Jan 18, 1978, DTIC accession
    number a055870.
