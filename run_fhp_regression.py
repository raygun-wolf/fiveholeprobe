#===================================
from fcn_fiveholeprobe import fhp_sim,fhp
from numpy import arange,concatenate
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# loop through h, mach, alpha, beta
h_alt = arange(0.0,11000,1000)
mach = arange(0.1,0.6,0.1)
alpha = arange(0.0,16,1)
beta = arange(-3.,3,1)

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

# linear regression    ========================================
                
# alpha regression
x1 = df['k_alpha'].values.reshape(-1,1)
y = df['alpha'].values.reshape(-1,1)

regressor = LinearRegression()
regressor.fit(x1,y)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(x1)
print(r2_score(y,y_pred))
plt.plot(x1,y,x1,y_pred)


# beta regression
x1 = df['k_beta'].values.reshape(-1,1)
x2 = df['k_alpha'].values.reshape(-1,1)
X = concatenate((x1,x2),axis=1)
y = df['beta'].values.reshape(-1,1)

regressor = LinearRegression()
regressor.fit(X,y)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X)
print(r2_score(y,y_pred))
plt.plot(x1,y,x1,y_pred)

# static pressure regression
x1 = df['k_p_static'].values.reshape(-1,1)
x2 = df['k_mach'].values.reshape(-1,1)
X = concatenate((x1,x2),axis=1)
y = df['p_static'].values.reshape(-1,1)

regressor = LinearRegression()
regressor.fit(X,y)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X)
print(r2_score(y,y_pred))
plt.plot(x1,y,x1,y_pred)


# total pressure regression
x1 = df['k_p_total'].values.reshape(-1,1)
x2 = df['k_p_static'].values.reshape(-1,1)
x3 = df['k_alpha'].values.reshape(-1,1)
x4 = df['k_beta'].values.reshape(-1,1)
X = concatenate((x1,x2,x3,x4),axis=1)
y = df['p_total'].values.reshape(-1,1)

regressor = LinearRegression()
regressor.fit(X,y)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X)
print(r2_score(y,y_pred))
plt.plot(x1,y,x1,y_pred)


