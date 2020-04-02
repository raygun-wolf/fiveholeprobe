"""
Create 5-hole probe neural network training and test data over a range of
altitudes, Mach numbers, angles of attack, and angles of sideslip. 

Build traditional regression models that take measured pressures from
the 5-hole probe, convert them into calibration coefficients, and finally
convert those into alt, Mach, alpha, beta.

Last part uses a neural network to build the regression from the training data.
"""

#===================================
from fcn_fiveholeprobe import fhp_sim,fhp
from numpy import arange,concatenate,sqrt,square
from numpy.random import uniform
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

# create a random uniform dataset, which should have more "continuous" data
# than the previous method, which created data at discrete setpoints.
for idx in range(10000):
    h = uniform(h_alt[0],h_alt[-1])
    m = uniform(mach[0],mach[-1])
    a = uniform(alpha[0],alpha[-1])
    b = uniform(beta[0],beta[-1])
    p_1,p_2,p_3,p_4,p_5,qc,q_inf,p_static,p_total=fhp_sim(h,m,a,b)
    k_mach,k_p_static,k_p_total,k_alpha,k_beta=fhp(p_1,p_2,p_3,p_4,p_5)
    df.loc[idx] = [h,m,a,b,p_1,p_2,p_3,p_4,p_5,qc,q_inf,p_static,p_total,k_mach,k_p_static,k_p_total,k_alpha,k_beta]

train,test = train_test_split(df, test_size=0.2, random_state=0) # random_state=0 is seed


# linear regression    ========================================
                
# alpha regression
x1 = train['k_alpha'].values.reshape(-1,1)
y = train['alpha'].values.reshape(-1,1)
regressor = LinearRegression()
regressor.fit(x1,y)
#print(regressor.intercept_)
#print(regressor.coef_)
y_pred = regressor.predict(x1)
print('alpha:')
print(f'R2 score is {r2_score(y,y_pred)}')
print(f'RMSE is {sqrt(mean_squared_error(y,y_pred))}')
plt.plot(x1,y,x1,y_pred)


# beta regression
x1 = train['k_beta'].values.reshape(-1,1)
x2 = train['k_alpha'].values.reshape(-1,1)
X = concatenate((x1,x2),axis=1)
y = train['beta'].values.reshape(-1,1)
regressor = LinearRegression()
regressor.fit(X,y)
#print(regressor.intercept_)
#print(regressor.coef_)
yhat = regressor.predict(X)
print('beta:')
print(f'R2 score is {r2_score(y,yhat)}')
print(f'RMSE is {sqrt(mean_squared_error(y,yhat))}')
plt.plot(x1,y,x1,y_pred)

# static pressure regression
x1 = train['k_p_static'].values.reshape(-1,1)
x2 = train['k_mach'].values.reshape(-1,1)
X = concatenate((x1,x2),axis=1)
y = train['p_static'].values.reshape(-1,1)
regressor = LinearRegression()
regressor.fit(X,y)
#print(regressor.intercept_)
#print(regressor.coef_)
yhat = regressor.predict(X)
print('p_static:')
print(f'R2 score is {r2_score(y,yhat)}')
print(f'RMSE is {sqrt(mean_squared_error(y,yhat))}')
plt.plot(x1,y,x1,yhat)


# total pressure regression
x1 = train['k_p_total'].values.reshape(-1,1)
x2 = train['k_p_static'].values.reshape(-1,1)
x3 = train['k_alpha'].values.reshape(-1,1)
x4 = train['k_beta'].values.reshape(-1,1)
X = concatenate((x1,x2,x3,x4),axis=1)
yhat = train['p_total'].values.reshape(-1,1)
regressor = LinearRegression()
regressor.fit(X,y)
#print(regressor.intercept_)
#print(regressor.coef_)
yhat = regressor.predict(X)
print('p_total:')
print(f'R2 score is {r2_score(y,yhat)}')
print(f'RMSE is {sqrt(mean_squared_error(y,yhat))}')
plt.plot(x1,y,x1,yhat)


#==============================================================
#==============================================================
# use neural network to generate regression

"""
#Multi-layer Perceptron is sensitive to feature scaling. If scaling is needed, 
# use this code to scale the data to mean 0 and standard deviation 1.
ss = StandardScaler()
ss.fit(train) # fit scaler to training data to be used by both the training and test sets
train_scaled = pd.DataFrame(ss.transform(train),columns = train.columns)
test_scaled = pd.DataFrame(ss.transform(test),columns = test.columns)

X = train_scaled[['k_alpha','k_beta']].values
y = train_scaled[['alpha','beta']].values.squeeze()

Xtest = test_scaled[['k_alpha','k_beta']].values
ytest = test_scaled[['alpha','beta']].values.squeeze()
"""

#==============================================================
# Alpha Beta Neural Network -- this section fits the calibration coefficients
# (k_alpha, k_beta) just like the linear regression section above.

# train NN with unscaled data
X = train[['k_alpha','k_beta']].values
y = train[['alpha','beta']].values.squeeze()

Xtest = test[['k_alpha','k_beta']].values
ytest = test[['alpha','beta']].values.squeeze()

max_iter = 10000 # changed to 10k because default of 200 wasn't long enough for convergence
hidden = ((6))  # number of neurons in ith hidden layer
model = MLPRegressor(hidden_layer_sizes=hidden,max_iter=max_iter).fit(X,y)

coefs = model.coefs_
yhat = model.predict(X)
yhattest = model.predict(Xtest)
loss = square(ytest - yhattest).mean()
R2 = model.score(Xtest,ytest)
print(f'{len(hidden)} hidden layer with {hidden} perceptrons, loss = {loss}, R2 = {R2}')

plt.plot(y,yhat,ytest,yhattest)
plt.legend(['train, alpha','train, beta','test, alpha','test, beta'])

#==============================================================
# use neural network to generate regression using measured pressures directly
# instead of the calibration coefficients output by fhp
# p_static and p_total

X = train[['p_1','p_2','p_3','p_4','p_5']].values
y = train[['p_static','p_total']].values.squeeze()

Xtest = test[['p_1','p_2','p_3','p_4','p_5']].values
ytest = test[['p_static','p_total']].values.squeeze()

max_iter = 10000
hidden = (20,20)  # number of neurons in ith hidden layer
model = MLPRegressor(hidden_layer_sizes=hidden,activation='relu',max_iter=max_iter).fit(X,y)

coefs = model.coefs_
yhat = model.predict(X)
yhattest = model.predict(Xtest)
R2 = model.score(Xtest,ytest)
loss = square(ytest - yhattest).mean()
loss_train = square(y-yhat).mean()

print(f'{len(hidden)} hidden layer with {hidden} perceptrons, loss = {loss}, R2 = {R2}')

plt.plot(y,yhat,'ro',ytest,yhattest,'bx')
plt.legend(['train, p_static','train, p_total','test, p_static','test, p_total'])

# Use NN p_static and p_total to calculate altitude and Mach number, compare 
# with truth values

p_static_hat = yhat[:,0]
p_total_hat = yhat[:,1]
p_static_hat_test = yhattest[:,0]
p_total_hat_test = yhattest[:,1]

machhat = qcparatio_to_mach((p_total_hat-p_static_hat)/p_static_hat)
machhattest = qcparatio_to_mach((p_total_hat_test-p_static_hat_test)/p_static_hat_test)
hhat = std_atm76_pressure_altitude(p_static_hat)
hhattest = std_atm76_pressure_altitude(p_static_hat_test)

out = train[['Mach','alt']].values
mach = out[:,0]
h = out[:,1]
out = test[['Mach','alt']].values
machtest = out[:,0]
htest = out[:,1]

error_h = hhat - h
error_htest = hhattest - htest
error_mach = machhat - mach
error_machtest = machhattest - machtest

s
