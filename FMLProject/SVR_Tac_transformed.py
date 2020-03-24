# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 07:41:57 2018

@author: Aliya Amirzhanova
SVR for transformed TAC dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from scipy.ndimage.interpolation import shift
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.preprocessing import FunctionTransformer
from sklearn import preprocessing
#Cauchy kernel is implmented with github library, where all necessary
#utils can be found (https://github.com/gmum/pykernels)
#from Reg import Cauchy

#load dataset
tac_index=np.load('tac_index.npy')
tac_index=np.reshape(tac_index, len(tac_index))
tac_index_r = tac_index.reshape(-1,1)

#MinMax Scaler for this data is not suitable
'''
scaler = preprocessing.MinMaxScaler()
scaler.fit(tac_index_r)
norm = scaler.transform(tac_index_r)
'''
#logarithmic transformation of the data np.log1p
transformer = FunctionTransformer(np.log1p)
transf = transformer.transform(tac_index_r)
transf = transf.reshape(-1)

#norm = norm.reshape(-1) used previously for MinMAx
#z_score_tac=stats.zscore(tac_index) not suitable for this data, 
#used for stationary series, which is not the case

#make the time series data to supervised learning
t_1 = shift(transf, 1, cval=np.NaN)
y = transf[1:]
X = t_1[1:]

#split into test and train sets
split = 1000
y_train = y[0:split]
y_test = y[split:]

X_train = X[0:split]
X_test = X[split:]


#reshape data for fitting
X_train_reshaped = X_train.reshape(-1,1)
X_test_reshaped = X_test.reshape(-1,1)


#fit SVR model, rbf kernel
clf = SVR(C=5.0, epsilon=0.001, gamma = 0.01, kernel = 'rbf')
clf.fit(X_train_reshaped, y_train) 
y_pred = clf.predict(X_test_reshaped)
#check performance MAE
y_pred = y_pred.reshape(-1,1)
y_test = y_test.reshape(-1,1)

#retransform data for plots and results
inv_y_pred = transformer.inverse_transform(y_pred)
inv_y_test = transformer.inverse_transform(y_test)

#MAE
error = mean_absolute_error(inv_y_test, inv_y_pred)
print('Test MAE: %.3f' % error)
with open('LNtr_RBF Kernel_Tac.txt', 'w') as f:
    print('Test MAE: %.3f' % error, file=f)

#r2_score
r2score = r2_score(inv_y_test, inv_y_pred)
print('Test r2_score: %.3f' % r2score)
with open('LNtr__RBF Kernel_Tac.txt', 'a') as f:
    print('Test r2_score: %.3f' % r2score, file=f)

#plots
plt.plot(inv_y_test, 'blue')
plt.plot(inv_y_pred, 'orange')
plt.title('Tac Index SVR prediction on test set')
#plt.savefig('Tac Index SVR prediction on test set.png')
plt.show()

#set parameters for Grid search
#set parameters for Grid search C, and e are the same for
#different types of kernel (poly, rbf, linear) in poly degrees has been choosen 
# as 2,3
#parts of the code was taken from the internet, sklearn GRID search 
#cv=10 
#for use Cauchy kernel it should be called
param_grid = ([{ 'kernel': [Cauchy()],
  'C': [1, 5, 10, 25, 50, 100, 1000], 'epsilon': 
      [0.1, 0.01, 0.001, 0.0001]
 }])
    
grid_search = GridSearchCV(SVR(), param_grid, cv = 10)
grid_search.fit(X_train_reshaped, y_train)                             

print("Grid scores on training set:")
print()
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, param_grid in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, param_grid))
print()

print('Best score: ', grid_search.best_score_)
print('Best parameter set:')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(best_parameters):
    print(param_name, best_parameters[param_name])

y_pred = grid_search.predict(X_test_reshaped)
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('R2_score: ', r2_score(y_test, y_pred))
print()


#fit SVR model, rbf kernel obtained best parameters
clf = SVR(C=100, epsilon=0.001, gamma = 0.001, kernel = 'rbf')
clf.fit(X_train_reshaped, y_train) 
y_pred = clf.predict(X_test_reshaped)
#reshape for retransform
y_pred = y_pred.reshape(-1,1)
y_test = y_test.reshape(-1,1)
#retransform
inv_y_pred = transformer.inverse_transform(y_pred)
inv_y_test =transformer.inverse_transform(y_test)
#plots on the test set
plt.plot(inv_y_test, 'blue')
plt.plot(inv_y_pred, 'orange')
plt.title('Tac Index SVR prediction on test set')
plt.legend(['Original test set', 'Predicted'], loc='upper left')
#plt.savefig('Tac Index SVR prediction on test set.png')
plt.show()

#forcasting part
#for 3 consecutive points in a gap of 10
#forecasting last point 1000
#clf = SVR(C=5.0, epsilon=0.001, kernel = 'rbf', gamma = 0.01)
clf.fit(X_train_reshaped, y_train) 
y_pred_forecasted_t1 = clf.predict(X_train_reshaped)
y_pred_forecasted_t1_last = y_pred_forecasted_t1[-1]
y_pred_forecasted_t1_last_r = y_pred_forecasted_t1_last.reshape(-1,1)
y_pred_forecasted_t2 = clf.predict(y_pred_forecasted_t1_last_r)
y_pred_forecasted_t2_r = y_pred_forecasted_t2.reshape(-1,1)
y_pred_forecasted_t3 = clf.predict(y_pred_forecasted_t2_r)
method_last1000 = np.vstack((y_pred_forecasted_t1[999], y_pred_forecasted_t2))
method_last1000 = np.vstack((method_last1000, y_pred_forecasted_t3))

method_last1000_r = method_last1000.reshape(-1,1)
y_test = y_test.reshape(-1,1)

inv_method_last1000_r = transformer.inverse_transform(method_last1000_r)
inv_y_test =transformer.inverse_transform(y_test)


#check performance MAE
error_for = mean_absolute_error(inv_y_test[:3], inv_method_last1000_r)
print('Test MAE: %.3f' % error_for)
#r2_score
r2score_for = r2_score(inv_y_test[:3], inv_method_last1000_r)
print('Test r2_score: %.3f' % r2score_for)

#forecasting last point 1010
X_r = X.reshape(-1,1)
clf.fit(X_r[:1010], y[:1010]) 
y_pred_for_1010_t1 = clf.predict(X_r[:1010])
y_pred_for_1010_t1_l = y_pred_for_1010_t1[-1]
y_pred_for_1010_t1_l_r = y_pred_for_1010_t1_l.reshape(-1,1)
y_pred_for_1010_t2 = clf.predict(y_pred_for_1010_t1_l_r)
y_pred_for_1010_t2_r = y_pred_for_1010_t2.reshape(-1,1)
y_pred_for_1010_t3 = clf.predict(y_pred_for_1010_t2_r)
method_last1010 = np.vstack((y_pred_for_1010_t1[1009], y_pred_for_1010_t2))
method_last1010 = np.vstack((method_last1010, y_pred_for_1010_t3))

method_last1010_r = method_last1010.reshape(-1,1)
y_R = y
y_R = y_R.reshape(-1,1)

inv_method_last1010_r = transformer.inverse_transform(method_last1010_r)
inv_y_R =transformer.inverse_transform(y_R)


#scores
error_for_1010 = mean_absolute_error(inv_y_R[1010:1013], inv_method_last1010_r)
print('Test MAE: %.3f' % error_for_1010)
#r2_score
r2score_for_1010 = r2_score(inv_y_R[1010:1013], inv_method_last1010_r)
print('Test r2_score: %.3f' % r2score_for_1010)

#forecasting last point 1020
#X_r = X.reshape(-1,1)
clf.fit(X_r[:1020], y[:1020]) 
y_pred_for_1020_t1 = clf.predict(X_r[:1020])
y_pred_for_1020_t1_l = y_pred_for_1020_t1[-1]
y_pred_for_1020_t1_l_r = y_pred_for_1020_t1_l.reshape(-1,1)
y_pred_for_1020_t2 = clf.predict(y_pred_for_1020_t1_l_r)
y_pred_for_1020_t2_r = y_pred_for_1020_t2.reshape(-1,1)
y_pred_for_1020_t3 = clf.predict(y_pred_for_1020_t2_r)
method_last1020 = np.vstack((y_pred_for_1020_t1[1019], y_pred_for_1020_t2))
method_last1020 = np.vstack((method_last1020, y_pred_for_1020_t3))


method_last1020_r = method_last1020.reshape(-1,1)
#y_R = y_R.reshape(-1,1)

inv_method_last1020_r = transformer.inverse_transform(method_last1020_r)
#inv_y_R =transformer.inverse_transform(y_R)


#scores
error_for_1020 = mean_absolute_error(inv_y_R[1020:1023], inv_method_last1020_r)
print('Test MAE: %.3f' % error_for_1020)
#r2_score
r2score_for_1020 = r2_score(inv_y_R[1020:1023], inv_method_last1020_r)
print('Test r2_score: %.3f' % r2score_for_1020)

#forecasting last point 1030
#X_r = X.reshape(-1,1)
clf.fit(X_r[:1030], y[:1030]) 
y_pred_for_1030_t1 = clf.predict(X_r[:1030])
y_pred_for_1030_t1_l = y_pred_for_1030_t1[-1]
y_pred_for_1030_t1_l_r = y_pred_for_1030_t1_l.reshape(-1,1)
y_pred_for_1030_t2 = clf.predict(y_pred_for_1030_t1_l_r)
y_pred_for_1030_t2_r = y_pred_for_1030_t2.reshape(-1,1)
y_pred_for_1030_t3 = clf.predict(y_pred_for_1030_t2_r)
method_last1030 = np.vstack((y_pred_for_1030_t1[1029], y_pred_for_1030_t2))
method_last1030 = np.vstack((method_last1030, y_pred_for_1030_t3))


method_last1030_r = method_last1030.reshape(-1,1)
inv_method_last1030_r = transformer.inverse_transform(method_last1030_r)

#scores
error_for_1030 = mean_absolute_error(inv_y_R[1030:1033], inv_method_last1030_r)
print('Test MAE: %.3f' % error_for_1030)
#r2_score
r2score_for_1030 = r2_score(inv_y_R[1030:1033], inv_method_last1030_r)
print('Test r2_score: %.3f' % r2score_for_1030)

#forecasting last point 1040
#X_r = X.reshape(-1,1)
clf.fit(X_r[:1040], y[:1040]) 
y_pred_for_1040_t1 = clf.predict(X_r[:1040])
y_pred_for_1040_t1_l = y_pred_for_1040_t1[-1]
y_pred_for_1040_t1_l_r = y_pred_for_1040_t1_l.reshape(-1,1)
y_pred_for_1040_t2 = clf.predict(y_pred_for_1040_t1_l_r)
y_pred_for_1040_t2_r = y_pred_for_1040_t2.reshape(-1,1)
y_pred_for_1040_t3 = clf.predict(y_pred_for_1040_t2_r)
method_last1040 = np.vstack((y_pred_for_1040_t1[1039], y_pred_for_1040_t2))
method_last1040 = np.vstack((method_last1040, y_pred_for_1040_t3))


method_last1040_r = method_last1040.reshape(-1,1)
inv_method_last1040_r = transformer.inverse_transform(method_last1040_r)

#scores
error_for_1040 = mean_absolute_error(inv_y_R[1040:1043], inv_method_last1040_r)
print('Test MAE: %.3f' % error_for_1040)
#r2_score
r2score_for_1040 = r2_score(inv_y_R[1040:1043], inv_method_last1040_r)
print('Test r2_score: %.3f' % r2score_for_1040)


#forecasting last point 1050
#X_r = X.reshape(-1,1)
clf.fit(X_r[:1050], y[:1050]) 
y_pred_for_1050_t1 = clf.predict(X_r[:1050])
y_pred_for_1050_t1_l = y_pred_for_1050_t1[-1]
y_pred_for_1050_t1_l_r = y_pred_for_1050_t1_l.reshape(-1,1)
y_pred_for_1050_t2 = clf.predict(y_pred_for_1050_t1_l_r)
y_pred_for_1050_t2_r = y_pred_for_1050_t2.reshape(-1,1)
y_pred_for_1050_t3 = clf.predict(y_pred_for_1050_t2_r)
method_last1050 = np.vstack((y_pred_for_1050_t1[1049], y_pred_for_1050_t2))
method_last1050 = np.vstack((method_last1050, y_pred_for_1050_t3))


method_last1050_r = method_last1050.reshape(-1,1)
inv_method_last1050_r = transformer.inverse_transform(method_last1050_r)


#scores
error_for_1050 = mean_absolute_error(inv_y_R[1050:1053], inv_method_last1050_r)
print('Test MAE: %.3f' % error_for_1050)
#r2_score
r2score_for_1050 = r2_score(inv_y_R[1050:1053], inv_method_last1050_r)
print('Test r2_score: %.3f' % r2score_for_1050)


# plot the forecasts in the context of the original dataset
ax = [10, 11, 12]
bx = [20, 21, 22]
cx = [30, 31, 32]
dx = [40, 41, 42]
ex = [50, 51, 52]
plt.plot(X_test, color = 'blue')
plt.plot(inv_method_last1000_r, color = 'red')
plt.plot(ax, inv_method_last1010_r, color = 'yellow')
plt.plot(bx, inv_method_last1020_r, color = 'black')
plt.plot(cx, inv_method_last1030_r, color = 'orange')
plt.plot(dx, inv_method_last1040_r, color = 'violet')
plt.plot(ex, inv_method_last1050_r, color = 'red')
plt.legend(['Original test', 'forecasted', 'forecasted', 'forecasted', 
'forecasted', 'forecasted', 'forecasted'], loc='upper left')
plt.title("Forecasting on TAC dataset using SVR and log transformation")
plt.show()

#obtained best parameters of Cauchy kernel
clf = SVR(C=1, epsilon=0.0001,  kernel = Cauchy())
clf.fit(X_train_reshaped, y_train) 
y_pred = clf.predict(X_test_reshaped)
#check performance MAE
y_pred = y_pred.reshape(-1,1)
y_test = y_test.reshape(-1,1)

inv_y_pred = transformer.inverse_transform(y_pred)
inv_y_test =transformer.inverse_transform(y_test)

error = mean_absolute_error(inv_y_test, inv_y_pred)
print('Test MAE: %.3f' % error)

#r2_score
r2score = r2_score(inv_y_test, inv_y_pred)
print('Test r2_score: %.3f' % r2score)

