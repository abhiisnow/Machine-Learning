
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from scipy.ndimage.interpolation import shift
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

#from sklearn.model_selection import cross_val_score

#load the data
tac_index=np.load('tac_index.npy')
tac_index=np.reshape(tac_index, len(tac_index))

#plots
plt.plot(tac_index)
plt.title('Tac Index')
plt.show()
#plt.savefig('Tac_Index.png')

#make the time series data to supervised learning
t_1 = shift(tac_index, 1, cval=np.NaN)
y = tac_index[1:]
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


#fit SVR model, rbf kernel (here kernel can be chosed as different types)
#also model parameters can be different C, e, d (for polynomial), gamma for rbf
clf = SVR(C=1.0, epsilon=0.001, kernel = 'rbf')
clf.fit(X_train_reshaped, y_train) 
y_pred = clf.predict(X_test_reshaped)

#check performance MAE
error = mean_absolute_error(y_test, y_pred)
print('Test MAE: %.3f' % error)
with open('RBF Kernel_Tac.txt', 'w') as f:
    print('Test MAE: %.3f' % error, file=f)

#r2_score
r2score = r2_score(y_test, y_pred)
print('Test r2_score: %.3f' % r2score)
with open('RBF Kernel_Tac.txt', 'a') as f:
    print('Test r2_score: %.3f' % r2score, file=f)

#make plot on the test set
plt.plot(y_test, 'blue')
plt.plot(y_pred, 'orange')
plt.title('Tac Index SVR prediction on test set')
#plt.savefig('Tac Index SVR prediction on test set.png')
plt.show()


#set parameters for Grid search C, and e are the same for
#different types of kernel (poly, rbf, linear) in poly degrees has been choosen 
# as 2,3
#parts of the code was taken from the internet, sklearn GRID search 
#cv=10
param_grid = ([{ 'kernel': ['rbf'],
  'C': [1, 5, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'epsilon': 
      [0.01, 0.001, 0.0001]
 }])
    
grid_search = GridSearchCV(SVR(), param_grid, cv =10)
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


#just as for comparison
#perform linear SVR
clf = SVR(C=1.0, kernel = 'linear')
clf.fit(X_train_reshaped, y_train) 
y_pred = clf.predict(X_test_reshaped)
with open('linear_Kernel_Tac.txt', 'w') as f:
    print(mean_absolute_error(y_test, y_pred), file=f)

#perform polynomial degree 2 and 3, SVR
clf = SVR(C=1.0, degree = 3, kernel = 'poly')
clf.fit(X_train_reshaped, y_train) 
y_pred = clf.predict(X_test_reshaped)
with open('polynomial_Kernel_Tac.txt', 'a') as f:
    print(mean_absolute_error(y_test, y_pred), file=f)

#parameters for linear kernel and grid search on linear kernel cv=10
param_grid = ([{ 'kernel': ['linear'],
  'C': [1, 5, 10, 100, 1000]
 }])
    
grid_search = GridSearchCV(SVR(), param_grid, cv =10)
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
print(mean_absolute_error(y_test, y_pred))
print()


#fit SVR best parameters model, rbf kernel
clf = SVR(C=5.0, epsilon=0.001, gamma = 0.01, kernel = 'rbf')
clf.fit(X_train_reshaped, y_train) 
y_pred = clf.predict(X_test_reshaped)

#check performance MAE
error = mean_absolute_error(y_test, y_pred)
print('Test MAE: %.3f' % error)
with open('BEST RBF Kernel_Tac.txt', 'w') as f:
    print('Test MAE: %.3f' % error, file=f)

#r2_score
r2score = r2_score(y_test, y_pred)
print('Test r2_score: %.3f' % r2score)
with open('BEST RBF Kernel_Tac.txt', 'a') as f:
    print('Test r2_score: %.3f' % r2score, file=f)

#make plot on the test set
plt.plot(y_test, 'blue')
plt.plot(y_pred, 'orange')
plt.title('Tac Index SVR prediction with best parameters on test set')
#plt.savefig('Tac Index SVR prediction on test set.png')
plt.show()



#forecasting last point 1000 based on the best parameters
clf = SVR(C=5.0, epsilon=0.001, kernel = 'rbf', gamma = 0.01)
clf.fit(X_train_reshaped, y_train) 
y_pred_forecasted_t1 = clf.predict(X_train_reshaped)
y_pred_forecasted_t1_last = y_pred_forecasted_t1[-1]
y_pred_forecasted_t1_last_r = y_pred_forecasted_t1_last.reshape(-1,1)
y_pred_forecasted_t2 = clf.predict(y_pred_forecasted_t1_last_r)
y_pred_forecasted_t2_r = y_pred_forecasted_t2.reshape(-1,1)
y_pred_forecasted_t3 = clf.predict(y_pred_forecasted_t2_r)
method_last1000 = np.vstack((y_pred_forecasted_t1[999], y_pred_forecasted_t2))
method_last1000 = np.vstack((method_last1000, y_pred_forecasted_t3))
error_for = mean_absolute_error(y_test[:3], method_last1000)
print('Test MAE: %.3f' % error_for)
#r2_score
r2score_for = r2_score(y_test[:3], method_last1000)
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
error_for_1010 = mean_absolute_error(y[1010:1013], method_last1010)
print('Test MAE: %.3f' % error_for_1010)
#r2_score
r2score_for_1010 = r2_score(y[1010:1013], method_last1010)
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
error_for_1020 = mean_absolute_error(y[1020:1023], method_last1020)
print('Test MAE: %.3f' % error_for_1020)
#r2_score
r2score_for_1020 = r2_score(y[1020:1023], method_last1020)
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
error_for_1030 = mean_absolute_error(y[1030:1033], method_last1030)
print('Test MAE: %.3f' % error_for_1030)
#r2_score
r2score_for_1030 = r2_score(y[1030:1033], method_last1030)
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
error_for_1040 = mean_absolute_error(y[1040:1043], method_last1040)
print('Test MAE: %.3f' % error_for_1040)
#r2_score
r2score_for_1040 = r2_score(y[1040:1043], method_last1040)
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
error_for_1050 = mean_absolute_error(y[1050:1053], method_last1050)
print('Test MAE: %.3f' % error_for_1050)
#r2_score
r2score_for_1050 = r2_score(y[1050:1053], method_last1050)
print('Test r2_score: %.3f' % r2score_for_1050)


# plot the forecasts in the context of the original dataset
ax = [10, 11, 12]
bx = [20, 21, 22]
cx = [30, 31, 32]
dx = [40, 41, 42]
ex = [50, 51, 52]
plt.plot(X_test, color = 'blue')
plt.plot(method_last1000, color = 'red')
plt.plot(ax, method_last1010, color = 'yellow')
plt.plot(bx, method_last1020, color = 'black')
plt.plot(cx, method_last1030, color = 'orange')
plt.plot(dx, method_last1040, color = 'violet')
plt.plot(ex, method_last1050, color = 'red')
plt.legend(['Original test', 'forecasted', 'forecasted', 'forecasted', 
'forecasted', 'forecasted', 'forecasted'], loc='upper left')
plt.title("Forecasting on TAC dataset using SVR")
plt.show()




'''
for i in range(len(method_last1000)):
	off_s = len(series) - n_test + i - 1
	off_e = off_s + len(forecasts[i]) + 1
	xaxis = [x for x in range(off_s, off_e)]
	yaxis = [series.values[off_s]] + forecasts[i]
	plt.plot(xaxis, yaxis, color='red')
# show the plot
plt.show()
'''

'''
#forecasting 1000+1 last point
clf = SVR(C=5.0, epsilon=0.001, kernel = 'rbf', gamma = 0.01)
clf.fit(X_train_reshaped, y_train) 
y_pred_forecm2_t1 = clf.predict(X_train_reshaped)
temp = np.append(y_train, y_pred_forecm2_t1[999])
temp_r = temp.reshape(-1,1)
y_pred_forecm2_t2 = clf.predict(temp_r)
temp = np.append(temp, y_pred_forecm2_t2[1000])
temp2_r = temp.reshape(-1,1)
y_pred_forecm2_t3 = clf.predict(temp2_r)
temp_3 = np.append(temp, y_pred_forecm2_t3[1001])
error_for_m2 = mean_absolute_error(y_test[:3], temp_3[1000:])
print('Test MAE: %.3f' % error_for_m2)
r2score_for_m2 = r2_score(y_test[:3], temp_3[1000:])
print('Test r2_score: %.3f' % r2score_for_m2)

#forecasting 1000+1 last point
clf = SVR(C=5.0, epsilon=0.001, kernel = 'rbf', gamma = 0.01)
clf.fit(X_train_reshaped, y_train) 
y_pred_forecm3_t1 = clf.predict(X_train_reshaped)
temp3 = np.append(X_train, y_pred_forecm3_t1[999])
temp3_r = temp3.reshape(-1,1)
y_pred_forecm3_t2 = clf.predict(temp3_r)
temp3 = np.append(temp3, y_pred_forecm3_t2[1000])
temp33_r = temp3.reshape(-1,1)
y_pred_forecm3_t3 = clf.predict(temp33_r)
temp33_3 = np.append(temp3, y_pred_forecm3_t3[1001])
error_for_m3 = mean_absolute_error(y_test[:3], temp33_3[1000:])
print('Test MAE: %.3f' % error_for_m3)
r2score_for_m3 = r2_score(y_test[:3], temp33_3[1000:])
print('Test r2_score: %.3f' % r2score_for_m3)
'''

'''
start = 1000
stop = len(tac_index)
interval = 10
number_forecasts = 3
clf = SVR(C=5.0, epsilon=0.001, kernel = 'rbf', gamma = 0.01)
tac_index_r = tac_index.reshape(-1,1)
for i in range (start, stop, interval):
    clf.fit(tac_index_r[i], y[i]) 
    y_t1 = clf.predict(tac_index_r[i])
    y_t1_last = 





y_pred_forecasted_t1_r = y_pred_forecasted_t1.reshape(-1,1)

y_pred_forecasted_t2 = clf.predict(y_pred_forecasted_t1_r)

y_pred_forecasted_t2_r = y_pred_forecasted_t2.reshape(-1,1)

y_pred_forecasted_t3 = clf.predict(y_pred_forecasted_t2_r)
'''




