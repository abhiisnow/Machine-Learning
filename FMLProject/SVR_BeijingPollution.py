
import numpy as np
from pandas import read_csv
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from scipy.ndimage.interpolation import shift
from scipy.interpolate import interp1d
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

#its better to use second uncommented loading
'''
#load data
reader = csv.reader(open("data.csv"))
header, data = next(reader), list(reader)
#drop the first 24hrs with NA values
data = data[24:]
def parse(val):
    return np.datetime64(val, '%Y %m %d %H')

data = np.asarray(data)
data = data[:,1:]
date_p = parse(data[:, 1:4])
#data = np.asarray(data)
#drop No
#data = data[:,1:]
'''
#load the dataset code ref only for loading 'data' set: 
#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
data =np.loadtxt('data.csv', delimiter=',', skiprows=1,usecols=2)
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('data.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('pollution.csv')



#load pollution dataset
reader = csv.reader(open("pollution.csv"))
header, data = next(reader), list(reader)

#convert to array
data = np.asarray(data)
#without date column
data = data[:,1:]

#categorical data to encode
categ_encode = preprocessing.LabelEncoder()
data[:,4] = categ_encode.fit_transform(data[:,4])

#change to float data
data = data.astype(float)
#take only pollution column, as it needs to be interpolated
poll = data[:,0]

#change 0 values to np.nan
for i in range (0, len(poll)):
    if poll[i]==0:
        poll[i] = np.nan
#https://stackoverflow.com/questions/13166914/linear-interpolation-using-numpy-interp
not_nan = np.logical_not(np.isnan(poll))
indices = np.arange(len(poll))
interp = interp1d(indices[not_nan], poll[not_nan])
#get interpolated pollution data
poll_interpolated=interp(indices)

#concatenate interpolated pollution with previous array of features
interp_data = np.column_stack([poll_interpolated,data[:, 1:]])

#plot datas except wind_dir, as it is categorical (need to work out y axes)
fig, (ax1, ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7)
ax1.plot(interp_data[:, 0])
ax2.plot(interp_data[:, 1])
ax3.plot(interp_data[:, 2])
ax4.plot(interp_data[:, 3])
ax5.plot(interp_data[:, 4])
ax6.plot(interp_data[:, 5])
ax7.plot(interp_data[:, 6])
plt.show()

#normalise data either use this or transformation
scalar = preprocessing.MinMaxScaler()
scalar.fit(interp_data)
norm = scalar.transform(interp_data)

#transformation technique, keep for only for 1 type 
#either normalisation or transformation
from sklearn.preprocessing import FunctionTransformer
from sklearn import preprocessing
from scipy import stats

transformer = FunctionTransformer(stats.zscore)
transformer.fit(interp_data)
transf = transformer.transform(interp_data)
#transf = transf.reshape(-1)
#norm = norm.reshape(-1)

#supervised t-1 w/o unnecessary columns
#if scaled, use norm, if transformed, use transf
temp = shift(transf, 1, cval=np.NaN)
temp = temp[:,1:] 
Last_column = shift(transf[:, 6], 1, cval=np.NaN) #reform to supervised
data_t1 = np.column_stack([temp, Last_column])

#train and test data + reformulation (without NAN)
#if scaled, use norm, if transformed, use transf
data_t1 = data_t1[1:15000]
Y = transf[1:15000, 0]
split = 12000
X_train = data_t1[:split,:]
y_train = Y[:split]
X_test = data_t1[split:,:]
y_test = Y[split:]

#fit SVR
clf = SVR(C=1000, epsilon=0.01, gamma = 0.01)
clf.fit(X_train, y_train) 

#reshaped_X_test_norm = X_test_norm[:,0].reshape(-1,1)

#predict
pred_test = clf.predict(X_test)
#denormalisation predicted test y
#if scaled, use scalar, if transformed, use transformer
inv_pred_test = np.column_stack([pred_test, X_test[:, 1:]])
inv_pred_test = transformer.inverse_transform(inv_pred_test)
inv_pred_test = inv_pred_test[:,0]

#denormalisation actual test y
#if scaled, use scalar, if transformed, use transformer
inv_y_test = np.column_stack([y_test, X_test[:, 1:]])
inv_y_test = transformer.inverse_transform(inv_y_test)
inv_y_test = inv_y_test[:,0]


#scores
err = mean_absolute_error(inv_y_test, inv_pred_test)
print('Test MAE: %.3f' % err)
with open('B SVR simple zsc poly.txt', 'w') as f:
    print('Test MAE: %.3f' % error, file=f)

#r2_score
r2scor = r2_score(inv_y_test, inv_pred_test)
print('Test r2_score: %.3f' % r2scor)
with open('B SVR simple zsc poly.txt', 'a') as f:
    print('Test r2_score: %.3f' % r2score, file=f)
#plot
plt.plot(y_test, 'blue')
plt.plot(pred_test, 'orange')
plt.show

#Grid search
#set parameters for Grid search
#set parameters for Grid search C, and e are the same for
#different types of kernel (poly, rbf, linear) in poly degrees has been choosen 
# as 2,3
#parts of the code was taken from the internet, sklearn GRID search 
#cv=10 
#
param_grid = ([{ 'kernel': ['rbf'],
  'C': [1, 5, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'epsilon': 
      [0.01, 0.001, 0.0001]
 }])
    
grid_search = GridSearchCV(SVR(), param_grid, cv =3)
grid_search.fit(X_train, y_train)  

pred_test = clf.predict(X_test)
# denormalisation predicted test y   
#if scaled, use scalar, if transformed, use transformer
inv_pred_test = np.column_stack([pred_test, X_test[:, 1:]])
inv_pred_test = transformer.inverse_transform(inv_pred_test)
inv_pred_test = inv_pred_test[:,0]

#denormalisation actual test y
#if scaled, use scalar, if transformed, use transformer
inv_y_test = np.column_stack([y_test, X_test[:, 1:]])
inv_y_test = transformer.inverse_transform(inv_y_test)
inv_y_test = inv_y_test[:,0]

#scores
error = mean_absolute_error(inv_y_test, inv_pred_test)
print('Test MAE: %.3f' % error)
with open('B SVR best gr zsc rbf.txt', 'w') as f:
    print('Test MAE: %.3f' % error, file=f)

#r2_score
r2score = r2_score(inv_y_test, inv_pred_test)
print('Test r2_score: %.3f' % r2score)
with open('B SVR best gr rbf.txt', 'a') as f:
    print('Test r2_score: %.3f' % r2score, file=f)

#plots on test set
plt.plot(inv_y_test, 'blue')
plt.plot(inv_pred_test, 'orange')
plt.legend(['Original test set', 'Predicted'], loc='upper left')
plt.title('Prediction using SVR on Beijing test set')
plt.show

#plots on test set only for the first 100 points
plt.plot(inv_y_test[:100], 'blue')
plt.plot(inv_pred_test[:100], 'orange')
plt.legend(['Original test set', 'Predicted'], loc='upper left')
plt.title('Prediction using SVR on Beijing test set')
plt.show


#forecasting part
#based on the best parameters found
#forecasting last point 12000
#clf = SVR(C=5.0, epsilon=0.001, kernel = 'rbf', gamma = 0.01)
clf.fit(X_train, y_train) 
y_pred_forecasted_t1 = clf.predict(X_train)
y_pred_forecasted_t1_last = y_pred_forecasted_t1[-1]
y_pred_forecasted_t1_last_r = y_pred_forecasted_t1_last.reshape(-1,1)
a = X_test[0, 1:]
a = a.reshape(1, 7)
y_pred_forecasted_t1_last_r = np.column_stack([y_pred_forecasted_t1_last_r, a])
y_pred_forecasted_t2 = clf.predict(y_pred_forecasted_t1_last_r)
y_pred_forecasted_t2_r = y_pred_forecasted_t2.reshape(-1,1)

y_pred_forecasted_t2_r = np.column_stack([y_pred_forecasted_t2_r, a])
y_pred_forecasted_t3 = clf.predict(y_pred_forecasted_t2_r)
method_last12000 = np.vstack((y_pred_forecasted_t1[11999], y_pred_forecasted_t2))
method_last12000 = np.vstack((method_last12000, y_pred_forecasted_t3))

method_last12000_r = method_last12000.reshape(-1,1)
y_test = y_test.reshape(-1,1)
'''
b = X_test[14,1:]
b = b.reshape(1,7
method_last12000_rr = np.column_stack([method_last12000_r, '''
#denormalisation / detransformation
#if scaled, use scalar, if transformed, use transformer
inv_method_last12000_r = transformer.inverse_transform(method_last12000_r)
inv_y_test = transformer.inverse_transform(y_test)
#scores
error_for = mean_absolute_error(inv_y_test[:3], inv_method_last12000_r)
print('Test MAE: %.3f' % error_for)
#r2_score
r2score_for = r2_score(inv_y_test[:3], inv_method_last12000_r)
print('Test r2_score: %.3f' % r2score_for)

#forecasting last point 12015
#data_t11 = data_t1[]
#Y = Y
#X_r = X.reshape(-1,1)
clf.fit(data_t1[:12015], Y[:12015]) 
y_pred_for_12015_t1 = clf.predict(data_t1[:12015])
y_pred_for_12015_t1_l = y_pred_for_12015_t1[-1]
y_pred_for_12015_t1_l_r = y_pred_for_12015_t1_l.reshape(-1,1)
b = X_test[14,1:]
b = b.reshape(1,7)
y_pred_for_12015_t1_l_r = np.column_stack([y_pred_for_12015_t1_l_r, b])
y_pred_for_12015_t2 = clf.predict(y_pred_for_12015_t1_l_r)
y_pred_for_12015_t2_r = y_pred_for_12015_t2.reshape(-1,1)
y_pred_for_12015_t2_r = np.column_stack([y_pred_for_12015_t2_r, b])

y_pred_for_12015_t3 = clf.predict(y_pred_for_12015_t2_r)

method_last12015 = np.vstack((y_pred_for_12015_t1[12014], y_pred_for_12015_t2))
method_last12015 = np.vstack((method_last12015, y_pred_for_12015_t3))

method_last12015_r = method_last12015.reshape(-1,1)
y_R = Y[:12100]
y_R = y_R.reshape(-1,1)
#y_R = y_R.reshape(-1,1)

#denormalisation / detransformation
#if scaled, use scalar, if transformed, use transformer
inv_method_last12015_r = transformer.inverse_transform(method_last12015_r)
inv_y_R =transformer.inverse_transform(y_R)
#scores
error_for_12015 = mean_absolute_error(inv_y_R[12015:12018], inv_method_last12015_r)
print('Test MAE: %.3f' % error_for_12015)
#r2_score
r2score_for_12015 = r2_score(inv_y_R[12015:12018], inv_method_last12015_r)
print('Test r2_score: %.3f' % r2score_for_12015)

#forecasting 12030
clf.fit(data_t1[:12030], Y[:12030]) 
y_pred_for_12030_t1 = clf.predict(data_t1[:12030])
y_pred_for_12030_t1_l = y_pred_for_12030_t1[-1]
y_pred_for_12030_t1_l_r = y_pred_for_12030_t1_l.reshape(-1,1)
c = X_test[29,1:]
c = c.reshape(1,7)
y_pred_for_12030_t1_l_r = np.column_stack([y_pred_for_12030_t1_l_r, c])
y_pred_for_12030_t2 = clf.predict(y_pred_for_12030_t1_l_r)
y_pred_for_12030_t2_r = y_pred_for_12030_t2.reshape(-1,1)
y_pred_for_12030_t2_r = np.column_stack([y_pred_for_12030_t2_r, c])

y_pred_for_12030_t3 = clf.predict(y_pred_for_12030_t2_r)

method_last12030 = np.vstack((y_pred_for_12030_t1[12029], y_pred_for_12030_t2))
method_last12030 = np.vstack((method_last12030, y_pred_for_12030_t3))

method_last12030_r = method_last12030.reshape(-1,1)
#y_R = Y[:12100]
#y_R = y_R.reshape(-1,1)

#denormalisation / detransformation
#if scaled, use scalar, if transformed, use transformer
inv_method_last12030_r = transformer.inverse_transform(method_last12030_r)
#inv_y_R =transformer.inverse_transform(y_R)

#scores
error_for_12030 = mean_absolute_error(inv_y_R[12030:12033], inv_method_last12030_r)
print('Test MAE: %.3f' % error_for_12030)
#r2_score
r2score_for_12030 = r2_score(inv_y_R[12030:12033], inv_method_last12030_r)
print('Test r2_score: %.3f' % r2score_for_12030)

#forecasting 12045
clf.fit(data_t1[:12045], Y[:12045]) 
y_pred_for_12045_t1 = clf.predict(data_t1[:12045])
y_pred_for_12045_t1_l = y_pred_for_12045_t1[-1]
y_pred_for_12045_t1_l_r = y_pred_for_12045_t1_l.reshape(-1,1)
d = X_test[44,1:]
d = d.reshape(1,7)
y_pred_for_12045_t1_l_r = np.column_stack([y_pred_for_12045_t1_l_r, d])
y_pred_for_12045_t2 = clf.predict(y_pred_for_12045_t1_l_r)
y_pred_for_12045_t2_r = y_pred_for_12045_t2.reshape(-1,1)
y_pred_for_12045_t2_r = np.column_stack([y_pred_for_12045_t2_r, d])

y_pred_for_12045_t3 = clf.predict(y_pred_for_12045_t2_r)

method_last12045 = np.vstack((y_pred_for_12045_t1[12044], y_pred_for_12045_t2))
method_last12045 = np.vstack((method_last12045, y_pred_for_12045_t3))

method_last12045_r = method_last12045.reshape(-1,1)
#y_R = Y[:12100]
#y_R = y_R.reshape(-1,1)

#denormalisation / detransformation
#if scaled, use scalar, if transformed, use transformer
inv_method_last12045_r = transformer.inverse_transform(method_last12045_r)
#inv_y_R =transformer.inverse_transform(y_R)

#scores
error_for_12045 = mean_absolute_error(inv_y_R[12045:12048], inv_method_last12045_r)
print('Test MAE: %.3f' % error_for_12045)
#r2_score
r2score_for_12045 = r2_score(inv_y_R[12045:12048], inv_method_last12045_r)
print('Test r2_score: %.3f' % r2score_for_12045)

#forecasting 12060
clf.fit(data_t1[:12060], Y[:12060]) 
y_pred_for_12060_t1 = clf.predict(data_t1[:12060])
y_pred_for_12060_t1_l = y_pred_for_12060_t1[-1]
y_pred_for_12060_t1_l_r = y_pred_for_12060_t1_l.reshape(-1,1)
e = X_test[59,1:]
e = e.reshape(1,7)
y_pred_for_12060_t1_l_r = np.column_stack([y_pred_for_12060_t1_l_r, e])
y_pred_for_12060_t2 = clf.predict(y_pred_for_12060_t1_l_r)
y_pred_for_12060_t2_r = y_pred_for_12060_t2.reshape(-1,1)
y_pred_for_12060_t2_r = np.column_stack([y_pred_for_12060_t2_r, e])

y_pred_for_12060_t3 = clf.predict(y_pred_for_12060_t2_r)

method_last12060 = np.vstack((y_pred_for_12060_t1[12059], y_pred_for_12060_t2))
method_last12060 = np.vstack((method_last12060, y_pred_for_12060_t3))

method_last12060_r = method_last12060.reshape(-1,1)
#y_R = Y[:12100]
#y_R = y_R.reshape(-1,1)

#denormalisation / detransformation
#if scaled, use scalar, if transformed, use transformer
inv_method_last12060_r = transformer.inverse_transform(method_last12060_r)
#inv_y_R =transformer.inverse_transform(y_R)

#scores
error_for_12060 = mean_absolute_error(inv_y_R[12060:12063], inv_method_last12060_r)
print('Test MAE: %.3f' % error_for_12060)
#r2_score
r2score_for_12060 = r2_score(inv_y_R[12060:12063], inv_method_last12060_r)
print('Test r2_score: %.3f' % r2score_for_12060)



#forecasting 12075
clf.fit(data_t1[:12075], Y[:12075]) 
y_pred_for_12075_t1 = clf.predict(data_t1[:12075])
y_pred_for_12075_t1_l = y_pred_for_12075_t1[-1]
y_pred_for_12075_t1_l_r = y_pred_for_12075_t1_l.reshape(-1,1)
f = X_test[74,1:]
f = f.reshape(1,7)
y_pred_for_12075_t1_l_r = np.column_stack([y_pred_for_12075_t1_l_r, f])
y_pred_for_12075_t2 = clf.predict(y_pred_for_12075_t1_l_r)
y_pred_for_12075_t2_r = y_pred_for_12075_t2.reshape(-1,1)
y_pred_for_12075_t2_r = np.column_stack([y_pred_for_12075_t2_r, f])

y_pred_for_12075_t3 = clf.predict(y_pred_for_12075_t2_r)

method_last12075 = np.vstack((y_pred_for_12075_t1[12074], y_pred_for_12075_t2))
method_last12075 = np.vstack((method_last12075, y_pred_for_12075_t3))

method_last12075_r = method_last12075.reshape(-1,1)
#y_R = Y[:12100]
#y_R = y_R.reshape(-1,1)

#denormalisation / detransformation
#if scaled, use scalar, if transformed, use transformer
inv_method_last12075_r = transformer.inverse_transform(method_last12075_r)
#inv_y_R =transformer.inverse_transform(y_R)

#scores
error_for_12075 = mean_absolute_error(inv_y_R[12075:12078], inv_method_last12075_r)
print('Test MAE: %.3f' % error_for_12075)
#r2_score
r2score_for_12075 = r2_score(inv_y_R[12075:12078], inv_method_last12075_r)
print('Test r2_score: %.3f' % r2score_for_12075)



#forecasting 12090
clf.fit(data_t1[:12090], Y[:12090]) 
y_pred_for_12090_t1 = clf.predict(data_t1[:12090])
y_pred_for_12090_t1_l = y_pred_for_12090_t1[-1]
y_pred_for_12090_t1_l_r = y_pred_for_12090_t1_l.reshape(-1,1)
g = X_test[89,1:]
g = g.reshape(1,7)
y_pred_for_12090_t1_l_r = np.column_stack([y_pred_for_12090_t1_l_r, g])
y_pred_for_12090_t2 = clf.predict(y_pred_for_12090_t1_l_r)
y_pred_for_12090_t2_r = y_pred_for_12090_t2.reshape(-1,1)
y_pred_for_12090_t2_r = np.column_stack([y_pred_for_12090_t2_r, g])

y_pred_for_12090_t3 = clf.predict(y_pred_for_12090_t2_r)

method_last12090 = np.vstack((y_pred_for_12090_t1[12089], y_pred_for_12090_t2))
method_last12090 = np.vstack((method_last12090, y_pred_for_12090_t3))

method_last12090_r = method_last12090.reshape(-1,1)
#y_R = Y[:12100]
#y_R = y_R.reshape(-1,1)

#denormalisation / detransformation
#if scaled, use scalar, if transformed, use transformer
inv_method_last12090_r = transformer.inverse_transform(method_last12090_r)
#inv_y_R =transformer.inverse_transform(y_R)

#scores
error_for_12090 = mean_absolute_error(inv_y_R[12090:12093], inv_method_last12090_r)
print('Test MAE: %.3f' % error_for_12090)
#r2_score
r2score_for_12090 = r2_score(inv_y_R[12090:12093], inv_method_last12090_r)
print('Test r2_score: %.3f' % r2score_for_12090)



# plot the forecasts in the context of the original dataset
ax = [15, 16, 17]
bx = [30, 31, 32]
cx = [45, 46, 47]
dx = [60, 61, 62]
ex = [75, 76, 77]
fx = [90, 91, 92]

plt.plot(inv_y_test[:100], color = 'blue')
plt.plot(inv_method_last12000_r, color = 'red')
plt.plot(ax, inv_method_last12015_r, color = 'yellow')
plt.plot(bx, inv_method_last12030_r, color = 'black')
plt.plot(cx, inv_method_last12045_r, color = 'orange')
plt.plot(dx, inv_method_last12060_r, color = 'violet')
plt.plot(ex, inv_method_last12075_r, color = 'red')
plt.plot(fx, inv_method_last12090_r, color = 'black')
plt.legend(['Original test', 'forecasted', 'forecasted', 'forecasted', 
'forecasted', 'forecasted', 'forecasted', 'forecasted'], loc='upper right')
plt.title("Forecasting on Beijing dataset using SVR and zscore transformation")
plt.show()



#previous part not necessary
'''
#supervised t-1 w/o unnecessary columns
temp = shift(norm, 1, cval=np.NaN)
temp = temp[:,1:] 
Last_column = shift(norm[:, 6], 1, cval=np.NaN) #reform to supervised
data_t1 = np.column_stack([temp, Last_column])

#train and test data + reformulation (without NAN)
data_t1 = data_t1[1:]
Y = norm[1:, 0]
split = 365*1*24
X_train = data_t1[:split,:]
y_train = Y[:split]
X_test = data_t1[split:,:]
y_test = Y[split:]


#reshaped 
#X_train_norm_t1_r = np.reshape(X_train_norm_t1[:,0], -1,1)
#X_train_norm_t1_resh =X_train_norm_t1_r.reshape(-1,1)

#SVR
clf = SVR(C=1.0, epsilon=0.001)
clf.fit(X_train, y_train) 

#reshaped_X_test_norm = X_test_norm[:,0].reshape(-1,1)

#predict
pred_test = clf.predict(X_test)

#plot
plt.plot(y_test, 'blue')
plt.plot(pred_test, 'orange')
plt.show

# denormalisation predicted test y   
inv_pred_test = np.column_stack([pred_test, X_test[:, 1:]])
inv_pred_test = scalar.inverse_transform(inv_pred_test)
inv_pred_test = inv_pred_test[:,0]

#denormalisation actual test y
inv_y_test = np.column_stack([y_test, X_test[:, 1:]])
inv_y_test = scalar.inverse_transform(inv_y_test)
inv_y_test = inv_y_test[:,0]

plt.plot(inv_y_test[:1000], 'blue')
plt.plot(inv_pred_test[:1000], 'orange')
plt.show

error = mean_squared_error(inv_y_test, inv_pred_test)
print('Test MSE: %.3f' % error)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(inv_y_test,inv_pred_test)

param_grid = ([{
  'C': [1, 5, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'epsilon': 
      [0.01, 0.001, 0.0001], 'degree': 10,
 }])
    
grid_search = GridSearchCV(SVR(), param_grid)
grid_search.fit(X_train, y_train)                             

print("Best parameters:")
print()
print(grid_search.best_estimator_.get_params())
print()
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

y_pred = grid_search.predict(X_test)

# denormalisation predicted test y   
inv_pred_test = np.column_stack([y_pred, X_test[:, 1:]])
inv_pred_test = scalar.inverse_transform(inv_pred_test)
inv_pred_test = inv_pred_test[:,0]

#denormalisation actual test y
inv_y_test = np.column_stack([y_test, X_test[:, 1:]])
inv_y_test = scalar.inverse_transform(inv_y_test)
inv_y_test = inv_y_test[:,0]

print(r2_score(inv_y_test, inv_pred_test))
print(explained_variance_score(inv_y_test, inv_pred_test))
'''

#perform adfuller test to know the stationarity of the data
from statsmodels.tsa.stattools import adfuller
adF_result = adfuller(interp_data[:,0])
print('ADF Statistic: %f' % adF_result[0])
print('p-value: %f' % adF_result[1])
print('Critical Values:')
for key, value in adF_result[4].items():
	print('\t%s: %.3f' % (key, value))
    
#perform acf plot
from statsmodels.graphics import tsaplots
tsaplots.plot_acf(interp_data[:, 0])
plt.show()

# 2 method perform kpss test to know the stationarity of the data
from statsmodels.tsa.stattools import kpss
result = kpss(interp_data[:,0])
print('KPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[3].items():
	print('\t%s: %.3f' % (key, value))
    