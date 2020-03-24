
import numpy as np
#import panda as pd
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from statsmodels.graphics import tsaplots
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer


#load the data tac_index
tac_index=np.load('tac_index.npy')
tac_index=np.reshape(tac_index, len(tac_index))
#plot of the data
plt.plot(tac_index)
plt.show()


#just used as for comparison transformed data 
#if to use umcomment the following strings, 
#also for scores evaluation the data should be retransformed back
'''
tac_index_r = tac_index.reshape(-1,1)
transformer = FunctionTransformer(np.log1p)
transf = transformer.transform(tac_index_r)
transf = transf.reshape(-1)
tac_index = transf
'''

#split into train and test sets
split = 1000
train_tac = tac_index[0:split]
test_tac = tac_index[split:]

#plots
'''
plt.plot(train_tac, 'blue')
plt.plot(test_tac, 'green')
plt.show()
'''


#trivia model P_t = P_t-1 for training set
#trivia = train_tac[1:]
trivia = shift(train_tac, 1, cval=np.NaN)

#plot of trivia is pred

#trivia model P_t = P_t-1 in test with plots
test_tac_tr_pred = shift(test_tac, 1, cval=np.NaN)
plt.plot(test_tac, 'green')
plt.plot(test_tac_tr_pred, 'red')
plt.show()


#as soon as Random walk for the Time series can be written as: P_t = P_t-1 + w_t
#then we can find w_t = P_t - P_t-1, w_t - random shock, error term
wtN = trivia - train_tac
#for consistency remove the last value, which was added artifically
#wtN = wt[0:len(wt)-1]

#need to find mean, std dev and ratio to evaluate t-stats
mean_wtN = wtN[1:].mean()
stdDev_wtN = wtN[1:].std()
st_err = stdDev_wtN / np.sqrt(len(wtN[1:]))
#t-stat testing to know whether the model with or without drift
ratio = mean_wtN / st_err

#plot of random shock
plt.plot(wtN)
plt.show()
#PLOT STATS AUTOCORR
tsaplots.plot_acf(wtN[1:])
plt.show()
#specify seed
np.random.seed(1)
#random steps history generation
wtN_random_steps = np.random.normal(0, stdDev_wtN, len(tac_index))

#pred random_walk on training set P_t = P_t-1 + w_t single 
random_pred = []
random_pred = np.add(trivia, wtN_random_steps[:len(train_tac)])
plt.plot(random_pred, 'orange')
plt.plot(train_tac, 'blue')
plt.legend(['Modelled', 'Original train set'], loc='upper left')
plt.title('Random walk on TAC dataset')
plt.show()


trivia_all = shift(tac_index, 1, cval=np.NaN)
#pred random_walk on all set  test P_t = P_t-1 + w_t single 
random_all = []
random_all = np.add(trivia_all, wtN_random_steps[:len(tac_index)])
plt.plot(random_all, 'orange')
plt.plot(tac_index, 'blue')
plt.title('Random walk on TAC dataset')
plt.legend(["Modelled", "Original"], loc='upper left')
plt.show()
 

#test set
random_test = []
random_test = np.add(test_tac, wtN_random_steps[:len(test_tac)])

#the following is used transformed data is used, uncomment for using
'''
random_test = random_test.reshape(-1,1)
inv_y_pred = transformer.inverse_transform(random_test[1:])
test_tac = test_tac.reshape(-1,1)
inv_y_test =transformer.inverse_transform(test_tac[1:])
'''

#plots of Random walk on test set
plt.plot(test_tac, 'blue')
plt.plot(random_test, 'orange')
plt.legend(['Original test', 'Predicted'], loc='upper left')
plt.title('Random walk on transformed TAC test dataset')
plt.show()

#evaluation (if transformed data is used, 
#then use instead of test_tac -> inv_y_test, 
#instead of random_test -> inv_y_pred)
error_test = mean_absolute_error(test_tac[1:], random_test[1:])
print('Test MAE: %.3f' % error_test)
with open('Random walk TAC pred.txt', 'w') as f:
    print('Test MAE: %.3f' % error, file=f)

#r2_score
r2score_train = r2_score(test_tac[1:], random_test[1:])
print('Test r2_score: %.3f' % r2score_train)
with open('Random walk TAC pred.txt', 'a') as f:
    print('Test r2_score: %.3f' % r2score, file=f)



# forecasting part uncomment this if using transformed data
'''
test_tac = test_tac.reshape(-1)
'''
## Forecasting part
trivia2 = shift(train_tac, 2, cval=np.NaN)


# need to redo Random walk for the second future point
# the same technique
wtN2 = trivia2 - trivia

mean_wtN2 = wtN2[2:].mean()
stdDev_wtN2 = wtN2[2:].std()
st_err2 = stdDev_wtN2 / np.sqrt(len(wtN2[2:]))
ratio2 = mean_wtN2 / st_err2

#random steps history
wtN_random_steps2 = np.random.normal(mean_wtN2, stdDev_wtN2, len(tac_index))
test_tac_tr_pred2 = shift(test_tac, 2, cval=np.NaN)
pred_1001 = np.add(test_tac_tr_pred2, wtN_random_steps2[:len(test_tac)])

#3d future point
trivia3 = shift(train_tac, 3, cval=np.NaN)
wtN3 = trivia3 - trivia2
mean_wtN3 = wtN3[3:].mean()
stdDev_wtN3 = wtN3[3:].std()
st_err3 = stdDev_wtN3 / np.sqrt(len(wtN3[3:]))
ratio3 = mean_wtN3 / st_err3
#random steps history
wtN_random_steps3 = np.random.normal(mean_wtN3, stdDev_wtN3, len(tac_index))
test_tac_tr_pred3 = shift(test_tac, 3, cval=np.NaN)
pred_1002 = np.add(test_tac_tr_pred3, wtN_random_steps3[:len(test_tac)])

#uncomment this for the transformed data
'''
random_test = random_test.reshape(-1)
'''
#1000 stack values for 3 consecutive points
p1000 = np.vstack((random_test[1], pred_1001[2]))
p1000 = np.vstack((p1000, pred_1002[3]))
p1010 = np.vstack((random_test[11], pred_1001[12]))
p1010 = np.vstack((p1010, pred_1002[13]))
p1020 = np.vstack((random_test[21], pred_1001[22]))
p1020 = np.vstack((p1020, pred_1002[23]))
p1030 = np.vstack((random_test[31], pred_1001[32]))
p1030 = np.vstack((p1030, pred_1002[33]))
p1040 = np.vstack((random_test[41], pred_1001[42]))
p1040 = np.vstack((p1040, pred_1002[43]))
p1050 = np.vstack((random_test[51], pred_1001[52]))
p1050 = np.vstack((p1050, pred_1002[53]))

#needed for x values for the graph
ax = [10, 11, 12]
bx = [20, 21, 22]
cx = [30, 31, 32]
dx = [40, 41, 42]
ex = [50, 51, 52]

#uncomment this and following if transformed data is used
'''
test_tac = test_tac.reshape(-1, 1)
p1000 = p1000.reshape(-1,1)
p1010 = p1010.reshape(-1,1)
p1020 = p1020.reshape(-1,1)
p1030 = p1030.reshape(-1,1)
p1040 = p1040.reshape(-1,1)
p1050 = p1050.reshape(-1,1)
'''

'''
inv_y_pred = transformer.inverse_transform(random_test)
inv_y_test =transformer.inverse_transform(test_tac)
p1000 = transformer.inverse_transform(p1000)
p1010 = transformer.inverse_transform(p1010)
p1020 = transformer.inverse_transform(p1020)
p1030 = transformer.inverse_transform(p1030)
p1040 = transformer.inverse_transform(p1040)
p1050 = transformer.inverse_transform(p1050)
'''

#plots of forecasted values 
#if transformed data is used, then use inv_y_test instead of test_tac
plt.plot(test_tac, color = 'blue')
plt.plot(p1000, color = 'red')
plt.plot(ax, p1010, color = 'yellow')
plt.plot(bx, p1020, color = 'black')
plt.plot(cx, p1030, color = 'orange')
plt.plot(dx, p1040, color = 'violet')
plt.plot(ex, p1050, color = 'red')
plt.legend(['Original test', 'forecasted', 'forecasted', 'forecasted', 
'forecasted', 'forecasted', 'forecasted'], loc='upper left')
plt.title("Forecasting on transformed TAC dataset using Random Walk")
plt.show()

#evaluate scores for forecasted values
#if transformed data is used, then use inv_y_test instead of test_tac
error1000 = mean_absolute_error(test_tac[:3], p1000)
print('Test MAE: %.3f' % error1000)

r2score1000 = r2_score(test_tac[:3], p1000)
print('Test r2_score: %.3f' % r2score1000)

error1010 = mean_absolute_error(test_tac[10:13], p1010)
print('Test MAE: %.3f' % error1010)

r2score1010 = r2_score(test_tac[10:13], p1010)
print('Test r2_score: %.3f' % r2score1010)

error1020 = mean_absolute_error(test_tac[20:23], p1020)
print('Test MAE: %.3f' % error1020)

r2score1020 = r2_score(test_tac[20:23], p1020)
print('Test r2_score: %.3f' % r2score1020)

error1030 = mean_absolute_error(test_tac[30:33], p1030)
print('Test MAE: %.3f' % error1030)

r2score1030 = r2_score(test_tac[30:33], p1030)
print('Test r2_score: %.3f' % r2score1030)

error1040 = mean_absolute_error(test_tac[40:43], p1040)
print('Test MAE: %.3f' % error1040)

r2score1040 = r2_score(test_tac[40:43], p1040)
print('Test r2_score: %.3f' % r2score1040)

error1050 = mean_absolute_error(test_tac[50:53], p1050)
print('Test MAE: %.3f' % error1050)

r2score1050 = r2_score(test_tac[50:53], p1050)
print('Test r2_score: %.3f' % r2score1050)








# for checking the time series stationarity
from statsmodels.tsa.stattools import adfuller
adF_result = adfuller(tac_index)
print('ADF Statistic: %f' % adF_result[0])
print('p-value: %f' % adF_result[1])
print('Critical Values:')
for key, value in adF_result[4].items():
	print('\t%s: %.3f' % (key, value))









 

