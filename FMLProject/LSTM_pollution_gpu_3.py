import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future =0, tol=0, window=3, optimize = False):
        outputs = []
        h_t = Variable(torch.zeros(1, 51).double(), requires_grad=False).cuda()#h_0 (batch, hidden_size)
        c_t = Variable(torch.zeros(1, 51).double(), requires_grad=False).cuda()#c_0 (batch, hidden_size)s
        h_t2 = Variable(torch.zeros(1, 51).double(), requires_grad=False).cuda()
        c_t2 = Variable(torch.zeros(1, 51).double(), requires_grad=False).cuda()
        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
       
        if future !=0:
            if optimize:
                out = torch.zeros(1,window).double()
                out[0]=input.data[-window:]
                output = Variable(input.data[-1].view(1,1), requires_grad=False).cuda()
            for i in range(future):# forecasting
                if optimize:
                    buffer =  out[0][-window:]
                    #buffer=[s.numpy() for s in buffer]
                    buffer= buffer.numpy()
                    #print(buffer)
                    slope,next_out = line_fit(buffer)
                
                h_t, c_t = self.lstm1(output, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                output = self.linear(h_t2)
                
                if optimize:
                    loss_t =(next_out-output.data).cpu().numpy()[0][0]
                    output = output + loss_t+tol
                
                outputs += [output]
                if optimize: out=torch.cat((out,output.data.cpu()),1)
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

def line_fit(y):
    def f(x, A, B): 
        return A*x + B

    norm_y= np.linalg.norm(y)
    y= y/norm_y
    x = np.arange(y.shape[0])
    x= x/np.linalg.norm(x)
    slope,intercept = curve_fit(f, x, y)[0]
    next_y = slope*(2*x[-1]-x[1])+intercept
    return slope,next_y*norm_y

if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(200)
    torch.manual_seed(200)
    def nan_helper(y):# from stack overflow
        return np.isnan(y), lambda z: z.nonzero()[0]
    data_ =np.genfromtxt('pollution.csv', delimiter=',', skip_header=25,usecols=5)
    data_=data_.reshape(data_.shape[0],1)
    data_ = data_[:15000,:]
    nans, x= nan_helper(data_)
    data_[nans]= np.interp(x(nans), x(~nans), data_[~nans])
    

    scaler = MinMaxScaler()
    scaler.fit(data_)
    data=scaler.transform(data_)

    init_test_len=3000
    end_point =2900
    trainset= data[0:-init_test_len,:]
    print('Initiating Training using',trainset.shape[0],'data points')
    #Normalizing data
    #std = np.std(trainset,axis=0)
    #mean = np.mean(trainset,axis=0)
    #trainset= (trainset -mean)/std
    #data= (data_-mean)/ std
    
    label = data[1:-init_test_len+1,:]
    input = Variable(torch.from_numpy(trainset), requires_grad=False).cuda()
    target = Variable(torch.from_numpy(label), requires_grad=False).cuda()
    
    model = LSTM_Model()
    model.double()
    model.cuda()
    criterion = nn.L1Loss()
    # use LBFGS as optimizer since we can load the whole data to train
    #optimizer = optim.LBFGS(model.parameters(), lr=0.8, history_size=1000)
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(torch.load('model_pollution_adam_4.t7'))
    #begin to train
    '''
    start_time = time.time()
    for i in range(40):
        print('Epoch: ', i)
        def closure():
            optimizer.zero_grad()
            out = model(input)
            loss = criterion(out, target)
            print('loss:', loss.data.cpu().numpy()[0])
            loss.backward()
            return loss
        optimizer.step(closure)
    print("Training finished in %s second(s)" % (time.time() - start_time))
    # Save the model
    torch.save(model.state_dict(),'model_pollution_adam_4.t7')
    print('Model saved.')
    '''
    #model.load_state_dict(torch.load('model_pollution_adam_no_normal.t7'))
    testset=  Variable(torch.from_numpy(data[-init_test_len:,0]), requires_grad=False).cuda()
    out = model(testset)
    loss = criterion(out[0,0:-1], testset[1:])
    print('Test loss Normalized:', loss.data.cpu().numpy()[0])
    #inv_data = scaler.inverse_transform(data)
    inv_out =  scaler.inverse_transform(out.data.cpu().numpy().T)
    print('Inv out shape',inv_out.shape)
    inv_out=Variable(torch.from_numpy(inv_out), requires_grad=False).cuda()
    print('Inv out torch size', inv_out.size())
    test_ = Variable(torch.from_numpy(data_[-init_test_len:,0]), requires_grad=False).cuda()
    loss = criterion(inv_out[0:-1,:], test_[1:])
    print('Test loss :', loss.data.cpu().numpy()[0])
    plt.figure()
    plt.title('Overall prediction over the test set')
    plt.plot(data_[-init_test_len:,0],label='actual')
    plt.plot(inv_out.data.cpu().numpy(),label='predicted')
    plt.legend(loc=1)
    plt.savefig('stack_prediction_test.jpg',format='jpg', dpi=600)
    print(inv_out.data.cpu().numpy().shape)
    k=data_[-init_test_len:,0]
    k=k.reshape(-1,1)
    print(k.shape)
    j=inv_out.data.cpu().numpy()
    r2_loss = r2_score(k,j)
    print('Prediction R2 loss:',r2_loss)

    mae_loss = mean_absolute_error(k.T,j.T)
    print('Prediction MAE loss:', mae_loss)

    plt.figure()
    forecast_gap= 15
    plt.figure()
    plt.title('Forecasting trend over in a gap of '+str(forecast_gap)+' days without optimization')
    temp=data_[data_.shape[0]-init_test_len:data_.shape[0]-end_point:,:]
    plt.plot(temp,label='actual')
    count=0
    print('Initiating Forecast by Cumulatively increasing input datapoints')

    for test_len in range(init_test_len,end_point,-forecast_gap):
        trainset= data[0:-test_len+1,:]
        label = trainset[1:,:]
        trainset=trainset[:-1,:]
        input = Variable(torch.from_numpy(trainset), requires_grad=False).cuda()
        target = Variable(torch.from_numpy(label), requires_grad=False).cuda()

        future =3
        pred = model(input, future = future,tol=0,window=3,optimize = False)
        future_target = Variable(torch.from_numpy(data[input.size()[0]:input.size()[0]+future,:]), requires_grad=False).cuda()
        f_loss = criterion(pred[:, -future:],future_target)
        print('Normalized Forecast loss:', f_loss.data.cpu().numpy()[0])
        
        y = pred.data.cpu().numpy()[0]
        y=y.reshape(-1,1)
        y= scaler.inverse_transform(y)
        r2_loss = r2_score(data_[input.size()[0]:input.size()[0]+future,:],y[ -future:,:])
        print('Forecast R2 loss:',r2_loss)

        mae_loss = mean_absolute_error(data_[input.size()[0]:input.size()[0]+future,:],y[ -future:,:])
        print('Forecast MAE loss:', mae_loss)

        #plotting
        plt.plot(np.arange(count*forecast_gap,count*forecast_gap+future),y[ -future:],label='forecast')
        count = count+1
    plt.legend(loc=1,prop={'size': 6})
    plt.savefig('stack_forecast_fut3.jpg',format='jpg', dpi=600)
    print('File Saved')
    #plt.show()
