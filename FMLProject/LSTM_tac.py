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
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future =0, tol=0, window=3, optimize = True):
        outputs = []
        h_t = Variable(torch.zeros(1, 51).double(), requires_grad=False)#h_0 (batch, hidden_size)
        c_t = Variable(torch.zeros(1, 51).double(), requires_grad=False)#c_0 (batch, hidden_size)s
        h_t2 = Variable(torch.zeros(1, 51).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(1, 51).double(), requires_grad=False)
        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
       
        if future !=0:
            if optimize:
                out = torch.zeros(1,window).double()
                out[0]=input.data[-window:]
                output = Variable(input.data[-1].view(1,1), requires_grad=False)
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
                    loss_t =(next_out-output.data).numpy()[0][0]
                    output = output + loss_t
                
                outputs += [output]
                if optimize: out=torch.cat((out,output.data),1)
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
    
    #data =np.loadtxt('tacHKG_EUR.csv', delimiter=',', skiprows=1,usecols=2)
    data= np.load('tac_index.npy')
    data=data.reshape(data.shape[0],1)
    init_test_len=58
    trainset= data[0:-init_test_len,:]
    print('Initiating Training using',trainset.shape[0],'data points')
    #Normalizing data
#    std = np.std(trainset,axis=0)
#    mean = np.mean(trainset,axis=0)
#    trainset= (trainset -mean)/std
#    data= (data-mean)/ std
    label = data[1:-init_test_len+1,:]
    input = Variable(torch.from_numpy(trainset), requires_grad=False)
    target = Variable(torch.from_numpy(label), requires_grad=False)
    
    model = LSTM_Model()
    model.double()
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)
    model.load_state_dict(torch.load('model.t7'))
    '''
    #begin to train
    start_time = time.time()
    for i in range(5):
        print('Epoch: ', i)
        def closure():
            optimizer.zero_grad()
            out = model(input)
            loss = criterion(out, target)
            print('loss:', loss.data.numpy()[0])
            loss.backward()
            return loss
        optimizer.step(closure)
    print("Training finished in %s second(s)" % (time.time() - start_time))
    # Save the model
    #torch.save(model.state_dict(),'model.t7')
    #print('Model saved.')
    

    out = model(input)
    loss = criterion(out, target)
    print('Train loss:', loss.data.numpy()[0])
    
    criterion = nn.L1Loss()
    '''

    testset = Variable(torch.from_numpy(data[data.shape[0]-init_test_len+1:,:]), requires_grad=False)
    test_label = Variable(torch.from_numpy(data[data.shape[0]+1-init_test_len:,:]), requires_grad=False)
    out = model(testset)
    loss = criterion(out, test_label)
    print('Test loss:', loss.data.numpy()[0])
    j= out.data.numpy().T
    r2_loss = r2_score(data[data.shape[0]+1-init_test_len:,:],j)
    print('Prediction R2 loss:',r2_loss)

    mae_loss = mean_absolute_error(data[data.shape[0]+1-init_test_len:,:],j)
    print('Prediction MAE loss:', mae_loss)
    '''
    plt.figure()
    plt.title('Overall prediction over the test set')
    plt.plot(data[data.shape[0]+1-init_test_len:,:],label = 'test set')
    plt.plot(out.data.numpy().T,label='prediction')
    plt.legend(loc=2)
    plt.savefig('tac_pred.jpg',format='jpg', dpi=600)
    '''
    forecast_gap= 5
    plt.figure()
    plt.title('Forecasting trend over in a gap of '+str(forecast_gap)+' days with optimization')
    plt.plot(data[data.shape[0]+1-init_test_len:,:],label = 'ground truth')
    count=0
    criterion = nn.L1Loss()
    r2_loss=0
    mae_loss=0
    print('Initiating Forecast by Cumulatively increasing input datapoints without optimization')
    for test_len in range(init_test_len,0,-forecast_gap):
        trainset= data[0:-test_len,:]
        #Normalizing data
    #    std = np.std(trainset,axis=0)
    #    mean = np.mean(trainset,axis=0)
    #    trainset= (trainset -mean)/std
    #    data= (data-mean)/ std
        label = data[1:-test_len+1,:]
        input = Variable(torch.from_numpy(trainset), requires_grad=False)
        target = Variable(torch.from_numpy(label), requires_grad=False)
             

        future =3
        pred = model(input, future = future,tol=0,window=3,optimize = True)
        future_target = Variable(torch.from_numpy(data[input.size()[0]:input.size()[0]+future,:]), requires_grad=False)
        f_loss = criterion(pred[:, -future:],future_target)
        print('Forecast loss:', f_loss.data.numpy()[0])
        
        y = pred.data.numpy()[0]
        temp =y[ -future:]
        temp = temp.reshape(-1,1)
        r2_loss += r2_score(data[input.size()[0]:input.size()[0]+future,:],temp)
        print('Forecast R2 loss:',r2_loss)

        mae_loss += mean_absolute_error(data[input.size()[0]:input.size()[0]+future,:],temp)
        print('Forecast MAE loss:', mae_loss)
        #plotting
        plt.plot(np.arange(count*forecast_gap,count*forecast_gap+future),y[ -future:],label= 'predicted')
        count = count+1
    print('Mean R2=',r2_loss/count)
    print('Mean MAE=',mae_loss/count)
    plt.legend(loc=2)
    #plt.savefig('Tac_forecast_opt.jpg' ,format='jpg', dpi=600)
    print('File Saved')

    #plt.show()
