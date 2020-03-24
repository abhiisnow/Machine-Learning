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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
#THIS IS LATEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
use_gpu = torch.cuda.is_available()
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTMCell(8, 51)
        self.lstm2 = nn.LSTMCell(51, 100)
        self.lstm3 = nn.LSTMCell(100,100)
        self.linear1 = nn.Linear(100, 1)

    def forward(self, input, future =0, tol=0, window=3, optimize = True):
        outputs = []
        if use_gpu:
            h_t = Variable(torch.zeros(1, 51).double(), requires_grad=False).cuda()#h_0 (batch, hidden_size)
            c_t = Variable(torch.zeros(1, 51).double(), requires_grad=False).cuda()#c_0 (batch, hidden_size)s
            h_t2 = Variable(torch.zeros(1, 100).double(), requires_grad=False).cuda()
            c_t2 = Variable(torch.zeros(1, 100).double(), requires_grad=False).cuda()
            h_t3 = Variable(torch.zeros(1, 100).double(), requires_grad=False).cuda()
            c_t3 = Variable(torch.zeros(1, 100).double(), requires_grad=False).cuda()
        else:
            h_t = Variable(torch.zeros(1, 51).double(), requires_grad=False)#h_0 (batch, hidden_size)
            c_t = Variable(torch.zeros(1, 51).double(), requires_grad=False)#c_0 (batch, hidden_size)s
            h_t2 = Variable(torch.zeros(1, 100).double(), requires_grad=False)
            c_t2 = Variable(torch.zeros(1, 100).double(), requires_grad=False)
            h_t3 = Variable(torch.zeros(1, 100).double(), requires_grad=False)
            c_t3 = Variable(torch.zeros(1, 100).double(), requires_grad=False)

        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear1(h_t3)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs, h_t3,c_t3

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTMCell(1,100)
        self.lstm2 = nn.LSTMCell(100, 100)
        self.linear1 = nn.Linear(100, 1)

    def forward(self, input,encoder_state_h,encoder_state_c, future =0, tol=0, window=3, optimize = False):
        outputs = []
        if use_gpu:
            #h_t = Variable(torch.zeros(1, 51).double(), requires_grad=False).cuda()#h_0 (batch, hidden_size)
            h_t= encoder_state_h.cuda()
            #c_t = Variable(torch.zeros(1, 100).double(), requires_grad=False).cuda()#c_0 (batch, hidden_size)
            c_t= encoder_state_c.cuda()
            h_t2 = Variable(torch.zeros(1, 100).double(), requires_grad=False).cuda()
            c_t2 = Variable(torch.zeros(1, 100).double(), requires_grad=False).cuda()
        else:
            #h_t = Variable(torch.zeros(1, 51).double(), requires_grad=False)#h_0 (batch, hidden_size)
            h_t= encoder_state_h
            #c_t = Variable(torch.zeros(1, 100).double(), requires_grad=False)#c_0 (batch, hidden_size)
            c_t= encoder_state_c
            h_t2 = Variable(torch.zeros(1, 100).double(), requires_grad=False)
            c_t2 = Variable(torch.zeros(1, 100).double(), requires_grad=False)

        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear1(h_t2)
            outputs += [output]


        if future !=0:
            if optimize:
                #out = torch.zeros(1,window).double()
                #out[0]=input.data[-window:]
                out= outputs[-window:]
                if use_gpu: output = Variable(input.data[-1].view(1,1), requires_grad=False).cuda()
                else: output = Variable(input.data[-1].view(1,1), requires_grad=False)
            for i in range(future):# forecasting
                if optimize:
                    #buffer =  out[0][-window:]
                    buffer=[s.data.cpu().numpy() for s in out]
                    #buffer= buffer.numpy()
                    buffer = np.stack(buffer,axis=1).tolist()
                    buffer=[s[0] for s in buffer[0]]
                    print(buffer)
                    slope,next_out = line_fit(buffer)

                h_t, c_t = self.lstm1(output, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                output = self.linear1(h_t2)

                if optimize:
                    loss_t =(next_out-output.data).cpu().numpy()[0][0]
                    output = output + loss_t+tol

                outputs += [output]
                #if optimize: out=torch.cat((out,output.data.cpu()),1)
                if optimize: out=out.append((out,output.data.cpu()))
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
    np.random.seed(1)
    torch.manual_seed(1)
    def nan_helper(y):# from stack overflow
        return np.isnan(y), lambda z: z.nonzero()[0]
    data =np.genfromtxt('pollution.csv', delimiter=',', skip_header=25,usecols=5)
    data=data.reshape(data.shape[0],1)
    data = data[:15000,:]
    nans, x= nan_helper(data)
    data[nans]= np.interp(x(nans), x(~nans), data[~nans])
    for col in range(6,13):
        if col !=  9:
            data_temp =np.genfromtxt('pollution.csv', delimiter=',', skip_header=25,usecols=col)
            data_temp=data_temp.reshape(data_temp.shape[0],1)
            data_temp = data_temp[:15000,:]
            nans, x= nan_helper(data_temp)
            data_temp[nans]= np.interp(x(nans), x(~nans), data_temp[~nans])
            data = np.hstack((data,data_temp))

    # integer encode direction
    encoder = LabelEncoder()
    data_temp =np.genfromtxt('pollution.csv', delimiter=',', skip_header=25,usecols=9,dtype="|U5")
    data_temp = data_temp[:15000]
    encoder.fit(data_temp)
    data_temp = encoder.transform(data_temp)
    data_temp=data_temp.reshape(data_temp.shape[0],1)
    data = np.hstack((data,data_temp))
    #THIS IS LATEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    d= data
    init_test_len=3000
    end_point =2900
    trainset_en= data[0:-init_test_len,:]
    label_en = data[1:-init_test_len+1,:]
    label_de = data[2:-init_test_len+2,:]

    scaler = MinMaxScaler()
    scaler.fit(trainset_en)
    trainset_en=scaler.transform(trainset_en)
    label_en = scaler.transform(label_en)[:,0]
    label_de = scaler.transform(label_de)[:,0]
    data = scaler.transform(data)
    print('Initiating Training using',trainset_en.shape[0],'data points')
    #Normalizing data
    '''
    std = np.std(trainset,axis=0)
    mean = np.mean(trainset,axis=0)
    trainset= (trainset -mean)/std
    data= (data_-mean)/ std
    '''


    if use_gpu:
        input_en = Variable(torch.from_numpy(trainset_en), requires_grad=False).cuda()
        target_en = Variable(torch.from_numpy(label_en.reshape(-1,1)), requires_grad=False).cuda()
        target_de = Variable(torch.from_numpy(label_de.reshape(-1,1)), requires_grad=False).cuda()
    else:
        input_en = Variable(torch.from_numpy(trainset_en), requires_grad=False)
        target_en = Variable(torch.from_numpy(label_en.reshape(-1,1)), requires_grad=False)
        target_de = Variable(torch.from_numpy(label_de.reshape(-1,1)), requires_grad=False)

    enco = Encoder()
    deco = Decoder()
    enco.double()
    deco.double()
    if use_gpu:
        enco.cuda()
        deco.cuda()
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    #optimizer = optim.LBFGS(model.parameters(), lr=0.8, history_size=1000)
    optimizer_en=torch.optim.Adam(enco.parameters(), lr=0.000001)
    optimizer_de=torch.optim.Adam(deco.parameters(), lr=0.000001)
    enco.load_state_dict(torch.load('model_pollution_en_oneLoss.t7'))
    deco.load_state_dict(torch.load('model_pollution_de_oneLoss.t7'))

    #begin to train
    '''
    start_time = time.time()
    for i in range(20):
        print('Epoch: ', i)
        #def closure():
        loss=0
        optimizer_en.zero_grad()
        optimizer_de.zero_grad()
        out_en,h,c = enco(input_en)
        out_en=out_en.transpose(0,1)
        #loss += criterion(out_en, target_en)
        #print('Encoder loss:', loss.data.cpu().numpy()[0])
        out_de = deco(out_en,h,c)
        loss = criterion(out_de.transpose(0,1), target_de)
        print('Total loss:', loss.data.cpu().numpy()[0])
        loss.backward()
        #return loss
        optimizer_en.step()
        optimizer_de.step()
    print("Training finished in %s second(s)" % (time.time() - start_time))
    # Save the model
    torch.save(enco.state_dict(),'model_pollution_en_oneLoss.t7')
    torch.save(deco.state_dict(),'model_pollution_de_oneLoss.t7')
    print('Model saved.')
    '''
    #THIS IS LATEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if use_gpu:
        testset=  Variable(torch.from_numpy(data[-init_test_len:,:]), requires_grad=False).cuda()
    else:
        testset=  Variable(torch.from_numpy(data[-init_test_len:,:]), requires_grad=False)
    out_en,h,c = enco(testset)
    loss = criterion(out_en[:,0:-1], testset[1:,0])
    print('Encoder loss:', loss.data.cpu().numpy()[0])
    out = deco(out_en.transpose(0,1),h,c)
    loss = criterion(out[:,0:-2], testset[2:,0])
    print('Decoder loss:', loss.data.cpu().numpy()[0])

    #inv_train = scaler.inverse_transform(d[input_en.size()[0]-1:d[input.size()[0]+2])
    scaler_out=MinMaxScaler()
    scaler_out.fit(d[2:-init_test_len+2,0].reshape(-1,1))
    inv_out =  scaler_out.inverse_transform(out.data.cpu().numpy().T)
    print(inv_out.shape)
    #print(type(inv_out))
    #test_ = Variable(torch.from_numpy(d[-init_test_len:,:]), requires_grad=False).cuda()
    #inv_out = Variable(torch.from_numpy(inv_out), requires_grad=False).cuda()
    #loss = criterion(inv_out[0:-2,:], test_[2:,0])
    #print('Decoder loss:', loss.data.cpu().numpy()[0])

    k=d[-init_test_len:,0]
    k=k.reshape(-1,1)
    r2_loss = r2_score(k,inv_out)
    print('Prediction R2 loss:',r2_loss)

    mae_loss = mean_absolute_error(k,inv_out)
    print('Prediction MAE loss:', mae_loss)

    plt.figure()
    plt.title('Overall prediction over the test set')
    plt.plot(data[-init_test_len+2:,0])
    plt.plot(out.data.cpu().numpy()[0],label='predicted')
    plt.legend(loc=1)
    #plt.plot(out[-3:,:])
    plt.savefig('prediction_sample.jpg',format='jpg', dpi=500)
    print('Prediction Saved')

    plt.figure()
    forecast_gap= 15
    future =3
    plt.figure()
    plt.title('Forecasting trend over '+str(future)+' days in a gap of '+str(forecast_gap)+' days')
    temp=d[data.shape[0]-init_test_len:data.shape[0]-end_point:,0]
    plt.plot(temp)
    count=0
    print('Initiating Forecast by Cumulatively increasing input datapoints')

    for test_len in range(init_test_len,end_point,-forecast_gap):
        trainset= data[0:-test_len+2,:]
       
        #label = trainset[2:,0]
        trainset=trainset[:-2,:]

        if use_gpu:
            input = Variable(torch.from_numpy(trainset), requires_grad=False).cuda()
            #target = Variable(torch.from_numpy(label), requires_grad=False).cuda()
            future_target = Variable(torch.from_numpy(data[input.size()[0]:input.size()[0]+future,0]), requires_grad=False).cuda()

        else:
            input = Variable(torch.from_numpy(trainset), requires_grad=False)
            #target = Variable(torch.from_numpy(label), requires_grad=False)
            future_target = Variable(torch.from_numpy(data[input.size()[0]:input.size()[0]+future,0]), requires_grad=False)


        out_en,h,c = enco(input)
        pred = deco(out_en.transpose(0,1),h,c,future =1,tol=0,window=3,optimize = False)
        y = pred.data.cpu().numpy()[0].T
        inv_y = np.concatenate((y.reshape(-1,1), data[2:input.size()[0]+future, 1:]), axis=1)
        y= scaler.inverse_transform(inv_y)[:,0]
        y=y.reshape(-1,1)
        '''
        target_ = Variable(torch.from_numpy(d[input.size()[0]:input.size()[0]+future,0]), requires_grad=False).cuda()
        y=  Variable(torch.from_numpy(y), requires_grad=False).cuda()
        print(y.size())
        f_loss = criterion(y[ -future:,:],target_)
        print('Forecast loss:', f_loss.data.cpu().numpy()[0])
        y = y.data.cpu().numpy()
        print(y.shape)
        '''
        r2_loss = r2_score(d[input.size()[0]:input.size()[0]+future,0],y[ -future:,:])
        print('Forecast R2 loss:',r2_loss)

        mae_loss = mean_absolute_error(d[input.size()[0]:input.size()[0]+future,0],y[ -future:,:])
        print('Forecast MAE loss:', mae_loss)
        #y= scaler.inverse_transform(y.reshape(-1,1))
        #plotting
        plt.plot(np.arange(count*forecast_gap,count*forecast_gap+future),y[ -future:],label='forecast')
        count = count+1
    plt.legend(loc=1)
    plt.savefig('forecast_sample_fut1.jpg',format='jpg', dpi=500)
    print('File Saved')

    #THIS IS LATEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
