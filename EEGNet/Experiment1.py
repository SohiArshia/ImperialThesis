import numpy as np 

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Flatten, DepthwiseConv2D, Dropout
from tensorflow.keras.layers import AveragePooling2D, SeparableConv2D,Activation
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers
from scipy.stats.stats import pearsonr   

import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import backend as K
import time 



# EXTRACT DATA EEG AND ENV
structEEG = {}


t = time.time()
for j in range(1,14):
    for i in range(1,16):
        adress = '/Desktop/Project/data/SubjectLeaveOneOut/Participant_%d_%d' %(j,i)
        name = '%d_%d' %(j,i)
        EEG = scipy.io.loadmat(adress)['EEG']
        structEEG[name] = EEG
        
structEnv = {}    
for i in range(1,16):
    adress = '/Desktop/Project/data/Envelope/Envelope_both_%d'  %i
    name = '%d' %i
    Env = scipy.io.loadmat(adress)['Envelope']
    structEnv[name] = Env
    
#FRAMING DATA

#creat dic to store framed data
Lall = {}
# CHOOSE WHICH SUBJECTS
for patient in range(1,16):
    #CHOOSE WHICH AUDIOBOOK TO HAVE AS TEST
    for scp in range(2,3):
        #FRAME SIZE, NUMBER OF CHANNELS AND OVERLAP(WINDOW)
        Fsize = 110
        ChanNum = 64
        Window = 0.25
        
        #HAVE A LIST FOR ALL THE TRAINING AUDIOBOOKS
        tst = scp
        trn = np.zeros(14)
        cc = 0
        
        for i in range(1,16):
            if i != tst:
                trn[cc] = i
                cc += 1
        
        #CALCULATE SIZE OF ARRAY NEEDED FOR TRAIN
        tN_Im = np.zeros(14)
        count = 0
        for ev in trn:
            name = '%d' %ev
            length = len(structEnv[name][0])
            
            tN_Im[count] = 1 + np.floor((length - Fsize)/(Fsize - round(Fsize*Window)))
            count += 1
        tot = int(sum(tN_Im))
        Env_Im = np.zeros((tot, Fsize))
        All_Im = np.zeros((tot, ChanNum, Fsize))
        #CALCULATE SIZE OF ARRAY NEEDED FOR TEST
        for ev in range(scp,scp+1):
            name = '%d' %ev  
            length  = len(structEnv[name][0])
            TestN_Im = 1 + np.floor((length - Fsize)/(Fsize - round(Fsize*Window)))
        
        Env_ImT = np.zeros((int(TestN_Im), Fsize))
        All_ImT = np.zeros((int(TestN_Im), ChanNum, Fsize))
  
        #FRAME AND CONCATENATE FOR TRAIN
        z = 0
        for pp in range(patient,patient+1):
            for j in trn:
                Envname = '%d' %j
                EEGname = '%d_%d' %(pp,j)
                Envelope = structEnv[Envname]
                EEG = structEEG[EEGname]
                N_Im = 1 + np.floor((len(Envelope[0]) - Fsize)/(Fsize - np.round(Fsize*Window)))
                     
        
                for i in range(int(N_Im)):
                    Env_Im[z,:] =  Envelope[0,i*(Fsize-round(Window*Fsize)):(i*(Fsize-round(Window*Fsize))+Fsize)]
                    All_Im[z,:,:] = EEG[:,i*(Fsize-round(Window*Fsize)):(i*(Fsize-round(Window*Fsize))+Fsize)]
                    z +=1
        #FRAME AND CONCATENATE FOR TEST
        z = 0     
        for pp in range(patient,patient+1):
            for j in range(tst,tst+1):    
                Envname = '%d' %j
                EEGname = '%d_%d' %(pp,j)
                Envelope = structEnv[Envname]
                EEG = structEEG[EEGname]
                N_Im = np.floor((len(Envelope[0]) - Fsize)/(Fsize - np.round(Fsize*Window)))
                for i in range(int(N_Im)):
                    Env_ImT[z,:] =  Envelope[0,i*(Fsize-round(Window*Fsize)):(i*(Fsize-round(Window*Fsize))+Fsize)]
                    All_ImT[z,:,:] = EEG[:,i*(Fsize-round(Window*Fsize)):(i*(Fsize-round(Window*Fsize))+Fsize)]            
                    z += 1 
        #STORE DATA IN DIC        
        Lname = '%d_%d' %(patient,scp)
        Lall[Lname] = [All_Im, Env_Im, All_ImT, Env_ImT]
                    
  
    



#DEFINE LOSS FUNCTION AND METRIC

#custom correlation loss function
def ccl(y_true, y_pred):
    
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    
    #mse = K.mean(K.square(y_pred-y_true)) 
           
    return 1-r

#correlation metric to observe
def correlation(y_true, y_pred):
    
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    
    return r



#NETWORK

def Net(rng,bs,F1,D):
    
     
    #List to store final values 
    Flist = []
    subsubL = []
    #choose subject to test
    for pat in range (10,11):
        subL = []
        #choose audiobook to test and the rest are for train
        for exe in range (2,3):
            #extract data
            scalers = {}
            nme = '%d_%d' %(pat,exe)
            x,y,xt,yt = Lall[nme]
            #normalise 
            for i in range(x.shape[1]):
                scalers[i] = StandardScaler()
                x[:, i, :] = scalers[i].fit_transform(x[:, i, :]) 
            
            
            for i in range(xt.shape[1]):
                xt[:, i, :] = scalers[i].fit_transform(xt[:, i, :])
                
                
            sc_y = StandardScaler()
            y = sc_y.fit_transform(y)        
                
            yt = sc_y.fit_transform(yt)   
                 
                
            #Add dimension 
            xtrainT = np.expand_dims(x,axis = 3)
            
            xtestT = np.expand_dims(xt, axis =3)
            
            #define input 
            inp = (xtrainT.shape[1],xtrainT.shape[2],xtrainT.shape[3])
            out = (y.shape[1])
            
            F2 = int(round(F1*D*0.5))
            model = tf.keras.Sequential()
            
            model.add(Conv2D(F1,(1,25), padding = 'same',
                               use_bias = False, input_shape=(inp)))
            model.add(BatchNormalization(axis = 1))
        
            
            model.add(DepthwiseConv2D((64,1), use_bias = False,
                                      depth_multiplier = D
                                      ,depthwise_constraint = max_norm(1.)))
            model.add(BatchNormalization(axis = 1))
            model.add(Activation('elu'))
            model.add(AveragePooling2D((1, 3)))
            model.add(Dropout(0.5))
        
            model.add(SeparableConv2D(F2 ,(1, 16) ,use_bias = False,
                                      padding = 'same'))
            model.add(BatchNormalization(axis = 1))
            model.add(Activation('elu'))
            model.add(AveragePooling2D((1,3)))
            model.add(Dropout(0.5))
            
            model.add(Flatten())
        
            model.add(Dense(out))
            model.compile(loss=ccl, optimizer="adam", metrics = [correlation])
            
            #set seed
            tf.random.set_seed(1)
            L = []
            #Rung rng number of epochs one by one and choose best validation value 
            for k in range(rng):
                model.fit( xtrainT, y, epochs=1, batch_size=bs, verbose = 0)
                ypred = model.predict(xtestT)
                nice = []
                for i in range(ypred.shape[0]):
                    nice.append(pearsonr(ypred[i], yt[i])[0])
                xx = np.mean(nice)
                # print(xx, '------', k)
                L.append(xx)
            subL.append(max(L))
        #calculate standard deviation    
        subsubL.append(np.std(subL))
        #calculate the performance for subject
        Flist.append(np.mean(subL)) 
    return [Flist, subsubL]
 
#Run
run1 = Net(20,1,6,6)
print(run1[0],run1[1])
print('PAUSE')


#calcuate time taken
dif = time.time() - t                  
print(dif)
