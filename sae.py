#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:26:19 2019

@author: Nan Ji
"""

import numpy as np
np.random.seed(1337)
import pandas as pd
import math
from preprocess import DataPreprocess as dp
import keras
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class SAE:
    def __init__(self):
        super(SAE, self).__init__()
        self.link = 257
        
        self.dp = dp()
        # self.trainX, self.trainY, self.trainY_nofilt = self.dp.get_data(data_type='train')
        # self.model = self.stacking_autoencoder(self.dp.inputstep* self.link, 400, 400, 400, self.link)[-1]
        # model.load_weights('path')
        self.finnal_sae = keras.models.load_model('/home/jinan/AutoKNN/SAE-model/SAE%dp%d.h5'%(self.dp.inputstep, self.dp.predstep))
        # self.finnal_sae.load_weights('/home/jinan/AutoKNN/SAE-model/SAE%dp%d-weights.h5'%(self.dp.inputstep, self.dp.predstep), by_name=True)
        
    def autoencoder(self, n_input, n_hidden, n_output):
        model = Sequential()
        model.add(Dense(n_hidden,input_dim=n_input,name='hidden'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(n_output))
        return model
    
    def stacking_autoencoder(self, n_input, n_h1, n_h2, n_h3, n_output):
        ae1 = self.autoencoder(n_input,n_h1,n_output)
        ae2 = self.autoencoder(n_h1,n_h2,n_output)
        ae3 = self.autoencoder(n_h2,n_h3,n_output)
        
        sae = Sequential()
        sae.add(Dense(n_h1,input_dim=n_input,name='hidden_1'))
        sae.add(Activation('relu'))
        sae.add(Dense(n_h2,name='hidden_2'))
        sae.add(Activation('relu'))
        sae.add(Dense(n_h3,name='hidden_3'))
        sae.add(Activation('relu'))
        sae.add(Dropout(0.2))
        sae.add(Dense(n_output))
        
        models = [ae1,ae2,ae3,sae]
        return models
    
    def predict_traffic(self, predX):
        predX = np.asarray(predX).reshape(-1, self.dp.inputstep * self.link,)
        return np.asarray(self.finnal_sae.predict(predX)).reshape(self.link,)

    def evaluate(self, y, y_):
        rmse = np.sqrt(np.mean(np.square(y-y_)))
        mae = np.mean(np.abs(y-y_))
        mape = np.mean(np.abs(y-y_)/y)*100
        return rmse, mae, mape
    
if __name__ == '__main__':
    sae = SAE()
    testX, testY, testY_nofilt = sae.dp.get_data(data_type='test')

    y = np.array([i * sae.dp.maxv for i in testY_nofilt]).reshape(-1, 257)
    df = pd.DataFrame(y)
    df.to_csv('model/saey-only-predY6p%d.csv'%(sae.dp.predstep), header=False, index=False)

    set_predY = []
    for x in testX:
        set_predY.append(sae.predict(x) * sae.dp.maxv)

    y_ = np.array(set_predY).reshape(-1, 257)
    df = pd.DataFrame(y_)
    df.to_csv('model/sae-only-predY6p%d.csv'%(sae.dp.predstep), header=False, index=False)

    rmse, mae, mape = sae.evaluate(y, y_)
    print('predstep={},RMSE = {:.4}, MAE = {:.4}, MAPE = {:.4}'.format(sae.dp.predstep, rmse, mae, mape))    