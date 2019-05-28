# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:33:38 2019

@author: Nan Ji
"""

import numpy as np
import random
import gym
from gym import spaces
from collections import deque
from preprocess import DataPreprocess as dp
from seknn import SEKNN
from knn import KNN
from sae import SAE
import keras
import warnings

warnings.filterwarnings('ignore')

class TrafficPrediction(gym.Env):
    def __init__(self):
        super(TrafficPrediction, self).__init__()
        
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(257,), dtype=np.float32)
        self.observation_space = spaces.Tuple([
                spaces.Box(low=0., high=1., shape=(6, 257), dtype=np.float32),
                spaces.Box(low=0., high=1., shape=(257,), dtype=np.float32)])
                #(last_state_point,)
        self.delta = 1.
        self.state = None
        
        self.predictor = SAE() #SEKNN #self.load_sae() #KNN()
        self.pointer = 0
        self.load_sae()
        
        self.link = 257
        self.predstep = self.predictor.dp.predstep
        self.maxv = np.asarray(self.predictor.dp.maxv)
        self.valiX, self.valiY, self.valiY_nofilt = self.predictor.dp.get_data(data_type='vali')
        self.testX, self.testY, self.testY_nofilt = self.predictor.dp.get_data(data_type='test')
        
        self.set_predY = []
        self.set_predY_ = []
        self.set_realY = []
    
    def load_sae(self):
        self.sae = keras.models.load_model('/home/jinan/AutoKNN/SAE-model/SAE%dp%d.h5'%(self.predictor.dp.inputstep, self.predictor.dp.predstep))
    
    def predict(self, predX):
        predX = np.asarray(predX).reshape(-1, self.predictor.dp.inputstep * self.link,)
        return self.sae.predict(predX).reshape(self.link,)
    
    def get_error(self, predY, realY):
        """
        Input: predY and realY are arrays.
        """
        rmse = np.sqrt(np.mean(np.square(predY-realY)))
        mae = np.mean(np.abs(predY-realY))
        mape = np.mean(np.abs(predY-realY)/realY) * 100
        return rmse, mae, mape
    
    def step(self, action, pointer, data_type='vali'):
        
        if data_type == 'vali':
            self.X = self.valiX[pointer]
            self.X_next = self.valiX[pointer+1]
            realY = self.valiY_nofilt[pointer]
        elif data_type == 'test':
            self.X = self.testX[pointer]
            self.X_next = self.testX[pointer+1]
            realY = self.testY_nofilt[pointer]
        
        predY = self.predict(self.X).reshape(257,)
        predY_ = np.clip(predY + action, 1e-2, 1).reshape(257,)
        
        self.set_predY.append(predY * self.maxv)
        self.set_predY_.append(predY_ * self.maxv)
        self.set_realY.append(realY * self.maxv)
        
        rmse, mae, mape = self.get_error(predY * self.maxv, realY * self.maxv)
        # print('MAE =%.3f, MAPE =%.3f'%(mae, mape))
        rmse_, mae_, mape_ = self.get_error(predY_ * self.maxv, realY * self.maxv)
        # print('MAE_=%.3f, MAPE_=%.3f'%(mae_, mape_))
        
        self.diff = (np.asarray(predY_) - np.asarray(realY)).reshape(self.link,)
        # self.diff_deque.append(diff)
        # self.diff = self.diff_deque[0]
        # print('self.diff shape (in StepFunc)',self.diff.shape)
        self.state = (self.X_next, self.diff)
        
        # done = (mae_ < (self.error[0] + 0.5 * self.delta) and mape_ < (self.error[1] + self.delta))
        # done = (mae_ <= (mae + 0.2 * self.delta)) and (mape_ <= (mape + self.delta))
        # done  = (mae_ <= 1.01*mae) and (mape_ <= 1.01*mape)
        done = (rmse_ <= rmse) and (mae_ <= mae) and (mape_ <= mape)
        done = bool(done)
        
        # update mae& mape
        self.error = (rmse_, mae_, mape_)
        
        reward = 1/3 * 100 * ((rmse-rmse_)/rmse + (mae-mae_)/mae + (mape-mape_)/mape)
        
        """
        if done:
            if ((mae_-mae) >= 0) or ((mape_-mape) >= 0):
                reward = 0
            else:
                reward = max(0,100*(mae-mae_)/mae) + max(0,100*(mape-mape_)/mape)
        else:
            reward = 100*(mae-mae_)/mae + 100*(mape-mape_)/mape
        """
        
        return self.state, reward, done, {}
    
    def reset(self, data_type='vali'):
        self.X = None
        self.X_next = None
        self.diff = None
        self.state = None
        # self.diff_deque = deque(maxlen=self.predstep+1)
        
        if self.set_predY:
            rmse, mae, mape = self.get_error(np.asarray(self.set_predY), np.asarray(self.set_realY))
            print('predictor: RMSE = {:.4f}, MAE = {:.4f}, MAPE = {:.4f}'.format(rmse, mae, mape))
            rmse, mae, mape = self.get_error(np.asarray(self.set_predY_), np.asarray(self.set_realY))
            print('I-predictor: RMSE = {:.4f}, MAE = {:.4f}, MAPE = {:.4f}'.format(rmse, mae, mape))

        self.set_predY = []
        self.set_predY_ = []
        self.set_realY = []
        
        if data_type == 'vali':
            self.pointer = random.randint(0, len(self.testX)-200-self.predstep) # the position of sample in test dataset.
            print('Env has been reset at %d'%(self.pointer))
            self.X = self.valiX[self.pointer]
            self.X_next = self.valiX[self.pointer+1]
            realY = self.valiY_nofilt[self.pointer]
        elif data_type == 'test':
            self.pointer = 0
            self.X = self.testX[self.pointer]
            self.X_next = self.testX[self.pointer+1]
            realY = self.testY_nofilt[self.pointer]
        
        # i = 0
        # while i <= self.predstep:
        #     if data_type == 'vali':
        #         self.X = self.valiX[self.pointer]
        #         self.X_next = self.valiX[self.pointer+1]
        #         realY = self.valiY_nofilt[self.pointer]
        #     elif data_type == 'test':
        #         self.X = self.testX[self.pointer,:,:]
        #         self.X_next = self.testX[self.pointer+1]
        #         realY = self.testY_nofilt[self.pointer]
                
        #     predY = self.predict(self.X)
        #     diff = (np.asarray(predY) - np.asarray(realY)).reshape(self.predictor.link,)
            # print('self.diff shape (in ResetFunc):',self.diff.shape)
        #     self.diff_deque.append(diff)
        #     self.pointer += 1
        #     i += 1
        # self.diff = self.diff_deque[0]
        
        predY = self.predict(self.X) # predict self.pointer-th sample using parameters in action.
        self.set_predY.append(predY * self.maxv)
        self.set_predY_.append(predY * self.maxv)
        self.set_realY.append(realY * self.maxv)
        
        # print('self.diff shape (in ResetFunc):',self.diff.shape)
        self.diff = (np.asarray(predY) - np.asarray(realY)).reshape(self.link,)
        self.state = (self.X_next, self.diff)
        
        return self.state
    
    def render(self, mode='human'):
        return None
    
    def close(self):
        return None