# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:33:38 2019

@author: Nan Ji
"""

import numpy as np
import random
import gym
from gym import spaces
from preprocess import DataPreprocess as dp
from seknn import SEKNN
import warnings

warnings.filterwarnings('ignore')

class TrafficPrediction(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(257,), dtype=np.float32)
        self.observation_space = spaces.Tuple([
                spaces.Box(low=0., high=1., shape=(6, 257), dtype=np.float32),
                spaces.Box(low=0., high=1., shape=(257,), dtype=np.float32)])
                #(last_state_point,)
        self.delta = 1.
        self.state = None
        self.seknn = SEKNN()
        self.pointer = 0
        
        self.predstep = self.seknn.dp.predstep
        self.maxv = np.asarray(self.seknn.dp.maxv)
        self.valiX, self.valiY, self.valiY_nofilt = self.seknn.dp.get_data(data_type='vali')
        self.testX, self.testY, self.testY_nofilt = self.seknn.dp.get_data(data_type='test')
        
        self.set_predY = []
        self.set_predY_ = []
        self.set_realY = []

    def get_error(self, predY, realY):
        """
        Input: predY and realY are arrays.
        """
        mae = np.mean(np.abs(predY - realY))
        mape = np.mean(np.abs(predY - realY) / (realY)) * 100
        return mae, mape
    
    def step(self, action, pointer, data_type='vali'):
        
        if data_type == 'vali':
            self.X = self.valiX[pointer,:,:]
            self.X_next = self.valiX[pointer+1,:,:]
            self.realY = self.valiY_nofilt[pointer,:]
        elif data_type == 'test':
            self.X = self.testX[pointer,:,:]
            self.X_next = self.testX[pointer+1,:,:]
            self.realY = self.testY_nofilt[pointer,:]
        
        predY = self.seknn.predict(self.X)
        predY_ = predY + action
        
        self.set_predY.append(predY * self.maxv)
        self.set_predY_.append(predY_ * self.maxv)
        self.set_realY.append(self.realY * self.maxv)
        
        mae, mape = self.get_error(predY * self.maxv, self.realY * self.maxv)
        # print('MAE =%.3f, MAPE =%.3f'%(mae, mape))
        mae_, mape_ = self.get_error(predY_ * self.maxv, self.realY * self.maxv)
        # print('MAE_=%.3f, MAPE_=%.3f'%(mae_, mape_))
        
        self.diff = (np.asarray(predY_) - np.asarray(self.realY)).reshape(self.seknn.link,)
        # print('self.diff shape (in StepFunc)',self.diff.shape)
        self.state = (self.X_next, self.diff)
        
        # done = (mae_ < (self.error[0] + 0.5 * self.delta) and mape_ < (self.error[1] + self.delta))
        # done = (mae_ <= (mae + 0.2 * self.delta)) and (mape_ <= (mape + self.delta))
        done = (mae_ <= (mae * 1.01)) and (mape_ <= (mape * 1.01))
        done = bool(done)
        
        # update mae& mape
        self.error = (mae_, mape_)
        
        if done:
            if ((mae_ - mae) >= 0) or ((mape_ - mape) >= 0):
                reward = 0
            else:
                reward = max(0, 100*(mae-mae_)/mae) + max(0, 100*(mape-mape_)/mape)
        else:
            reward = 100*(mae-mae_)/mae + 100*(mape-mape_)/mape
#        if done:
#            if mae_ < 3 and mape_ < 10:
#                reward = 10
#            elif (mae_ > 3 and mae_ < 4) and (mape_ > 10 and mape_ < 15):
#                reward = 8
#            elif (mae_ > 4 and mae_ < 5) and (mape_ > 15 and mape_ < 20):
#                reward = 6
#            elif mae_ < 5 and (mape_ > 20 and mape_ < 25):
#                reward = 4
#            else:
#                reward = 1
#        else:
#            reward = -10
        
        return self.state, reward, done, {}
    
    def reset(self, data_type='vali'):
        self.state = None
        if self.set_predY:
            mae, mape = self.get_error(np.asarray(self.set_predY), np.asarray(self.set_realY))
            print('SEKNN: MAE = {:.4f}, MAPE = {:.4f}'.format(mae, mape))
            mae, mape = self.get_error(np.asarray(self.set_predY_), np.asarray(self.set_realY))
            print('ISEKNN: MAE = {:.4f}, MAPE = {:.4f}'.format(mae, mape))

        self.set_predY = []
        self.set_predY_ = []
        self.set_realY = []
        
        if data_type == 'vali':
            self.pointer = random.randint(0, len(self.testX)-201) # the position of sample in test dataset.
            print('Env has been reset at %d'%(self.pointer))
            self.X = self.valiX[self.pointer,:,:]
            self.X_next = self.valiX[self.pointer+1,:,:]
            self.realY = self.valiY_nofilt[self.pointer,:]

        elif data_type == 'test':
            self.pointer = 0
            self.X = self.testX[self.pointer,:,:]
            self.X_next = self.valiX[self.pointer+1,:,:]
            self.realY = self.testY_nofilt[self.pointer,:]
        
        self.predY = self.seknn.predict(self.X) # predict self.pointer-th sample using parameters in action.
        
        self.diff = (np.asarray(self.predY) - np.asarray(self.realY)).reshape(self.seknn.link,)
        # print('self.diff shape (in ResetFunc):',self.diff.shape)
        self.state = (self.X_next, self.diff)
        
        return self.state
    
    def render(self, mode='human'):
        return None
    
    def close(self):
        return None