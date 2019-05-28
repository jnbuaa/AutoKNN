# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:47:35 2019

@author: Nan Ji
"""
import numpy as np
import pandas as pd
import math
from preprocess import DataPreprocess as dp
import warnings

warnings.filterwarnings('ignore')

class SEKNN:
    def __init__(self):
        self.link = 257
        
        self.dp = dp()
        self.trainX, self.trainY, self.trainY_nofilt = self.dp.get_data(data_type='train')
        self.trainX_cl = self.delete_corr(self.trainX)
        self.trainX_incr = self.increment(self.trainX, self.trainY)
                
        
    def delete_corr(self, X):
        corr_link = pd.read_csv(r'/home/jinan/delcorrlink.csv')
        corr_link = list(np.asarray(corr_link)[:,1])
        if len(np.shape(X)) == 3:
            return X[:,:,corr_link]
        else:
            return X[:,corr_link]
    
    def trend(self, X):
        return X[-1] - X[0]
    
    def increment(self, X, Y):
        incr = []
        for i in range(len(X)):
            incr.append(Y[i]-X[i][-1])
        return incr
    
    def distance(self, X1, X2, dtype='ed'):
        """calcuate distance between X1 and X2.
        
        Arguments:
            X1, X2: the objects whose distance would be calcuated.
            dtype: the type of distacne, `ed` for Euclidean distance, or `cd` for consine distace.
            
        Returns:
            d: distance between X1 and X2.
        """
        X1, X2 = np.asarray(X1), np.asarray(X2)
        
        if dtype == 'ed': # Euclidean distance
            d = np.linalg.norm(X1-X2, ord=2)
        elif dtype == 'cd': # Cosine distance
            d = 1 - np.dot(X1, X2)/(np.linalg.norm(X1, ord=2)*np.linalg.norm(X2, ord=2))
        else:
            raise Exception('Wrong or missing argument of `dtype`, please use `ed` for Euclidean distance, or `cd` for consine distace.')
        return d
    
    
    def index(self, l):
        """create a index dict of list 'l' ().
        
        Arguments:
            l: a list.
            
        Returns:
            index: a dict, whose keys are elements of l, and values are indexes of l's elements.
        """
        index = {}
        for i in range(len(l)):
            index[l[i]] = i
        return index
    
    def quickSort(self, l):
        if len(l) < 2:
            return l
        else:
            pivot = l[0]
            left = [i for i in l[1:] if i <= pivot]
            right = [i for i in l[1:] if i >= pivot]
        return self.quickSort(left) + [pivot] + self.quickSort(right)
    
    def index_sorted(self, l):
        """return original index with a sorted order.
        
        Arguments:
            
        Returns:
            ind_sorted: the new index of sorted l
        """
        index_of_l = self.index(l)
        sorted_l = self.quickSort(l)
        ind_sorted = []
        for element in sorted_l:
            ind_sorted.append(index_of_l[element])
        
        return ind_sorted
    
    
    def predict(self, predX):
        """use SEKNN to predict sample predX
        
        Arguments:
            #params: parameters, incluing (k, alpha), i.e. action in RL.
            predX: sample. (Attention: predX is a single sameple (one time step), not a dataset)
            
        Returns:
            pred: predicted valule of X.
        """
        params = {0:(0.9, 97), 1:(0.9, 57), 2:(0.8, 57), 3:(0.8, 54), 4:(0.7, 54)}
        
        alpha, k = params[self.dp.predstep]
        predX_cl = self.delete_corr(predX)
        
        ED = []
        CD = []
        for index_t, t in enumerate(self.trainX_cl):
            ED.append(self.distance(t[-1], predX_cl[-1], dtype='ed'))
            CD.append(self.distance(self.trend(t), self.trend(predX_cl), dtype='cd'))
        maxED, minED = max(ED), min(ED)
        ED01 = [2*(d-minED)/(maxED-minED) for d in ED]
        D = list(map(lambda ed, cd: alpha*ed + (1-alpha)*cd, ED01, CD))
        D_topK = self.index_sorted(D)[:k] # the index value
        
        nn_topK = np.asarray(self.trainX_incr)[D_topK].reshape(-1, self.link)
        w = [math.e**(-D[i]**2/(2*1.33**2)) for i in D_topK]
        
        predY = np.dot(w, nn_topK + predX[-1]) / np.sum(w)
        return predY

    def evaluate(self, y, y_):
        rmse = np.sqrt(np.mean(np.square(y-y_)))
        mae = np.mean(np.abs(y-y_))
        mape = np.mean(np.abs(y-y_)/y)*100
        return rmse, mae, mape


# FUNCTIONAL TEST

if __name__ == '__main__':
    model = SEKNN()
    for p in [4]:# range(5):
        model.dp.predstep = p
        testX, testY, testY_nofilt = model.dp.get_data(data_type='test')

        y = np.array([i * model.dp.maxv for i in testY_nofilt]).reshape(-1, 257)
        df = pd.DataFrame(y)
        df.to_csv('model/testy-only-predY6p%d.csv'%(model.dp.predstep), header=False, index=False)

        set_predY = []
        for x in testX:
            set_predY.append(model.predict(x) * model.dp.maxv)

        y_ = np.array(set_predY).reshape(-1, 257)
        df = pd.DataFrame(y_)
        df.to_csv('model/seknn-only-predY6p%d.csv'%(model.dp.predstep), header=False, index=False)

        rmse, mae, mape = model.evaluate(y, y_)
        print('predstep={},RMSE = {:.4}, MAE = {:.4}, MAPE = {:.4}'.format(p, rmse, mae, mape))
