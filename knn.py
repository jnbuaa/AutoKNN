#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:57:51 2019

@author: Nan Ji
"""

import numpy as np
import pandas as pd
import math
from preprocess import DataPreprocess as dp
import warnings

warnings.filterwarnings('ignore')

class KNN():
    def __init__(self):
        self.link = 257
        
        self.dp = dp()
        self.trainX, self.trainY, self.trainY_nofilt = self.dp.get_data(data_type='train')
        self.trainX_cl = self.delete_corr(self.trainX)
        
    def delete_corr(self, X):
        corr_link = pd.read_csv(r'/home/jinan/delcorrlink.csv')
        corr_link = list(np.asarray(corr_link)[:,1])
        if len(np.shape(X)) == 3:
            return X[:,:,corr_link]
        else:
            return X[:,corr_link]
    
    def distance(self, X1, X2):
        X1, X2 = np.asarray(X1), np.asarray(X2)        
        return np.linalg.norm(X1-X2, ord=2)
        
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
            predX: sample. (Attention: predX is a single sameple (one time step), not a dataset)
            
        Returns:
            pred: predicted valule of X.
        """
        D = []
        for i, x in enumerate(self.trainX_cl):
            D.append(self.distance(self.delete_corr(predX), x))
        topK_index = self.index_sorted(D)[:15]
        topK_y = np.asarray(self.trainY[topK_index]).reshape(-1, self.link)
        
        w = [math.e**(-D[i]**2/(2*1.33**2)) for i in topK_index]
        
        predY = np.dot(w, topK_y) / np.sum(w)
        return predY
    
    def evaluate(self, y, y_):
        rmse = np.sqrt(np.mean(np.square(y-y_)))
        mae = np.mean(np.abs(y-y_))
        mape = np.mean(np.abs(y-y_)/y)*100
        return rmse, mae, mape

    
# FUNCTIONAL TEST

if __name__ == '__main__':
    model = KNN()
    for p in range(5):
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
        df.to_csv('model/knn-only-predY6p%d.csv'%(model.dp.predstep), header=False, index=False)

        rmse, mae, mape = model.evaluate(y, y_)
        print('predstep={},RMSE = {:.4}, MAE = {:.4}, MAPE = {:.4}'.format(p, rmse, mae, mape))
