# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:30:02 2015

@author: Administrator
"""
import numpy as np
import pandas as pd
import MySQLdb
import time
from collections import Counter


class Helper():
    
    def __init__(self):
        try:
            self.__conn = MySQLdb.connect(host='', user='', passwd='')
            self.__conn.select_db('')
            self.__cur = self.__conn.cursor()
        except MySQLdb.Error, e:
            print e
    
    def fetch_data(self, query):
        self.__cur.execute(query)
        data = []
        for i in range(self.__cur.rowcount):
            row = self.__cur.fetchone()
            data.append(row)
        return data
        
    def fetch_csv(self, file_path):
        csv = pd.read_csv(file_path, header=None)
        return csv
        
    def close_connection(self):
        if self.__conn:
            self.__conn.close()
            
class Cart:
   
    def __init__(self, sample, max_depth):
        self.n_depth = max_depth
        self.sample = sample
        self.nr_instance = np.shape(self.sample)[0]
        self.nr_feature = np.shape(self.sample)[1]
        self.split_list = []
        self.gamma = {}
        self.tree = dict()
       
    def fit(self):
        splited_feat = []
        dd = dict()
        for level in range(self.n_depth):
            for tnode in range(2 ** level):
                for feat in range(1, self.nr_feature):
                    if feat in splited_feat:
                        continue
                    dd[feat] = self.split_feat(feat)
               
#                dd = dict(filter(lambda a: a[0] not in splited_feat, dd.iteritems()))
                for i in splited_feat:                                         #remove splitted features
                    if dd.has_key(i):
                        dd.pop(i)
                        
                split_feat_ = sorted(dd.iteritems(), key=lambda a: a[1][1])
                if len(split_feat_) == 0:
                    break
                feat_, pair_ = split_feat_[0]
               
                splited_feat.append(feat_)
                self.split_list.append(dict([[feat_, pair_[0]]]))
#            print "level: %s ->" %level
#            print self.split_list[(2**level-1):]
           
        yhat = self.split_data()
        new_y = self.gradient_boost(yhat)

        return new_y
        
    def split_feat(self, feat):
        feat_dict = Counter(self.sample[:, feat])
        d = dict()
        for key in feat_dict:
            indices = (self.sample[:, feat] <= key)
            c1 = np.mean(self.sample[indices, 0].astype('float'))
            c2 = np.mean(self.sample[~indices, 0].astype('float'))
            d[key] = sum((self.sample[indices, 0].astype('float') - c1) ** 2) + sum((self.sample[~indices, 0].astype('float') - c2) ** 2)
           
        return sorted(d.iteritems(), key=lambda a: a[1])[0]
       
    def split_data(self):
        yhat_ = dict()
        
        for single_sample in self.sample:
            
            tree_index = 0
            for _ in range(int(np.ceil(np.log2(len(self.split_list)))) - 1):
                split_index = self.split_list[tree_index].keys()
                split_value = self.split_list[tree_index].values()
                if single_sample[split_index] <= split_value:
                    tree_index =  2 * tree_index + 1
                else:
                    tree_index = 2 * tree_index + 2

            if not yhat_.has_key(tree_index):
                yhat_[tree_index] = np.array([single_sample])
            else:
                yhat_[tree_index] = np.vstack((yhat_[tree_index], single_sample))
        
        return yhat_
        
    def gradient_boost(self, yhat):
        for k in yhat:
            tmp = yhat.get(k)
            gamma_ = sum(tmp[:, 0].astype('float')) * 1.0 / sum(map(lambda a: a*(2-a), tmp[:, 0].astype('float')))
            yhat.get(k)[:, 0] = tmp[:,0].astype('float') + gamma_
            
            self.gamma[k] = gamma_            
            self.tree[k] = list()
            for val in tmp[:,1]:
                self.tree[k].append(val)
        
        l = [k for v in yhat.values() for k in v]
        return np.array(l)
       
class Gbdt:
   
    def __init__(self,tree_size):
        self.n_estimator = tree_size
        self.n_trees = []
   
    def fit(self, sample=None):
        max_depth = 5
        fx = np.empty((np.shape(sample)[0], 1))
        for i in range(self.n_estimator):
            if i == 0:                          #initialize Fx
                ybar = np.mean(sample[:, 0].astype('float'))
                fx.fill(0.5 * np.log((1+ybar) / (1-ybar)))
            else:
                fx = sample[:, 0].astype('float')
#            print sample
            sample[:, 0] = 2*sample[:, 0].astype('float') / (1 + np.exp(np.dot(2*sample[:, 0].astype('float'), fx)))
            cart_tree = Cart(sample, max_depth)
            sample = cart_tree.fit()
            self.n_trees.append(cart_tree.tree)
            
    def predict(self):
        pass
    
def main():
    helper = Helper()
    
    start = time.clock()
    
    negative = helper.fetch_data()
    positive = helper.fetch_data()
    helper.close_connection()
    
    x = np.vstack((positive, negative))
    pos_y = np.ones((len(positive), 1))
    neg_y = np.zeros((len(negative), 1)) - 1
    y = np.vstack((pos_y, neg_y))
    training = np.hstack((y, x))
    
    end = time.clock()
    print "Data preparation elapsed: %s s" %(end - start) 
    
    start = time.clock()
    gbdt = Gbdt(30)
    gbdt.fit(training)
    end = time.clock()
    print "Training time: %s s" %(end - start)
        
if __name__ == '__main__':
    
    main()