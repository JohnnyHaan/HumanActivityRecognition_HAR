#
# Human activity recognition for feature selecting
# 
# Author: J.Haan
# Data: June_26Th_2020
# 

import numpy as np
from features_register import *
from utils.utilities import *
from har_cnn import *

class SelectFeatures:
    def __init__(self, mode=0):
        self.fr = FeaturesRegister()
        self.max_feat_nums = len(self.fr.reg_funcs)
        self.choice_method = mode
    
    def choise_sequence_features(self): 
        ## choise x,y,z three features at least  
        for f in range(self.max_feat_nums, self.max_feat_nums+1):
            func_list = []
            for i in range(0, f):
                print(self.fr.reg_funcs[i])
                func_list.insert(i, self.fr.reg_funcs[i])
            
            # get data for training or testing
            X_train, labels_train, list_ch_train = read_data(data_path="./data/", split="train") # train
            print(np.shape(X_train))            
            X_train = extend_features(X_train, self.fr, func_list)
            print(np.shape(X_train))           
            X_test, labels_test, list_ch_test = read_data(data_path="./data/", split="test") # test
            X_test = extend_features(X_test, self.fr, func_list)            
            print(list_ch_train)
            assert list_ch_train == list_ch_test, "Mistmatch in channels!"
            
            run_algo(X_train, labels_train, list_ch_train, X_test, labels_test, list_ch_test)
            # train
        return 
        
    def choise_random_features(self):
        # TODO
        return 


    def run(self):
        #obj = SelectFeatures()
        if self.choice_method == 0:
            return self.choise_sequence_features()
        else :
            return self.choise_random_features()

if __name__ == "__main__":
    obj = SelectFeatures()
    obj.run()