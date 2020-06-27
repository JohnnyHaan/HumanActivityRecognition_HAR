#
# Features Register
# Human activity recognition for feature register system
# Just using accelerator signals to extend the features to get more accuracy result
# 
# Author: J.Haan
# Data: June_26Th_2020
# 
import numpy as np
import math

class FeaturesRegister:
    def __init__(self):
        self.reg_funcs = [
        'get_xFFT','get_yFFT','get_zFFT',
        'get_xyzVector','get_xyzDiffVector'
        ]
    
    def get_xFFT(self, x, y, z):
        result = np.zeros((len(x), 0)) 
        result = np.insert(result, 0, np.fft.fft(x, n=len(x),axis=-1)) ## neglect the complexy part  
        return np.reshape(result,(len(x),1))
        
    def get_yFFT(self, x, y, z):
        result = np.zeros((len(y), 0)) 
        result = np.insert(result, 0, np.fft.fft(y, n=len(y),axis=-1)) ## neglect the complexy part
        return np.reshape(result,(len(y),1))

    def get_zFFT(self, x, y, z):
        result = np.zeros((len(z), 0)) 
        result = np.insert(result, 0, np.fft.fft(z, n=len(z),axis=-1)) ## neglect the complexy part
        return np.reshape(result,(len(z),1))
        
    def get_xyzVector(self, x, y, z):
        result = np.zeros((len(x), 1))
        for i in range(0, len(x)):
            result[i,] = math.sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])
        return result        

    def get_xyzDiffVector(self, x, y, z):
        hold_x = 0
        hold_y = 0
        hold_z = 0
        result = np.zeros((len(x), 1))
        for i in range(0, len(x)):
            diff_x=x[i]-hold_x
            diff_y=y[i]-hold_y
            diff_z=z[i]-hold_z
            result[i,] = math.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
            hold_x = x[i]
            hold_y = y[i]
            hold_z = z[i]            
        return result
