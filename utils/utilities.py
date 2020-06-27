# HAR classification 
# Author: Burak Himmetoglu
# Modify: add the fuction of extend_features by J.Haan 
# 8/15/2017

import pandas as pd 
import numpy as np
import os
import attr

def extend_features(XYZ_, obj, funcs_):
    extend_feat_group = 3
    extend_len = len(funcs_) * extend_feat_group
    XYZ_hold = np.zeros((np.shape(XYZ_)[0], np.shape(XYZ_)[1], np.shape(XYZ_)[2] + extend_len))
    batch = 0
    for XYZ in XYZ_:
        acc_x = XYZ[:,0].copy()
        acc_y = XYZ[:,1].copy()
        acc_z = XYZ[:,2].copy()
        
        gyro_x = XYZ[:,3].copy()
        gyro_y = XYZ[:,4].copy()
        gyro_z = XYZ[:,5].copy() 
        
        total_x = XYZ[:,6].copy()
        total_y = XYZ[:,7].copy()
        total_z = XYZ[:,8].copy()        
        for func_name in funcs_:
            func_ = getattr(obj,func_name)        
            value = func_(acc_x, acc_y, acc_z)           
            XYZ = np.append(XYZ, value, axis=-1)  

        for func_name in funcs_:
            func_ = getattr(obj,func_name)        
            value = func_(gyro_x, gyro_y, gyro_z)           
            XYZ = np.append(XYZ, value, axis=-1) 
                   
        for func_name in funcs_:
            func_ = getattr(obj,func_name)        
            value = func_(total_x, total_y, total_z)           
            XYZ = np.append(XYZ, value, axis=-1)             

        XYZ_hold[batch:,:] =  XYZ 
        batch += 1            
    #print(np.shape(XYZ_hold))
    return XYZ_hold

def read_data(data_path, split = "train"):
	""" Read data """

	# Fixed params
	n_class = 6
	n_steps = 128

	# Paths
	path_ = os.path.join(data_path, split)
	path_signals = os.path.join(path_, "Inertial_Signals")

	# Read labels and one-hot encode
	label_path = os.path.join(path_, "y_" + split + ".txt")
	labels = pd.read_csv(label_path, header = None)

	# Read time-series data
	channel_files = os.listdir(path_signals)
	channel_files.sort()
	n_channels = len(channel_files)
	posix = len(split) + 5

	# Initiate array
	list_of_channels = []
	X = np.zeros((len(labels), n_steps, n_channels))
	i_ch = 0
	for fil_ch in channel_files:
		channel_name = fil_ch[:-posix]
		dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)
		# X[:,:,i_ch] = dat_.as_matrix()
		X[:,:,i_ch] = dat_.iloc[:,:,].values
		#print(X[:,:,i_ch])
		# Record names
		list_of_channels.append(channel_name)

		# iterate
		i_ch += 1

	# Return 
	return X, labels[0].values, list_of_channels

def standardize(train, test):
	""" Standardize data """

	# Standardize train and test
	X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
	X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

	return X_train, X_test

def one_hot(labels, n_class = 6):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"

	return y

def get_batches(X, y, batch_size = 100):
	""" Return a generator for batches """
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size]
	




