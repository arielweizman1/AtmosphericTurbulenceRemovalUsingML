'''
Warping Function for Turbulence Simulator

Python version
of original code by Stanley Chan

Zhiyuan Mao and Stanley Chan
Copyright 2021
Purdue University, West Lafayette, In, USA.
'''

import numpy as np
from skimage.transform import resize


def motion_compensate(img, Mvx, Mvy, pel):
	m, n =  np.shape(img)[0], np.shape(img)[1]
	img = resize(img, (np.int32(m/pel), np.int32(n/pel)), mode = 'reflect' )
	Blocksize = np.floor(np.shape(img)[0]/np.shape(Mvx)[0])
	m, n =  np.shape(img)[0], np.shape(img)[1]
	M, N =  np.int32(np.ceil(m/Blocksize)*Blocksize), np.int32(np.ceil(n/Blocksize)*Blocksize)

	f = img[0:M, 0:N]


	Mvxmap = resize(Mvy, (N,M))
	Mvymap = resize(Mvx, (N,M))


	xgrid, ygrid = np.meshgrid(np.arange(0,N-0.99), np.arange(0,M-0.99))
	X = np.clip(xgrid+np.round(Mvxmap/pel),0,N-1)
	Y = np.clip(ygrid+np.round(Mvymap/pel),0,M-1)

	idx = np.int32(Y.flatten()*N + X.flatten())
	f_vec = f.flatten()
	g = np.reshape(f_vec[idx],[N,M])

	g = resize(g, (np.shape(g)[0]*pel,np.shape(g)[1]*pel))
	return g