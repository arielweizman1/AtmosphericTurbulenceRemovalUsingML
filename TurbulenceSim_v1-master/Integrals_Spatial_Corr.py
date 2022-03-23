'''
Spatial Correlation Functions for Tilts

Zhiyuan Mao and Stanley Chan
Copyright 2021
Purdue University, West Lafayette, In, USA.
'''

import numpy as np
import scipy.integrate as integrate
from scipy.special import jv
import os
from math import gamma


def genTiltPSD(s_max, spacing, N, **kwargs):
	D = 0.1
	D_r0 = 20
	wavelength = kwargs.get('wavelength', 500 * (10 ** (-9)))
	L = 7000
	delta0 = L * wavelength / (2 * D)
	delta0 *= 1  # Adjusting factor
	# Constants in [Channan, 1992]
	c1 = 2 * ((24 / 5) * gamma(6 / 5)) ** (5 / 6)
	c2 = 4 * c1 / np.pi * (gamma(11 / 6)) ** 2
	s_arr = np.arange(0, s_max, spacing)
	I0_arr = np.float32(s_arr * 0)
	I2_arr = np.float32(s_arr * 0)
	for i in range(len(s_arr)):
		I0_arr[i] = I0(s_arr[i])
		I2_arr[i] = I2(s_arr[i])
	i, j = np.int32(N / 2), np.int32(N / 2)
	[x, y] = np.meshgrid(np.arange(1, N + 0.01, 1), np.arange(1, N + 0.01, 1))
	s = np.sqrt((x - i) ** 2 + (y - j) ** 2)
	s *= spacing
	C0 = (In_m(s, spacing*100, I0_arr) + In_m(s, spacing*100, I2_arr)) / I0(0)
	C0[i, j] = 1
	C0_scaled = C0 * I0(0) * c2 * ((D_r0) ** (5 / 3)) / (2 ** (5 / 3)) * (
				(2 * wavelength / (np.pi * D)) ** 2) * 2 * np.pi
	Cfft = np.fft.fft2(C0_scaled)
	S_half = np.sqrt(Cfft)
	S_half_max = np.max(np.max(np.abs(S_half)))
	S_half[np.abs(S_half) < 0.0001 * S_half_max] = 0

	return S_half


def I0(s):
	# z = np.linspace(1e-6, 1e3, 1e5)
	# f_z = (z**(-14/3))*jv(0,2*s*z)*(jv(2,z)**2)
	# I0_s = np.trapz(f_z, z)

	I0_s, _ = integrate.quad( lambda z: (z**(-14/3))*jv(0,2*s*z)*(jv(2,z)**2), 1e-4, np.inf, limit = 100000)
	# print('I0: ',I0_s)
	return I0_s


def I2(s):
	# z = np.linspace(1e-6, 1e3, 1e5)
	# f_z = (z**(-14/3))*jv(2,2*s*z)*(jv(2,z)**2)
	# I2_s = np.trapz(f_z, z)

	I2_s, _ = integrate.quad( lambda z: (z**(-14/3))*jv(2,2*s*z)*(jv(2,z)**2), 1e-4, np.inf, limit = 100000)
	# print('I2: ',I2_s)
	return I2_s


def save_In(s_max, spacing):
	s_arr = np.arange(0,s_max,spacing)
	I0_arr = np.float32(s_arr*0)
	I2_arr = np.float32(s_arr*0)
	for i in range(len(s_arr)):
		I0_arr[i] = I0(s_arr[i])
		I2_arr[i] = I2(s_arr[i])
	os.makedirs('temp/In_arr', exist_ok=True)

	np.save('temp/In_arr/I0_%d_%d.npy'%(s_max,spacing*1000),I0_arr)
	np.save('temp/In_arr/I2_%d_%d.npy'%(s_max,spacing*1000),I2_arr)


def In_m(s, spacing, In_arr):
	idx = np.int32(np.floor(s.flatten()/spacing))
	M,N = np.shape(s)[0], np.shape(s)[1]
	In = np.reshape(np.take(In_arr, idx), [M,N])

	return In


def In_arr(s, spacing, In_arr):
	idx = np.int32(np.floor(s/spacing))
	In = np.take(In_arr, idx)

	return In

