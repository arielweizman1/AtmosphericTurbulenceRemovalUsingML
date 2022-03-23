'''
Collection of Optical Functions for Turbulence Simulator

Nicholas Chimitt and Stanley H. Chan "Simulating anisoplanatic turbulence
by sampling intermodal and spatially correlated Zernike coefficients," Optical Engineering 59(8), Aug. 2020

ArXiv:  https://arxiv.org/abs/2004.11210

Nicholas Chimitt and Stanley Chan
Copyright 2021
Purdue University, West Lafayette, In, USA.
'''

import numpy as np
import math
import scipy.signal
import scipy.interpolate
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from scipy.interpolate import RectBivariateSpline
from scipy.special import jv
import scipy.integrate as integrate
import Integrals_Spatial_Corr as ISC
from math import gamma
import Motion_Compensate as MC


def p_obj(N, D, L, r0, wvl, obj_size):
    """
    The parameter "object". It is really just a list of some useful parameters.

    :param N: size of pixels of one image dimension (assumed to be square image N x N).
    :param D: size of aperture diameter (meters)
    :param L: length of propagation (meters)
    :param r0: Fried parameter (meters)
    :param wvl: wavelength (meters)
    :return: returns the parameter object
    """
    a = {}
    a['N'] = N
    a['D'] = D
    a['L'] = L
    a['wvl'] = wvl
    a['r0'] = r0
    a['Dr0'] = D/r0
    a['delta0'] = L*wvl/(2*D)
    a['k'] = 2*np.pi/wvl
    a['smax'] = a['delta0']/D*N
    a['spacing'] = a['delta0']/D
    a['ob_s'] = obj_size
    a['scaling'] = obj_size / (N * a['delta0'])

    a['smax'] *= a['scaling']
    a['spacing'] *= a['scaling']
    a['ob_s'] *= a['scaling']
    a['delta0'] *= a['scaling']

    return a


def gen_PSD(p_obj):
    """
    This function generates the PSD necessary for the tilt values (both x and y pixel shifts). The PSD is **4 times**
    the size of the image, this is to simplify the generation of the random vector using a property of Toeplitz
    matrices. This is further highlighted in the genTiltImg() function, where only 1/4 of the entire grid is used
    (this is because of symmetry about the origin -- hence why the PSD is quadruple the size).

    All that is required is the parameter list, p_obj.

    :param p_obj:
    :return: PSD
    """
    N = 2 * p_obj['N']
    smax = p_obj['delta0'] / p_obj['D'] * N
    c1 = 2 * ((24 / 5) * gamma(6 / 5)) ** (5 / 6)
    c2 = 4 * c1 / np.pi * (gamma(11 / 6)) ** 2
    s_arr = np.linspace(0, smax, N)
    I0_arr = np.float32(s_arr * 0)
    I2_arr = np.float32(s_arr * 0)
    for i in range(len(s_arr)):
        I0_arr[i] = ISC.I0(s_arr[i])
        I2_arr[i] = ISC.I2(s_arr[i])
    i, j = np.int32(N / 2), np.int32(N / 2)
    [x, y] = np.meshgrid(np.arange(1, N + 0.01, 1), np.arange(1, N + 0.01, 1))
    s = np.sqrt((x - i) ** 2 + (y - j) ** 2)
    C = (ISC.In_m(s, p_obj['delta0'] / p_obj['D'] * N , I0_arr) + ISC.In_m(s, p_obj['delta0'] / p_obj['D'] * N, I2_arr)) / ISC.I0(0)
    C[round(N / 2), round(N / 2)] = 1
    C = C * ISC.I0(0) * c2 * (p_obj['Dr0']) ** (5 / 3) / (2 ** (5 / 3)) * (2 * p_obj['wvl'] / (np.pi * p_obj['D'])) ** 2 * 2 * np.pi
    Cfft = np.fft.fft2(C)
    S_half = np.sqrt(Cfft)
    S_half_max = np.max(np.max(np.abs(S_half)))
    S_half[np.abs(S_half) < 0.0001 * S_half_max] = 0

    return S_half


def genTiltImg(img, p_obj):
    """
    This function takes the p_obj (with the PSD!) and applies it to the image. If no PSD is found, one will be
    generated. However, it is **significantly** faster to generate the PSD once and then use it to draw the values from.
    This is also done automatically, because it is significantly faster.

    :param img: The input image (assumed to be N x N pixels)
    :param p_obj: The parameter object -- with the PSD is preferred
    :return: The output, tilted image
    """
    flag_noPSD = 0
    if (p_obj.get('S') == None).any():
        S = gen_PSD(p_obj)
        p_obj['S'] = S
        flag_noPSD = 1
    MVx = np.real(np.fft.ifft2(p_obj['S'] * np.random.randn(2 * p_obj['N'], 2 * p_obj['N']))) * np.sqrt(2) * 2 * p_obj['N'] * (p_obj['L'] / p_obj['delta0'])
    MVx = MVx[round(p_obj['N'] / 2) :2 * p_obj['N'] - round(p_obj['N'] / 2), 0: p_obj['N']]
    #MVx = 1 / p_obj['scaling'] * MVx[round(p_obj['N'] / 2):2 * p_obj['N'] - round(p_obj['N'] / 2), 0: p_obj['N']]
    MVy = np.real(np.fft.ifft2(p_obj['S'] * np.random.randn(2 * p_obj['N'], 2 * p_obj['N']))) * np.sqrt(2) * 2 * p_obj['N'] * (p_obj['L'] / p_obj['delta0'])
    MVy = MVy[0:p_obj['N'], round(p_obj['N'] / 2): 2 * p_obj['N'] - round(p_obj['N'] / 2)]
    #MVy = 1 / p_obj['scaling'] * MVy[0:p_obj['N'], round(p_obj['N'] / 2): 2 * p_obj['N'] - round(p_obj['N'] / 2)]
    img_ = MC.motion_compensate(img, MVx - np.mean(MVx), MVy - np.mean(MVy), 0.5)
    #plt.quiver(MVx[::10,::10], MVy[::10,::10], scale=60)
    #plt.show()

    if flag_noPSD == 1:
        return img_, p_obj
    else:
        return img_, p_obj


def genBlurImage(p_obj, img):
    smax = p_obj['delta0'] / p_obj['D'] * p_obj['N']
    temp = np.arange(1,101)
    patchN = temp[np.argmin((smax*np.ones(100)/temp - 2)**2)]
    patch_size = round(p_obj['N'] / patchN)
    xtemp = np.round_(p_obj['N']/(2*patchN) + np.linspace(0, p_obj['N'] - p_obj['N']/patchN + 0.001, patchN))
    xx, yy = np.meshgrid(xtemp, xtemp)
    xx_flat, yy_flat = xx.flatten(), yy.flatten()
    NN = 32 # For extreme scenarios, this may need to be increased
    img_patches = np.zeros((p_obj['N'], p_obj['N'], int(patchN**2)))
    den = np.zeros((p_obj['N'], p_obj['N']))
    patch_indx, patch_indy = np.meshgrid(np.linspace(-patch_size, patch_size+0.001, num=2*patch_size+1), np.linspace(-patch_size, patch_size+0.001, num=2*patch_size+1))

    for i in range(int(patchN**2)):
        aa = genZernikeCoeff(36, p_obj['Dr0'])
        temp, x, y, nothing, nothing2 = psfGen(NN, coeff=aa, L=p_obj['L'], D=p_obj['D'], z_i=1.2, wavelength=p_obj['wvl'])
        psf = np.abs(temp) ** 2
        psf = psf / np.sum(psf.ravel())
        # focus_psf, _, _ = centroidPsf(psf, 0.95) : Depending on the size of your PSFs, you may want to use this
        psf = resize(psf, (round(NN/p_obj['scaling']), round(NN/p_obj['scaling'])))
        patch_mask = np.zeros((p_obj['N'], p_obj['N']))
        patch_mask[round(xx_flat[i]), round(yy_flat[i])] = 1
        patch_mask = scipy.signal.fftconvolve(patch_mask, np.exp(-patch_indx**2/patch_size**2)*np.exp(-patch_indy**2/patch_size**2)*np.ones((patch_size*2+1, patch_size*2+1)), mode='same')
        den += scipy.signal.fftconvolve(patch_mask, psf, mode='same')
        img_patches[:,:,i] = scipy.signal.fftconvolve(img * patch_mask, psf, mode='same')

    out_img = np.sum(img_patches, axis=2) / (den + 0.000001)
    return out_img


def genZernCorrHighOrder_v2():
    """
    This function generates the correlation functions found in [Chimitt-Chan 2020] and [Takato et al. 1992].
    There are no inputs to this function, and is provided primarily for (i) convenience (ii) further modification,
    specifically with respect to the "A" parameter below, which should be modified simply to include (n+1)*(npr+1) upon
    calculation of each row.

    :return: There is no return of this function in its current form. Instead, the functions are saved as it takes a
    good amount of time to generate. The functions evaluated at sufficiently long distance and resolution are provided
    in the download for this simulator package [LOCATION].
    """
    lamb = 0.525e-6
    Cn2 = 1e-15
    L = 7000
    A = 0.00969 * (2 * np.pi / lamb ) ** 2 * Cn2 * L
    D = 0.2034
    L0 = np.inf
    k0 = 1 / L0

    MM = 65
    NN = 150

    svalarr = np.hstack((-np.logspace(-2,2,num=int((MM-1)/2))[::-1], 0,  np.logspace(-2,2,num=int((MM-1)/2))))
    fij = np.zeros((MM,MM))
    outxx, outyy = np.meshgrid(np.linspace(-5, 5, NN), np.linspace(-5, 5, NN))
    smat = np.sqrt(outxx**2 + outyy**2)
    thetamat = np.arctan2(outyy, outxx + 0.0001)
    xx = np.linspace(-np.max(np.max(smat)),np.max(np.max(smat)),NN)
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, NN, endpoint=True), np.linspace(-1, 1, NN, endpoint=True))
    mask = np.sqrt(x_grid ** 2 + y_grid ** 2) <= 1

    for i in np.arange(27,37):
        #print(i)
        n,m = nollToZernInd(i)
        for j in range(4, 37):
            print(i,j)
            npr, mpr = nollToZernInd(j)
            line1int = np.zeros((len(xx),len(xx)))
            line2int = np.zeros((len(xx),len(xx)))
            line3int = np.zeros((len(xx),len(xx)))
            line4int = np.zeros((len(xx),len(xx)))
            line5int = np.zeros((len(xx),len(xx)))
            line6int = np.zeros((len(xx),len(xx)))
            line7int = np.zeros((len(xx),len(xx)))
            line1rot = np.zeros((len(xx),len(xx)))
            line2rot = np.zeros((len(xx),len(xx)))
            line3rot = np.zeros((len(xx),len(xx)))
            line4rot = np.zeros((len(xx),len(xx)))
            line5rot = np.zeros((len(xx),len(xx)))
            line6rot = np.zeros((len(xx),len(xx)))
            line7rot = np.zeros((len(xx),len(xx)))
            for xxi in range(len(xx)):
                #print(xxi/len(xx))
                x0, kap, mu, nu, ss = D * np.pi * k0, m + mpr, n + 1, npr + 1, xx[xxi]
                line1int[int(NN/2 - 1), xxi], _ = integrate.quad(lambda x: jv(kap, 2 * ss * x) * jv(mu, x) * jv(nu, x) / (x * (x ** 2 + x0 ** 2) ** (11 / 6)), 1E-20, np.inf, limit=10**8)

                x0, kap, mu, nu, ss = D * np.pi * k0, abs(m - mpr), n + 1, npr + 1, xx[xxi]
                line2int[int(NN/2 - 1), xxi], _ = integrate.quad(
                    lambda x: jv(kap, 2 * ss * x) * jv(mu, x) * jv(nu, x) / (x * (x ** 2 + x0 ** 2) ** (11 / 6)), 1E-20,
                    np.inf, limit=10**8)

                line3int[int(NN/2 - 1), xxi] = line1int[int(NN/2 - 1), xxi]

                line4int[int(NN/2 - 1), xxi] = line2int[int(NN/2 - 1), xxi]

                x0, kap, mu, nu, ss = D * np.pi * k0, m, n + 1, npr + 1, xx[xxi]
                line5int[int(100/2 - 1), xxi], _ = integrate.quad(
                    lambda x: jv(kap, 2 * ss * x) * jv(mu, x) * jv(nu, x) / (x * (x ** 2 + x0 ** 2) ** (11 / 6)), 1E-20,
                    np.inf, limit=10**8)

                line6int[int(NN/2 - 1), xxi] = line5int[int(NN/2 - 1), xxi]

                x0, kap, mu, nu, ss = D * np.pi * k0, 0, n + 1, npr + 1, xx[xxi]
                line7int[int(NN/2 - 1), xxi], _ = integrate.quad(
                    lambda x: jv(kap, 2 * ss * x) * jv(mu, x) * jv(nu, x) / (x * (x ** 2 + x0 ** 2) ** (11 / 6)), 1E-20,
                    np.inf, limit=10**8)

            base_1s = np.zeros((NN,NN))
            base_1s[int(NN/2 - 1), :] = 1
            normalize_1s = np.zeros((NN,NN))
            for anglee in np.arange(0, 360, 0.5):
                base_rot = scipy.ndimage.rotate(base_1s, anglee, reshape=False)
                line1rot += base_rot * scipy.ndimage.rotate(line1int, anglee, reshape=False)
                line2rot += base_rot * scipy.ndimage.rotate(line2int, anglee, reshape=False)
                line3rot += base_rot * scipy.ndimage.rotate(line3int, anglee, reshape=False)
                line4rot += base_rot * scipy.ndimage.rotate(line4int, anglee, reshape=False)
                line5rot += base_rot * scipy.ndimage.rotate(line5int, anglee, reshape=False)
                line6rot += base_rot * scipy.ndimage.rotate(line6int, anglee, reshape=False)
                line7rot += base_rot * scipy.ndimage.rotate(line7int, anglee, reshape=False)
                normalize_1s += base_rot

            line1rot = mask*line1rot/normalize_1s
            line2rot = mask * line2rot / normalize_1s
            line3rot = mask * line3rot / normalize_1s
            line4rot = mask * line4rot / normalize_1s
            line5rot = mask * line5rot / normalize_1s
            line6rot = mask * line6rot / normalize_1s
            line7rot = mask * line7rot / normalize_1s
            #plt.imshow(line1rot)
            #plt.show()

            if m != 0 and mpr != 0 and np.mod(i, 2) == 0 and np.mod(j, 2) == 0:
                fij = (-1) ** ((n + npr - m + mpr) / 2) * np.cos((m + mpr) * thetamat) * line1int + \
                                (-1) ** ((n + npr + 2 * m + abs(m - mpr)) / 2) * np.cos((m - mpr) * thetamat) * line2int
            if m != 0 and mpr != 0 and np.mod(i, 2) != 0 and np.mod(j, 2) != 0:
                fij = -(-1) ** ((n + npr - m + mpr) / 2) * np.cos((m + mpr) * thetamat) * line1int + \
                                (-1) ** ((n + npr + 2 * m + abs(m - mpr)) / 2) * np.cos((m - mpr) * thetamat) * line2int
            if m != 0 and mpr != 0 and np.mod(i + j, 2) != 0:
                fij = (-1) ** ((n + npr - m + mpr) / 2) * np.sin((m + mpr) * thetamat) * line3int + \
                                (-1) ** ((n + npr + 2 * m + abs(m - mpr)) / 2) * np.sin((m - mpr) * thetamat) * line4int
            if mpr == 0 and np.mod(i, 2) == 0 and (not m == 0):
                fij = (-1) ** ((n + npr - m) / 2) * np.sqrt(2) * np.cos(m * thetamat) * line5int
            if mpr == 0 and np.mod(i, 2) != 0 and (not m == 0):
                fij = (-1) ** ((n + npr - m) / 2) * np.sqrt(2) * np.sin(m * thetamat) * line6int
            if m == 0 and np.mod(j, 2) == 0 and (not mpr == 0):
                fij = (-1) ** ((n + npr - mpr) / 2) * np.sqrt(2) * np.cos(mpr * thetamat) * line5int
            if m == 0 and np.mod(j, 2) != 0 and (not mpr == 0):
                fij = (-1) ** ((n + npr - mpr) / 2) * np.sqrt(2) * np.sin(mpr * thetamat) * line6int
            if m == 0 and mpr == 0:
                fij = (-1) ** ((n + npr) / 2) * np.sqrt(2) * line7int

            np.save('./corr_zerns_v3/Zern{}_Zern{}_v2_-5_5_150points'.format(i,j), fij)


def focusPsf(psf, thresh):
    x = np.linspace(0, 1, psf.shape[0])
    y = np.linspace(0, 1, psf.shape[1])
    col, row = np.meshgrid(x, y)
    # Don't have to normalize as psf should already be normalized
    cen_row = np.uint8(np.sum(row * psf))
    cen_col = np.uint8(np.sum(col * psf))
    temp_sum = 0
    radius = 0
    psf /= np.sum(psf)
    while temp_sum < thresh:
        radius += 1
        return_psf = psf[cen_row-radius:cen_row+radius+1, cen_col-radius:cen_col+radius+1]
        temp_sum = np.sum(return_psf)
        #print(temp_sum)
        #print(radius, temp_sum)

    return return_psf, cen_row, cen_col


def centroidPsf(psf, thresh):
    x = np.linspace(0, psf.shape[0], psf.shape[0])
    y = np.linspace(0, psf.shape[1], psf.shape[1])
    psf /= np.sum(psf)
    col, row = np.meshgrid(x, y)
    cen_row = np.uint8(np.sum(row * psf))
    cen_col = np.uint8(np.sum(col * psf))
    temp_sum = 0
    radius = 0
    while temp_sum < thresh:
        radius += 1
        return_psf = psf[cen_row-radius:cen_row+radius+1, cen_col-radius:cen_col+radius+1]
        temp_sum = np.sum(return_psf)
        #print(radius, temp_sum)

    return return_psf, cen_row, cen_col


def genZernikeCoeff(num_zern, D_r0):
    '''
    Just a simple function to generate random coefficients as needed, conforms to Zernike's Theory. The nollCovMat()
    function is at the heart of this function.

    A note about the function call of nollCovMat in this function. The input (..., 1, 1) is done for the sake of
    flexibility. One can call the function in the typical way as is stated in its description. However, for
    generality, the D/r0 weighting is pushed to the "b" random vector, as the covariance matrix is merely scaled by
    such value.

    :param num_zern: This is the number of Zernike basis functions/coefficients used. Should be numbers that the pyramid
    rows end at. For example [1, 3, 6, 10, 15, 21, 28, 36]
    :param D_r0:
    :return:
    '''
    C = nollCovMat(num_zern, 1, 1)
    e_val, e_vec = np.linalg.eig(C)
    R = np.real(e_vec * np.sqrt(e_val))

    b = np.random.randn(int(num_zern), 1) * D_r0 ** (3.0/10.0)
    a = np.matmul(R, b)

    return a

def genZernikeCoeff_chrom(num_zern, D_r0, wvls):
    '''
    Just a simple function to generate random coefficients as needed, conforms to Zernike's Theory.
    Uses smallest wavelength as the reference (D/r0 is scaled accordingly)

    :param num_zern:
    :param D_r0:
    :return:
    '''
    C = nollCovMat(num_zern, 1, 1) # The way I wrote it, has to be done :(
    e_val, e_vec = np.linalg.eig(C)
    R = np.real(e_vec * np.sqrt(e_val))
    wvl_rel = np.sort(wvls)
    out_scale = np.zeros(len(wvls))

    b = np.random.randn(int(num_zern), 1) * D_r0 ** (3.0/10.0)
    for i in range(len(wvls)):
        scale_Dr0 = ((2*np.pi/wvl_rel[0])**(6.0/5.0)) / ((2*np.pi/wvl_rel[i])**(6.0/5.0))
        out_scale[i] = ((D_r0 * scale_Dr0) ** (3.0/10.0))/(D_r0 ** (3.0/10.0))

    a = np.matmul(R, b)

    return a, out_scale


def nollCovMat(Z, D, fried):
    """
    This function generates the covariance matrix for a single point source. See the associated paper for details on
    the matrix itself.

    :param Z: Number of Zernike basis functions/coefficients, determines the size of the matrix.
    :param D: The diameter of the aperture (meters)
    :param fried: The Fried parameter value
    :return:
    """
    C = np.zeros((Z,Z))
    for i in range(Z):
        for j in range(Z):
            ni, mi = nollToZernInd(i+1)
            nj, mj = nollToZernInd(j+1)
            if (abs(mi) == abs(mj)) and (np.mod(i - j, 2) == 0):
                num = math.gamma(14.0/3.0) * math.gamma((ni + nj - 5.0/3.0)/2.0)
                den = math.gamma((-ni + nj + 17.0/3.0)/2.0) * math.gamma((ni - nj + 17.0/3.0)/2.0) * \
                      math.gamma((ni + nj + 23.0/3.0)/2.0)
                coef1 = 0.0072 * (np.pi ** (8.0/3.0)) * ((D/fried) ** (5.0/3.0)) * np.sqrt((ni + 1) * (nj + 1)) * \
                        ((-1) ** ((ni + nj - 2*abs(mi))/2.0))
                C[i, j] = coef1*num/den
            else:
                C[i, j] = 0
    C[0,0] = 1
    return C


def psfGen(N, **kwargs):
    '''
    EXAMPLE USAGE - TESTING IN MAIN
    temp, xx, yy, pupil = util.psfGen(256, pad_size=1024)
    psf = np.abs(temp) ** 2
    print(np.min(xx.ravel()), np.max(xx.ravel()))
    plt.imshow(psf / np.max(psf.ravel()), extent=[np.min(xx.ravel()), np.max(xx.ravel()),
                                                  np.min(yy.ravel()), np.max(yy.ravel())])
    plt.show()

    :param N:
    :param kwargs:
    :return:
    '''
    wavelength = kwargs.get('wavelength', 500 * (10 ** (-9)))
    pad_size = kwargs.get('pad_size', 0)
    D = kwargs.get('D', 0.1)
    L = kwargs.get('L', -1)
    z_i = kwargs.get('z_i', 1.2)
    vec = kwargs.get('coeff', np.asarray([[1], [0], [0], [0], [0], [0], [0], [0], [0]]))
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), np.linspace(-1, 1, N, endpoint=True))
    mask = np.sqrt(x_grid ** 2 + y_grid ** 2) <= 1
    zernike_stack = zernikeGen(N, vec)
    phase = np.sum(zernike_stack, axis=2)
    wave = np.exp((1j * 2 * np.pi * phase)) * mask

    pad_wave = np.pad(wave, int(pad_size/2), 'constant', constant_values=0)
    #c_psf = np.fft.fftshift(np.fft.fft2(pad_wave))
    h = np.fft.fftshift(np.fft.ifft2(pad_wave))
    #pad_wave = np.abs(pad_wave) ** 2
    # numpy.correlate(x, x, mode='same')

    #plt.imshow(phase * mask)
    #plt.show()

    M = pad_size + N
    fs = N * wavelength * z_i / D
    temp = np.linspace(-fs/2, fs/2, M)
    x_samp_grid, y_samp_grid = np.meshgrid(temp, -temp)

    if L == -1:
        return h, x_samp_grid, y_samp_grid, phase * mask,
    else:
        return h, (L/z_i)*x_samp_grid, (L/z_i)*y_samp_grid, phase * mask, wave


def zernikeGen(N, coeff, **kwargs):
    # Generating the Zernike Phase representation.
    #
    # This implementation uses Noll's indices. 1 -> (0,0), 2 -> (1,1), 3 -> (1, -1), 4 -> (2,0), 5 -> (2, -2), etc.

    num_coeff = coeff.size
    #print(num_coeff)

    # Setting up 2D grid
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), np.linspace(-1, 1, N, endpoint=True))
    #mask = np.sqrt(x_grid **2 + y_grid ** 2) <= 1
    #x_grid = x_grid * mask
    #y_grid = y_grid * mask

    zern_out = np.zeros((N,N,num_coeff))
    for i in range(num_coeff):
        zern_out[:,:,i] = coeff[i]*genZernPoly(i+1, x_grid, y_grid)

    return zern_out


def nollToZernInd(j):
    """
    This function maps the input "j" to the (row, column) of the Zernike pyramid using the Noll numbering scheme.

    Authors: Tim van Werkhoven, Jason Saredy
    See: https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
    """
    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")
    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))

    return n, m


def genZernPoly(index, x_grid, y_grid):
    """
    This function simply

    :param index:
    :param x_grid:
    :param y_grid:
    :return:
    """
    n,m = nollToZernInd(index)
    radial = radialZernike(x_grid, y_grid, (n,m))
    #print(n,m)
    if m < 0:
        return np.multiply(radial, np.sin(-m * np.arctan2(y_grid, x_grid)))
    else:
        return np.multiply(radial, np.cos(m * np.arctan2(y_grid, x_grid)))


def radialZernike(x_grid, y_grid, z_ind):
    rho = np.sqrt(x_grid ** 2 + y_grid ** 2)
    radial = np.zeros(rho.shape)
    n = z_ind[0]
    m = np.abs(z_ind[1])

    for k in range(int((n - m)/2 + 1)):
        #print(k)
        temp = (-1) ** k * np.math.factorial(n - k) / (np.math.factorial(k) * np.math.factorial((n + m)/2 - k)
                                                       * np.math.factorial((n - m)/2 - k))
        radial += temp * rho ** (n - 2*k)

    # radial = rho ** np.reshape(np.asarray([range(int((n - m)/2 + 1))]), (int((n - m)/2 + 1), 1, 1))

    return radial
