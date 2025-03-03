"The functions in this file all come from 'phase_retrieval.py'."


import numpy as np
import cupy as cp  # no GPU, so it's been changed from 'import cupy as cp'
import matplotlib.pyplot as plt

import hcipy as hp
import poppy as pp
# import matlab.engine as mat
from scipy.ndimage import gaussian_filter
from skimage.transform import resize, downscale_local_mean
from skimage.filters import threshold_otsu
from copy import deepcopy

from functools import partial
import multiprocessing as mp
from time import sleep

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fdpr2')


from scipy.optimize import minimize
from scipy.ndimage import binary_erosion #, shift, center_of_mass
from poppy import zernike




# This function is to get the type of the hardware used by computation.
def get_array_module(arr):
    if cp is not None: # determine if the system imports 'cupy'
        return cp.get_array_module(arr) # use 'get_array_module' in 'cupy' to check if 'arr' is on CPU (np) or GPU (cp)
    else:
        return np

# This function is used to get the extra wavefront at the pupil introduced by the defocus.
# fitmask: P * P binary matrix showing the pupil region.
# vals_waves: 1 x N vector showing defocus magnitudes in waves.
def get_defocus_probes(fitmask, vals_waves):
    zmodes = zernike.arbitrary_basis(fitmask, nterms=4, outside=0) # give the bases of the first 4 zernike polynomials inside 'fitmask', and set the data outside the mask to be 0
    return cp.exp(1j*zmodes[-1]*2*cp.pi*cp.asarray(vals_waves)[:,None,None]) # the map having data equal to e^(j2πn*defocus_phase)

# This function is used to estimate the wavefront error at the pupil plane.
# Imeas: x * H * H measured intensity maps on the image planes (PSFs).
# fitmask: P * P matrix representing the mask at the pupil plane. It indicates the pupil region.
# tol: the parameter showing the tolerance used to terminate the iteration.
# reg: the constant controlling the strength of regularization penalty to avoid overfitting in error and gradient.
# wreg: the parameter used to prevent infinite weights calculated from the reciprocal of 0 intensity.
# Eprobes: x * P * P tensor showing the extra wavefront at the pupil introduced by the defocus.
# init_params: initial values of the to-be-fitted parameters. It's a vector. Its length is either the pixel number of the pupil region or the number of Zernike terms.
# bounds: indicate if we need to set boundaries for fitting coefficients. It should be either 'True' or 'False'.
# modes: there are 2 data fitting modes. The first one is to fit the wavefront pixel by pixel; the second one is to use zernike polynomials to fit the wavefront layer by layer. It should give either 'None' or bases of zernike polynomials.
# fit_amp: indicate if we need to fit the amplitude. It should be either 'True' or 'False'.
def run_phase_retrieval(Imeas, fitmask, tol, reg, wreg, Eprobes, init_params=None, bounds=True, modes=None, fit_amp=True):
    xp = get_array_module(Imeas) # get the type of the hardware used by computation
    if modes is None: # determine the method used to fit the wavefront at the pupil
        N = np.count_nonzero(fitmask) # give pixel numbers of data fitting
    else:
        N = len(modes) # give layer numbers for fitting data
    # Initialize to-be-fitted coefficients if they were not given.
    if init_params is None:
        if modes is None:
            fitsmooth = gauss_convolve(binary_erosion(fitmask, iterations=3), 3) # blur the mask after implementing image erosion. It simulates the actual pupil amplitude.
            init_params = np.concatenate([fitsmooth[fitmask], fitsmooth[fitmask]*0], axis=0) # extract pupil amplitude ('True') points row by row and splice them into a 1D array, and give the same number of 0 phase points
        else:
            amp0 = np.zeros(len(modes)) # amplitude
            amp0[0] = 1 # piston
            ph0 = np.zeros(len(modes)) # 0 phase
            init_params = np.concatenate([amp0, ph0], axis=0) # 1D amplitude + phase
    # Give weights for fitting data. It's an x * H * H tensor.
    weights = 1/(Imeas + wreg) * get_han2d_sq(Imeas[0].shape[0], fraction=0.7, xp=xp) # the reciprocal of the amplitude weakens the impact of the central region, while the Hanning window restricts the scope and adjusts the weights
    weights /= np.max(weights,axis=(-2,-1))[:,None,None] # normalization
    # Give fitting boundaries for the to-be-fitted coefficients.
    if bounds:
        bounds = [(0,None),]*N + [(None,None),]*N # 0 is the lower boundary for the amplitude
    else:
        bounds = None
    # Force all previous data to right kinds of arrays.
    Eprobes = xp.asarray(Eprobes, dtype=xp.complex128)
    Imeas = xp.asarray(Imeas, dtype=xp.float64)
    weights = xp.asarray(weights, dtype=xp.float64)
    fitmask_cp = xp.asarray(fitmask)
    if modes is not None:
        modes_cp = xp.asarray(modes)
    else:
        modes_cp = None
    if not fit_amp:
        init_params = init_params[N:] # just keep the phase part
        bounds = bounds[N:]

    # Fit the parameters used to constitute the wavefront.
    errfunc = get_sqerr_grad # get the error function
    # This is used to minimize the output parameters (error and gradient) of the target function 'errfunc'.
    # 'init_params' includes the parameters used in calculation of 'errfunc' in each iteration.
    # 'args' includes all other input parameters required by 'errfunc'.
    # 'method' shows the iteration method 'L-BFGS-B'. It supports the boundary constraints.
    # 'jac' equals to 'True' means 'errfunc' will provide gradient in addition to error.
    # 'bounds' shows the boundary constraints of 'init_params'.
    # 'tol' is the tolerance. When the error is smaller than 'tol', the iteration will terminate.
    # 'options' gives extra tolerances for terminating the iteration. 'ftol': tolerance of the error difference in 2 adjacent iterations; 'gtol': tolerance of the norm of the gradient; 'maxls': maximum iteration times.
    fitdict = minimize(errfunc, init_params, args=(fitmask_cp, fitmask_cp,
                        Eprobes, weights, Imeas, N, reg, modes_cp, fit_amp),
                        method='L-BFGS-B', jac=True, bounds=bounds,
                        tol=tol, options={'ftol' : tol, 'gtol' : tol, 'maxls' : 100})
    fitdict['x'] = cp.asarray(fitdict['x']) # optimized coefficients of the wavefront

    # Sort the output data.
    phase_est = cp.zeros(fitmask.shape)
    amp_est = cp.zeros(fitmask.shape)
    if fit_amp: # amplitude and phase
        if modes is None: # parameters of pixels
            phase_est[fitmask] = fitdict['x'][N:] # phase map (from [N, 2N-1] of fitdict['x'])
            amp_est[fitmask] = fitdict['x'][:N] # amplitude map (from [0, N-1] of fitdict['x'])
        else: # parameters of Zernike polynomials
            phase_est = cp.sum(fitdict['x'][N:,None,None] * modes, axis=0) # phase map (from Zernike polynomials * [N, 2N-1] of fitdict['x'])
            amp_est = cp.sum(fitdict['x'][:N,None,None] * modes, axis=0) # amplitude map (from Zernike polynomials * [0, N-1] of fitdict['x'])
    else: # only phase
        if modes is None:
            phase_est[fitmask] = fitdict['x'][:N]
            amp_est = None
        else:
            phase_est = cp.sum(fitdict['x'][:N,None,None] * modes, axis=0)
            amp_est = None
    return {
        'phase_est' : phase_est, # phase map
        'amp_est' : amp_est, # amplitude map
        'obj_val' : fitdict['fun'], # output of the error function
        'fit_params' : fitdict['x'] # fitted parameters
    }

# This function implements gaussian blur in frequency domain.
# image: the P * P image that is going to be blurred.
# sigma: the parameter showing the gaussian distribution width for the image blur.
# force_real: determine if we need to remove the imaginary part of the obtained blurred image. The default is 'True'.
def gauss_convolve(image, sigma, force_real=True):
    if cp is not None: # determine the hardware used for calculation
        xp = cp.get_array_module(image)
    else:
        xp = np
    g = get_gauss(sigma, image.shape, xp=xp) # get the gaussian kernel
    return convolve_fft(image, g, force_real=force_real) # blur the image in frequency domain and keep the real part

# This function generates the gaussian kernel used for image blur.
# sigma: the parameter showing the standard deviation (width) of the gaussian distribution.
# shape: (rows, cols) showing numbers of rows and columns of the kernel.
# cenyx: [y x] showing central coordinates of the gaussian distribution in the kernel.
# xp: numpy or cupy that is used for calculation. The default is 'np'.
def get_gauss(sigma, shape, cenyx=None, xp=cp):
    if cenyx is None:
        cenyx = xp.asarray([(shape[0])/2., (shape[1])/2.]) # central coordinates of the kernel
    yy, xx = xp.indices(shape).astype(float) - cenyx[:,None,None] # 'xp.indices().astype()' generates 2 arrays showing y and x coordinates in specific data type
    g = xp.exp(-0.5*(yy**2+xx**2)/sigma**2) # gaussian distribution in the kernel: e^(-(x^2+y^2))/(2*sigma^2))
    return g / xp.sum(g) # return the normalized gaussian kernel

# This function is used to implement gaussian blur in frequency domain.
# in1/in2: P * P to-be-blurred image and gaussian kernel. They should have the same size.
# force_real: determine if we need to remove the imaginary number. The default is 'False'.
def convolve_fft(in1, in2, force_real=False):
    out = ifft2_shiftnorm(fft2_shiftnorm(in1,norm=None)*fft2_shiftnorm(in2,norm=None),norm=None) # gaussian blur in frequency domain
    if force_real:
        return out.real # just keep the real number
    else:
        return out

# This function is used to do inverse Fast Fourier Transform for the image.
# image: P * P to-be-transformed image.
# axes: the dimensions that implement FFT. Default choice 'None' equals to (-2, -1).
# norm: normalization method. The default choice is orthogonal normalization 'ortho'.
# shift: indicate if the image needs to be shifted for the FFT. The default is 'True'.
def ifft2_shiftnorm(image, axes=None, norm='ortho', shift=True):
    if axes is None:
        axes = (-2, -1) # the last 2 dimensions
    if isinstance(image, np.ndarray): # check if 'image' is numpy array
        xp = np
    else:
        xp = cp
    if shift: # check if it requires shifting 'image'
        shiftfunc = xp.fft.fftshift # centralize the image
        ishiftfunc = xp.fft.ifftshift # decentralize the image
    else:
        shiftfunc = ishiftfunc = lambda x, axes=None: x # use 'lambda' to create a function who has 2 input parameters: x, axes (default=None), the output is 'x'
    if isinstance(image, np.ndarray): # implement IFFT according to the data type of 'image'
        t = np.fft.ifft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm)
        return shiftfunc(t, axes=axes)
    else:
        t = cp.fft.ifft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm)
        return shiftfunc(t, axes=axes)

# This function gives a Hanning window for smoothing the image.
# N: side length of the mask.
# fraction: the coverage ratio of the Hanning window in a single direction. 1 for the inscribed circle (default); sqrt(2) for the circumscribed circle.
# xp: the type of the hardware used by computation, which should be 'np' or 'cp'.
def get_han2d_sq(N, fraction=1., xp=np):
    x = xp.linspace(-N / 2., N / 2., num=N)
    rmax = N / 2. * fraction # radius of the Hanning window
    scaled = (1 - x / rmax) * xp.pi / 2.
    window = xp.sin(scaled) ** 2 # 1D
    window[xp.abs(x) > rmax] = 0
    return xp.outer(window, window) # 2D

# This function is used to calculate the error and gradient for the iteration.
# params: current values of the to-be-fitted parameters. It's a vector. Its length is either the pixel number of the pupil region or the number of Zernike terms.
# pupil: P * P binary matrix showing the pupil region.
# mask: P * P binary mask showing the pupil region.
# Eprobes: x * P * P tensor showing extra wavefronts introduced by different defocus magnitudes.
# weights: x * H * H tensor determining the influences of different components in comparing the differences between the model result and measured result.
# Imeas: x * H * H measured intensity maps on the image planes (PSFs).
# N: number of to-be-fitted parameters.
# lambdap: the constant controlling the strength of regularization penalty to avoid overfitting in error and gradient.
# modes: there are 2 data fitting modes. The first one is to fit the wavefront pixel by pixel; the second one is to use zernike polynomials to fit the wavefront layer by layer. It should give either 'None' or bases of zernike polynomials.
# fit_amp: indicate if we need to fit the amplitude. It should be either 'True' or 'False'.
def get_sqerr_grad(params, pupil, mask, Eprobes, weights, Imeas, N, lambdap, modes, fit_amp):
    # Use current values of the to-be-fitted parameters to generate the wavefront at the pupil.
    xp = get_array_module(Eprobes) # get the type of the hardware used by computation
    if xp is cp and isinstance(params, np.ndarray): # if there's GPU and the to-be-fitted coefficients are saved in numpy array
        params = cp.array(params) # convert coefficients to be in cupy array
    if fit_amp: # if we need to fit the amplitude
        params_amp = params[:N] # split the data
        params_phase = params[N:]
    else:
        params_amp = np.ones(N) # if not, just set the amplitudes to constants
        params_phase = params[:N]
    A = xp.zeros(mask.shape) # amplitude map
    phi = xp.zeros(mask.shape) # phase map
    if modes is None: # parameters fitted pixel by pixel
        A[mask] = params_amp # give the non-zero points of 'mask' in 'A' the values of 'params_amp' row by row
        phi[mask] = params_phase # do the same thing to phase 'phi'
    else: # parameters fitted by Zernike polynomials
        if fit_amp:
            A = xp.sum(modes * params_amp[:, None, None], axis=0) # use 'params_amp' and Zernike polynomials in 'modes' to calculate 'A'
        else:
            A = pupil.astype(float) # convert 'pupil' to float data and give it to 'A'
        phi = xp.sum(modes * params_phase[:, None, None], axis=0) # do the same thing to phase 'phi'
    Eab = A * np.exp(1j * phi) # mirror shape error (wavefront) predicted by the last iteration
    # Use the forward model, pupil wavefront and defocus wavefronts to calculate the PSFs on the image planes.
    Imodel, Efocals, Epupils = forward_model(pupil, Eprobes, Eab)

    # Calculate the error. 'lambdap * xp.sum(params ** 2)' is the L2 regularization term working for error, which penalizes the large parameter values, effectively reducing the overfitting risk. 'lambdap' controls the strength of regularization penalty.
    err = get_err(Imeas, Imodel, weights) + lambdap * xp.sum(params ** 2)

    # Update to-be-fitted parameters using gradient for the next iteration. '2 * lambdap * params' is the L2 regularization term working for gradient, which penalizes the large parameter values, effectively reducing the overfitting risk.
    if fit_amp: # fit amplitude and phase
        gradA, gradphi = get_grad(Imeas, Imodel, Efocals, Eprobes, A, phi, weights, fit_amp=True)
        if modes is None: # pixel fitting mode
            grad_Aphi = xp.concatenate([gradA[mask], gradphi[mask]], axis=0) + 2 * lambdap * params # update 'params'
        else: # Zernike coefficient fitting mode
            gradAmodal = xp.sum(gradA * modes, axis=(-2, -1))
            gradphimodal = xp.sum(gradphi * modes, axis=(-2, -1))
            grad_Aphi = xp.concatenate([gradAmodal, gradphimodal], axis=0) + 2 * lambdap * params  # update 'params'
    else: # only fit phase
        gradphi = get_grad(Imeas, Imodel, Efocals, Eprobes, A, phi, weights, fit_amp=False)
        if modes is None:
            grad_Aphi = gradphi[mask] + 2 * lambdap * params
        else:
            gradphimodal = xp.sum(gradphi * modes, axis=(-2, -1))
            grad_Aphi = gradphimodal + 2 * lambdap * params
    return err, grad_Aphi

# This function is used to calculate the wavefront and intensity maps on the image plane.
# pupil: P * P binary matrix showing the pupil region.
# probes: x * P * P tensor showing extra wavefronts introduced by different defocus magnitudes.
# wavefront: P * P matrix showing the wavefront introduced by the mirror shape error.
def forward_model(pupil, probes, wavefront):
    xp = get_array_module(wavefront) # get the hardware used for data computation
    Epupils = pupil * wavefront * probes # layer-by-layer multiplication which obtains an x * P * P tensor showing the OPD at the pupil
    Epupils /= xp.mean(xp.abs(Epupils),axis=(-2,-1))[:,None,None] # divide pupil data by the mean of each layer for normalization
    Efocals = fft2_shiftnorm(Epupils, axes=(-2,-1)) # wavefront at the image plane
    Ifocals = xp.abs(Efocals)**2 # intensity at the image plane
    return Ifocals, Efocals, Epupils

# This function is used to do Fast Fourier Transform for the image.
# image: it's usually a P * P matrix showing the wavefront at the pupil.
# axes: the dimensions that implement FFT. Default choice 'None' equals to (-2, -1).
# norm: normalization method. The default choice is orthogonal. It is 'ortho'.
# shift: indicate if the image needs to be shifted for the FFT. The default is 'True'.
def fft2_shiftnorm(image, axes=None, norm='ortho', shift=True):
    if axes is None:
        axes = (-2, -1) # the last 2 dimensions
    if isinstance(image, np.ndarray): # check if 'image' is numpy array
        xp = np
    else:
        xp = cp
    if shift: # check if it requires shifting 'image'
        shiftfunc = xp.fft.fftshift # centralize the image
        ishiftfunc = xp.fft.ifftshift # decentralize the image
    else:
        shiftfunc = ishiftfunc = lambda x, axes=None: x # use 'lambda' to create a function who has 2 input parameters: x, axes (default=None), the output is 'x'
    if isinstance(image, np.ndarray): # implement FFT according to the data type of 'image'
        t = np.fft.fft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm)
        return shiftfunc(t,axes=axes)
    else:
        t = cp.fft.fft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm)
        return shiftfunc(t, axes=axes)




# !!!
# This function is used to calculate the error between the predicted PSFs and measured PSFs. It comes from Eq. (12) of paper 'Amplitude metrics for field retrieval with hard-edged and uniformly illuminated apertures'.
# Imeas: x * H * H measured PSFs.
# Imodel: x * H * H predicted PSFs from the forward model.
# weights: x * H * H tensor determining the influences of different components in comparing the differences between the model result and measured result.
def get_err(Imeas, Imodel, weights):
    xp = get_array_module(Imeas)
    K = len(weights)
    t1 = xp.sum(weights * Imodel * Imeas, axis=(-2,-1))**2
    t2 = xp.sum(weights * Imeas**2, axis=(-2,-1))
    t3 = xp.sum(weights * Imodel**2, axis=(-2,-1))
    return 1 - 1/K * np.sum(t1/(t2*t3), axis=0) # Eq. (12)

# !!!
# This function is used to calculate the gradient of the error and give the updated parameters for the following iteration. The primary component comes from Eq. (A1) of paper 'Amplitude metrics for field retrieval with hard-edged and uniformly illuminated apertures'.
# Imeas: x * H * H measured PSFs.
# Imodel: x * H * H predicted PSFs (intensity of the wavefront) from the forward model.
# Efocals: x * H * H predicted wavefront from the forward model.
# Eprobes: x * P * P tensor showing extra wavefronts introduced by different defocus magnitudes.
# A: x * P * P tensor showing the amplitude maps predicted by the last iteration.
# phi: x * P * P tensor showing the phase maps predicted by the last iteration.
# weights: x * H * H tensor determining the influences of different components in comparing the differences between the model result and measured result.
# fit_amp: indicate if we need to fit the amplitude. It should be either 'True' or 'False'.
def get_grad(Imeas, Imodel, Efocals, Eprobes, A, phi, weights, fit_amp=True):
    xp = get_array_module(Imeas)

    # common gradient terms
    Ibar = get_Ibar_model(Imeas, Imodel, weights)
    Ehatbar = 2 * Efocals * Ibar
    Ebar = ifft2_shiftnorm(Ehatbar, axes=(-2, -1))

    # --- get Eab ---
    Eabbar = Ebar * Eprobes.conj()
    # get amplitude
    expiphi = np.exp(1j * phi)
    Abar = Eabbar * expiphi.conj()
    # get phase
    expiphibar = Eabbar * A
    phibar = xp.imag(expiphibar * expiphi.conj())

    # sum terms (better be purely real, should double check this!!!)
    gradA = xp.sum(Abar, axis=0).real
    gradphi = xp.sum(phibar, axis=0).real

    if fit_amp:
        return gradA, gradphi
    else:
        return gradphi

# !!!
# This function is used to calculate the gradient of the error. It comes from Eq. (A1) of paper 'Amplitude metrics for field retrieval with hard-edged and uniformly illuminated apertures'.
# Imeas: x * H * H measured PSFs.
# Imodel: x * H * H predicted PSFs from the forward model.
# weights: x * H * H tensor determining the influences of different components in comparing the differences between the model result and measured result.
def get_Ibar_model(Imeas, Imodel, weights):
    xp = get_array_module(Imeas)
    K = len(weights)
    t1 = xp.sum(weights * Imodel * Imeas, axis=(-2,-1))[:,None,None]
    t2 = xp.sum(weights * Imeas**2, axis=(-2,-1))[:,None,None]
    t3 = xp.sum(weights * Imodel**2, axis=(-2,-1))[:,None,None]
    return 2/K * weights * t1 / (t2 * t3**2) * (Imodel * t1 - Imeas * t3) # Eq. (A1)

