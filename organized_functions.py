

# This function is to get the type of the hardware used by computation.
def get_array_module(arr):
    if cp is not None: # determine if the system imports 'cupy'
        return cp.get_array_module(arr) # use 'get_array_module' in 'cupy' to check if 'arr' is on CPU (np) or GPU (cp)
    else:
        return np

# This function is used to get the extra wavefront at the pupil introduced by the defocus.
# fitmask: binary matrix showing the pupil region.
# vals_waves: 1 x N vector showing defocus magnitudes in waves.
def get_defocus_probes(fitmask, vals_waves):
    zmodes = zernike.arbitrary_basis(fitmask, nterms=4, outside=0) # give the bases of the first 4 zernike polynomials inside 'fitmask', and set the data outside the mask to be 0
    return cp.exp(1j*zmodes[-1]*2*cp.pi*cp.asarray(vals_waves)[:,None,None]) # the map having data equal to e^(j2πn*defocus_phase)

# This function is used to calculate the wavefront and intensity maps on the image plane.
# pupil: N * N binary matrix showing the pupil region.
# probes: M * N * N tensor showing extra wavefronts introduced by different defocus magnitudes.
# wavefront: N * N matrix showing the extra wavefront introduced by the mirror shape error.
def forward_model(pupil, probes, wavefront):
    xp = get_array_module(wavefront) # get the hardware used for data computation
    Epupils = pupil * wavefront * probes # layer-by-layer multiplication which obtains an M * N * N tensor showing the OPD at the pupil
    Epupils /= xp.mean(xp.abs(Epupils),axis=(-2,-1))[:,None,None] # divide pupil data by the mean of each layer for normalization
    Efocals = fft2_shiftnorm(Epupils, axes=(-2,-1)) # wavefront at the image plane
    Ifocals = xp.abs(Efocals)**2 # intensity at the image plane
    return Ifocals, Efocals, Epupils

# This function is used to do Fast Fourier Transform for the image.
# image: it's usually the wavefront at the pupil.
# axes: the dimensions that implement FFT.
# norm: normalization method. The default choice is orthogonal.
# shift: indicate if the image needs to be shifted for the FFT.
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




# Imeas: intensity maps on the image planes.
# fitmask: the mask at the pupil plane. It indicates the pupil region.
# init_params: initial values of the to-be-fitted coefficients.
# modes: there are 2 data fitting modes. The first one is to fit the wavefront pixel by pixel; the second one is to use zernike polynomials to fit the wavefront layer by layer. It should give either 'None' or bases of zernike polynomials.
def run_phase_retrieval(Imeas, fitmask, tol, reg, wreg, Eprobes, init_params=None, bounds=True, modes=None, fit_amp=True):
    xp = get_array_module(Imeas) # get the type of the hardware used by computation
    if modes is None: # determine the method used to fit the wavefront at the pupil
        N = np.count_nonzero(fitmask) # give pixel numbers of data fitting
    else:
        N = len(modes) # give layer numbers for fitting data
    # Initialize to-be-fitted coefficients if they were not given.
    if init_params is None:
        if modes is None:
            fitsmooth = gauss_convolve(binary_erosion(fitmask, iterations=3), 3) # blur the mask after implementing image erosion. It simulates the actual pupil intensity.
            init_params = np.concatenate([fitsmooth[fitmask], fitsmooth[fitmask]*0], axis=0) # extract pupil intensity points and give the same number of 0 phase points
        else:
            amp0 = np.zeros(len(modes)) # intensity
            amp0[0] = 1 # piston
            ph0 = np.zeros(len(modes)) # 0 phase
            init_params = np.concatenate([amp0, ph0], axis=0) # 1D intensity + phase


    # compute weights?
    weights = 1/(Imeas + wreg) * get_han2d_sq(Imeas[0].shape[0], fraction=0.7)
    weights /= np.max(weights,axis=(-2,-1))[:,None,None]

    if bounds:
        bounds = [(0,None),]*N + [(None,None),]*N
    else:
        bounds = None

    # get probes

    # force all to right kind of array (numpy or cupy)
    Eprobes = xp.asarray(Eprobes, dtype=xp.complex128)
    Imeas = xp.asarray(Imeas, dtype=xp.float64)
    weights = xp.asarray(weights, dtype=xp.float64)
    fitmask_cp = xp.asarray(fitmask)
    if modes is not None:
        modes_cp = xp.asarray(modes)
    else:
        modes_cp = None

    if not fit_amp:
        init_params = init_params[N:]
        bounds = bounds[N:]

    errfunc = get_sqerr_grad
    fitdict = minimize(errfunc, init_params, args=(fitmask_cp, fitmask_cp,
                        Eprobes, weights, Imeas, N, reg, modes_cp, fit_amp),
                        method='L-BFGS-B', jac=True, bounds=bounds,
                        tol=tol, options={'ftol' : tol, 'gtol' : tol, 'maxls' : 100})

    # construct amplitude and phase
    phase_est = np.zeros(fitmask.shape)
    amp_est = np.zeros(fitmask.shape)

    if fit_amp:
        if modes is None:
            phase_est[fitmask] = fitdict['x'][N:]
            amp_est[fitmask] = fitdict['x'][:N]
        else:
            phase_est = np.sum(fitdict['x'][N:,None,None] * modes, axis=0)
            amp_est = np.sum(fitdict['x'][:N,None,None] * modes, axis=0)
    else:
        if modes is None:
            phase_est[fitmask] = fitdict['x'][:N]
            amp_est = None
        else:
            phase_est = np.sum(fitdict['x'][:N,None,None] * modes, axis=0)
            amp_est = None

    return {
        'phase_est' : phase_est,
        'amp_est' : amp_est,
        'obj_val' : fitdict['fun'],
        'fit_params' : fitdict['x']
    }



# This function implements gaussian blur in frequency domain.
# image: the image that is going to be blurred.
# sigma: indicate the gaussian distribution width for the image blur.
# force_real: determine if we need to remove the imaginary part of the obtained blurred image.
def gauss_convolve(image, sigma, force_real=True):
    if cp is not None: # determine the hardware used for calculation
        xp = cp.get_array_module(image)
    else:
        xp = np
    g = get_gauss(sigma, image.shape, xp=xp) # get the gaussian kernel
    return convolve_fft(image, g, force_real=force_real) # blur the image in frequency domain and keep the real part

# This function generates the gaussian kernel used for image blur.
# sigma: standard deviation (width) of the gaussian distribution.
# shape: (rows, cols) showing numbers of rows and columns of the kernel.
# cenyx: [y x] showing central coordinates of the gaussian distribution in the kernel.
# xp: numpy or cupy that is used for calculation.
def get_gauss(sigma, shape, cenyx=None, xp=np):
    if cenyx is None:
        cenyx = xp.asarray([(shape[0])/2., (shape[1])/2.]) # central coordinates of the kernel
    yy, xx = xp.indices(shape).astype(float) - cenyx[:,None,None] # 'xp.indices().astype()' generates 2 arrays showing y and x coordinates in specific data type
    g = xp.exp(-0.5*(yy**2+xx**2)/sigma**2) # gaussian distribution in the kernel: e^(-(x^2+y^2))/(2*sigma^2))
    return g / xp.sum(g) # return the normalized gaussian kernel

# This function is used to implement gaussian blur in frequency domain.
# in1/in2: to-be-blurred image and gaussian kernel. They should have the same size.
# force_real: determine if we need to remove the imaginary number.
def convolve_fft(in1, in2, force_real=False):
    out = ifft2_shiftnorm(fft2_shiftnorm(in1,norm=None)*fft2_shiftnorm(in2,norm=None),norm=None) # gaussian blur in frequency domain
    if force_real:
        return out.real # just keep the real number
    else:
        return out

# This function is used to do inverse Fast Fourier Transform for the image.
# image: to-be-transformed image.
# axes: the dimensions that implement FFT.
# norm: normalization method. The default choice is orthogonal normalization.
# shift: indicate if the image needs to be shifted for the FFT.
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





def get_han2d_sq(N, fraction=1. / np.sqrt(2), normalize=False):

    '''
    Radial Hanning window scaled to a fraction
    of the array size.
    Fraction = 1. for circumscribed circle
    Fraction = 1/sqrt(2) for inscribed circle (default)
    '''

    x = xp.linspace(-N / 2., N / 2., num=N)
    rmax = N * fraction
    scaled = (1 - x / rmax) * xp.pi / 2.
    window = xp.sin(scaled) ** 2
    window[xp.abs(x) > rmax] = 0
    return xp.outer(window, window)


get_han2d_sq(Imeas[0].shape[0], fraction=0.7)