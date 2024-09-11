

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





# This function
# pupil: N * N binary matrix showing the pupil region.
# probes: M * N * N tensor showing extra wavefronts introduced by different defocus magnitudes.
# wavefront: N * N matrix showing the extra wavefront introduced by the mirror.
def forward_model(pupil, probes, wavefront):
    xp = get_array_module(wavefront) # get the hardware used for data computation
    pupils = pupil * wavefront * probes # layer-by-layer multiplication which obtains an M * N * N tensor showing the OPD at the pupil
    pupils /= xp.mean(xp.abs(pupils),axis=(-2,-1))[:,None,None] # divide pupil data by the mean of each layer for normalization
    Efocals = fft2_shiftnorm(pupils, axes=(-2,-1)) # k simultaneous FFTs
    Ifocals = xp.abs(Efocals)**2
    return Ifocals, Efocals, pupils



def fft2_shiftnorm(image, axes=None, norm='ortho', shift=True):
    if axes is None:
        axes = (-2, -1)
    if isinstance(image, np.ndarray): # CPU or GPU
        xp = np
    else:
        xp = cp
    if shift:
        shiftfunc = xp.fft.fftshift
        ishiftfunc = xp.fft.ifftshift
    else:
        shiftfunc = ishiftfunc = lambda x, axes=None: x
    if isinstance(image, np.ndarray):
        t = np.fft.fft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm) #pyfftw.builders.fft
        return shiftfunc(t,axes=axes)
    else:
        return shiftfunc(cp.fft.fft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm), axes=axes)



