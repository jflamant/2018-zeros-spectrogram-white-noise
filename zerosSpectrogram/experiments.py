import numpy as np
import scipy.signal as sg

from .utils import *
from .spatialstats import *

# experiment definition
def demoSpectrogramWhiteNoise(base, mode='complex'):
    N = 4 * base
    Nfft = 2 * base
    t = np.arange(N)

    # Noise only
    if mode == 'complex':
        w = np.random.randn(N)+1j*np.random.randn(N)
    elif mode == 'real':
        w = np.random.randn(N)
    else:
        return ValueError('mode should be either real or complex')
    # window
    g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
    g = g/g.sum()
    _, _, stft = sg.stft(w, window=g, nperseg=Nfft, noverlap=Nfft-1, return_onesided=False)
    Sww = np.abs(stft)**2
    print("STFT computed")
    # detection
    begin = base
    block = base
    x = []
    y = []
    th = 1e-14
    print("searching for zeros...")
    while begin < N-base:

        y_e, x_e = extr2minth(Sww[:, begin:begin+block], th) # on the log for greater precision?

        x = x + list(x_e+begin)
        y = y + list(y_e)
        begin += block
    print('finished')
    #imshow(log(Sww[:, base:N-base]), cmap='viridis')
    #scatter(np.array(x)-base, np.array(y), color='red', marker='s')

    u = (np.array(x)-base)/np.sqrt(2*base)
    v = (np.array(y))/np.sqrt(2*base)

    pos = np.zeros((len(x), 2))
    pos[:, 0] = u
    pos[:, 1] = v

    return pos, [Sww, x, y]
