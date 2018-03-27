import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import cmocean
import seaborn as sns

from .utils import *
from .spatialstats import *
from mpl_toolkits.axes_grid.inset_locator import inset_axes

sns.set(style="ticks", color_codes=True)

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
        raise ValueError('mode should be either real or complex')
    # window
    g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
    g = g/g.sum()
    _, _, stft = sg.stft(w, window=g, nperseg=Nfft, noverlap=Nfft-1, return_onesided=False)

    Sww_t = np.abs(stft)**2
    print("STFT computed")

    tmin = base
    tmax = 3*base

    Sww = Sww_t[:, tmin:tmax+1]

    # detection
    th = 1e-14
    y, x = extr2minth(Sww, th)

    u = (np.array(x))/np.sqrt(2*base)
    v = (np.array(y))/np.sqrt(2*base)

    pos = np.zeros((len(x), 2))
    pos[:, 0] = u
    pos[:, 1] = v

    return pos, [Sww, x, y]



def rankEnvelopePowerSNR(amp, spread, folder='../data/', base='128', MCH0=199, MCsignal=200, method_list=['F', 'L'], k=10, inset_pos=[0.5, 0.5, 0.4, 0.4]):

    SNR = amp # SNR here corresponds to the amplitude of the signal wrt to white noise
    # error bars
    error_ind = np.arange(1024)[::300]+100

    # colors
    c = ['b', 'g']


    fig, ax = plt.subplots(figsize=(4, 4)) # create figure

    for m, method in enumerate(method_list):
        names = folder+'spatialStats/base'+str(base)+'MCH0'+str(MCH0)+'MCsignal'+str(MCsignal)+'SNR'+str(SNR)+'spread'+str(spread)+'method'+method

        radius = np.load(names+'_radius.npy')
        H = np.load(names+'.npy')

        # compute mean

        H_0_est = np.mean(H, axis=1)

        # step 2: compute integrated errors


        # loop on experiments
        t2_exp = np.zeros((MCsignal, len(radius)))

        for nexp in range(MCsignal):

            # compute H0 statistics
            t2 = np.zeros((MCH0, len(radius)))

            for i in range(1, MCH0):

                t2[i, :] = np.cumsum((H[nexp, i, :] - H_0_est[nexp, :])**2)

            # sorting
            t2 = np.sort(t2, axis=0)[::-1, :]

            # generate data from experiment
            t2_exp[nexp, :] = np.cumsum((H[nexp, 0, :] - H_0_est[nexp, :])**2)

        mask = np.zeros((MCsignal, len(radius)), dtype=bool)

        for nexp in range(MCsignal):
            mask[nexp, :] = t2_exp[nexp, :] > t2[k, :]

        beta = np.mean(mask, axis=0)

        # plot
        l, = ax.plot(radius, beta, alpha=0.8, color=c[m])
        label_line(l, r'$'+method+'$', near_x=3.5, offset=(0.25, 0))

        # compute errors bars
        error_r = radius[error_ind]
        k1 = (np.array(beta[error_ind]*MCsignal)).astype(int)

        error_beta = np.array(clopper_pearson(k1, MCsignal, 0.05/2/len(error_ind)))

        yerr1 = np.abs(error_beta - beta[error_ind])
        yerr1[np.isnan(yerr1)] = 0

        (_, caps, _) = ax.errorbar(error_r, beta[error_ind], yerr=yerr1, fmt='.',color=c[m], markersize=8, elinewidth=1, capsize=3)
        for cap in caps:
            cap.set_markeredgewidth(1)



    # load examples exp
    names2 = folder+'patterns/base'+str(base)+'SNR'+str(SNR)+'spread'+str(spread)
    Sww = np.load(names2+'Sww.npy')
    pos = np.load(names2+'pos.npy')

    # inset_axes
    if SNR > 5:
        loc = 4
    else:
        loc = 1

    #im_ax = inset_axes(ax, width="50%", height="50%", loc=7)
    im_ax = fig.add_axes(inset_pos)
    im_ax.imshow(np.log10(Sww), cmap=cmocean.cm.deep, origin='lower')
    im_ax.set_xticks([])
    im_ax.set_xticklabels([])
    im_ax.set_yticks([])
    im_ax.set_yticklabels([])

    im_ax.scatter(pos[:, 0], pos[:, 1], color='w')
    im_ax.set_xlim([0, Sww.shape[0]])
    im_ax.set_ylim([0, Sww.shape[0]])
    SNRt = SNR**2/2
    im_ax.set_ylabel('SNR = '+str(SNRt))

    ax.set_ylim([0, 1.05])
    #ax[i].set_xlim([0, radius[-1]])

    ax.set_ylabel('Test power')
    sns.despine(ax=ax, offset=10)

    # spines trick
    ax.spines['left'].set_bounds(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.set_xlim([0, 3.8])
    ax.set_xlabel(r'$r_{\mathrm{max}}$')

    fig.subplots_adjust(right=.95, bottom=0.18, left=0.2, top=0.98)


# rank envelope
def demoSpectrogramSignal(SNR, duration, viz=False, shrink=True):
    base = 128
    b = 150
    a = base - b

    N = 4 * base
    Nfft = 2 * base
    t = np.arange(N)

    # Noise only
    w = np.random.randn(N)
    # window
    g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
    g = g/g.sum()

    # bounds for detection (larger)
    fmin = 0
    fmax = base

    #tmin = 2*base - (base-trunc) // 2
    #tmax = 2*base + (base-trunc) // 2
    tmin = base
    tmax = 3*base

    #chirp
    duration = int(np.floor(duration))
    if duration > base:
        raise ValueError('Duration should be lesser than base')


    start_s = 2*base-duration //2
    end_s = 2*base+duration //2
    chirp = np.zeros(N)

    freq = (a + b*t[start_s:end_s]/N)*t[start_s:end_s]/N
    chirp[start_s:end_s] = sg.tukey(duration)*np.cos(2*np.pi*freq)

    x0 = np.sqrt(2*SNR)*chirp + w
    #spectro = STFT(x0, g, Nfft)
    _, _, spectro = sg.stft(x0, window=g, nperseg=Nfft, noverlap=Nfft-1)
    Sww_t = abs(spectro)**2
    #print("STFT computed")

    Sww = Sww_t[fmin:fmax+1, tmin:tmax+1]


    # detection
    th = 1e-14
    y0, x0 = extr2minth(Sww, th)
    if shrink is True:
        # boundary conditions
        side = 110 # size of square; equivalent to trunc
        fmin_b = (max(0, (base-side)//2))
        fmax_b = (min(base, (base+side)//2))
        tmin_b = (base-side//2)
        tmax_b = (base+side//2)

        mask = (y0 > fmin_b)*(y0 < fmax_b)*(x0 > tmin_b)*(x0 < tmax_b)
        u = x0[mask]/np.sqrt(2*base)
        v = y0[mask]/np.sqrt(2*base)
    else:
        u = x0/np.sqrt(2*base)
        v = y0/np.sqrt(2*base)

    pos = np.zeros((len(u), 2))
    pos[:, 0] = u
    pos[:, 1] = v

    if viz is True:
        side = 110 # size of square; equivalent to trunc
        fmin = (max(0, (base-side)//2))/np.sqrt(2*base)
        fmax = (min(base, (base+side)//2))/np.sqrt(2*base)
        tmin = (base-side//2)/np.sqrt(2*base)
        tmax = (base+side//2)/np.sqrt(2*base)

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.imshow(np.log10(Sww), origin='lower', extent=[0, (2*base)/np.sqrt(2*base), 0, (base)/np.sqrt(2*base)], cmap=cmocean.cm.deep)
        ax.scatter(pos[:, 0], pos[:, 1], color='w', s=40)

        ax.set_xlim([tmin, tmax])
        ax.set_ylim([fmin, fmax])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.subplots_adjust(left=0.04, bottom=0.05)

        return Sww, pos, spectro, chirp

    else:
        return pos
