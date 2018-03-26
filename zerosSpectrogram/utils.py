import numpy as np
import scipy.stats as spst
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
import scipy.signal as sg

from math import atan2
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull, Delaunay

def clopper_pearson(k, n, alpha):
    """
    Clopper-Pearson confidence interval for Bernoulli parameter
    alpha: confidence level
    k: number of successes
    n: number of observations
    """
    lb = spst.beta.ppf(alpha/2, k, n-k+1)
    ub = spst.beta.ppf(1 - alpha/2, k+1, n-k)

    return lb, ub

def extr2minth(M,th):

    C,R = M.shape

    Mid_Mid = np.zeros((C,R), dtype=bool)

    for c in range(1, C-1):
        for r in range(1, R-1):
            T = M[c-1:c+2,r-1:r+2]
            Mid_Mid[c, r] = (np.min(T) == T[1, 1]) * (np.min(T) > th)
            #Mid_Mid[c, r] = (np.min(T) == T[1, 1])

    x, y = np.where(Mid_Mid)
    return x, y


# Graphics and plot
def label_line(line, label_text, near_i=None, near_x=None, near_y=None, rotation_offset=0, offset=(0,0)):
    """call
        l, = plt.loglog(x, y)
        label_line(l, "text", near_x=0.32)
    """
    def put_label(i, axis):
        """put label at given index"""
        i = min(i, len(x)-2)
        dx = sx[i+1] - sx[i]
        dy = sy[i+1] - sy[i]
        rotation = (np.rad2deg(atan2(dy, dx)) + rotation_offset)*0
        pos = [(x[i] + x[i+1])/2. + offset[0], (y[i] + y[i+1])/2 + offset[1]]
        axis.text(pos[0], pos[1], label_text, size=12, rotation=rotation, color = line.get_color(),
        ha="center", va="center", bbox = dict(ec='1',fc='1', alpha=1., pad=0))

    x = line.get_xdata()
    y = line.get_ydata()
    ax = line.axes
    if ax.get_xscale() == 'log':
        sx = np.log10(x)    # screen space
    else:
        sx = x
    if ax.get_yscale() == 'log':
        sy = np.log10(y)
    else:
        sy = y

    # find index
    if near_i is not None:
        i = near_i
        if i < 0: # sanitize negative i
            i = len(x) + i
        put_label(i, ax)
    elif near_x is not None:
        for i in range(len(x)-2):
            if (x[i] < near_x and x[i+1] >= near_x) or (x[i+1] < near_x and x[i] >= near_x):
                put_label(i, ax)
    elif near_y is not None:
        for i in range(len(y)-2):
            if (y[i] < near_y and y[i+1] >= near_y) or (y[i+1] < near_y and y[i] >= near_y):
                put_label(i, ax)
    else:
        raise ValueError("Need one of near_i, near_x, near_y")


def plotRankEnvRes(radius, k, t2, tinfty, t2_exp, tinfty_exp):
    lsize=16 # labelsize

    fig, ax = plt.subplots(figsize=(4, 4))

    lk, = ax.plot(radius,  t2[k, :], color='g', alpha=1)
    ax.fill_between(radius,0,  t2[k, :], color='g', alpha=.8)

    lexp, = ax.plot(radius, t2_exp, color='k')

    label_line(lexp, r'$t_{\mathrm{exp}}$', near_i=49, offset=(0.3, 0))
    label_line(lk, r'$t_k$', near_i=49, offset=(0.3, 0))

    ax.set_ylabel(r'$T_2$' + r'$\mathrm{-statistic}$', fontsize=lsize)
    ax.set_xlabel(r'$r_{\mathrm{max}}$', fontsize=lsize)
    ax.set_xlim([0, radius[-1]])
    ax.set_yticks(np.linspace(0, 0.40, 5))

    sns.despine(offset=10)
    fig.subplots_adjust(left=.25, right=0.9, bottom=0.2, top=0.97)


    fig, ax = plt.subplots(figsize=(4, 4))
    lk, = ax.plot(radius,  tinfty[k, :], color='g', alpha=1)
    ax.fill_between(radius, 0, tinfty[k, :], color='g', alpha=.8)
    lexp, = ax.plot(radius, tinfty_exp, color='k')
    label_line(lexp, r'$t_{\mathrm{exp}}$', near_i=49, offset=(0.3, 0))
    label_line(lk, r'$t_k$', near_i=49, offset=(0.3, -0.005))

    ax.set_ylabel(r'$T_\infty$' + r'$\mathrm{-statistic}$', fontsize=lsize)
    ax.set_xlabel(r'$r_{\mathrm{max}}$', fontsize=lsize)
    ax.set_xlim([0, radius[-1]])
    sns.despine(offset=10)
    fig.subplots_adjust(left=.25, right=0.9, bottom=0.2, top=0.97)


class ProgressBar(object):
     def __init__(self, total=100, stream=sys.stderr):
         self.total = total
         self.stream = stream
         self.last_len = 0
         self.curr = 0

     def count(self):
         self.curr += 1
         self.print_progress(self.curr)

     def print_progress(self, value):
         self.stream.write('\r' * self.last_len)
         pct = 100 * self.curr / self.total
         out = '{:.2f}% [{}/{}]'.format(pct, self.curr, self.total)
         self.last_len = len(out)
         self.stream.write(out)
         self.stream.flush()


def findCenterEmptyBalls(Sww, pos_exp, radi_seg, base=128):

    # define a kd-tree with zeros
    kdpos = KDTree(pos_exp)

    # define a grid corresponding to the time-frequency paving
    vecx = (np.arange(0, Sww.shape[0])/np.sqrt(2*base))
    vecy = (np.arange(0, Sww.shape[1])/np.sqrt(2*base))
    g = np.transpose(np.meshgrid(vecy, vecx))

    result = kdpos.query_ball_point(g, radi_seg).T

    empty = np.zeros(result.shape, dtype=bool)
    for i in range(len(vecx)):
        for j in range(len(vecy)):
            empty[i,j] = len(result[i, j]) < 1


    # then plot
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(g[..., 0], g[..., 1], s=0.1, color='w')
    ax.imshow(empty, extent=[0, (2*base)/np.sqrt(2*base), 0, (base)/np.sqrt(2*base)], origin='lower', cmap=cmocean.cm.deep_r)
    ax.scatter(pos_exp[:,0], pos_exp[:, 1], color='w', s=30)

    side = 110
    fmin = (max(0, (base-side)//2))/np.sqrt(2*base)
    fmax = (min(base, (base+side)//2))/np.sqrt(2*base)
    tmin = (base-side//2)/np.sqrt(2*base)
    tmax = (base+side//2)/np.sqrt(2*base)

    ax.set_xlim([tmin, tmax])
    ax.set_ylim([fmin, fmax])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.subplots_adjust(left=0.04, bottom=0.05)

    return empty


def getConvexHull(Sww, pos_exp, empty_mask, radi_expand=0.5, base=128):

    # extract region of interest
    side = 110
    fmin = max(0, (base-side)//2)
    fmax = min(base, (base+side)//2)
    tmin = base-side//2
    tmax = base+side//2

    sub_empty = empty_mask[fmin:fmax, tmin:tmax]

    # optional smoothing, comment if necessary
    u, v = np.where(sub_empty)
    vec = np.arange(0, side)
    g = np.transpose(np.meshgrid(vec, vec))

    kdpos = KDTree(np.array([u, v]).T)
    result = kdpos.query_ball_point(g, radi_expand*np.sqrt(2*base)).T

    sub_empty = np.zeros(result.shape, dtype=bool)
    for i in range(len(vec)):
        for j in range(len(vec)):
            sub_empty[j, i] = len(result[i, j]) > 0

    u, v = np.where(sub_empty)
    points = np.array([u, v]).T
    hull = ConvexHull(points)
    hull_d = Delaunay(points) # for convenience

    # plot
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(np.log10(Sww), origin='lower', extent=[0, (2*base)/np.sqrt(2*base), 0, (base)/np.sqrt(2*base)], cmap=cmocean.cm.deep)
    ax.scatter(pos_exp[:, 0], pos_exp[:, 1], color='w', s=40)

    for simplex in hull.simplices:
        plt.plot((points[simplex, 1]+tmin)/np.sqrt(2*base), (points[simplex, 0]+fmin)/np.sqrt(2*base), 'g-', lw=4)

    ax.set_xlim([tmin/np.sqrt(2*base), tmax/np.sqrt(2*base)])
    ax.set_ylim([fmin/np.sqrt(2*base), fmax/np.sqrt(2*base)])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.subplots_adjust(left=0.04, bottom=0.05)

    return hull_d


def reconstructionSignal(hull_d, stft, chirp, base=128):

    side = 110
    fmin = max(0, (base-side)//2)
    fmax = min(base, (base+side)//2)
    tmin = base-side//2
    tmax = base+side//2
    # sub mask : check which points are in the convex hull
    vec = (np.arange(0, side))
    g = np.transpose(np.meshgrid(vec, vec))
    sub_mask = hull_d.find_simplex(g)>=0

    # create a mask
    mask = np.zeros(stft.shape, dtype=bool)
    mask[fmin:fmax, base+tmin:base+tmax] = sub_mask

    # reconstruction
    Nfft = 2*base # as in the simulation
    g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
    g = g/g.sum()
    t, xorigin = sg.istft(stft, window=g,  nperseg=Nfft, noverlap=Nfft-1)
    t, xr = sg.istft(mask*stft, window=g,  nperseg=Nfft, noverlap=Nfft-1)


    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(t, xorigin+15)
    ax.plot(t, chirp, color='k')
    ax.plot(t, xr-15, color='g')

    ax.set_xlim(162, 350)
    ax.set_ylim(-25, 25)
    sns.despine(offset=10)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.text((350+162)/2, -26.5,r'$\mathsf{time}$', fontsize=24, ha='center', backgroundcolor='white')
    ax.text(158.5, 0,r'$\mathsf{amplitude}$', fontsize=24, ha='center', va='center', rotation=90, backgroundcolor='white')

    ax.text(350, 19, r'$\mathsf{noisy\: signal}$', color='b', fontsize=20, ha='right', backgroundcolor='white' )
    ax.text(350, 2, r'$\mathsf{original}$', color='k', fontsize=20, ha='right', backgroundcolor='white' )
    ax.text(350, -13, r'$\mathsf{reconstructed}$', color='g', fontsize=20, ha='right', backgroundcolor='white' )
    #ax.legend(fontsize=18, loc=5)
    fig.tight_layout()
    fig.subplots_adjust(left=0.06, bottom=0.06)
