import numpy as np
from scipy.integrate import cumtrapz
# for R to work
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr



def spatialStatsFromR(pos):

    # load spatstat
    spatstat = importr('spatstat')

    u_r = robjects.FloatVector(pos[:, 0])
    v_r = robjects.FloatVector(pos[:, 1])

    bounds_u = np.array([np.min(pos[:, 0]), np.max(pos[:, 0])])
    bounds_v = np.array([np.min(pos[:, 1]), np.max(pos[:, 1])])

    b_u = robjects.FloatVector(bounds_u)
    b_v = robjects.FloatVector(bounds_v)

    ppp_r = spatstat.ppp(u_r, v_r, b_u, b_v)

    K_r = spatstat.Kest(ppp_r)
    L_r = spatstat.Lest(ppp_r)
    pcf_r = spatstat.pcf(ppp_r)

    radius = np.array(K_r[0])
    Kborder = np.array(K_r[2])

    if len(pos[:, 0]) < 1024:
        Ktrans = np.array(K_r[3])
        Kiso = np.array(K_r[4])

        K = [Kborder, Ktrans, Kiso]
    else:
        K = [Kborder]

    Lborder = np.array(L_r[2])
    Ltrans = np.array(L_r[3])
    Liso = np.array(L_r[4])

    L = [Lborder, Ltrans, Liso]

    pcftrans = np.array(pcf_r[2])
    pcfiso = np.array(pcf_r[3])

    pcf = [pcftrans, pcfiso]

    return radius, K, L, pcf

def pairCorrPlanarGaf(r, L):

    a = 0.5*L*r**2
    num = (np.sinh(a)**2+L**2/4*r**4)*np.cosh(a)-L*r**2*np.sinh(a)
    den = np.sinh(a)**3
    rho = num/den

    if r[0] == 0:
        rho[0] = 0
    return rho

def Kfunction(r, rho):
    K = np.zeros(len(rho))
    K[1:] = 2*np.pi*cumtrapz(r*rho, r)

    return K

def ginibreGaf(r, c):

    rho = 1-np.exp(-c*r**2)
    return rho
