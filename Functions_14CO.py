#basic imports and ipython setup
import matplotlib.pyplot as plt
import numpy as np

from MCEq.core import MCEqRun
#import mceq_config as config
from MCEq import config
import crflux.models as pm

from tqdm import tqdm

from MCEq.geometry.density_profiles import GeneralizedTarget

import daemonflux

import proposal as pp

from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy import stats

import scipy.sparse

#import mute.constants as mtc
#import mute.underground as mtu
    
## PRIMARY FUNCTIONS

def get_primary(Prop, *primary_model):
    if Prop is None: # run function with Prop=None to get input stage
        return ''

    Prop.mceq.set_primary_model(*primary_model)
    pname = Prop.mceq.pman.pname2pref
    return np.array([[Prop.mceq._phi0[pname['p+'].lidx:pname['p+'].uidx], Prop.mceq._phi0[pname['n0'].lidx:pname['n0'].uidx]]])

#def load_primary(Prop, p_models=None): # primary CR intensities
    #if Prop is None: # run function with Prop=None to get input stage
        #return ''

    #if p_models is None:
        #p_models = Prop.p_models
        
    #return np.array([get_primary(Prop, *p) for p in p_models])

def set_primary_identity(Prop):
    if Prop is None: # run function with Prop=None to get input stage
        return ''

    # sets up primary flux matrix to test each energy of p+ and n0 individually

    Phi0 = np.zeros((2*len(Prop.E), 2, len(Prop.E)))
    for i in range(2):
        for j in range(len(Prop.E)):
            Prop.Phi0[len(Prop.E)*i+j, i, j] = 1.

    return Phi0

## ATMOSPHERIC FUNCTIONS

def judge_nash(Prop, K_mu = 1.268):
    if Prop is None: # run function with Prop=None to get input stage
        return ''

    """


    Parameters
    ----------------
    Phi0 - 

    K_mu - 

    """

    E0 = np.reshape(Prop.E_mu, (1,1,-1))
    cosTH = np.reshape(Prop.cosTH, (-1,1,1))

    H_pi = 114 #GeV
    H_K = 851 #GeV
    gamma_pi = 2.7
    gamma_K = gamma_pi
    A_pi = 0.28
    A_K = 0.0455
    y0 = 1000 # Atmosphere depth at sea level? (g/cm^2)
    r_pi = 0.76 #muon to parent pion energy ratio
    r_K = 0.523 #muon to parent kaon energy ratio
    q = 2.2e-3 #mean energy loss of the muon in the atmosphere (GeV cm^2/g)
    tau_mu = 2.2e-6 #mean muon lifetime (s)
    tau_pi = 2.61e-8 #mean pion lifetime (s)
    tau_K = 1.24e-8 #mean kaon lifetime (s)
    g = 981.3 #acceleration due to gravity (cm/s^2)
    R = 8.314e7 #gas constant (g cm^2/s^2 /K /mol)
    m_mu = 105.659e-3 #muon rest mass (GeV/c^2)
    m_pi = 139.580e-3 #pion rest mass (GeV/c^2)
    m_K = 493.800e-3 #kaon rest mass (GeV/c^2)
    c = 3e10 #speed of light (cm/s)
    M = 28.966 #effective molecular weight of air (g/mol)

    #effective mean temperature of the atmosphere as experienced at zenith angle theta
    T_e = 220 #220 #from Chatzidakis 2015

    H_mu = R*m_mu*T_e / c / M / g / tau_mu

    E_pi = (E0 + q*y0*(1/cosTH - 0.0874))/r_pi

    #H_pi = R*m_pi*c*T_e / M/g/tau_pi

    W_pi = (0.0874 * cosTH * (1- q * (y0/cosTH - 90)/r_pi/E_pi))**(H_mu/cosTH/(r_pi*E_pi+90*q))

    E_K = (E0 + q*y0*(1/cosTH - 0.0874))/r_K

    #H_K = R*m_K*c*T_e / M / g /tau_K

    W_K = (0.0874 * cosTH * (1- q * (y0/cosTH - 0.0874)/r_K/E_K))**(H_mu/cosTH/(r_K*E_K+90*q))

    Phi_J = A_pi*W_pi*E_pi**(-gamma_pi)*H_pi / (E_pi*cosTH + H_pi) + A_K*W_K*E_K**(-gamma_K)*H_K / (E_K*cosTH + H_K)

    Phi_mu = Phi_J * np.reshape([K_mu/(K_mu+1), 1/(K_mu+1)], (1,-1,1))

    Phi_mu = np.reshape(Phi_mu, (1, len(Prop.cosTH), 2, -1))

    return np.nan_to_num(Phi_mu)

def bugaev_reyna(Prop, K_mu = 1.268):
    if Prop is None: # run function with Prop=None to get input stage
        return ''

    """


    Parameters
    ---------------
    Phi0 - 

    K_mu - 

    """

    #Phi0
    #axis0 - Primary Model (Energy spectrum & Time dependence)
    #axis1 - Particle Species (proton, neutron)
    #axis2 - Primary Energy

    E0 = np.reshape(Prop.E_mu, (1,1,-1))
    cosTH = np.reshape(Prop.cosTH, (-1,1,1))

    A_B = 0.00253
    a0 = 0.2455
    a1 = 1.288
    a2 = -0.2555
    a3 = 0.0209

    y = np.log10(E0*cosTH)

    Phi_R = cosTH**3 * A_B * (E0*cosTH)**(-(a3*y**3 + a2*y**2 + a1*y + a0))

    Phi_mu = Phi_R * np.reshape([K_mu/(K_mu+1), 1/(K_mu+1)], (1,-1,1))

    Phi_mu = np.reshape(Phi_mu, (1, len(Prop.cosTH), 2, -1))

    return Phi_mu

def SDC(Prop, K_mu = 1.268):
    if Prop is None: # run function with Prop=None to get input stage
        return ''

    """


    Parameters
    ----------------
    Phi0 - 

    K_mu - 

    """

    # Smith & Duller / Chatzidakis

    #Phi0
    #axis0 - Primary Model (Energy spectrum & Time dependence)
    #axis1 - Particle Species (proton, neutron)
    #axis2 - Primary Energy

    E0 = np.reshape(Prop.E_mu, (1,1,-1))
    cosTH = np.reshape(Prop.cosTH, (-1,1,1))

    A = 0.002382 # fitting parameter
    r = 0.76 # Ratio of muon energy to pion energy
    a = 2.500e-3 # Muon rate of energy loss in air (GeV / (g/cm^2))
    y0 = 1000 # Atmosphere depth at sea level (g/cm^2)
    gamma = 8/3 # fitting parameter
    b_mu = 0.800 # Correction factor related to atmospheric temperature
    m_mu = 105.659e-3 # Rest mass of muon (GeV/c^2)
    tau_mu0 = 2.2e-6 # Mean lifetime of muon at rest (s)
    rho0 = 0.00123 # Density of atmosphere at sea level (g/cm^3)
    c = 3e10 # speed of light (cm/s)
    lambda_pi = 120 # Absorption mean free path of pions (g/cm^2)
    b = 0.771 # coefficient to modify the isothermal atmosphere approximation
    tau0 = 2.61e-8 # Mean lifetime of pion at rest (s)
    m_pi = 139.580e-3 # Rest mass of pion (GeV/c^2)
    j_pi = 148.16 # m_pi * y0 * c / (tau0 * rho0) GeV

    # energy of pion that produced muon
    E_pi = (E0 + a*y0*(1/cosTH - 0.100))/r

    B_mu = b_mu * m_mu * y0 / (c * tau_mu0 * rho0)

    # Probability for muons to reach sea level
    P_mu = (0.100 * cosTH * (1-a*(y0/cosTH - 100)/(r * E_pi)))**(B_mu / ((r*E_pi + 100*a)*cosTH))

    Phi_S = A * E_pi**(-gamma) * P_mu * lambda_pi * b * j_pi/(E_pi* cosTH + b * j_pi)

    Phi_mu = Phi_S * np.reshape([K_mu/(K_mu+1), 1/(K_mu+1)], (1,-1,1))

    Phi_mu = np.reshape(Phi_mu, (1, len(Prop.cosTH), 2, -1))

    return Phi_mu

def att_L(h):

    """


    Parameters
    --------------
    h - 

    """

    # effective atmospheric attenuation length for muons at mass depth h

    # define range/momentum relation for atmospheric attenuation length used in Balco calculation

    # table for muons in standard rock in Groom and others 2001
    # g/cm^2
    h_range = np.array([8.516e-1, 1.542e0, 2.866e0, 5.698e0, 9.145e0, 2.676e1, 3.696e1, 5.879e1, 9.332e1, 1.524e2,
                        2.115e2, 4.418e2, 5.534e2, 7.712e2, 1.088e3, 1.599e3, 2.095e3, 3.998e3, 4.920e3, 6.724e3,
                        9.360e3, 1.362e4, 1.776e4, 3.343e4, 4.084e4, 5.495e4, 7.459e4, 1.040e5, 1.302e5, 2.129e5])

    # MeV/c
    momentum = np.array([4.704e1, 5.616e1, 6.802e1, 8.509e1, 1.003e2, 1.527e2, 1.764e2, 2.218e2, 2.868e2, 3.917e2,
                         4.945e2, 8.995e2, 1.101e3, 1.502e3, 2.103e3, 3.104e3, 4.104e3, 8.105e3, 1.011e4, 1.411e4,
                         2.011e4, 3.011e4, 4.011e4, 8.011e4, 1.001e5, 1.401e5, 2.001e5, 3.001e5, 4.001e5, 8.001e5])

    P_MeVc = np.exp(np.interp(np.log(np.clip(h,1e-5,None)*100), np.log(h_range), np.log(momentum)))

    return 263 + 150 * P_MeVc/1000

def phi_vert_slhl(h):
    """Empirical fit to vertical muon flux at sea level, presented in

    B Heisinger et al. “Production of selected cosmogenic radionuclides by muons: 1. Fast muons”.
    In: Earth and Planetary Science Letters 200.3 (2002), pp. 345–355. issn: 0012-821X.
    doi: https://doi.org/10.1016/S0012-821X(02)00640-4.

    where it was modified from the parameterization in

    A.I. Barbouti, B.C. Rastin, A study of the absolute intensity of muons at sea level and under
    various thicknesses of absorber, J. Phys. G 9 (1983) 1577-1595.

    Parameters
    -----------
    h : float or array of floats
        mass depth below surface (hg/cm^2)

    Returns
    --------
    Phi_v : float or array of floats
        Vertical muon flux (cm^-2 s^-1 sr^-1)
    """
    #parameters
    p = [258.5,  #p0
        -5.5e-4, #p1
        210,     #p2
        10,      #p3
        1.66,    #p4
        75]      #p5

    a = np.exp(p[1] * h)
    b = h + p[2]
    c = (h+p[3])**p[4] + p[5]

    Phi_v = p[0] * a / b / c  # cm^-2 s^-1 sr^-1

    return Phi_v

def R_vert_slhl(h):
    """Analytic derivative of above vertical muon flux function with respect to mass depth,
    derived in

    B Heisinger et al. “Production of selected cosmogenic radionuclides by muons: 1. Fast muons”.
    In: Earth and Planetary Science Letters 200.3 (2002), pp. 345–355. issn: 0012-821X.
    doi: https://doi.org/10.1016/S0012-821X(02)00640-4.

    Parameters
    -----------
    h : float or array of floats
        mass depth below surface (hg/cm^2)

    Returns
    --------
    R_v : float or array of floats
        Vertical muon stopping rate (hg^-1 s^-1 sr^-1)
    """
    #parameters
    p = [258.5,  #p0
        -5.5e-4, #p1
        210,     #p2
        10,      #p3
        1.66,    #p4
        75]      #p5

    a = np.exp(p[1] * h)
    b = h + p[2]
    c = (h+p[3])**p[4] + p[5]

    dadh = p[1] * a
    dbdh = 1.
    dcdh = p[4] * (h+p[3])**(p[4]-1)

    R_v = -p[0] * (b*c*dadh - a*c*dbdh - a*b*dcdh)/ b**2 / c**2  # hg^-1 s^-1 sr^-1

    return R_v

def phi_vert_site(h, dh, H, h_end=2e3):

    """


    Parameters
    -----------------
    h - 

    dh - 

    H - 

    h_end - 


    Returns
    ----------------
    Phi_site - 

    R_site - 

    """

    Phi_v= phi_vert_slhl(h)

    R_v = R_vert_slhl(h)

    R_site = R_v * np.exp(H/att_L(h))

    Phi_end = phi_vert_slhl(h_end)

    dh_ext = 1
    h_ext = np.arange(h[-1]+dh_ext, h_end+dh_ext, dh_ext)

    h_int = np.append(h, h_ext)
    dh_int = np.append(dh, dh_ext + 0*h_ext)

    R_int = R_vert_slhl(h_int) * np.exp(H/att_L(h_int))

    Phi_site = np.flip(np.cumsum(np.flip(R_int * dh_int))) + (1-np.exp(H/att_L(h_end)))*Phi_end

    Phi_site = Phi_site[:len(h)]

    return Phi_site, R_site

def cos_pow(h, H):

    """


    Parameters
    -------------------
    h - 

    H - 


    Returns
    --------------------
    n - 

    dndh - 

    """

    #parameters
    p = [3.21,     #p0
         0.297,    #p1
         42,       #p2
         1.21e-3]  #p3

    #H = (1013.25 - pressure/100) * 1.019716
    #h_mod = h + H (atmospheric depth diff. from sea level)

    # shouldn't H be subtracted here?

    n = p[0] - p[1]*np.log(h + H/100 + p[2]) + p[3] * (h + H/100)

    dndh = -p[1]/(h + H/100 + p[2]) + p[3]

    return n, dndh

def f_mu_neg():
    K_mu = 1.268 # +/- 0.008 + 0.002 * E[GeV]

    return 1/(K_mu+1)

def phi_all(h, dh, H):

    """


    Parameters
    ------------------
    h - 

    dh - 

    H - 

    cos_pow_func - 

    f_func - 


    Returns
    ---------------------
    Phi - 

    R - 

    """

    n, dndh = cos_pow(h, H)

    Phi_v, R_v = phi_vert_site(h, dh, H)

    Phi = 2*np.pi/(n+1) * Phi_v

    R = f_mu_neg() * (2*np.pi * R_v + Phi*dndh) / (n+1)

    return Phi, R # cm^-2 s^-1

def Heisinger(h):

    """


    Parameters
    ----------------
    h - 


    Returns
    ---------------
    E_pred - 

    Beta_pred - 

    """

    #parameters
    a = 7.6
    b = 321.7
    c = 8.059e-4
    d = 50.7
    e = 5.05e-5

    f = 0.846
    g = 0.015
    i = 0.003139

    # Heisinger's fit for average Energy
    E_pred = a + b * (1-np.exp(-c*h)) + d*(1-np.exp(-e*h))

    # Heisinger's Beta correction term
    Beta_pred = f - g*np.log(h+1)+i*np.log(h+1)**2

    return E_pred, Beta_pred

def E_surf(Prop, 
           E_d, 
           X, 
           a=None, 
           b=None
          ):
    if a is None:
        a = Prop.a
    if b is None:
        b = Prop.b

    """
    Returns
    ------------------
    E_surf - 

    """

    return ((E_d + a/b)*np.exp(X*b)-a/b).clip(min=Prop.E_bins[0])

def Heisinger_ice(Prop, 
                  norm=True, 
                  mode=0, 
                  a=None, 
                  b=None, 
                  H=None
                 ):
    if Prop is None: # run function with Prop=None to get input stage
        return 'atm'
    
    if a is None:
        a = Prop.a
    if b is None:
        b = Prop.b
    if H is None:
        H = Prop.H

    """
    Returns
    ----------------------
    Phi_proj - 

    """

    # project under ice w/ Gaisser-Stanev
    # normalize proportional to Heisinger depth fit times total surface flux
    
    Phi_atm = Prop.Phi['atm']
    #Phi_atm
    #axis0 - Atmospheric Model
    #axis1 - Zenith Angle
    #axis2 - Muon Charge (positive, negative)
    #axis3 - Muon Energy

    X = np.reshape(Prop.h_bins,(1,1,-1))/np.reshape(Prop.cosTH,(1,-1,1)) # slant-depth
    
    E_surface = (Prop.E_mu.reshape((-1,1,1)) + a/b)*np.exp(X*b)-a/b
    E_bounds = E_surf(Prop, np.reshape(Prop.E_mu_bins,(-1,1,1)), X, a, b) # Energy bins at depth projected back to their surface energies
    #axis0 - Projected Energy
    #axis1 - Zenith Angle
    #axis2 - Depth (top -> bottom)
    
    # interpolate phi at surface energy
    # phi(E_underground) dE_underground = phi(E_surface) dE_surface
    # phi(E_underground) = phi(E_surface) * E_surface / E_underground
    # Don't forget to integrate over 2pi cos theta
    if mode==0: # log-linear interp
        Phi_proj = np.sum(
            [np.nan_to_num(np.exp(interp1d(np.log(Prop.E_mu), np.log(Phi_atm[:,i]), axis=-1, bounds_error=False, assume_sorted=True)(np.log(E_surface[:,i])))) * np.exp(X[0,i]*b).reshape((1,1,1,-1)) * Prop.dcosTH[i]
             for i in tqdm(range(len(Prop.cosTH)))], axis=0) * 2 * np.pi
    #axis0 - Atmopsheric Model
    #axis1 - Muon Charge (positive, negative)
    #axis2 - Muon Energy
    #axis3 - Depth (top -> bottom)
    
    elif mode==1: # linear interp
        Phi_proj = np.sum(
            [interp1d(Prop.E_mu, Phi_atm[:,i], axis=-1, bounds_error=False, fill_value=0., assume_sorted=True)(E_surface[:,i]) * np.exp(X[0,i]*b).reshape((1,1,1,-1)) * Prop.dcosTH[i]
             for i in range(len(Prop.cosTH))], axis=0) * 2 * np.pi
    #axis0 - Atmopsheric Model
    #axis1 - Muon Charge (positive, negative)
    #axis2 - Muon Energy
    #axis3 - Depth (top -> bottom)
    
    elif mode==2: # new, direct sum calculation (uses more memory)
        E_proj = np.expand_dims(E_bounds.swapaxes(0,1), (1,2))

        Phi_proj = np.array([np.sum(np.expand_dims(p, (3,4)) * Prop.dcosTH.reshape((-1,1,1,1,1)) * ( (E_proj[:,:,:,1:] - Prop.E_mu_bins[:-1].reshape((1,1,-1,1,1))).clip(0,Prop.dE_mu.reshape(1,1,-1,1,1)) - (E_proj[:,:,:,:-1] - Prop.E_mu_bins[:-1].reshape((1,1,-1,1,1))).clip(0,Prop.dE_mu.reshape(1,1,-1,1,1)) ), axis=(0,2)) for p in tqdm(Phi_atm)]) * 2 * np.pi / Prop.dE_mu.reshape((1,1,-1,1))
        # Phi_proj_i dE_i = sum_j Phi_atm_j 2pi cosTH_j [ (E_proj_(i+1) - E_j).clip(0,dE_j) - (E_proj_i - E_j).clip(0,dE_j) ]
        #axis0 - Atmopsheric Model
        #axis1 - Zenith Angle (SUM)
        #axis2 - Muon Charge (positive, negative)
        #axis3 - Surface Energy (SUM)
        #axis4 - Projected Energy
        #axis5 - Depth (top -> bottom)
        
    else: # old looping calculation
        Phi_proj = np.zeros((np.shape(Phi_atm)[0], 2, len(Prop.E_mu), len(Prop.h_bins)))

        # Now, this is going to look incomprehensible, but...
        for i in tqdm(range(len(Prop.cosTH))):
            for j in range(len(Prop.h_bins)):
                E_proj = E_bounds[:,i,j]
                deep = True # Starting from a Phi_proj energy bin edge?  False means Phi_atm
                #print(i,j,E_proj[0])
                k=np.arange(len(Prop.E_mu_bins))[Prop.E_mu_bins<=E_proj[0]][-1] # current Phi_atm energy bin
                l=0 # current Phi_proj energy bin
                while k < len(Prop.E_mu_bins)-1 and l < len(E_proj)-1: # step through energy bin edge, one by one, putting muons from Phi_atm into the Phi_proj bin corresponding to their projected underground energy
                    if Prop.E_bins[k+1]<=E_proj[l+1]: # if the next bin edge is from Phi_atm
                        Phi_proj[:, :, l, j] += Phi_atm[:,i,:,k] * (Prop.E_mu_bins[k+1]-(E_proj[l] if deep else Prop.E_mu_bins[k])) * Prop.dcosTH[i] * 2 * np.pi
                        k += 1 # Start the next step from Phi_atm's bin edge
                        deep = False
                    else: # if the next bin boundary is from Phi_proj
                        Phi_proj[:, :, l, j] += Phi_atm[:,i,:,k] * (E_proj[l+1]-(E_proj[l] if deep else Prop.E_mu_bins[k])) * Prop.dcosTH[i] * 2 * np.pi
                        l += 1 # Start the next step from Phi_proj's bin edge
                        deep = True
                # Hate to use a While loop, but it should have to stop before 2*len(E_bins) steps

        Phi_proj = Phi_proj / np.reshape(Prop.dE_mu,(1,1,-1,1))
        #axis0 - Atmospheric Model
        #axis1 - Muon Charge (positive, negative)
        #axis2 - Muon Energy
        #axis3 - depth (top -> bottom)

    # Normalize the result to Heisinger's total flux fit (elevation adjusted by Balco)
    if norm:
        return Phi_proj / np.sum(Phi_proj * np.reshape(Prop.dE_mu,(1,1,-1,1)), axis=(1,2), keepdims=True) * np.reshape(phi_all(Prop.h_bins,np.append(Prop.dh,Prop.dh[-1]),Prop.H)[0], (1,1,1,-1))
    
    return Phi_proj


def Heisinger_full(Prop, H=None):
    if Prop is None: # run function with Prop=None to get input stage
        return ''
    
    if H is None:
        H = Prop.H

    """


    Parameters
    ---------------------
    Phi0 - 

    H - 

    f_factors - 


    Returns
    ---------------
    P_14C - 

    """

    # Standard Heisinger calculation
    # normalize proportional to total primary flux

    #Phi0
    #axis0 - Primary Model (Energy spectrum & Time dependence)
    #axis1 - Particle Species (proton, neutron)
    #axis2 - Primary Energy

    # Currently not normalizing to Phi0

    E_pred, Beta_pred = Heisinger(Prop.h)

    Phi, R = phi_all(Prop.h, Prop.dh, H) # Total Muon Flux, Negative Muon Stopping Rate

    P_neg = R * Prop.f_tot

    P_fast = Prop.sigma_0 * Phi * E_pred**Prop.alpha * Beta_pred * Prop.N

    P_14C = np.reshape([P_fast, P_neg], (1,2,-1)) /100 * 60 * 60 * 24 * 365.25 # g^-1, a^-1

    #rescale = np.ones((np.shape(Phi0)[0], 1))

    #P_14C
    #axis1 - Primary Model
    #axis2 - Production Mode (fast, neg)
    #axis3 - depth (top -> bottom)

    return P_14C

def get_mu_cut(mceq):
    M = (mceq.int_m != 0)+(mceq.dec_m != 0)+scipy.sparse.identity(mceq.int_m.shape[0])
    pname = mceq.pman.pname2pref
    cut = np.arange(pname['mu+_l'].lidx, pname['mu-_r'].uidx)
    cut_len = len(cut)
    
    converge = False
    i = 0
    while not converge and i<100:
        cut = np.array(np.sum(M[cut], axis=0)>0)[0]
        if np.sum(cut)==cut_len:
            converge = True
        cut_len = np.sum(cut)
        i+=1
    
    return cut

def solve_mceq(mceq, int_grid=None, grid_var='X', use_tqdm=False):

    """


    Parameters
    ---------------
    mceq - MCEqRun object

    int_grid - 

    grid_var - string

    use_tqdm - bool

    """
    
    cut = get_mu_cut(mceq)
    
    mceq._calculate_integration_path(int_grid=int_grid, grid_var=grid_var)

    nsteps, dX, rho_inv, grid_idcs = mceq.integration_path
    int_m = mceq.int_m[cut][:,cut]
    dec_m = mceq.dec_m[cut][:,cut]

    dXaccum = 0.
    grid_sol = np.zeros((len(grid_idcs), *np.shape(mceq._phi0))) # grid_sol begins with the right shape, to avoid restructuring
    grid_step = 0

    phc = np.copy(mceq._phi0[cut])

    for step in (tqdm(range(nsteps)) if use_tqdm else range(nsteps)): # added option for tqdm progress bar
        phc += (int_m.dot(phc) + dec_m.dot(rho_inv[step] * phc)) * dX[step]
        phc[phc<1e-250] = 0. # exreme low values set to 0, improving efficiency for large slant depths

        if (grid_idcs and grid_step < len(grid_idcs)
                and grid_idcs[grid_step] == step):
            grid_sol[grid_step,cut] = np.copy(phc) # grid_sol no longer appends
            grid_step += 1
    
    mceq._solution = np.zeros(mceq._phi0.shape)
    mceq._solution[cut], mceq.grid_sol = phc, grid_sol

    return

def get_int_path(mceq, cth):
    mceq.set_theta_deg(180*np.arccos(cth)/np.pi)
    mceq._calculate_integration_path(int_grid=None, grid_var='X')
    nsteps, dX, rho_inv = mceq.integration_path[:3]
    
    return [nsteps, dX, rho_inv]

#interaction_models = ["SIBYLL-2.3c","SIBYLL-2.3","SIBYLL-2.1","EPOS-LHC","QGSJET-II-04","DPMJET-III",'DPMJETIII191']
#density_models = [('CORSIKA', ('USStd', None)), ('CORSIKA',('SouthPole', 'December'))]
#density_names = ['CORSIKA_USStd', 'CORSIKA_SP_Dec']
def MCEq_atm(Prop, interaction_model="SIBYLL-2.3c", density_model=('CORSIKA', ('USStd', None)), elev=None, solver='default'):
    if Prop is None: # run function with Prop=None to get input stage
        return 'primary'
    
    if elev is None:
        elev = Prop.elev

    # Use MCEq to propagate primary flux to atmospheric muons
    Phi0 = Prop.Phi['primary']
    #Phi0
    #axis0 - Primary Model (Energy spectrum & Time dependence)
    #axis1 - Particle Species (proton, neutron)
    #axis2 - Primary Energy

    #import mceq_config as config
    from MCEq import config
    config.debug_level = 0
    config.h_obs = elev # elevation in (m) of Dome-C
    config.enable_default_tracking = False
    config.e_min = Prop.E_bins[1]
    config.e_max = Prop.E_bins[-1]
    config.max_density = 0.001225
    config.dedx_material = 'air'

    mceq = MCEqRun(
        interaction_model=interaction_model,
        theta_deg = 0,
        density_model = density_model,
        #medium=medium,
        primary_model = (pm.GaisserHonda, None),
    )

    pname = mceq.pman.pname2pref

    phi0 = np.zeros((len(mceq._phi0), Phi0.shape[0]))
    phi0[pname['p+'].lidx:pname['p+'].uidx] = Phi0[:,0].T
    phi0[pname['n0'].lidx:pname['n0'].uidx] = Phi0[:,1].T
    
    # This is the output array
    phi_mu = np.zeros((np.shape(Phi0)[0],len(Prop.cosTH),2,len(Prop.E_mu)))
    
    for i in tqdm(range(len(Prop.cosTH))):

        mceq.set_theta_deg(180*np.arccos(Prop.cosTH[i])/np.pi)
        mceq._phi0 = phi0

        if solver == 'default':
            solve_mceq(mceq)
        else:
            # 'numpy', 'cuda', or 'mkl'
            config.kernel_config = solver
            mceq.solve()
        
        # mceq._solution has the same shape as our phi0
        phi_surf = mceq._solution.T

        phi_mu[:,i,0] += phi_surf[:,pname['mu+'].lidx:pname['mu+'].lidx+len(Prop.E_mu)]
        phi_mu[:,i,0] += phi_surf[:,pname['mu+_l'].lidx:pname['mu+_l'].lidx+len(Prop.E_mu)]
        phi_mu[:,i,0] += phi_surf[:,pname['mu+_r'].lidx:pname['mu+_r'].lidx+len(Prop.E_mu)]

        phi_mu[:,i,1] += phi_surf[:,pname['mu-'].lidx:pname['mu-'].lidx+len(Prop.E_mu)]
        phi_mu[:,i,1] += phi_surf[:,pname['mu-_l'].lidx:pname['mu-_l'].lidx+len(Prop.E_mu)]
        phi_mu[:,i,1] += phi_surf[:,pname['mu-_r'].lidx:pname['mu-_r'].lidx+len(Prop.E_mu)]

    return phi_mu
    """
    # Build 2D array for mceq primary particles
    # We need the 2nd to last axis to be Particle Species & Energy
    # for the matrix multiplication to line up
    phi0 = np.zeros((Phi0.shape[0], len(Prop.cosTH), len(mceq._phi0)))
    phi0[:,:,pname['p+'].lidx:pname['p+'].uidx] = Phi0[:,0].reshape((Phi0.shape[0],1,-1))
    phi0[:,:,pname['n0'].lidx:pname['n0'].uidx] = Phi0[:,1].reshape((Phi0.shape[0],1,-1))
    phi0 = phi0.T.reshape((len(mceq._phi0),-1))
    # axis0 - Species & Energy
    # axis1 - CosTH & primary model
    
    int_paths = np.array([get_int_path(mceq, cth) for cth in tqdm(Prop.cosTH)], dtype=object)
    # axis0 - zenith angle
    # axis1 - [nsteps, dX, rho_inv]
    nsteps = np.max(int_paths[:,0])
    dX = np.array([np.append(x,[0.]*(nsteps-len(x))) for x in int_paths[:,1]]).T.reshape((nsteps,1,-1,1)).repeat(Phi0.shape[0],axis=-1).reshape((nsteps,1,-1))
    rho_inv = np.array([np.append(x,[0.]*(nsteps-len(x))) for x in int_paths[:,2]]).T.reshape((nsteps,1,-1,1)).repeat(Phi0.shape[0],axis=-1).reshape((nsteps,1,-1))
    
    int_m = mceq.int_m
    dec_m = mceq.dec_m
    
    phc = np.copy(phi0)
    
    print(nsteps, int_m.shape, phc.shape, dX.shape, rho_inv.shape)

    for step in tqdm(range(nsteps)):
        phc += (int_m @ phc + dec_m @ (rho_inv[step] * phc)) * dX[step]
        phc[phc<1e-250] = 0. # exreme low values set to 0, improving efficiency for large slant depths
    
    phi_surf = phc.reshape((len(mceq._phi0), len(Prop.cosTH), Phi0.shape[0])).T
    
    # This is the output array
    phi_mu = np.zeros((np.shape(Phi0)[0],len(Prop.cosTH),2,len(Prop.E_mu)))
    
    phi_mu[:,:,0] += phi_surf[:,:,pname['mu+'].lidx:pname['mu+'].lidx+len(Prop.E_mu)]
    phi_mu[:,:,0] += phi_surf[:,:,pname['mu+_l'].lidx:pname['mu+_l'].lidx+len(Prop.E_mu)]
    phi_mu[:,:,0] += phi_surf[:,:,pname['mu+_r'].lidx:pname['mu+_r'].lidx+len(Prop.E_mu)]

    phi_mu[:,:,1] += phi_surf[:,:,pname['mu-'].lidx:pname['mu-'].lidx+len(Prop.E_mu)]
    phi_mu[:,:,1] += phi_surf[:,:,pname['mu-_l'].lidx:pname['mu-_l'].lidx+len(Prop.E_mu)]
    phi_mu[:,:,1] += phi_surf[:,:,pname['mu-_r'].lidx:pname['mu-_r'].lidx+len(Prop.E_mu)]
    
    return phi_mu
    
"""

def MCEq_mu_ice(Prop,
                #interaction_model="SIBYLL-2.3c",
                dEdX = None, # Use mceq's default muon energy loss function? (If false, uses Heisinger)
                #allow_int = False, # Allow interactions?
                ignore_decays = False, # Allow decays?
                Phi_atm=None # surface flux
               ):
    if Prop is None: # run function with Prop=None to get input stage
        return 'atm'
    if Phi_atm is None:
        Phi_atm = Prop.Phi['atm']
    #Phi_atm
    #axis0 - Atmospheric Model
    #axis1 - Zenith Angle
    #axis2 - Muon Charge (positive, negative)
    #axis3 - Muon Energy
    #print(dEdX_mceq, allow_int, allow_dec)

    #import mceq_config as config
    from MCEq import config
    config.debug_level = 0
    config.enable_default_tracking = False
    config.e_min = Prop.E_mu_bins[1]
    config.e_max = Prop.E_mu_bins[-1]
    config.max_density = Prop.rho_ice
    config.dedx_material='ice'
    config.leading_process = "interactions"
    medium = 'ice'

    target = GeneralizedTarget(len_target=Prop.z_bins[-1]*100, env_density = Prop.rho_ice, env_name = 'ice')
    
    interaction_model="SIBYLL-2.3c"
    
    mceq = MCEqRun(
        interaction_model=interaction_model,
        theta_deg = 0,
        density_model = target,
        medium=medium,
        primary_model = (pm.GaisserHonda, None),
    )

    pname = mceq.pman.pname2pref
    
    # Build array for mceq primary particles
    # We need the 2nd to last axis to be Particle Species & Energy
    # for the matrix multiplication to line up
    phi0 = Phi_atm.reshape((*Phi_atm.shape[:-2],-1,1))
    
    phi_cut = np.append(np.arange(pname['mu+'].lidx,pname['mu+'].uidx), np.arange(pname['mu-'].lidx,pname['mu-'].uidx))

    #phi_mu = np.zeros((np.shape(Phi_atm)[0],2,len(Prop.E_mu),len(Prop.h_bins)))
        
    # Calculate interaction & decay matrices
    #print('Calculating interaction & decay matrices...')

    if not (dEdX is None): # sets dE/dX to Heisinger approximation of Gaisser+Stanev energy loss function
        if dEdX == "Heisinger":
            dEdX = -(Prop.a + Prop.b*mceq._energy_grid.c)/100
        mceq.matrix_builder._pman[-13].dEdX = dEdX
        mceq.matrix_builder._pman[13].dEdX = dEdX

    int_m, dec_m = mceq.matrix_builder.construct_matrices()

    int_m = int_m[phi_cut][:, phi_cut].toarray()
    dec_m = np.diag(dec_m[phi_cut][:, phi_cut].toarray()).reshape((-1,1))
    
    if ignore_decays:
        dec_m = 0
    
    # Calculate integration path for max zenith angle
    #print('Calculating integration path...')
    #target = GeneralizedTarget(len_target=Prop.z_bins[-1]*100/Prop.cosTH[-1], env_density = Prop.rho_ice, env_name = 'ice')
    #target.mat_list = [[Prop.z_bins[j]*100/Prop.cosTH[-1], Prop.z_bins[j+1]*100/Prop.cosTH[-1], Prop.rho[j], 'ice'] for j in range(len(Prop.z_bins)-1)]
    #target._update_variables()
    
    #mceq.set_density_model(target)
    
    #mceq._calculate_integration_path(int_grid=Prop.h_bins/Prop.cosTH[-1]*100, grid_var='X')
    #nsteps, dX_max, rho_inv, grid_idcs = mceq.integration_path
    
    dXmax = min(config.stability_margin / mceq.matrix_builder.max_lint, config.dXmax)
    dX_grid = [[dXmax]*int(dX//dXmax)+[dX%dXmax] for dX in np.append([0], Prop.dh/Prop.cosTH[-1]*100)]
    
    dX_max = np.concatenate(dX_grid)
    X = np.cumsum(dX_max)
    nsteps = len(dX_max)
    rho_inv = 1/Prop.rho[np.digitize(X*Prop.cosTH[-1], Prop.h_bins[:-1])-1]
    grid_idcs = (np.cumsum([len(x) for x in dX_grid])-1).tolist()
    
    phc = np.copy(phi0)
    grid_sol = np.zeros((len(grid_idcs), *np.shape(phc))) # grid_sol begins with the right shape, to avoid restructuring
    
    dX = dX_max.reshape((-1,1,1,1,1)) * (Prop.cosTH[-1]/Prop.cosTH).reshape((1,1,-1,1,1))
    grid_step = 0
    #print('Integrating...')
    
    for step in tqdm(range(nsteps)): # added option for tqdm progress bar
        phc += (int_m @ phc + rho_inv[step] * dec_m * phc) * dX[step]
        phc[phc<1e-250] = 0. # exreme low values set to 0, improving efficiency for large slant depths

        if (grid_idcs and grid_step < len(grid_idcs)
                and grid_idcs[grid_step] == step):
            grid_sol[grid_step] = np.copy(phc) # grid_sol no longer appends
            grid_step += 1
            
    phi_mu = np.sum(np.moveaxis(grid_sol, 0, -1).reshape((*Phi_atm.shape,-1))*Prop.dcosTH.reshape((1,-1,1,1,1)), axis=1) * 2 * np.pi
    
    return phi_mu
"""
    for i in tqdm(range(len(Prop.cosTH))):
        target = GeneralizedTarget(len_target=Prop.z_bins[-1]*100/Prop.cosTH[i], env_density = Prop.rho_ice, env_name = 'ice')
        target.mat_list = [[Prop.z_bins[j]*100/Prop.cosTH[i], Prop.z_bins[j+1]*100/Prop.cosTH[i], Prop.rho[j], 'ice'] for j in range(len(Prop.z_bins)-1)]
        target._update_variables()

        mceq.set_density_model(target)

        int_grid=Prop.h_bins/Prop.cosTH[i]*100
        
        mceq._calculate_integration_path(int_grid=int_grid, grid_var='X')

        nsteps, dX, rho_inv, grid_idcs = mceq.integration_path

        phc = np.copy(phi0[:,i])

        dXaccum = 0.
        grid_sol = np.zeros((len(grid_idcs), *np.shape(phc))) # grid_sol begins with the right shape, to avoid restructuring
        grid_step = 0

        for step in range(nsteps): # added option for tqdm progress bar
            phc += (int_m @ phc + dec_m @ (rho_inv[step] * phc)) * dX[step]
            phc[phc<1e-250] = 0. # exreme low values set to 0, improving efficiency for large slant depths

            if (grid_idcs and grid_step < len(grid_idcs)
                    and grid_idcs[grid_step] == step):
                grid_sol[grid_step] = np.copy(phc) # grid_sol no longer appends
                grid_step += 1
        
        # mceq.grid_sol shape is (z, models, charge&energies, N/A)
        # we need (models, charge, energies, z)
        phi_mu += np.moveaxis(grid_sol, 0, -1).reshape(phi_mu.shape)*Prop.dcosTH[i]
    
    return phi_mu * 2 * np.pi
"""

def MCEq_ice(Prop, interaction_model="SIBYLL-2.3c", solver='default'):
    if Prop is None: # run function with Prop=None to get input stage
        return 'atm'

    """


    Parameters
    ------------------
    Phi_atm - 

    interaction_model - 

    solver - 


    Returns
    -----------------------
    phi_mu - 

    """

    # Use MCEq to propagate atmospheric muons underground
    Phi_atm = Prop.Phi['atm']
    #Phi_atm
    #axis0 - Atmospheric Model
    #axis1 - Zenith Angle
    #axis2 - Muon Charge (positive, negative)
    #axis3 - Muon Energy

    #import mceq_config as config
    from MCEq import config
    config.debug_level = 0
    config.enable_default_tracking = False
    config.e_min = Prop.E_mu_bins[1]
    config.e_max = Prop.E_mu_bins[-1]
    config.max_density = Prop.rho_ice
    config.dedx_material='ice'
    medium = 'ice'

    target = GeneralizedTarget(len_target=Prop.z_bins[-1]*100, env_density = Prop.rho_ice, env_name = 'ice')

    mceq = MCEqRun(
        interaction_model=interaction_model,
        theta_deg = 0,
        density_model = target,
        medium=medium,
        primary_model = (pm.GaisserHonda, None),
    )

    pname = mceq.pman.pname2pref
    
    # Build 2D array for mceq primary particles
    # We need the 2nd to last axis to be Particle Species & Energy
    # for the matrix multiplication to line up
    phi0 = np.zeros((len(Prop.cosTH), len(mceq._phi0), np.shape(Phi_atm)[0]))
    phi0[:,pname['mu+'].lidx:pname['mu+'].uidx] = np.moveaxis(Phi_atm[:,:,0], 0, -1)
    phi0[:,pname['mu-'].lidx:pname['mu-'].uidx] = np.moveaxis(Phi_atm[:,:,1], 0, -1)

    phi_mu = np.zeros((np.shape(Phi_atm)[0],2,len(Prop.E_mu),len(Prop.h_bins)))

    for i in tqdm(range(len(Prop.cosTH))):
        target = GeneralizedTarget(len_target=Prop.z_bins[-1]*100/Prop.cosTH[i], env_density = Prop.rho_ice, env_name = 'ice')
        target.mat_list = [[Prop.z_bins[j]*100/Prop.cosTH[i], Prop.z_bins[j+1]*100/Prop.cosTH[i], Prop.rho[j], 'ice'] for j in range(len(Prop.z_bins)-1)]
        target._update_variables()

        mceq.set_density_model(target)
        mceq._phi0 = phi0[i]

        if solver == 'default':
            solve_mceq(mceq, int_grid=Prop.h_bins/Prop.cosTH[i]*100)
        else:
            # 'numpy', 'cuda', or 'mkl'
            config.kernel_config = solver
            mceq.solve(int_grid=Prop.h_bins/Prop.cosTH[i]*100)
        
        # mceq.grid_sol shape is (z, energies, models)
        # we need (models, energies, z)
        phi_deep = np.swapaxes(mceq.grid_sol, 0, -1)

        phi_mu[:,0] += phi_deep[:,pname['mu+'].lidx:pname['mu+'].uidx]*Prop.dcosTH[i]
        phi_mu[:,0] += phi_deep[:,pname['mu+_l'].lidx:pname['mu+_l'].uidx]*Prop.dcosTH[i]
        phi_mu[:,0] += phi_deep[:,pname['mu+_r'].lidx:pname['mu+_r'].uidx]*Prop.dcosTH[i]

        phi_mu[:,1] += phi_deep[:,pname['mu-'].lidx:pname['mu-'].uidx]*Prop.dcosTH[i]
        phi_mu[:,1] += phi_deep[:,pname['mu-_l'].lidx:pname['mu-_l'].uidx]*Prop.dcosTH[i]
        phi_mu[:,1] += phi_deep[:,pname['mu-_r'].lidx:pname['mu-_r'].uidx]*Prop.dcosTH[i]

    return phi_mu * 2 * np.pi

def MCEq_mu_atmice(Prop, interaction_model="SIBYLL-2.3c", density_model=('CORSIKA', ('USStd', None)), elev=None, dEdX=None, ignore_decays=False):
    if Prop is None: # run function with Prop=None to get input stage
        return 'primary'
    
    Phi_atm = MCEq_atm(Prop, interaction_model, density_model, elev)
    
    Phi_ice = MCEq_mu_ice(Prop, interaction_model, dEdX, ignore_decays, Phi_atm)
    
    return Phi_ice
    

def MCEq_atmice(Prop, interaction_model="SIBYLL-2.3c", density_model=('CORSIKA', ('USStd', None)), elev=None, solver='default'):
    if Prop is None: # run function with Prop=None to get input stage
        return 'primary'

    """


    Parameters
    ------------------
    Phi0 - 

    interaction_model - 

    density_model - 

    elev - 

    solver - 


    Returns
    ---------------------
    phi_mu - 

    """

    # Use MCEq to propagate primary flux to atmospheric muons to underground muons
    if elev is None:
        elev = Prop.elev
    
    Phi0 = Prop.Phi['primary']
    #Phi0
    #axis0 - Primary Model (Energy spectrum & Time dependence)
    #axis1 - Particle Species (proton, neutron)
    #axis2 - Primary Energy

    #import mceq_config as config
    from MCEq import config
    config.debug_level = 0
    config.h_obs = elev # elevation in (m) of Dome-C
    config.enable_default_tracking = False
    config.e_min = Prop.E_bins[1]
    config.e_max = Prop.E_bins[-1]
    config.max_density = 0.001225
    config.dedx_material = 'air'

    mceq_air = MCEqRun(
        interaction_model=interaction_model,
        theta_deg = 0,
        density_model = density_model,
        #medium=medium,
        primary_model = (pm.GaisserHonda, None),
    )

    #import mceq_config as config
    from MCEq import config
    config.debug_level = 0
    config.enable_default_tracking = False
    config.e_min = Prop.E_mu_bins[1]
    config.e_max = Prop.E_mu_bins[-1]
    config.max_density = Prop.rho_ice
    config.dedx_material='ice'
    medium = 'ice'

    target = GeneralizedTarget(len_target=Prop.z_bins[-1]*100, env_density = Prop.rho_ice, env_name = 'ice')

    mceq_ice = MCEqRun(
        interaction_model=interaction_model,
        theta_deg = 0,
        density_model = target,
        medium=medium,
        primary_model = (pm.GaisserHonda, None),
    )

    pname = mceq_air.pman.pname2pref
    
    # Build 2D array for mceq primary particles
    # We need the 2nd to last axis to be Particle Species & Energy
    # for the matrix multiplication to line up
    phi0 = np.zeros((len(mceq_air._phi0), np.shape(Phi0)[0]))
    phi0[pname['p+'].lidx:pname['p+'].uidx] = Phi0[:,0].T
    phi0[pname['n0'].lidx:pname['n0'].uidx] = Phi0[:,1].T

    phi_mu = np.zeros((np.shape(Phi0)[0],2,len(Prop.E_mu),len(Prop.h_bins)))

    for i in tqdm(range(len(Prop.cosTH))):
        #dX_air, dz_air = get_mceq_path(mceq_air, cosTH[i])
        #phi_surf = mceq_integrate(phi0, dX_air, dz_air, int_m_air, dec_m_air)
        mceq_air.set_theta_deg(180*np.arccos(Prop.cosTH[i])/np.pi)
        mceq_air._phi0 = phi0

        if solver == 'default':
            solve_mceq(mceq_air)
        else:
            # 'numpy', 'cuda', or 'mkl'
            config.kernel_config = solver
            mceq_air.solve()

        target = GeneralizedTarget(len_target=Prop.z_bins[-1]*100/Prop.cosTH[i], env_density = Prop.rho_ice, env_name = 'ice')
        target.mat_list = [[Prop.z_bins[j]*100/Prop.cosTH[i], Prop.z_bins[j+1]*100/Prop.cosTH[i], Prop.rho[j], 'ice'] for j in range(len(Prop.z_bins)-1)]
        target._update_variables()

        mceq_ice.set_density_model(target)
        mceq_ice._phi0 = mceq_air._solution[np.arange(len(phi0))%len(Prop.E)<len(Prop.E_mu)]

        if solver == 'default':
            solve_mceq(mceq_ice, int_grid=Prop.h_bins/Prop.cosTH[i]*100)
        else:
            # 'numpy', 'cuda', or 'mkl'
            config.kernel_config = solver
            mceq_ice.solve(int_grid=Prop.h_bins/Prop.cosTH[i]*100)

        # mceq.grid_sol shape is (z, energies, models)
        # we need (models, energies, z)
        phi_deep = np.swapaxes(mceq_ice.grid_sol, 0, -1)
        
        pname = mceq_ice.pman.pname2pref

        phi_mu[:,0] += phi_deep[:,pname['mu+'].lidx:pname['mu+'].lidx+len(Prop.E_mu)]*Prop.dcosTH[i]
        phi_mu[:,0] += phi_deep[:,pname['mu+_l'].lidx:pname['mu+_l'].lidx+len(Prop.E_mu)]*Prop.dcosTH[i]
        phi_mu[:,0] += phi_deep[:,pname['mu+_r'].lidx:pname['mu+_r'].lidx+len(Prop.E_mu)]*Prop.dcosTH[i]

        phi_mu[:,1] += phi_deep[:,pname['mu-'].lidx:pname['mu-'].lidx+len(Prop.E_mu)]*Prop.dcosTH[i]
        phi_mu[:,1] += phi_deep[:,pname['mu-_l'].lidx:pname['mu-_l'].lidx+len(Prop.E_mu)]*Prop.dcosTH[i]
        phi_mu[:,1] += phi_deep[:,pname['mu-_r'].lidx:pname['mu-_r'].lidx+len(Prop.E_mu)]*Prop.dcosTH[i]

    return phi_mu * 2 * np.pi

def daemonflux_atm(Prop):
    if Prop is None: # run function with Prop=None to get input stage
        return ''

    """


    Parameters
    --------------------


    Returns
    --------------------
    Phi_atm - 

    """
    
    df_cut = (Prop.E_mu <= 1e9)

    daemon_flux_pos = daemonflux.Flux(location='generic').flux(Prop.E_mu[df_cut], np.arccos(Prop.cosTH)*180/np.pi, 'mu+')/np.reshape(Prop.E_mu[df_cut]**3, (-1,1))
    daemon_flux_neg = daemonflux.Flux(location='generic').flux(Prop.E_mu[df_cut], np.arccos(Prop.cosTH)*180/np.pi, 'mu-')/np.reshape(Prop.E_mu[df_cut]**3, (-1,1))

    Phi_atm = np.zeros((1, len(Prop.cosTH), 2, len(Prop.E_mu)))
    #pname = self.mceq.pman.pname2pref
    Phi_atm[0, :, 0, df_cut] = daemon_flux_pos # positive muons
    Phi_atm[0, :, 1, df_cut] = daemon_flux_neg # negative muons

    #Phi_atm
    #axis0 - Primary Model
    #axis1 - Zenith Angle
    #axis2 - Muon Charge (positive, negative)
    #axis3 - Muon Energy

    return Phi_atm

def get_proposal(Prop, mu_pos=True):
    if mu_pos:
        mu = pp.particle.MuPlusDef()
    else:
        mu = pp.particle.MuMinusDef()
    cuts = pp.EnergyCutSettings(500, 0.05, True)

    medium = pp.medium.Water()

    args = {"particle_def": mu, "target": medium, "interpolate": True, "cuts": cuts}

    # Initialise standard cross-sections, then specify and set parametrisation models

    cross_sections = pp.crosssection.make_std_crosssection(**args)

    brems_param = pp.parametrization.bremsstrahlung.KelnerKokoulinPetrukhin(lpm=False)
    epair_param = pp.parametrization.pairproduction.KelnerKokoulinPetrukhin(lpm=False)
    ionis_param = pp.parametrization.ionization.BetheBlochRossi(energy_cuts=cuts)
    shado_param = pp.parametrization.photonuclear.ShadowButkevichMikheyev()
    photo_param = pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97(
        shadow_effect=shado_param
    )

    cross_sections[0] = pp.crosssection.make_crosssection(brems_param, **args)
    cross_sections[1] = pp.crosssection.make_crosssection(epair_param, **args)
    cross_sections[2] = pp.crosssection.make_crosssection(ionis_param, **args)
    cross_sections[3] = pp.crosssection.make_crosssection(photo_param, **args)

    # Propagation utility

    collection = pp.PropagationUtilityCollection()

    collection.interaction = pp.make_interaction(cross_sections, True)
    collection.displacement = pp.make_displacement(cross_sections, True)
    collection.time = pp.make_time(cross_sections, mu, True)
    collection.decay = pp.make_decay(cross_sections, mu, True)

    pp.PropagationUtilityCollection.cont_rand = False

    utility = pp.PropagationUtility(collection=collection)

    # Other settings

    pp.do_exact_time = False

    # Set up geometry

    detector = pp.geometry.Sphere(
        position=pp.Cartesian3D(0, 0, 0), radius=10000000, inner_radius=0
    )
    density_distr = pp.density_distribution.density_homogeneous(
        mass_density=Prop.rho_ice
    )

    return pp.Propagator(mu, [(detector, utility, density_distr)])

def proposal_loop(Prop, propagator, N, energy):
    
    mu_initial = pp.particle.ParticleState()
    mu_initial.energy = (energy + Prop.mu_mass) * 1e3 # Muon Total Energy (MeV)
    mu_initial.position = pp.Cartesian3D(0, 0, 0)
    mu_initial.direction = pp.Cartesian3D(0, 0, -1)

    slant_depth = Prop.h_bins[-1]/Prop.cosTH[-1]/Prop.rho_ice * 1e2 # convert meters-water-equivalent to cm
    
    print ('Running {} Simulations at {:.1e} GeV...'.format(N, energy))

    tracks = [propagator.propagate(mu_initial, slant_depth) for i in tqdm(range(N))]
    
    E = np.concatenate([np.array(t.track_energies()) * 1e-3 - Prop.mu_mass for t in tracks]) # convert MeV Total to GeV Kinetic
    D = np.concatenate([np.array(t.track_propagated_distances()) * Prop.rho_ice * 1e-2 for t in tracks]) # convert cm to m.w.e slant depth
    
    counts = np.zeros((len(Prop.cosTH), len(Prop.E_mu), len(Prop.h_bins)), dtype=int)
    
    i_depths = np.digitize(D*Prop.cosTH.reshape((-1,1)), Prop.h_bins, right=True)
    i_energies = np.digitize(E, Prop.E_mu_bins[:-1])-1
    
    # Count muons into energy bins for each depth and zenith angle
    # This method takes advantage of the fact that each track starts at 0 distance traveled
    # Thus, if the next recorded event occurred at 0 distance, it's from the next track, and so the current one is the last event of this track

    # Loop over each event recorded
    # We start at -1 because it makes indexing easier
    for i in tqdm(range(-1, len(D)-1)):
        if i_energies[i] != -1:
            for j,d in enumerate(i_depths):
                if d[i]<d[i+1]:
                    counts[j, i_energies[i], d[i]:d[i+1] if D[i+1]!=0. else d[i]:] += 1
                    
    # multiply by dOmega for each zenith angle and divide by dE for each underground energy
    return counts/N * Prop.dcosTH.reshape((-1,1,1)) / Prop.dE_mu.reshape((1,-1,1)) * 2 * np.pi

def get_survival_tensor(Prop, N=10**5, E_max=1e3):
    survival_tensor = np.zeros((2,len(Prop.E_mu),len(Prop.cosTH),len(Prop.E_mu),len(Prop.h_bins)))
    # axis0 - muon charge
    # axis1 - muon surface energy
    # axis2 - zenith angle
    # axis3 - muon underground energy
    # axis4 - depth
    
    for i,mu_pos in enumerate([True, False]):
        propagator = get_proposal(Prop, mu_pos)
        for j,energy in enumerate(Prop.E_mu[Prop.E_mu<E_max]):
            survival_tensor[i,j] = proposal_loop(Prop, propagator, N, energy)
    
    return survival_tensor * Prop.dE_mu.reshape((1,-1,1,1,1))

def proposal_ice(Prop, file='survival_tensor_TEST.npy', new_tensor=False):
    if Prop is None:
        return 'atm'
    
    if new_tensor:
        survival_tensor = get_survival_tensor(Prop)
        
        if not file is None:
            np.save(file, survival_tensor)
    else:
        survival_tensor = np.load(file)
    
    phi_atm = Prop.Phi['atm']
    
    PA = phi_atm.swapaxes(1,2).reshape((len(phi_atm),2,-1))
    ST = survival_tensor.swapaxes(1,3).reshape((2,len(Prop.E_mu),-1,len(Prop.h_bins)))
    
    phi_ice = np.moveaxis([PA[:,i] @ S for i,S in enumerate(ST)], 2,0)
    
    return phi_ice

"""
def mute_ice(Prop):
    if Prop is None:
        return ''
    
    K_mu = 1.268 # positive-to-negative muon ratio (mute doesn't track muon charge it seems)
    
    mtc.clear()
    mtc.set_overburden('flat')
    mtc.shallow_extrapolation = True # lets mute extrapolate to depths above 500 m.w.e.
    mtc.set_medium('ice')
    #mtc.set_density(Prop.rho_ice) # once this is depricated, change to below
    mtc.set_reference_density(Prop.rho_ice)
    
    mtc._E_BINS = Prop.E_mu_bins*1e3 # units: MeV
    mtc._E_WIDTHS = Prop.dE_mu*1e3
    mtc.ENERGIES = Prop.E_mu*1e3
    
    Phi_ice = np.zeros((1,2,len(Prop.E_mu),len(Prop.h)))
    
    for i in tqdm(range(len(Prop.h))):
        mtc._vertical_depth = Prop.h[i]
        mtc.slant_depths = Prop.h[i]/Prop.cosTH_bins[:-1]
        mtc.angles = np.degrees(np.arccos(Prop.cosTH_bins[:-1]))
        Phi_ice[0,0,:,i] = mtu.calc_u_e_spect()
    
    # split up intensity between positive and negative muons
    Phi_ice[0,1] = Phi_ice[0,0] * 1/(K_mu+1)
    Phi_ice[0,0] = Phi_ice[0,0] * K_mu/(K_mu+1)
    
    Phi_ice *= 1e3 #convert units from (cm^2 s MeV)^-1 to (cm^2 s GeV)^-1
    
    #Phi_ice
    #axis0 - Underice Model
    #axis1 - Muon Charge (positive, negative)
    #axis2 - Muon Energy
    #axis3 - depth (top -> bottom)
    
    return Phi_ice
"""

def Dyonisius_prod(Prop, sigma_E = None, E_sigma = None, alpha = None, N = None, f_tot = None):
    if Prop is None: # run function with Prop=None to get input stage
        return 'ice'
    
    if sigma_E is None:
        sigma_E = Prop.sigma_E
    if E_sigma is None:
        E_sigma = Prop.E_sigma
    if alpha is None:
        alpha=Prop.alpha
    if N is None:
        N = Prop.N
    if f_tot is None:
        f_tot = Prop.f_tot

    """


    Parameters
    --------------------
    Phi_ice - 

    sigma_E - 

    alpha - 

    N - 

    f_tot - 


    Returns
    --------------------
    P_14C - 

    """

    # Calculate production rates
    
    Phi_ice = Prop.Phi['ice']
    #Phi_ice
    #axis0 - Underice Model
    #axis1 - Muon Charge (positive, negative)
    #axis2 - Muon Energy
    #axis3 - depth (top -> bottom)

    # NOTE: depth starts measured on the bin EDGES and is returned on the bin CENTERS
    # (This is because we need to take a derivative)

    sigma_0 = sigma_E / E_sigma**alpha

    P_neg = f_tot * -np.diff(np.sum(Phi_ice[:,1] * np.reshape(Prop.dE_mu, (1,-1,1)), axis=1), axis=-1)/np.reshape(Prop.dh, (1,-1))

    P_fast = sigma_0 * N * np.sum((Phi_ice[:,:,:,1:]+Phi_ice[:,:,:,:-1])/2 * np.reshape(Prop.E_mu**alpha * Prop.dE_mu, (1,1,-1,1)), axis=(1,2))

    return np.moveaxis([P_fast, P_neg], 0, 1) /100 * 60 * 60 * 24 * 365.25 # g^-1, a^-1

## CO Profiles

def diag_sum(A, off=None, axis1=-2, axis2=-1):
    # sums along the upper diagonals of two axes in an array
    # the new axis replaces axis1; axis2 is eliminated.
    if off is None:
        off = np.flip(range(np.shape(A)[axis2]))
    return np.moveaxis(np.array([np.trace(A, offset=i, axis1=axis1, axis2=axis2) for i in tqdm(off)]), 0, axis1 if axis1>=0 else axis1+1)

def Basic_flow(Prop, f_factors = None, f_t = None, lambd=None):
    if Prop is None: # run function with Prop=None to get input stage
        return 'prod'
    
    if f_factors is None:
        f_factors = Prop.f_factors
    if f_t is None:
        f_t = np.ones(len(Prop.t[Prop.i_start:]))
    if lambd is None:
        lambd = Prop.lambd

    """


    Parameters
    -----------------
    P_14C - 

    f_factors - 

    f_t - 

    lambd - 


    Returns
    -------------------
    CO - 

    """

    # Shift past 14CO down and decay
    # (AKA multiply by survival fraction and sum along upper diagonal)
    
    P_14C = np.copy(Prop.Phi['prod'])[:,:,Prop.i_start:]
    #P_14C - 14C Production Rate
    #axis0 - Production Model
    #axis1 - Production Mode (fast, neg)
    #axis2 - depth (top -> bottom)

    lambda_dt = np.reshape((1-lambd)**(Prop.t[Prop.i_start:][-1]-Prop.t[Prop.i_start:]) * Prop.dt[Prop.i_start:] * f_t, (1,-1))
    
    P_f = np.sum(P_14C*np.reshape(f_factors, (1,-1,1)), axis=1)
    
    return np.moveaxis([np.sum(P_f[:,:i+1]*lambda_dt[:,-i-1:], axis=-1) for i in tqdm(range(np.shape(P_f)[-1]))], 0, -1)
    #return diag_sum(np.expand_dims(np.sum(P_14C[:,:,Prop.i_start:]*np.reshape(f_factors, (1,-1,1)), axis=1), axis=-1) * lambda_dt)

def flow_14C(Prop, f_t = None, lambd=None, use_tqdm=True):
    if Prop is None: # run function with Prop=None to get input stage
        return 'prod'
    
    if f_t is None:
        f_t = np.ones(len(Prop.t[Prop.i_start:]))
    if lambd is None:
        lambd = Prop.lambd

    """


    Parameters
    -----------------
    P_14C - 

    f_t - 

    lambd - 


    Returns
    -------------------
    CO - 

    """

    # Shift past 14CO down and decay
    # (AKA multiply by survival fraction and sum along upper diagonal)
    
    P_14C = np.copy(Prop.Phi['prod'])[:,:,Prop.i_start:]
    #P_14C - 14C Production Rate
    #axis0 - Production Model
    #axis1 - Production Mode (fast, neg)
    #axis2 - depth (top -> bottom)

    lambda_dt = np.reshape((1-lambd)**(Prop.t[-1]-Prop.t[Prop.i_start:]) * Prop.dt[Prop.i_start:] * f_t, (1,1,-1))
    
    return np.moveaxis([np.sum(P_14C[:,:,:i+1]*lambda_dt[:,:,-i-1:], axis=-1) for i in (tqdm(range(np.shape(P_14C)[-1])) if use_tqdm else range(np.shape(P_14C)[-1]))], 0, -1)
    #return diag_sum(np.expand_dims(P_14C[:,:,Prop.i_start:], axis=-1) * lambda_dt)
    
def flow_14C_response(Prop, lambd=None):
    if Prop is None: # run function with Prop=None to get input stage
        return 'prod'
    
    if lambd is None:
        lambd = Prop.lambd

    # Shift past 14CO down and decay
    # (AKA multiply by survival fraction and sum along upper diagonal)
    
    P_14C = np.copy(Prop.Phi['prod'])[:,:,Prop.i_start:]
    #P_14C - 14C Production Rate
    #axis0 - Production Model
    #axis1 - Production Mode (fast, neg)
    #axis2 - depth (top -> bottom)

    C_response = np.zeros((*np.shape(P_14C),np.shape(P_14C)[-1]))
    #axis0 - Production Model
    #axis1 - Production Mode (fast, neg)
    #axis2 - Time (past -> present)
    #axis3 - depth (top -> bottom)
    
    for i in range(len(Prop.t[Prop.i_start:])):
        C_response[:,:,i,-i-1:] = P_14C[:,:,:i+1]
    
    return C_response * np.reshape(np.exp(-lambd*(Prop.t_bins[-1]-Prop.t[Prop.i_start:])) * Prop.dt[Prop.i_start:], (1,1,-1,1))
    #return diag_sum(np.expand_dims(P_14C[:,:,Prop.i_start:], axis=-1) * lambda_dt)

# Reproducition of 14C analysis at Taylor Glacier, Antarctica by Dyonisius et al. (2023)
# https://tc.copernicus.org/articles/17/843/2023/
def Taylor_flow(
    Prop, # Propagator object, contains data from calculation of production rates
    flow='trim', # flowpaths to use ('sample', 'all', or 'trim')
    f_t=None, # muon flux scaling factor over time
    lambd=None # 14C decay constant (default = 1.216e-4 yr^-1)
):
    if Prop is None: # run function with Prop=None to get input stage
        return 'prod'
    if lambd is None:
        lambd = Prop.lambd
        
    z_P = Prop.z_bins
    
    P_14C = interp1d(Prop.z, Prop.Phi['prod'], bounds_error=False, assume_sorted=True)(z_P)
    #P_14C - 14C Production Rates [molecules /g /yr]
    #axis0 - Production Model
    #axis1 - Production Mode (fast, neg)
    #axis2 - depth (top -> bottom)
    
    if flow=='best' or flow=='sample':
        flowpath = loadmat('matlab/MC_14CO_mod/flowpath_sample.mat')
        h_age = flowpath['h_age']
    elif flow=='all':
        flowpath = loadmat('matlab/MC_14CO_mod/flowpath_all.mat')
        h_age = flowpath['h_age']
    else:
        flowpath = loadmat('matlab/MC_14CO_mod/flowpath_trim.mat')
        h_age = np.swapaxes(flowpath['h_trim'], 1,2)
    # h_age : ice parcel depth as a function of age [m]
    #axis0 - sample depth
    #axis1 - flowpath
    #axis2 - ice parcel age
    
    age = flowpath['age'][0]
    # age : ice parcel ages [years]
    #axis0 - ice parcel age
    
    z_baseline = -699.3415
    z_init = 575. - (h_age[:,:,-1] - z_baseline)
    
    dt = 0.5
    t = np.arange(age[0], age[-1], dt) # time integration steps
    
    if f_t is None:
        f_t = np.ones(len(t))
    
    parcel_depth = np.flip(interp1d(age, h_age, assume_sorted=True)(t+dt)*-1., axis=-1)
    # parcel_depth : depth of parcel over integration time [m]
    # axis0 - sample depth
    # axis1 - flowpath
    # axis2 - integration time (= flip(age))
    
    # interpolator function for Production Rates vs depth
    P_interp = interp1d(z_P, P_14C, assume_sorted=True)
    
    C = P_interp(z_init)/lambd # equilibrium 14CO concentration at starting depth
    
    # integrate
    for i in tqdm(range(len(t))):
        C += (P_interp(parcel_depth[:,:,i])*f_t[i] - C*lambd)*dt
    
    C = np.moveaxis(C, -1,0)
    # C : final 14CO concentration at sample depth [molecules /g]
    #axis0 - flowpath
    #axis1 - Production Model
    #axis2 - Production Mode (fast, neg)
    #axis3 - sample depth
    
    return C

def Taylor_flow_response(
    Prop, # Propagator object, contains data from calculation of production rates
    flow='trim', # flowpaths to use ('sample', 'all', or 'trim')
    t_bins=None,
    lambd=None # 14C decay constant (default = 1.216e-4 yr^-1)
):
    if Prop is None: # run function with Prop=None to get input stage
        return 'prod'
    
    if lambd is None:
        lambd = Prop.lambd
        
    z_P = Prop.z
    
    P_14C = np.copy(Prop.Phi['prod'])
    #P_14C - 14C Production Rates [molecules /g /yr]
    #axis0 - Production Model
    #axis1 - Production Mode (fast, neg)
    #axis2 - depth (top -> bottom)
    
    if flow=='best' or flow=='sample':
        flowpath = loadmat('matlab/MC_14CO_mod/flowpath_sample.mat')
        h_age = flowpath['h_age']
    elif flow=='all':
        flowpath = loadmat('matlab/MC_14CO_mod/flowpath_all.mat')
        h_age = flowpath['h_age']
    else:
        flowpath = loadmat('matlab/MC_14CO_mod/flowpath_trim.mat')
        h_age = np.swapaxes(flowpath['h_trim'], 1,2)
    # h_age : ice parcel depth as a function of age [m]
    #axis0 - sample depth
    #axis1 - flowpath
    #axis2 - ice parcel age
    
    age = flowpath['age'][0]
    # age : ice parcel ages [years]
    #axis0 - ice parcel age
    
    z_baseline = -699.3415
    z_init = 575. - (h_age[:,:,-1] - z_baseline)
    
    if t_bins is None:
        dt = 0.5
        t_bins = -np.flip(np.arange(0, age[-1]-age[0]+dt, dt)) # time integration steps (from -t_width (past) to 0 (present))
    t = (t_bins[:-1]+t_bins[1:])/2
    dt = np.diff(t_bins)
    
    #parcel_depth = np.flip(interp1d(age, h_age, assume_sorted=True)(-t)*-1., axis=-1)
    parcel_depth = interp1d(age, h_age, assume_sorted=True, fill_value=(h_age[:,:,0], h_age[:,:,-1]), bounds_error=False)(-t)*-1.
    # parcel_depth : depth of parcel over integration time [m]
    # axis0 - sample depth
    # axis1 - flowpath
    # axis2 - integration time
    
    # interpolator function for Production Rates vs depth
    P_interp = interp1d(z_P, P_14C, assume_sorted=True)
    
    # equilibrium 14CO concentration at starting depth
    C_init = np.moveaxis(P_interp(z_init)/lambd * np.exp(-lambd*(t_bins[-1]-t_bins[0])), -1, 0)
    # C : final 14CO concentration at sample depth [molecules /g]
    #axis0 - flowpath
    #axis1 - Production Model
    #axis2 - Production Mode (fast, neg)
    #axis3 - sample depth
    
    # response matrix
    C_response = np.moveaxis(P_interp(parcel_depth) * (np.exp(-lambd*(t_bins[-1]-t)) * dt).reshape((1,1,1,1,-1)), (-2, -1), (0, -2))
    #axis0 - flowpath
    #axis1 - Production Model
    #axis2 - Production Mode (fast, neg)
    #axis3 - integration time
    #axis4 - sample depth
    
    return C_response, C_init

"""
def general_flow(
    Prop,
    age = None,
    h_age = None,
    z_init = None,
    f_t = None,
    lambd = None
):
    if Prop is None: # run function with Prop=None to get input stage
        return 'prod'
    
    if age is None:
        age = Prop.t_bins
    if h_age is None:
        h_age = Prop.z_bins
    if f_t is None:
        f_t = np.ones(len(age))
    if lambd is None:
        lambd = Prop.lambd
    
    # time integration steps
    t_bins = Prop.t_bins
    t = Prop.t
    dt = Prop.dt
    
    age_end = interp1d(h_age, age)(z_init)
    age_int = # age = age_end-
    
    parcel_depth = interp1d(age, h_age, assume_sorted=True)(age_int)
    # parcel_depth : depth of parcel over integration time [m]
    # axis0 - sample depth
    # axis1 - flowpath
    # axis2 - integration time (= flip(age))
    
    # interpolator function for Production Rates vs depth
    P_interp = interp1d(z_P, P_14C, assume_sorted=True)
    
    if z_init = None:
        C = 0.
    else:
        C = P_interp(z_init)/lambd # equilibrium 14CO concentration at starting depth
    
    # integrate
    for i in tqdm(range(len(t))):
        C += (P_interp(parcel_depth[:,:,i]) - C*lambd)*dt
    
    C = np.moveaxis(C, -1,0)
    # C : final 14CO concentration at sample depth [molecules /g]
    #axis0 - Production Model
    #axis1 - Production Mode (fast, neg)
    #axis2 - sample depth
    
    return C
"""
    
def load_prod(Prop, fast_file='Production Rates/P_fast_0m.csv', neg_file='Production Rates/P_neg_0m.csv'):
    if Prop is None:
        df_Pfast = pd.read_csv(fast_file)
        return '', list(df_Pfast.columns)

def load_profile(Prop, file='balco_14co_const_models.fits', i=68):
    if Prop is None: # run function with Prop=None to get input stage
        return ''

    """


    Parameters
    --------------------
    Phi0 - 

    file - 

    i - 


    Returns
    -----------------------
    CO - 

    """

    #Phi0
    #axis0 - Primary Model (Energy spectrum & Time dependence)
    #axis1 - Particle Species (proton, neutron)
    #axis2 - Primary Energy

    hdus = fits.open(file)
    return np.reshape(hdus['CO14'].data[i][1:], (1,-1))
    
def calc_gauss_pred(Prop, c_samp = None, c_weights=None, output=False):
    if c_samp is None:
        if Prop.c_samp is None: # calculate the predicted 14CO samples if it hasn't been done already
            c_samp = Prop.Phi['CO']@Prop.S_mat[Prop.i_start:]
        else:
            c_samp = Prop.c_samp
    if c_weights is None:
        if Prop.c_weights is None: # if weights haven't been provided, assume all models are equal
            c_weights = np.ones(len(c_samp))
        elif len(Prop.c_weights)==len(c_samp): # if weights have already been used and fit the number of models here, use them
            c_weights = Prop.c_weights
        else: # if all else fails, assume all models are equal
            c_weights = np.ones(len(c_samp))

    # 14CO sample predictions for each model
    Prop.c_samp = c_samp

    # relative weight of each model
    Prop.c_weights = c_weights

    # average of the 14CO predictions
    Prop.c_pred = np.average(c_samp, weights=c_weights, axis=0)

    # covariance matrix for the 14CO predictions
    Prop.cov_pred = np.cov(c_samp, aweights=c_weights, rowvar=False)

    if output:
        return Prop.c_pred, Prop.cov_pred
    return

def get_samples(Prop, N=1, c_pred=None, cov_pred=None, rel_err=None):
    if c_pred is None:
        if Prop.c_pred is None:
            Prop.calc_gauss_pred()
        c_pred = Prop.c_pred
    if cov_pred is None:
        if Prop.cov_pred is None:
            Prop.calc_gauss_pred()
        cov_pred = Prop.cov_pred
    if rel_err is None:
        rel_err = Prop.rel_err

    # generate random 14CO predictions
    c = np.random.multivariate_normal(c_pred, cov_pred, N)

    # generate random relative experimental errors
    s = np.random.normal(1., rel_err, (N,len(c_pred)))

    # return simulated 14CO samples w/ experimental error
    return c*s, c*s * rel_err

def log_likelihood(Prop, c, c_err=None, c_pred=None, cov_pred=None):
    N_samp = len(Prop.z_samp)

    c = np.reshape(c, (-1,N_samp))
    if c_err is None:
        c_err = c * Prop.rel_err
    else:
        c_err = np.reshape(c_err, np.shape(c))
    if c_pred is None:
        c_pred = Prop.c_pred
    c_pred = np.reshape(c_pred, (1,N_samp))
    if cov_pred is None:
        cov_pred = Prop.cov_pred

    # Calculate total systematic uncertainty
    # Sigma = cov_pred + np.diag(c_err**2)
    # I can't find a quick way to diagonalize axis1 into axis 1&2 while preserving axis0
    Sigma = np.reshape(cov_pred, (1,N_samp,N_samp)) * np.ones((len(c_err),1,1))
    for i in range(N_samp): # add experimental variance along diagonal
        Sigma[:,i,i] += c_err[:,i]**2
    Sigma_inv = np.linalg.inv(Sigma)

    # difference between measurements and prediction
    x = (c-np.reshape(c_pred, (1,N_samp))).reshape((-1,N_samp,1))

    # Calculate chi-square
    chi2 = (T(x)@Sigma_inv@x)[:,0,0]

    # Convert to log_likelihood
    logL = -chi2/2 - np.log(np.linalg.det(2*np.pi*Sigma))/2

    # Convert to p-value
    # (using chi_square to calculate, becauSe I can't think how to convert log_likelihood off the top of my head)
    p = stats.chi2.sf(chi2, N_samp)

    # Convert to # of standard deviations
    sig = stats.norm.isf(p/2.)

    return logL, p, sig

def get_sensitivity(Prop, f_var='linear', error=0.02, amp=None, a_weights=None, P_14C=None, p_weights=None, f_file='factors_2sigma_hull.csv', Pres_conv=True, Normalize=True, Gauss_prod=True, Gauss_f=True, Int_f_all=True, Sample_f=True):

    return