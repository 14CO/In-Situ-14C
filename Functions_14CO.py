#basic imports and ipython setup
import matplotlib.pyplot as plt
import numpy as np

from MCEq.core import MCEqRun
import mceq_config as config
import crflux.models as pm

from tqdm import tqdm

from MCEq.geometry.density_profiles import GeneralizedTarget

import daemonflux
    
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

def E_surf(Prop, E_d, X, a=None, b=None):
    if a is None:
        a = Prop.a
    if b is None:
        b = Prop.b

    """


    Parameters
    -----------------
    E_d - 

    X - 

    a - 

    b - 


    Returns
    ------------------
    E_surf - 

    """

    return ((E_d + a/b)*np.exp(X*b)-a/b).clip(min=Prop.E_bins[0])

def Heisinger_ice(Prop, norm=True, a=None, b=None, H=None):
    if Prop is None: # run function with Prop=None to get input stage
        return 'atm'
    
    if a is None:
        a = Prop.a
    if b is None:
        b = Prop.b
    if H is None:
        H = Prop.H

    """


    Parameters
    ------------------
    Phi_atm - 

    a - 

    b - 

    norm - 

    H - 


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

    X = np.reshape(Prop.h_bins,(1,1,-1))/np.reshape(Prop.cosTH,(1,-1,1))

    E_bounds = E_surf(Prop, np.reshape(Prop.E_mu_bins,(-1,1,1)), X, a, b) # Energy bins at depth projected back to their surface energies

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

def solve_mceq(mceq, int_grid=None, grid_var='X', use_tqdm=False):

    """


    Parameters
    ---------------
    mceq - MCEqRun object

    int_grid - 

    grid_var - string

    use_tqdm - bool

    """

    mceq._calculate_integration_path(int_grid=int_grid, grid_var=grid_var)

    nsteps, dX, rho_inv, grid_idcs = mceq.integration_path
    int_m = mceq.int_m
    dec_m = mceq.dec_m

    dXaccum = 0.
    grid_sol = np.zeros((len(grid_idcs), *np.shape(mceq._phi0))) # grid_sol begins with the right shape, to avoid restructuring
    grid_step = 0

    phc = np.copy(mceq._phi0)

    for step in (tqdm(range(nsteps)) if use_tqdm else range(nsteps)): # added option for tqdm progress bar
        phc += (int_m.dot(phc) + dec_m.dot(rho_inv[step] * phc)) * dX[step]
        phc[phc<1e-250] = 0. # exreme low values set to 0, improving efficiency for large slant depths

        if (grid_idcs and grid_step < len(grid_idcs)
                and grid_idcs[grid_step] == step):
            grid_sol[grid_step] = np.copy(phc) # grid_sol no longer appends
            grid_step += 1

    mceq._solution, mceq.grid_sol = phc, grid_sol

    return

#interaction_models = ["SIBYLL-2.3c","SIBYLL-2.3","SIBYLL-2.1","EPOS-LHC","QGSJET-II-04","DPMJET-III",'DPMJETIII191']
#density_models = [('CORSIKA', ('USStd', None)), ('CORSIKA',('SouthPole', 'December'))]
#density_names = ['CORSIKA_USStd', 'CORSIKA_SP_Dec']
def MCEq_atm(Prop, interaction_model="SIBYLL-2.3c", density_model=('CORSIKA', ('USStd', None)), elev=None, solver='default'):
    if Prop is None: # run function with Prop=None to get input stage
        return 'primary'
    
    if elev is None:
        elev = Prop.elev

    """


    Parameters
    -----------------
    Phi0 - numpy array

    interaction_model - string

    density_model - tuple, shape ('MODEL_NAME', parameters)

    elev - int or float

    solver - string


    Returns
    ---------------------
    phi_mu - 

    """

    # Use MCEq to propagate primary flux to atmospheric muons
    Phi0 = Prop.Phi['primary']
    #Phi0
    #axis0 - Primary Model (Energy spectrum & Time dependence)
    #axis1 - Particle Species (proton, neutron)
    #axis2 - Primary Energy

    import mceq_config as config
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
    
    # Build 2D array for mceq primary particles
    # We need the 2nd to last axis to be Particle Species & Energy
    # for the matrix multiplication to line up
    phi0 = np.zeros((len(mceq._phi0), np.shape(Phi0)[0]))
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

    import mceq_config as config
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

    import mceq_config as config
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

    import mceq_config as config
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

def flow_14C(Prop, f_t = None, lambd=None):
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
    
    return np.moveaxis([np.sum(P_14C[:,:,:i+1]*lambda_dt[:,:,-i-1:], axis=-1) for i in tqdm(range(np.shape(P_14C)[-1]))], 0, -1)
    #return diag_sum(np.expand_dims(P_14C[:,:,Prop.i_start:], axis=-1) * lambda_dt)

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