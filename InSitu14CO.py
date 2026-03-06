#basic imports and ipython setup
import matplotlib.pyplot as plt
import numpy as np

#import os.path

from MCEq.core import MCEqRun
from MCEq import config
import crflux.models as pm

from tqdm import tqdm

import pandas as pd
#from MCEq.geometry.density_profiles import GeneralizedTarget

#import matplotlib as mpl
#from cycler import cycler
#from astropy.io import fits

import Functions_14CO as F

#from scipy import stats

# could split this into:
# Sites
# Calculation Steps (Models)
# Datasets

# I can speed up in-ice calculations by capping muon energy at 100 TeV

def T(A):
    return np.swapaxes(A, -1,-2)

def argnear_below(x, a): 
    # returns the index of the nearest value to x in the array a
    # such that a[i] <= x
    # assuming a is sorted low -> high

    # works by interpolating the inverse function of a[i]
    return max(min(int(np.interp(x, a, np.arange(len(a)))), len(a)-1), 0)

def argnear_above(x, a): 
    # returns the index of the nearest value to x in the array a
    # such that a[i] >= x
    # assuming a is sorted low -> high

    # works by interpolating the inverse function of a[i]
    return max(min(int(np.interp(x, a, np.arange(len(a))))+1, len(a)-1), 0)

class ModelStep:

    def __init__(self, function=None, params=None, names=''):
        
        if type(params) is list or type(params) is np.ndarray:
            self.params = params
        else:
            self.params = [params]
            
        if type(names) is list or type(names) is np.ndarray:
            self.names = names
        else:
            self.names = [names]
            
        if callable(function):
            self.function = function
            a = self.function(None)
            if type(a) is tuple:
                self.input = a[0]
                n = a[1]
            else:
                self.input = a
                n = None
            if names=='' and not (n is None):
                self.names = n
        else:
            self.function = None
            self.input = ''
        
        if len(self.names) > len(self.params):
            self.names = self.names[:len(self.params)]
        elif len(self.names) < len(self.params):
            if self.names[0] == '':
                self.names = ['']*len(self.params)
            else:
                self.names = ['{}-{}'.format(self.names[0], p) for p in self.params]
            
        if self.input != '':
            self.names = ['_'+n if n!='' else n for n in self.names]
        
        # params naming convention?
        # plotting format?
    
    def run(self, Prop):
        if self.function is None:
            return np.array(self.params)
        
        return np.concatenate([self.run_solo(Prop, p) for p in self.params])
    
    def run_solo(self, Prop, p):
        if type(p) is tuple:
            return self.function(Prop, *p)
        elif type(p) is dict:
            return self.function(Prop, **p)
        elif p is None:
            return self.function(Prop)
        else:
            return self.function(Prop, p)

class Dataset:

    def __init__(self, x, y=None, z=None, x_err=None, y_err=None, z_err=None, name=None):
        self.x = x
        self.y = y
        self.z = z
        self.x_err = x_err
        self.y_err = y_err
        self.z_err = z_err
        self.name = name
        # plotting format?


class Propagator:
    """
    Class for propagating primary cosmic rays to atmospheric muons, underground muons, in-situ 14C production rates, and 14CO profiles
    
    Instance Variables
    --------------------
    
    Grid Values
    ---------------------------------------------------
    self.z_bins : numpy array, shape (#z+1), dtype float
        depth bin edges [m]
        Ranges from z_min to z_deep
    self.z : numpy array, shape (#z), dtype float
        depth bin centers [m]
        z = (z_bins[:-1] + z_bins[1:])/2
    self.dz : numpy array, shape (#z), dtype float
        depth bin widths [m]
        dz = np.diff(z_bins)
    
    self.h_bins : numpy array, shape (#z+1), dtype float
        mass depth bin edges [meters-water-equivalent (m.w.e.) = hg/cm^2]
        corresponds to z_bins
    self.h : numpy array, shape (#z), dtype float
        mass depth bin centers [meters-water-equivalent (m.w.e.) = hg/cm^2]
        h = (h_bins[:-1] + h_bins[1:])/2
    self.dh : numpy array, shape (#z), dtype float
        mass depth bin widths [meters-water-equivalent (m.w.e.) = hg/cm^2]
        dh = np.diff(h_bins)
    
    self.t_bins : numpy array, shape (#z+1), dtype float
        ice age bin edges [years]
        corresponds to z_bins
    self.t : numpy array, shape (#z), dtype float
        ice age bin centers [years]
        t = (t_bins[:-1] + t_bins[1:])/2
    self.dt : numpy array, shape (#z), dtype float
        ice age bin widths [years]
        dt = np.diff(t_bins)
        
    self.z_samp_bins : numpy array, shape (#samp+1), dtype float
        sample depth bin edges [m]
    self.z_samp : numpy array, shape (#samp), dtype float
        sample depth bin centers [m]
    self.dz_samp : numpy array, shape (#samp), dtype float
        sample depth bin widths [m]
        
    self.S_mat : numpy array, shape (#z, #samp), dtype float
        Matrix averaging over the depth bins in a core sample [unitless]
        Given an array A whose final axis ranges over depth,
        A_samp = A @ S_mat
        Where A_samp lists the average value of A in each core sample.
    self.i_start : int
        depth index where 14CO accumulation starts [unitless]
    
    self.cosTH_bins : numpy array, shape (#cosTH+1), dtype float
        cosine zenith angle bin edges [unitless]
        Ranges from 1. to 0.
    self.cosTH : numpy array, shape (#cosTH), dtype float
        cosine zenith angle bin centers [unitless]
        cosTH = (cosTH_bins[:-1] + cosTH_bins[1:])/2
    self.dcosTH : numpy array, shape (#cosTH), dtype float
        cosine zenith angle bin widths [unitless]
        dcosTH = np.diff(cosTH_bins)
    
    self.E_bins : numpy array, shape (#E+1), dtype float
        particle energy bin edges [GeV]
        E_bins = 10.**logE_bins
    self.E : numpy array, shape (#E), dtype float
        particle energy bin centers [GeV]
        E = 10.**logE
    self.dE : numpy array, shape (#E), dtype float
        particle energy bin widths [GeV]
        dE = np.diff(E_bins)
        
    Muon Propagation Parameters
    ------------------------------------------------------
    self.rho_ice : float
        Density of solid ice [g/cm^3]
        
    self.pressure : float
        atmospheric pressure at site [Pa]
        used to calculate H in Balco
    self.H : float
        atmospheric depth above sea level [m.w.e. = hg/cm^2]
        H = (1013.25 - pressure/100)*1.019716
        
    self.h_range : numpy array, shape (30), dtype float
        Lithospheric depth corresponding to momentum array [g/cm^2]
    self.momentum : numpy array, shape (30), dtype float
        Average momentum of muons at depth [GeV/c]
        Used for atmospheric attenuation length calculation in Balco
        From a table for muons in standard rock in Groom and others 2001
        
    self.a : float
        energy loss due to ionization [GeV cm^2/hg]
    self.b : float
        sum of fractional radiation losses in solid rock [cm^2/hg]
        value averaged from Gaisser-Stanev table
        for ~30GeV muons (see Heisinger)
    self.b_ice : float
        sum of fractional radiation losses in ice [cm^2/hg]
        value averaged from Gaisser-Stanev table
        for ~30GeV muons
    
    self.elev : float or int
        elevation above sea level [m]
        For use in MCEq atmospheric profile
    self.mceq : MCEqRun object
        dummy MCEq instance to get info from
        
   
    14CO Production Parameters
    ------------------------------------------------------
    self.sigma_E : float
        fast muon interaction cross section measurement [cm^2]
        default value = 4.5e-28
        (see Heisinger)
    self.E_sigma : float
        energy of cross section measurement [GeV]
        default value = 190.
    self.alpha : float
        cross section energy scaling factor [unitless]
        sigma(E) = sigma_0 * E**alpha
        default value = 0.75
    self.sigma_0 : float
        fast muon interaction cross section at 1 GeV [cm^2]
        sigma_0 = sigma_E / E_sigma**alpha
    self.N : float
        density of fast muon interaction targets (oxygen nucleii) [hg^-1]
        #oxgyen nucleii per molecule (1) / molecular mass (0.1802 / 6.022e23)
    self.f_tot : float
        effective probability of 14C production by capture of a stopped negative muon [unitless]
        f_tot = f_C * f_D * f_star
            f_C : 
            f_D : 
            f_star : 
        
    self.f_factors : numpy array, shape (2), dtype float
        coefficients scaling 14CO production via fast and negative muon interactions [unitless]
        f_factors = [f_fast, f_neg]
        
    self.lambd : float
        14C annual loss to radioactive decay [year^-1]
        default value = 1.21e-4
        14C_end = 14C_start * (1-lambd)**Delta_t
    
    
    Production Models
    -------------------------------------------------------------------
    self.stages : list of strings
        
            'primary' : 
            'atm' : 
            'ice' : 
            'prod' : 
            'CO' : 
    
    self.Phi : dict of numpy arrays
    
    self.models : dict of ModelSteps
    
    self.model_names : dict of lists of strings
    
    """
    def __init__(self,
                 pressure = 65800, # atmospheric pressure at site [Pa], used in Balco elevation adjustment for Heisinger calculation
                 elev=3233, #Elevation above sea level [m]
                 rho_ice = 0.9239, # Density of solid ice [g/cm^3]
                 f_factors = [0.072, 0.066], #coefficients scaling 14CO production via fast (f_fast) and negative (f_neg) muon interactions [unitless]
                 ice_eq_depth_file = 'Real_vs_ice_eq_depth.csv', #.csv table converting real depths to ice-equivalent depths
                    #(Ice-equivalent depth is defined as the mass per square centimeter above that depth, divided by the density of ice)
                    #Columns:
                        #z - real depth [m]
                        #ice_eq_depth - corresponding ice-equivalent depth [meters-ice-eq]
                 age_scale_file = 'DomeC_age_scale_Apr2023.csv', #.csv table converting depth to ice age
                    #Columns:
                        #depths_real - depth of ice [m]
                        #ages - age of ice [years]
                 z_min = 0, # Minimum depth of density profile [m] (Should always be 0?)
                 z_start = 96.5, #Depth at which 14CO accumulation starts [m]
                 z_deep = 300, #Maximum depth of calculation [m]
                 sample_depths = (100.,301.,20.), # depths defining sample bins [m]
                 #tuple : parameters for numpy.arange to define a 1D array of sample bin edges
                #1D array : sample bin edges, connected
                #2D array : sample bin edges in form [[min0, min1, ...], [max0, max1, ...]]
                 N_ang = 10, # Number of zenith angle bins [unitless] (zenith angle bins are equally spaced in solid angle)
                 logE_min = -1, #log base 10 of the minimum tracked particle energy [log10 GeV]
                 logE_max = 11, #log base 10 of the maximum tracked particle energy [log10 GeV]
                 logE_mu_max = 7.5, #log base 10 of the maximum tracked  MUON energy [log10 GeV]
                ):
        
        # load in depth, mass depth, and time bins (default location - Dome C, Antarctica)
        self.load_ice_profile(ice_eq_depth_file, age_scale_file, rho_ice, z_min, z_deep, z_start, sample_depths)
        
        # set zenith angle bins (default 10 equally spaced in solid angle)
        self.set_zenith_bins(N_ang)
        
        # set energy bins (default 120 equally space between logE = 1e-1 and 1e11)
        self.set_energy_bins(logE_min, logE_max, logE_mu_max, 0.1)
        
        # set pressure used in Balco calculation (default = 65800 Pa for Dome C, Antarctica)
        self.set_pressure(pressure)
        
        # parameters for Gaisser-Stanev Energy loss
        self.a = 0.227 #energy loss due to ionization (GeV cm^2/hg)
        self.b = 2.44e-4 #sum of fractional radiation losses (cm^2/hg)
        self.b_ice = 2.04e-4 #ice value
        
        self.set_cross_sections()

        # Production rate adjustment from Taylor Glacier data
        self.f_factors = np.array(f_factors)
        
        self.setup_mceq(elev)
        
        # Muon mass
        self.mu_mass = 1.056583745e-1 # GeV
        
        # 14C Decay parameter
        self.lambd = 1.216e-4 #yr^-1
        
        self.stages = ['primary',
                       'atm',
                       'ice',
                       'prod',
                       'CO'
                      ]
        
        self.Phi = dict()
        self.clear_Phi()
        
        self.data = dict()
        self.set_data()
        
        self.models = dict()
        self.model_names = dict()
        self.set_models()
        
    
    #Loads ice profile data from .csv files to setup depth bins
    def load_ice_profile(self, ice_eq_depth_file, age_scale_file, rho_ice = None, z_min = None, z_deep = None, z_start = None, sample_depths = None):
        if rho_ice is None:
            rho_ice = self.rho_ice
        if z_min is None:
            z_min = self.z_min
        if z_deep is None:
            z_deep = self.z_deep
        if z_start is None:
            z_start = self.z_start
        if sample_depths is None:
            sample_depths = self.sample_depths

        # read age-scale file
        age_scale = pd.read_csv(age_scale_file)
        ages = np.array(age_scale['ages']) # years
        depths_real = np.array(age_scale['depths_real']) # meters

        # read ice-eq-depth file
        ice_eq_depth = pd.read_csv(ice_eq_depth_file)
        real_z = np.array(ice_eq_depth['z']) # meters
        ice_eq_z = np.array(ice_eq_depth['ice_eq_z']) # meters (ice-eq) aka mass-depth / ice density
        
        self.rho_ice = rho_ice # density of solid ice at Dome C (g/cm^3)
        
        self.set_mass_depth(depths_real, np.interp(depths_real, real_z, ice_eq_z)*self.rho_ice, ages, z_min, z_deep, z_start, sample_depths)
        
        return
    
    
    # Sets up depth bins using real and water-equivalent depths
    def set_mass_depth(self, z_bins, h_bins, t_bins = None, z_min = None, z_deep = None, z_start = None, sample_depths = None):
        if t_bins is None:
            t_bins = np.arange(len(z_bins))
        if z_min is None:
            z_min = self.z_min
        if z_deep is None:
            z_deep = self.z_deep
        if z_start is None:
            z_start = self.z_start
        if sample_depths is None:
            sample_depths = self.sample_depths
        
        self.z_min = z_min # starting depth for plots (m)
        self.z_deep = z_deep # end depth (m)

        i_min = argnear_below(self.z_min, z_bins) # nearest depths_real index to z_min
        i_end = argnear_above(self.z_deep, z_bins) # nearest depths_real index to z_end
        
        # Define depth bins
        self.z_bins = z_bins[i_min:i_end+1] # depth bin edges in steps of 1-year ice age (m)
        self.z = (self.z_bins[:-1]+self.z_bins[1:])/2 # bin-average of z (m)
        self.dz = np.diff(self.z_bins) # bin-width of z (m)

        # Define mass depth bins
        self.h_bins = h_bins[i_min:i_end+1] # mass depth bin edges corresponding to z bins (m.w.e = hg/cm^2)
        self.h = (self.h_bins[:-1]+self.h_bins[1:])/2 # bin-average of h (m.w.e = hg/cm^2)
        self.dh = np.diff(self.h_bins) # bin-width of h (m.w.e = hg/cm^2)

        self.rho = self.dh/self.dz # density of depth bins

        # Define time bins
        self.t_bins = t_bins[i_min:i_end+1] # ice age bins corresponding to z array (years)
        self.t = (self.t_bins[:-1]+self.t_bins[1:])/2 # bin-average of t (years)
        self.dt = np.diff(self.t_bins) # bin-width of t (years)
        
        self.setup_sample_bins(z_start, sample_depths)
        
        return
    
    # Loads ice density data from .csv files to setup depth bins
    def load_density(self, density_file, age_scale_file = None, z_min = None, z_deep = None, z_start = None, sample_depths = None):
        if z_min is None:
            z_min = self.z_min
        if z_deep is None:
            z_deep = self.z_deep
        if z_start is None:
            z_start = self.z_start
        if sample_depths is None:
            sample_depths = self.sample_depths
            
        self.density_file = density_file # relationship bewteen ice-equivalent-depth and real-depth at Dome-C

        # read ice-eq-depth file
        density_scale = pd.read_csv(self.density_file)
        real_z = np.array(density_scale['z']) # meters
        rho = np.array(density_scale['rho']) # 
        
        if age_scale_file is None:
            t_bins = None
        else:
            self.age_scale_file = age_scale_file # relationship between age and depth of ice at Dome-C
            # read age-scale file
            age_scale = pd.read_csv(self.age_scale_file)
            ages = np.array(age_scale['ages']) # years
            depths_real = np.array(age_scale['depths_real']) # meters
            t_bins = np.interp(real_z, depths_real, ages)
        
        self.set_density(real_z, rho, t_bins, z_min, z_deep, z_start, sample_depths)
    
    
    # Sets up depth bins using real depth and density profile
    def set_density(self, z_bins, rho, t_bins = None, z_min = None, z_deep = None, z_start = None, sample_depths = None):
        if t_bins is None:
            t_bins = np.arange(len(z_bins))
        if z_min is None:
            z_min = self.z_min
        if z_deep is None:
            z_deep = self.z_deep
        if z_start is None:
            z_start = self.z_start
        if sample_depths is None:
            sample_depths = self.sample_depths
        
        self.z_min = z_min # starting depth for plots (m)
        self.z_deep = z_deep # end depth (m)

        i_min = argnear_below(self.z_min, z_bins) # nearest depths_real index to z_min
        i_end = argnear_above(self.z_deep, z_bins) # nearest depths_real index to z_end
        
        # Define depth bins
        self.z_bins = z_bins[i_min:i_end+1] # depth bin edges in steps of 1-year ice age (m)
        self.z = (self.z_bins[:-1]+self.z_bins[1:])/2 # bin-average of z (m)
        self.dz = np.diff(self.z_bins) # bin-width of z (m)
        
        self.rho = (rho[i_min:i_end]+rho[i_min+1:i_end+1])/2 # density of depth bins

        # Define mass depth bins - assumes constant density above z_bins[0]
        self.dh = self.rho * self.dz # bin-width of h (m.w.e = hg/cm^2)
        self.h_bins = np.cumsum(np.concatenate(([np.sum(self.z_bins[:i_min+1]*rho[:i_min+1])], self.dh))) # mass depth bin edges corresponding to z bins (m.w.e = hg/cm^2)
        self.h = (self.h_bins[:-1]+self.h_bins[1:])/2 # bin-average of h (m.w.e = hg/cm^2)

        # Define time bins
        self.t_bins = t_bins[i_min:i_end+1] # ice age bins corresponding to z array (years)
        self.t = (self.t_bins[:-1]+self.t_bins[1:])/2 # bin-average of t (years)
        self.dt = np.diff(self.t_bins) # bin-width of t (years)
        
        self.setup_sample_bins(z_start, sample_depths)
        
        return
    
    
    # Sets up transformation from depths used in calculation to sample depth bins
    def setup_sample_bins(self, z_start = None, sample_depths = None):
        if z_start is None:
            z_start = self.z_start
        if sample_depths is None:
            sample_depths = self.sample_depths
        
        self.z_start = z_start # starting depth of 14C accumulation (m) - close-off depth beneath firn layer
        self.sample_depths = sample_depths # length of ice core samples (m)

        self.i_start = argnear_below(self.z_start, self.z_bins) # index of first bin above starting point for 14C accumulation
        
        if type(self.sample_depths) is tuple:
            z_samp_bins = np.array([np.arange(*self.sample_depths)[:-1], np.arange(*self.sample_depths)[1:]])
        elif len(np.shape(self.sample_depths)) == 1:
            z_samp_bins = np.array([self.sample_depths[:-1], self.sample_depths[1:]])
        elif len(np.shape(self.sample_depths)) == 2:
            z_samp_bins = np.array(self.sample_depths)[:2]
        else:
            print('Invalid Sample Depths Format')
            z_samp_bins = np.array([[self.z_start],[self.z_deep]])
        
        self.z_samp_bins = z_samp_bins
        self.z_samp = (self.z_samp_bins[0] + self.z_samp_bins[1])/2
        self.dz_samp = self.z_samp_bins[1]-self.z_samp_bins[0]
        
        i_samp_bins = np.expand_dims(np.interp(self.z_samp_bins, self.z_bins, np.arange(len(self.z_bins))), axis=1)
        i = np.reshape(np.arange(len(self.dh)), (-1,1))
        
        # matrix of the mass/cm^2 of each depth bin which is within the bounds of each sample bin
        dh_samp_mat = np.reshape(self.dh, (-1,1)) * ( (i_samp_bins[1]-i).clip(0,1) - (i_samp_bins[0]-i).clip(0,1) )
        
        self.S_mat = dh_samp_mat / np.sum(dh_samp_mat, axis=0, keepdims=True)
        
        return
    
    
    # Sets up zenith angle bins
    def set_zenith_bins(self, N_ang = 10):
        
        self.N_ang = N_ang

        # Define zenith angle bins
        self.cosTH_bins = np.linspace(1,0,self.N_ang+1)
        self.cosTH = (self.cosTH_bins[:-1]+self.cosTH_bins[1:])/2
        self.dcosTH = -np.diff(self.cosTH_bins)
        
        return
    
    
    # Sets up energy bins
    def set_energy_bins(self, logE_min = -1, logE_max = 11, logE_mu_max = 7, dlogE = 0.1):
        
        self.logE_min = logE_min # minimum energy (log10 GeV)
        self.logE_max = logE_max # maximum energy (log10 GeV)
        self.dlogE = dlogE # energy bin width (log10 GeV)
        self.logE_mu_max = logE_mu_max
        

        # Define energy bins
        self.logE_bins = np.arange(self.logE_min, self.logE_max+self.dlogE, self.dlogE) # log10 GeV
        self.logE = (self.logE_bins[:-1]+self.logE_bins[1:])/2 # log10 GeV
        self.E_bins = 10.**(self.logE_bins) # GeV
        self.E = 10.**(self.logE) # bin-average of E (GeV)
        self.dE = np.diff(self.E_bins) # bin-width of E (GeV)
        
        self.logE_mu_bins = np.arange(self.logE_min, self.logE_mu_max+self.dlogE, self.dlogE)
        self.logE_mu = (self.logE_mu_bins[:-1]+self.logE_mu_bins[1:])/2 # log10 GeV
        self.E_mu_bins = 10.**(self.logE_mu_bins) # GeV
        self.E_mu = 10.**(self.logE_mu)
        self.dE_mu = np.diff(self.E_mu_bins)

        # how to average E?  Currently doing geometric mean, but maybe there's a better way.
        
        self.setup_mceq()
        
        return
    
    
    # Sets up a dummy MCEq instance to pull data from
    def setup_mceq(self, elev=None):
        if not (elev is None):
            self.elev = elev
            config.h_obs = self.elev

        interaction_model = "SIBYLL-2.3c"

        density_model, density_name = ('CORSIKA', ('USStd', None)), 'CORSIKA_USStd'
        
        config.debug_level = 0
        config.enable_default_tracking = False
        config.e_min = self.E_bins[0]*10.**0.1
        config.e_max = self.E_bins[-1]

        config.max_density = 0.001225
        config.dedx_material='air'

        self.mceq = MCEqRun(
            interaction_model=interaction_model,
            theta_deg = 0,
            density_model = density_model,
            medium = 'air',
            primary_model = (pm.GlobalSplineFitBeta, None),
        )
        
        return
    
    
    # Sets up parameters for production rates calculations
    def set_cross_sections(self,
                           sigma_E = None, #fast muon interaction cross section measurement [cm^2] (default value = 4.5e-28, see Heisinger 2002)
                           E_sigma = None, #energy of cross section measurement [GeV] (default value = 190.)
                           alpha = None, #cross section energy scaling factor [unitless] (default value = 0.75)
                               #sigma(E) = sigma_0 * E**alpha
                           N = None, #density of fast muon interaction targets (oxygen nucleii) [hg^-1]
                                #oxgyen nucleii per molecule (1) / molecular mass (0.1802 / 6.022e23)
                           f_tot = None #effective probability of 14C production by capture of a stopped negative muon [unitless]
                                #f_tot = f_C (Chemcial factor) * f_D (Decay factor) * f_star ()
                          ):
        if sigma_E is None:
            if hasattr(self, 'sigma_E'):
                sigma_E = self.sigma_E
            else:
                sigma_E = 4.5e-28
        if E_sigma is None:
            if hasattr(self, 'E_sigma'):
                E_sigma = self.E_sigma
            else:
                E_sigma = 190.
        if alpha is None:
            if hasattr(self, 'alpha'):
                alpha = self.alpha
            else:
                alpha = 0.75
        if N is None:
            if hasattr(self, 'N'):
                N = self.N
            else:
                N = 6.022e23 / 0.1802 # hg^-1
        if f_tot is None:
            if hasattr(self, 'f_tot'):
                f_tot = self.f_tot
            else:
                f_tot = 1 * 0.1828 * 0.137

        # Fast Muon Interaction parameters (Heisinger)
        self.sigma_E = sigma_E #cm^2
        #self.dsigma_E = 2.5e-28
        self.E_sigma = E_sigma
        
        self.alpha = alpha
        self.sigma_0 = self.sigma_E / self.E_sigma**self.alpha #8.8e-30 +/- 4.9e-30 cm^2 = 8.8 +/- 4.9 mu b
        self.N = N
        
        # Negative Muon Capture parameters (Heisinger)
        #self.f_C = 1 # value should be absolute
        #self.f_D = 0.1828 #error unknown
        #self.f_star = 0.137
        #self.df_star = 0.011
        
        #f_star, df_star = 4.4e-3/f_C/f_D, 2.6e-3/f_C/f_D
        self.f_tot = f_tot
        
        return
    
    # Sets up Balco elevation adjustment factors, starting from pressure
    def set_pressure(self,
                     pressure # atmospheric pressure at site [Pa]
                    ):
        
        self.pressure = pressure # surface pressure in Pa, should be 65800 for Dome C

        # figure the difference in atmospheric depth from sea level in g/cm2
        self.H = (1013.25 - self.pressure/100)*1.019716 # the 1.019716 number is basically just 1/g accounting for needed unit conversions
        
        return
    
    # Sets up Balco elevation adjustment factors, starting from atmospheric depth above sea level
    def set_H(self,
              H #atmospheric depth above sea level [m.w.e. = hg/cm^2]
              # H = (1013.25 - pressure/100)*1.019716
             ):
        
        self.H = H
        
        self.pressure = (1013.25 - self.H/1.019716)*100
        
        return
    
    #
    def set_models(self,
                   clear=True, # Clear existing models?
                   update_names=True, # Update model names?
                   **kwargs
                  ):
        
        for s in self.stages:
            new_models = [ModelStep(*m) for m in kwargs.get(s, [])]
            if clear:
                self.models[s] = new_models
            else:
                self.models[s] += new_models
        
        if update_names:
            self.build_model_names()
        
        return
    
    #
    def add_models(self, **kwargs):
        self.set_models(clear=False, **kwargs)
        
    #
    def clear_models(self):
        self.set_models()
     
    #
    def build_model_names(self):
        for s in self.stages:
            self.model_names[s] = sum([sum([['{}{}'.format(i,n) for i in self.model_names.get(m.input,[''])] for n in m.names], []) for m in self.models[s]], [])
        
    #
    def set_data(self, clear=True, **kwargs):
        
        for s in self.stages:
            new_data = [DataSet(*d) for d in kwargs.get(s, [])]
            if clear:
                self.data[s] = new_data
            else:
                self.data[s] += new_data
        
        return
    #
    def add_data(self, **kwargs):
        self.set_data(clear=False, **kwargs)
    
    #
    def clear_data(self):
        self.set_data()
    
    #
    def clear_Phi(self):
        
        self.Phi[self.stages[0]] = np.zeros((0,2,len(self.E)))
        #Phi0
        #axis0 - Primary Model
        #axis1 - Particle Species (proton, neutron)
        #axis2 - Primary Energy (E)
        
        self.Phi[self.stages[1]] = np.zeros((0,len(self.cosTH),2,len(self.E)))
        #Phi_atm
        #axis0 - Atmospheric Model
        #axis1 - Zenith Angle
        #axis2 - Muon Charge (positive, negative)
        #axis3 - Muon Energy (E_mu)
        
        self.Phi[self.stages[2]] = np.zeros((0,len(self.cosTH),2,len(self.E),len(self.z_bins)))
        #Phi_ice
        #axis0 - Underice Model
        #axis1 - Muon Charge (positive, negative)
        #axis2 - Muon Energy (E_mu)
        #axis3 - depth bin EDGES (top -> bottom)
        
        self.Phi[self.stages[3]] = np.zeros((0,2,len(self.z)))
        #P_14C
        #axis0 - Production Model
        #axis1 - Production Mode (fast, neg)
        #axis2 - depth (top -> bottom)
        
        self.Phi[self.stages[4]] = np.zeros((0,len(self.z)))
        #CO
        #axis0 - 14CO Model
        #axis1 - depth (top -> bottom)
        
        return

    #
    def set_primary(self, Phi0, clear=True, run=False):

        self.set_models(clear=clear, **{self.stages[0]: Phi0})
        
        if run:
            self.calculate(start=0, end=0)
        
        return
    
    #
    def load_primary(self, p_models=None, clear=True, run=False): # primary CR intensities
        
        if p_models is None:
            p_models = self.p_models
            
        Phi0 = F.load_primary(self, p_models)
        
        self.set_models(clear=clear, **{self.stages[0]: Phi0})

        if run:
            self.calculate(start=0, end=0)
        return

    # sets up primary flux matrix to test each energy of p+ and n0 individually
    def set_primary_identity(self, run=False):

        Phi0 = F.set_primary_identity(self)
        
        self.set_models(**{self.stages[0]: Phi0})

        if run:
            self.calculate(start=0, end=0)
        return

    #
    def calculate(self,
                  start=0, # name or index of first stage to run
                  end=-1, # name or index of last stage to run
                  models=None, # dictionary of lists of ModelSteps to run; if None, runs self.Models
                  output=False, # return self.Phi after calculation?
                  clear=True, # overwrite past calculations?  If False, appends new calculations
                  **kwargs
                 ):
        
        if isinstance(start, str):
            start = self.stages.index(start)
        if isinstance(end, str):
            end = self.stages.index(end)
        if end == -1:
            end = len(self.stages)-1
        if models is None:
            models = self.models
        
        for s in self.stages[start:end+1]:
            print('Running {} stage...'.format(s))
            if clear:
                self.Phi[s] = np.concatenate([m.run(self) for m in models[s]]) if len(models[s])>0 else np.zeros((0,*np.shape(self.Phi[s])[1:]))
                self.model_names[s] = sum([sum([['{}{}'.format(i,n) for i in self.model_names.get(m.input,[''])] for n in m.names], []) for m in models[s]], [])
            else:
                self.Phi[s] = np.append(self.Phi[s], np.concatenate([m.run(self) for m in models[s]]), axis=0) if len(models[s])>0 else self.Phi[s]
                self.model_names[s] += sum([sum([['{}{}'.format(i,n) for i in self.model_names.get(m.input,[''])] for n in m.names], []) for m in models[s]], [])
            print('{} stage complete'.format(s))
        
        if output:
            return self.Phi
        return
    
    # def set_primary_data
    
    # def set_atm_data
    
    # def set_ice_data
    
    # def set_prod_data
    
    # def set_CO_data
    
    # def set_array
    
    # def save_primary_to_csv
    
    # def save_atm_to_csv
    
    # def save_ice_to_csv
    
    def save_prod_to_csv(self,
                         tag='' # Label for Production Rates (usually location, such as DomeC)
                        ):
        if tag != '':
            tag = '_'+tag
        
        # Note: this method doesn't work if two production rates have the same name
        
        P_fast = dict()
        P_neg = dict()
        for i,n in enumerate(self.model_names['prod']):
            P_fast[n] = self.Phi['prod'][i,0]
            P_neg[n] = self.Phi['prod'][i,1]
            
        df_fast = pd.DataFrame(P_fast)
        df_neg = pd.DataFrame(P_neg)
        df_fast.to_csv('Production Rates/P_fast{}_{}m.csv'.format(tag, self.elev), index=False)
        print('Saved to:  Production Rates/P_fast{}_{}m.csv'.format(tag, self.elev))
        df_neg.to_csv('Production Rates/P_neg{}_{}m.csv'.format(tag, self.elev), index=False)
        print('Saved to:  Production Rates/P_neg{}_{}m.csv'.format(tag, self.elev))
        
        return
    
    # def save_CO_to_csv