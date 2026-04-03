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

def make_bins(x): # takes bin edges and returns edges, centers, & widths
    X = np.array(x)
    return X, (X[:-1]+X[1:])/2, np.abs(np.diff(X))

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
        if p in self.params:
            print(self.names[self.params.index(p)])
        else:
            print(self.names[0])
        
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
    self.z_grid : numpy array, shape (#z_grid), dtype float
    
    self.h_grid : numpy array, shape (#z_grid), dtype float
    
    self.age_grid : numpy array, shape (#z_grid), dtype float
    
    self.z_bins : numpy array, shape (#z+1), dtype float
        production depth bin edges [m]
        Ranges from z_min to z_deep
    self.z : numpy array, shape (#z), dtype float
        production depth bin centers [m]
        z = (z_bins[:-1] + z_bins[1:])/2
    self.dz : numpy array, shape (#z), dtype float
        production depth bin widths [m]
        dz = np.diff(z_bins)
    
    self.h_bins : numpy array, shape (#z+1), dtype float
        production mass depth bin edges [meters-water-equivalent (m.w.e.) = hg/cm^2]
        corresponds to z_bins
    self.h : numpy array, shape (#z), dtype float
        production mass depth bin centers [meters-water-equivalent (m.w.e.) = hg/cm^2]
        h = (h_bins[:-1] + h_bins[1:])/2
    self.dh : numpy array, shape (#z), dtype float
        production mass depth bin widths [meters-water-equivalent (m.w.e.) = hg/cm^2]
        dh = np.diff(h_bins)
    
    self.rho : numpy array, shape (#z), dtype float
        average density in production depth bins [g/cm^3]
        rho = dh/dz
        
    self.z_accum_bins : numpy array, shape (#z_accum+1), dtype float
        accumulation depth bin edges [m]
        Ranges from shallowest to deepest sample depth
    self.z_accum : numpy array, shape (#z_accum), dtype float
        accumulation depth bin centers [m]
        z = (z_bins[:-1] + z_bins[1:])/2
    self.dz_accum : numpy array, shape (#z_accum), dtype float
        accumulation depth bin widths [m]
        dz = np.diff(z_bins)
    
    self.h_accum_bins : numpy array, shape (#z_accum+1), dtype float
        accumulation mass depth bin edges [meters-water-equivalent (m.w.e.) = hg/cm^2]
    self.h_accum : numpy array, shape (#z_accum), dtype float
        accumulation mass depth bin centers [meters-water-equivalent (m.w.e.) = hg/cm^2]
    self.dh_accum : numpy array, shape (#z_accum), dtype float
        accumulation mass depth bin widths [meters-water-equivalent (m.w.e.) = hg/cm^2]
    
    self.t_int_bins : numpy array, shape (#t_int+1), dtype float
        integration time bin edges [yrs]
        Ranges from age(z_start) - age(z_deep) to 0 (present)
    self.t_int : numpy array, shape (#t_int), dtype float
        integration time bin centers [yrs]
    self.dt_int : numpy array, shape (#t_int), dtype float
        integration time bin widths [yrs]
    
    self.z_samp_bins : numpy array, shape (#samp+1), dtype float
        sample depth bin edges [m]
    self.z_samp : numpy array, shape (#samp), dtype float
        sample depth bin centers [m]
    self.dz_samp : numpy array, shape (#samp), dtype float
        sample depth bin widths [m]
    
    self.h_samp_bins : numpy array, shape (#samp+1), dtype float
        sample mass depth bin edges [meters-water-equivalent (m.w.e.) = hg/cm^2]
    self.h_samp : numpy array, shape (#samp), dtype float
        sample mass depth bin centers [meters-water-equivalent (m.w.e.) = hg/cm^2]
    self.dh_samp : numpy array, shape (#samp), dtype float
        sample mass depth bin widths [meters-water-equivalent (m.w.e.) = hg/cm^2]
    
    self.S_mat : numpy array, shape (#z_accum, #z_samp), dtype float
        Matrix averaging over the depth bins in a core sample [unitless]
        Given an array A whose final axis ranges over accumulation depth,
        A_samp = A @ S_mat
        Where A_samp lists the average value of A in each core sample.
    
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
        
    self.E_mu_bins : numpy array, shape (#E_mu+1), dtype float
        muon energy bin edges [GeV]
    self.E_mu : numpy array, shape (#E_mu), dtype float
        muon energy bin centers [GeV]
    self.dE_mu : numpy array, shape (#E_mu), dtype float
        muon energy bin widths [GeV]
        
    Muon Propagation Parameters
    ------------------------------------------------------
    self.rho_ice : float
        Density of solid ice [g/cm^3]
        
    self.z_close : float
        Close-off depth, below which ice accumulates 14CO [m]
        
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
                 #f_factors = [0.072, 0.066], #coefficients scaling 14CO production via fast (f_fast) and negative (f_neg) muon interactions [unitless]
                 ice_eq_depth_file = 'Real_vs_ice_eq_depth.csv', #.csv table converting real depths to ice-equivalent depths
                    #(Ice-equivalent depth is defined as the mass per square centimeter above that depth, divided by the density of ice)
                    #Columns:
                        #z - real depth [m]
                        #ice_eq_depth - corresponding ice-equivalent depth [meters-ice-eq]
                 age_scale_file = 'DomeC_age_scale_Apr2023.csv', #.csv table converting depth to ice age
                    #Columns:
                        #depths_real - depth of ice [m]
                        #ages - age of ice [years]
                 density_file = None,
                 z_grid = None,
                 h_grid = None,
                 age_grid = None,
                 #z_min = 0, # Minimum depth of density profile [m] (Should always be 0?)
                 z_close = 96.5, #Depth at which 14CO accumulation starts [m]
                 #z_deep = 300, #Maximum depth of calculation [m]
                 h_bins = None,
                 sample_depths = (100.,301.,20.), # depths defining sample bins [m]
                 h_accum_bins = None,
                 t_int_bins = None,
                 #tuple : parameters for numpy.arange to define a 1D array of sample bin edges
                 #1D array : sample bin edges, connected
                 #2D array : sample bin edges in form [[min0, min1, ...], [max0, max1, ...]]
                 cosTH_bins = 10, # Number of zenith angle bins [unitless] (zenith angle bins are equally spaced in solid angle)
                 logE_min = -1, #log base 10 of the minimum tracked particle energy [log10 GeV]
                 logE_max = 11, #log base 10 of the maximum tracked particle energy [log10 GeV]
                 logE_mu_max = 7.5, #log base 10 of the maximum tracked  MUON energy [log10 GeV]
                ):
        
        self.rho_ice = rho_ice
        self.z_close = z_close
        
        # load in depth, mass depth, and time bins (default location - Dome C, Antarctica)
        if not (z_grid is None) and not (h_grid is None):
            self.set_ice_profile(z_grid, h_grid, age_grid, h_bins, sample_depths, h_accum_bins)
        elif not density_file is None:
            self.load_density(density_file, age_scale_file, h_bins, sample_depths, h_accum_bins)
        else:
            self.load_ice_profile(ice_eq_depth_file, age_scale_file, rho_ice, h_bins, sample_depths, h_accum_bins)
        
        self.set_integration_time(t_int_bins)
        
        # set zenith angle bins (default 10 equally spaced in solid angle)
        self.set_zenith_bins(cosTH_bins)
        
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
        #self.f_factors = np.array(f_factors)
        
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
        
    # interpolate between z, h, and age grids
    def z_to_h(self, z):
        return np.interp(z, self.z_grid, self.h_grid)
    def z_to_age(self, z):
        return np.interp(z, self.z_grid, self.age_grid)
    def h_to_z(self, h):
        return np.interp(h, self.h_grid, self.z_grid)
    def h_to_age(self, h):
        return np.interp(h, self.h_grid, self.age_grid)
    def age_to_z(self, age):
        return np.interp(age, self.age_grid, self.z_grid)
    def age_to_h(self, age):
        return np.interp(age, self.age_grid, self.h_grid)
    
    def h_prime(self, h, t): # mass depth of an ice parcel at time t if it ends at mass depth h
        return self.age_to_h(self.h_to_age(h)+t)
    
    #Loads ice profile data from .csv files to setup depth bins
    def load_ice_profile(self, ice_eq_depth_file, age_scale_file, rho_ice = None, h_bins = None, sample_depths = None, h_accum_bins = None):
        if rho_ice is None:
            rho_ice = self.rho_ice
        else:
            self.rho_ice = rho_ice # density of solid ice at Dome C (g/cm^3)

        # read age-scale file
        age_scale = pd.read_csv(age_scale_file)
        ages = np.array(age_scale['ages']) # years
        depths_real = np.array(age_scale['depths_real']) # meters

        # read ice-eq-depth file
        ice_eq_depth = pd.read_csv(ice_eq_depth_file)
        real_z = np.array(ice_eq_depth['z']) # meters
        ice_eq_z = np.array(ice_eq_depth['ice_eq_z']) # meters (ice-eq) aka mass-depth / ice density
        
        self.set_ice_profile(depths_real, np.interp(depths_real, real_z, ice_eq_z)*self.rho_ice, ages, h_bins, sample_depths, h_accum_bins)
        
        return
    
    # Loads ice density data from .csv files to setup depth bins
    def load_density(self, density_file, age_scale_file = None, h_bins = None, sample_depths = None, h_accum_bins = None):
            
        self.density_file = density_file # relationship bewteen ice-equivalent-depth and real-depth at Dome-C

        # read ice-eq-depth file
        density_scale = pd.read_csv(self.density_file)
        real_z = np.array(density_scale['z']) # meters
        rho = np.array(density_scale['rho']) # 
        h_grid = np.append(rho[0]*real_z[0],np.cumsum((rho[:-1]+rho[1:])/2*np.diff(real_z)))
        
        if age_scale_file is None:
            age_grid = None
        else:
            self.age_scale_file = age_scale_file # relationship between age and depth of ice at Dome-C
            # read age-scale file
            age_scale = pd.read_csv(self.age_scale_file)
            ages = np.array(age_scale['ages']) # years
            depths_real = np.array(age_scale['depths_real']) # meters
            age_grid = np.interp(real_z, depths_real, ages)
        
        self.set_ice_profile(real_z, h_grid, age_grid, h_bins, sample_depths, h_accum_bins)
        
    def set_ice_profile(self, z_grid, h_grid, age_grid = None, h_bins = None, sample_depths = None, h_accum_bins = None):
        
        self.z_grid = z_grid
        self.h_grid = h_grid
        self.age_grid = age_grid
        
        self.set_production_depth(h_bins)
        self.set_sample_depth(sample_depths, h_accum_bins)
        
        return
    
    def set_production_depth(self, h_bins = None):
        if h_bins is None:
            if hasattr(self, 'h_bins'):
                h_bins = self.h_bins
            else:
                dh = 1.
                h_bins = np.arange(self.h_grid[0], self.h_grid[-1], dh)
            
        self.h_bins, self.h, self.dh = make_bins(h_bins)
        self.z_bins, self.z, self.dz = make_bins(self.h_to_z(h_bins))
        
        self.rho = self.dh/self.dz
        
        return
    
    def set_sample_depth(self, sample_depths = None, h_accum_bins = None):
        # Setup sample depths
        if sample_depths is None and hasattr(self, 'z_samp_bins'):
            z_samp_bins = self.z_samp_bins
        else:
            if type(sample_depths) is tuple:
                z_samp_bins = np.array([np.arange(*sample_depths)[:-1], np.arange(*sample_depths)[1:]])
            elif len(np.shape(self.sample_depths)) == 1:
                z_samp_bins = np.array([self.sample_depths[:-1], self.sample_depths[1:]])
            elif len(np.shape(self.sample_depths)) == 2:
                z_samp_bins = np.array(self.sample_depths)[:2]
            else:
                print('Invalid Sample Depths Format')
                z_samp_bins = np.array([self.z_accum_bins[:-1], self.z_accum_bins[1:]])

            self.z_samp_bins = z_samp_bins
            self.z_samp = (self.z_samp_bins[0] + self.z_samp_bins[1])/2
            self.dz_samp = self.z_samp_bins[1]-self.z_samp_bins[0]
        
        self.h_samp_bins = self.z_to_h(z_samp_bins)
        self.h_samp = (self.h_samp_bins[0] + self.h_samp_bins[1])/2
        self.dh_samp = self.h_samp_bins[1]-self.h_samp_bins[0]
        
        # Setup accumulation depths
        if h_accum_bins is None:
            if hasattr(self, 'h_accum_bins'):
                h_accum_bins = self.h_accum_bins
            else:
                dh = 1.
                h_accum_bins = np.arange(self.h_samp_bins[0,0], self.h_samp_bins[1,-1]+dh, dh)
            
        self.h_accum_bins, self.h_accum, self.dh_accum = make_bins(h_accum_bins)
        self.z_accum_bins, self.z_accum, self.dz_accum = make_bins(self.h_to_z(h_accum_bins))
        
        # Setup matrix to average samples over accumulation depths
        #i_samp_bins = np.expand_dims(np.interp(self.z_samp_bins, self.z_accum_bins, np.arange(len(self.z_accum_bins))), axis=1)
        #i = np.reshape(np.arange(len(self.dh_accum)), (-1,1))
        
        # matrix of the mass/cm^2 of each depth bin which is within the bounds of each sample bin
        #dh_samp_mat = np.reshape(self.dh_accum, (-1,1)) * ( (i_samp_bins[1]-i).clip(0,1) - (i_samp_bins[0]-i).clip(0,1) )
        dh_samp_mat = (self.h_samp_bins[1].reshape((1,-1))-self.h_accum_bins[:-1].reshape((-1,1))).clip(0,self.dh_accum.reshape((-1,1))) - (self.h_samp_bins[0].reshape((1,-1))-self.h_accum_bins[:-1].reshape((-1,1))).clip(0,self.dh_accum.reshape((-1,1)))
  
        self.S_mat = dh_samp_mat / np.sum(dh_samp_mat, axis=0, keepdims=True)
        
        return
    
    def set_integration_time(self, t_int_bins=None):
        if t_int_bins is None:
            if self.age_grid is None:
                return
            else:
                t_int_bins = -np.flip(np.arange(0, self.z_to_age(self.z_accum_bins[-1])-self.z_to_age(self.z_close)+1))
        
        self.t_int_bins, self.t_int, self.dt_int = make_bins(t_int_bins)
        
        return
    
    # Sets up zenith angle bins
    def set_zenith_bins(self, cosTH_bins = 10):
        if isinstance(cosTH_bins, int):
            N = cosTH_bins
            cosTH_bins = np.linspace(1,0,N+1)

        # Define zenith angle bins
        self.cosTH_bins, self.cosTH, self.dcosTH = make_bins(cosTH_bins)
        
        return
    
    
    # Sets up energy bins
    def set_energy_bins(self, logE_min = -1, logE_max = 11, logE_mu_max = 7.5, dlogE = 0.1):
        
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
        
        self.Phi[self.stages[1]] = np.zeros((0,len(self.cosTH),2,len(self.E_mu)))
        #Phi_atm
        #axis0 - Atmospheric Model
        #axis1 - Zenith Angle
        #axis2 - Muon Charge (positive, negative)
        #axis3 - Muon Energy (E_mu)
        
        self.Phi[self.stages[2]] = np.zeros((0,2,len(self.E_mu),len(self.z_bins)))
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
        
        self.Phi[self.stages[4]] = np.zeros((0,len(self.z_accum)))
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
            if len(models[s])>0:
                if clear:
                    self.Phi[s] = np.concatenate([m.run(self) for m in models[s]], axis = 0)
                    self.model_names[s] = []
                else:
                    self.Phi[s] = np.append(self.Phi[s], np.concatenate([m.run(self) for m in models[s]], axis = 0))
                self.model_names[s] += ['{}{}'.format(i,n) for m in models[s] for n in m.names for i in self.model_names.get(m.input,[''])]
            print('{} stage complete'.format(s))
            print()
        
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