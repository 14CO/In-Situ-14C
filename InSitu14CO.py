#basic imports and ipython setup
import matplotlib.pyplot as plt
import numpy as np

#from abc import ABC

import os.path

from MCEq.core import MCEqRun
import mceq_config as config
import crflux.models as pm

from tqdm import tqdm

#from scipy.sparse import find

import pandas as pd
from MCEq.geometry.density_profiles import GeneralizedTarget

import matplotlib as mpl
#from cycler import cycler
from astropy.io import fits

import daemonflux

import Functions_14CO as F

# could split this into:
# Sites
# Calculation Steps (Models)
# Datasets

# I can speed up in-ice calculations by capping muon energy at 100 TeV

class ModelStep:

    def __init__(self, function=None, params=None, name=''):
        self.function = function
        if type(params) is list or type(params) is np.ndarray:
            self.params = params
        else:
            self.params = [params]
        if type(name) is list or type(name) is np.ndarray:
            self.names = name
        else:
            self.names = [name]
        
        if len(self.names) > len(self.params):
            self.names = self.names[:len(self.params)]
        elif len(self.names) < len(self.params):
            if self.names[0] == '':
                self.names = ['']*len(self.params)
            else:
                self.names = ['{}-{}'.format(self.names[0], p) for p in self.params]
        
        if self.function is None:
            self.input = ''
        else:
            self.input = self.function(None)
            
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

# class Site:
class Propagator:
    """
    Class for propagating primary cosmic rays to atmospheric muons, underground muons, in-situ 14C production rates, and 14CO profiles
    
    Instance Variables
    --------------------
    self.z_bins - numpy array, shape (#z+1), dtype float
        depth bin edges [m]
        Ranges from z_min to z_deep
    self.z - numpy array, shape (#z), dtype float
        depth bin centers [m]
        z = (z_bins[:-1] + z_bins[1:])/2
    self.dz - numpy array, shape (#z), dtype float
        depth bin widths [m]
        dz = np.diff(z_bins)
        
    self.rho_ice - float
        Density of solid ice [g/cm^3]
    
    self.h_bins - numpy array, shape (#z+1), dtype float
        mass depth bin edges [meters-water-equivalent (m.w.e.) = hg/cm^2]
        corresponds to z_bins
    self.h - numpy array, shape (#z), dtype float
        mass depth bin centers [meters-water-equivalent (m.w.e.) = hg/cm^2]
        h = (h_bins[:-1] + h_bins[1:])/2
    self.dh - numpy array, shape (#z), dtype float
        mass depth bin widths [meters-water-equivalent (m.w.e.) = hg/cm^2]
        dh = np.diff(h_bins)
    
    self.t_bins - numpy array, shape (#z+1), dtype float
        ice age bin edges [years]
        corresponds to z_bins
    self.t - numpy array, shape (#z), dtype float
        ice age bin centers [years]
        t = (t_bins[:-1] + t_bins[1:])/2
    self.dt - numpy array, shape (#z), dtype float
        ice age bin widths [years]
        dt = np.diff(t_bins)
        
    self.z_samp_bins - numpy array, shape (#samp+1), dtype float
        sample depth bin edges [m]
    self.z_samp - numpy array, shape (#samp), dtype float
        sample depth bin centers [m]
    self.dz_samp - numpy array, shape (#samp), dtype float
        sample depth bin widths [m]
        
    self.S_mat = numpy array, shape (#z, #samp), dtype float
        Matrix averaging over the depth bins in a core sample [unitless]
        Given an array A whose final axis ranges over depth,
        A_samp = A @ S_mat
        Where A_samp lists the average value of A in each core sample.
    self.i_start - int
        depth index where sampling starts [unitless]
    
    self.cosTH_bins - numpy array, shape (#cosTH+1), dtype float
        cosine zenith angle bin edges [unitless]
        Ranges from 1. to 0.
    self.cosTH - numpy array, shape (#cosTH), dtype float
        cosine zenith angle bin centers [unitless]
        cosTH = (cosTH_bins[:-1] + cosTH_bins[1:])/2
    self.dcosTH - numpy array, shape (#cosTH), dtype float
        cosine zenith angle bin widths [unitless]
        dcosTH = np.diff(cosTH_bins)
        
    self.logE_bins - numpy array, shape (#E+1), dtype float
        log10 of particle energy bin edges [log10 GeV]
        Ranges from -1. to 11. by default
    self.logE - numpy array, shape (#E), dtype float
        log10 of particle energy bin centers [log10 GeV]
        logE = (logE_bins[:-1] + logE_bins[1:])/2
    self.dlogE - numpy array, shape (#E), dtype float
        log10 of particle energy bin widths [log10 GeV]
        dlogE = np.diff(dlogE_bins)
    
    self.E_bins - numpy array, shape (#E+1), dtype float
        particle energy bin edges [GeV]
        E_bins = 10.**logE_bins
    self.E - numpy array, shape (#E), dtype float
        particle energy bin centers [GeV]
        E = 10.**logE
    self.dE - numpy array, shape (#E), dtype float
        particle energy bin widths [GeV]
        dE = np.diff(E_bins)
        
    self.pressure - float
        atmospheric pressure at site [Pa]
        used to calculate H in Balco
    self.H - float
        atmospheric depth above sea level [m.w.e. = hg/cm^2]
        H = (1013.25 - pressure/100)*1.019716
        
    self.h_range - numpy array, shape (30), dtype float
        Lithospheric depth corresponding to momentum array [g/cm^2]
    self.momentum - numpy array, shape (30), dtype float
        Average momentum of muons at depth [GeV/c]
        Used for atmospheric attenuation length calculation in Balco
        From a table for muons in standard rock in Groom and others 2001
        
    self.a - float
        energy loss due to ionization [GeV cm^2/hg]
    self.b - float
        sum of fractional radiation losses in solid rock [cm^2/hg]
        value averaged from Gaisser-Stanev table
        for ~30GeV muons (see Heisinger)
    self.b_ice - float
        sum of fractional radiation losses in ice [cm^2/hg]
        value averaged from Gaisser-Stanev table
        for ~30GeV muons
    
    self.elev - float or int
        elevation above sea level [m]
        For use in MCEq atmospheric profile
    self.mceq - MCEqRun object
        dummy MCEq instance to get info from
        
    self.sigma_E - float
        fast muon interaction cross section measurement [cm^2]
        default value = 4.5e-28
        (see Heisinger)
    self.E_sigma - float
        energy of cross section measurement [GeV]
        default value = 190.
    self.alpha - float
        cross section energy scaling factor [unitless]
        sigma(E) = sigma_0 * E**alpha
        default value = 0.75
    self.sigma_0 - float
        fast muon interaction cross section at 1 GeV [cm^2]
        sigma_0 = sigma_E / E_sigma**alpha
    self.N - float
        density of fast muon interaction targets (oxygen nucleii) [hg^-1]
        #oxgyen nucleii per molecule (1) / molecular mass (0.1802 / 6.022e23)
    self.f_tot - float
        effective probability of 14C production by capture of a stopped negative muon [unitless]
        f_tot = f_C * f_D * f_star
        f_C - 
        f_D - 
        f_star - 
        
    self.f_factors - numpy array, shape (2), dtype float
        coefficients scaling 14CO production via fast and negative muon interactions [unitless]
        f_factors = [f_fast, f_neg]
        
    self.lambd - float
        14C annual loss to radioactive decay [year^-1]
        default value = 1.21e-4
        14C_end = 14C_start * (1-lambd)**Delta_t
    
    self.p_models - list of tuples, shape [(), ...]
        
    self.p_names - list of strings
        
    
    self.atm - dictionary, shape {'name':(function, (param_1, param_2, param_3,...)), ...}
        Dictionary of functions to be run and their parameters, indexed by a name
        primary CR flux -> atmospheric muon flux
    self.ice - dictionary, shape {'name':(function, (param_1, param_2, param_3,...)), ...}
        Dictionary of functions to be run and their parameters, indexed by a name
        atmospheric muon flux -> muon flux underground (underice)
    self.atmice - dictionary, shape {'name':(function, (param_1, param_2, param_3,...)), ...}
        Dictionary of functions to be run and their parameters, indexed by a name
        primary CR flux -> muon flux underground (underice)
    self.prod - dictionary, shape {'name':(function, (param_1, param_2, param_3,...)), ...}
        Dictionary of functions to be run and their parameters, indexed by a name
        muon flux underground (underice) -> 14C production rates
    self.prodfull - dictionary, shape {'name':(function, (param_1, param_2, param_3,...)), ...}
        Dictionary of functions to be run and their parameters, indexed by a name
        primary CR flux -> 14C production rates
    self.flow - dictionary, shape {'name':(function, (param_1, param_2, param_3,...)), ...}
        Dictionary of functions to be run and their parameters, indexed by a name
        14C production rates -> 14CO profile
    self.flowfull - dictionary, shape {'name':(function, (param_1, param_2, param_3,...)), ...}
        Dictionary of functions to be run and their parameters, indexed by a name
        primary CR flux -> 14CO profile
    """
    def __init__(self, pressure = 65800, elev=3120, rho_ice = 0.9239, f_factors = [0.072, 0.066], ice_eq_depth_file = 'Real_vs_ice_eq_depth.csv', age_scale_file = 'DomeC_age_scale_Apr2023.csv', z_min = 0, z_deep = 300, z_start = 96.5, sample_length = 20, N_ang = 10, logE_min = -1, logE_max = 11, logE_mu_max = 7, dlogE = 0.1):
        """
        
        Parameters
        ------------------
        pressure - int or float
            atmospheric pressure at site [Pa]
            Used in Balco elevation adjustment for Heisinger calculation
        elev - int or float
            Elevation above sea level [m]
            Used in Matlab atmospheric calculation
        rho_ice - float
            Density of solid ice [g/cm^3]
            Used to convert ice-equivalent depth to water-equivalent
        f_factors - list of two floats
            coefficients scaling 14CO production via fast and negative muon interactions [unitless]
            f_factors = [f_fast, f_neg]
        ice_eq_depth_file - string
            .csv table converting real depths to ice-equivalent depths
            (Ice-equivalent depth is defined as the mass per square centimeter above that depth, divided by the density of ice)
            Columns:
                z - real depth [m]
                ice_eq_depth - corresponding ice-equivalent depth [meters-ice-eq]
        age_scale_file - string
            .csv table converting depth to ice age
            Columns:
                depths_real - depth of ice [m]
                ages - age of ice [years]
        z_min - int or float
            Minimum depth of density profile [m]
            (Should always be 0?)
        z_start - int or float
            Depth at which sampling starts [m]
        z_deep - int or float
            Maximum depth of calculation [m]
        sample_length - int or float
            Length of core samples [m]
        N_ang - int
            Number of zenith angle bins [unitless]
            zenith angle bins are equally spaced in solid angle
        logE_min - int or float
            log base 10 of the minimum tracked particle energy [log10 GeV]
        logE_max - int or float
            log base 10 of the maximum tracked particle energy [log10 GeV]
        dlogE - int or float
            Width of particle energy bins in log base 10 [log10 GeV]
        """
        
        # load in depth, mass depth, and time bins (default location - Dome C, Antarctica)
        self.load_ice_profile(ice_eq_depth_file, age_scale_file, rho_ice, z_min, z_deep, z_start, sample_length)
        
        # set zenith angle bins (default 10 equally spaced in solid angle)
        self.set_zenith_bins(N_ang)
        
        # set energy bins (default 120 equally space between logE = 1e-1 and 1e11)
        self.set_energy_bins(logE_min, logE_max, logE_mu_max, dlogE)
        
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
        
        # 14C Decay parameter
        self.lambd = 1.21e-4 #yr^-1
            
        self.p_models = [(pm.GlobalSplineFitBeta, None), (pm.HillasGaisser2012, "H3a"), (pm.HillasGaisser2012, "H4a"), (pm.PolyGonato, False),
                   (pm.GaisserStanevTilav, "3-gen"), (pm.GaisserStanevTilav, "4-gen"), (pm.CombinedGHandHG, "H3a"),
                   (pm.ZatsepinSokolskaya, "pamela"), (pm.ZatsepinSokolskaya, "default"), (pm.GaisserHonda, None),
                   (pm.Thunman, None), (pm.SimplePowerlaw27, None)]
        self.p_names = ['GlobalSplineFitBeta', 'HillasGaisser2012 H3a', 'HillasGaisser2012 H4a', 'PolyGonato',
                      'GaisserStanevTilav 3-gen', 'GaisserStanevTilav 4-gen', 'CombinedGHandHG H3a',
                      'ZatsepinSokolskaya pamela', 'ZatsepinSokolskaya default', 'GaisserHonda',
                      'Thunman', 'SimplePowerlaw27']
        
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
        

    def argnear_below(self, x, a): 
        # returns the index of the nearest value to x in the array a
        # such that a[i] <= x
        # assuming a is sorted low -> high
        
        # works by interpolating the inverse function of a[i]
        return max(min(int(np.interp(x, a, np.arange(len(a)))), len(a)-1), 0)

    def argnear_above(self, x, a): 
        # returns the index of the nearest value to x in the array a
        # such that a[i] >= x
        # assuming a is sorted low -> high
        
        # works by interpolating the inverse function of a[i]
        return max(min(int(np.interp(x, a, np.arange(len(a))))+1, len(a)-1), 0)
    
    def load_ice_profile(self, ice_eq_depth_file, age_scale_file, rho_ice = None, z_min = None, z_deep = None, z_start = None, sample_length = None):
        if rho_ice is None:
            rho_ice = self.rho_ice
        if z_min is None:
            z_min = self.z_min
        if z_deep is None:
            z_deep = self.z_deep
        if z_start is None:
            z_start = self.z_start
        if sample_length is None:
            sample_length = self.sample_length
            
        """
        Loads ice profile data from .csv files to setup depth bins
        
        Parameters
        ----------------
        ice_eq_depth_file - string
            .csv table converting real depths to ice-equivalent depths
            (Ice-equivalent depth is defined as the mass per square centimeter above that depth, divided by the density of ice)
            Columns:
                z - real depth [m]
                ice_eq_depth - corresponding ice-equivalent depth [meters-ice-eq]
        age_scale_file - string
            .csv table converting depth to ice age
            Columns:
                depths_real - depth of ice [m]
                ages - age of ice [years]
        rho_ice - float
            Density of solid ice [g/cm^3]
            Used to convert ice-equivalent depth to water-equivalent
        z_min - int or float
            Minimum depth of density profile [m]
            (Should always be 0?)
        z_start - int or float
            Depth at which sampling starts [m]
        z_deep - int or float
            Maximum depth of calculation [m]
        sample_length - int or float
            Length of core samples [m]
        """
        
        #self.age_scale_file = age_scale_file # relationship between age and depth of ice at Dome-C
        #self.ice_eq_depth_file = ice_eq_depth_file # relationship bewteen ice-equivalent-depth and real-depth at Dome-C

        # read age-scale file
        age_scale = pd.read_csv(age_scale_file)
        ages = np.array(age_scale['ages']) # years
        depths_real = np.array(age_scale['depths_real']) # meters

        # read ice-eq-depth file
        ice_eq_depth = pd.read_csv(ice_eq_depth_file)
        real_z = np.array(ice_eq_depth['z']) # meters
        ice_eq_z = np.array(ice_eq_depth['ice_eq_z']) # meters (ice-eq) aka mass-depth / ice density
        
        self.rho_ice = rho_ice # density of solid ice at Dome C (g/cm^3)
        
        self.set_mass_depth(depths_real, np.interp(depths_real, real_z, ice_eq_z)*self.rho_ice, ages, z_min, z_deep, z_start, sample_length)
        
        return
    
    def set_mass_depth(self, z_bins, h_bins, t_bins = None, z_min = None, z_deep = None, z_start = None, sample_length = None):
        if t_bins is None:
            t_bins = np.arange(len(z_bins))
        if z_min is None:
            z_min = self.z_min
        if z_deep is None:
            z_deep = self.z_deep
        if z_start is None:
            z_start = self.z_start
        if sample_length is None:
            sample_length = self.sample_length
            
        """
        Sets up depth bins using real and water-equivalent depths
        
        Parameters
        ----------------
        z_bins - numpy array, shape (#z+1), dtype float
            depth bin edges [m]
        h_bins - numpy array, shape (#z+1), dtype float
            mass depth bin edges [meters-water-equivalent (m.w.e.) = hg/cm^2]
            corresponds to z_bins
        t_bins - numpy array, shape (#z+1), dtype float
            ice ages by depth [years]
            corresponds to z_bins
        z_min - int or float
            Minimum depth of density profile [m]
            (Should always be 0?)
        z_start - int or float
            Depth at which sampling starts [m]
        z_deep - int or float
            Maximum depth of calculation [m]
        sample_length - int or float
            Length of core samples [m]
        """
        
        self.z_min = z_min # starting depth for plots (m)
        self.z_deep = z_deep # end depth (m)

        i_min = self.argnear_below(self.z_min, z_bins) # nearest depths_real index to z_min
        i_end = self.argnear_above(self.z_deep, z_bins) # nearest depths_real index to z_end
        
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
        
        self.setup_sample_bins(z_start, sample_length)
        
        return
    
    def load_density(self, density_file, age_scale_file = None, z_min = None, z_deep = None, z_start = None, sample_length = None):
        if z_min is None:
            z_min = self.z_min
        if z_deep is None:
            z_deep = self.z_deep
        if z_start is None:
            z_start = self.z_start
        if sample_length is None:
            sample_length = self.sample_length
            
        """
        Loads ice density data from .csv files to setup depth bins
        
        Parameters
        ----------------
        density_file - string
            .csv table of ice densities
            Columns:
                z - depth [m]
                rho - ice density [g/cm^3]
        age_scale_file - string
            .csv table converting depth to ice age
            Columns:
                depths_real - depth of ice [m]
                ages - age of ice [years]
        z_min - int or float
            Minimum depth of density profile [m]
            (Should always be 0?)
        z_start - int or float
            Depth at which sampling starts [m]
        z_deep - int or float
            Maximum depth of calculation [m]
        sample_length - int or float
            Length of core samples [m]
        """
            
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
        
        self.set_density(real_z, rho, t_bins, z_min, z_deep, z_start, sample_length)
    
    def set_density(self, z_bins, rho, t_bins = None, z_min = None, z_deep = None, z_start = None, sample_length = None):
        if t_bins is None:
            t_bins = np.arange(len(z_bins))
        if z_min is None:
            z_min = self.z_min
        if z_deep is None:
            z_deep = self.z_deep
        if z_start is None:
            z_start = self.z_start
        if sample_length is None:
            sample_length = self.sample_length
            
        """
        Sets up depth bins using real and water-equivalent depths
        
        Parameters
        ----------------
        z_bins - numpy array, shape (#z+1), dtype float
            depth bin edges [m]
        rho - numpy array, shape (#z+1), dtype float
            ice density [g/cm^3]
            corresponds to z_bins
        t_bins - numpy array, shape (#z+1), dtype float
            ice ages by depth [years]
            corresponds to z_bins
        z_min - int or float
            Minimum depth of density profile [m]
            (Should always be 0?)
        z_start - int or float
            Depth at which sampling starts [m]
        z_deep - int or float
            Maximum depth of calculation [m]
        sample_length - int or float
            Length of core samples [m]
        """
        
        self.z_min = z_min # starting depth for plots (m)
        self.z_deep = z_deep # end depth (m)

        i_min = self.argnear_below(self.z_min, z_bins) # nearest depths_real index to z_min
        i_end = self.argnear_above(self.z_deep, z_bins) # nearest depths_real index to z_end
        
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
        
        self.setup_sample_bins(z_start, sample_length)
        
        return
    
    def setup_sample_bins(self, z_start = None, sample_length = None):
        if z_start is None:
            z_start = self.z_start
        if sample_length is None:
            sample_length = self.sample_length
            
        """
        Sets up transformation from depths used in calculation to sample depth bins
        
        Parameters
        ---------------
        z_start - int or float
            Depth at which sampling starts [m]
        sample_length - int or float
            Length of core samples [m]
        """
        
        self.z_start = z_start # starting depth of 14C accumulation (m) - close-off depth beneath firn layer
        self.sample_length = sample_length # length of ice core samples (m)

        self.i_start = self.argnear_below(self.z_start, self.z_bins) # index of first bin beneath starting point for 14C accumulation

        z_samp_ideal = np.arange(self.z_bins[self.i_start],self.z_bins[-1],self.sample_length)
        i_samp = np.append(np.argmin(abs(self.z_bins.reshape((1,-1))-z_samp_ideal.reshape(-1,1)), axis=1), len(self.z_bins)-1)
        i_samp[0] = self.i_start # just making sure

        # Define sample depth bins
        self.z_samp_bins = self.z_bins[i_samp] # sample depth bin edges (m)
        self.z_samp = (self.z_samp_bins[:-1]+self.z_samp_bins[1:])/2 # bin-average of sample depth (m)
        self.dz_samp = np.diff(self.z_samp_bins) # bin-width of sample depth (m)

        # Define sample compression matrix
        dh_samp = np.zeros(len(i_samp)-1)
        self.S_mat = np.zeros((len(self.z_bins)-1, len(i_samp)-1))
        for i in range(len(i_samp)-1):
            dh_samp[i] = np.sum(self.dh[i_samp[i]:i_samp[i+1]])
            self.S_mat[i_samp[i]:i_samp[i+1], i] = self.dh[i_samp[i]:i_samp[i+1]]/dh_samp[i]
        
        return
    
    def set_zenith_bins(self, N_ang = 10):
        
        """
        Sets up zenith angle bins
        
        Parameters
        ---------------
        N_ang - int
            Number of zenith angle bins [unitless]
            zenith angle bins are equally spaced in solid angle
        """
        
        self.N_ang = N_ang

        # Define zenith angle bins
        self.cosTH_bins = np.linspace(1,0,self.N_ang+1)
        self.cosTH = (self.cosTH_bins[:-1]+self.cosTH_bins[1:])/2
        self.dcosTH = -np.diff(self.cosTH_bins)
        
        return
    
    def set_energy_bins(self, logE_min = -1, logE_max = 11, logE_mu_max = 7, dlogE = 0.1):
        
        """
        Sets up energy bins
        
        Parameters
        ---------------
        logE_min - int or float
            log base 10 of the minimum tracked particle energy [log10 GeV]
        logE_max - int or float
            log base 10 of the maximum tracked particle energy [log10 GeV]
        dlogE - int or float
            Width of particle energy bins in log base 10 [log10 GeV]
        """
        
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
    
    def setup_mceq(self, elev=None):
        if not (elev is None):
            self.elev = elev
            config.h_obs = self.elev
            
        """
        Sets up a dummy MCEq instance to pull data from
        
        Parameters
        ----------------
        elev - int or float
            Elevation above sea level [m]
            Used in Matlab atmospheric calculation
        """

        interaction_model = "SIBYLL-2.3c"
        #interaction_model = "SIBYLL-2.3"
        #interaction_model = "SIBYLL-2.1"
        #interaction_model = "EPOS-LHC"
        #interaction_model = "QGSJET-II-04"
        #interaction_model = "DPMJET-III"
        #interaction_model = 'DPMJETIII191'

        density_model, density_name = ('CORSIKA', ('USStd', None)), 'CORSIKA_USStd'
        #density_model, density_name = ('CORSIKA',('SouthPole', 'June')), 'CORSIKA_SP_Jun'
        #density_model, density_name = ('CORSIKA',('SouthPole', 'December')), 'CORSIKA_SP_Dec'
        
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
    
    def set_cross_sections(self, sigma_E = None, E_sigma = None, alpha = None, N = None, f_tot = None):
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
                
        """
        Sets up parameters for production rates calculations
        
        Parameters
        ----------------
        sigma_E - float
            fast muon interaction cross section measurement [cm^2]
            default value = 4.5e-28
            (see Heisinger)
        E_sigma - float
            energy of cross section measurement [GeV]
            default value = 190.
        alpha - float
            cross section energy scaling factor [unitless]
            sigma(E) = sigma_0 * E**alpha
            default value = 0.75
        N - float
            density of fast muon interaction targets (oxygen nucleii) [hg^-1]
            #oxgyen nucleii per molecule (1) / molecular mass (0.1802 / 6.022e23)
        f_tot - float
            effective probability of 14C production by capture of a stopped negative muon [unitless]
            f_tot = f_C * f_D * f_star
            f_C - 
            f_D - 
            f_star - 
        """

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
    
    def set_pressure(self, pressure):
        
        """
        Sets up Balco elevation adjustment factors, starting from pressure
        
        Parameters
        --------------
        pressure - float
            atmospheric pressure at site [Pa]
            used to calculate H in Balco
        """
        
        self.pressure = pressure # surface pressure in Pa, should be 65800 for Dome C

        # figure the difference in atmospheric depth from sea level in g/cm2
        self.H = (1013.25 - self.pressure/100)*1.019716 # the 1.019716 number is basically just 1/g accounting for needed unit conversions
        
        return
    
    def set_H(self, H):
        
        """
        Sets up Balco elevation adjustment factors, starting from atmospheric depth above sea level
        
        Parameters
        ----------------
        H - float
            atmospheric depth above sea level [m.w.e. = hg/cm^2]
            H = (1013.25 - pressure/100)*1.019716
        """
        
        self.H = H
        
        self.pressure = (1013.25 - self.H/1.019716)*100
        
        return
    
    def set_models(self, clear=True, update_names=True, **kwargs):
        
        for s in self.stages:
            new_models = [ModelStep(*m) for m in kwargs.get(s, [])]
            if clear:
                self.models[s] = new_models
            else:
                self.models[s] += new_models
        
        if update_names:
            self.build_model_names()
        
        return
    
    def add_models(self, **kwargs):
        
        self.set_models(clear=False, **kwargs)
        
        return
        
    def clear_models(self):
        
        self.set_models()
        
    def build_model_names(self):
        for s in self.stages:
            self.model_names[s] = sum([sum([['{}{}'.format(i,n) for i in self.model_names.get(m.input,[''])] for n in m.names], []) for m in self.models[s]], [])
        
    def set_data(self, clear=True, **kwargs):
        
        for s in self.stages:
            new_data = [DataSet(*d) for d in kwargs.get(s, [])]
            if clear:
                self.data[s] = new_data
            else:
                self.data[s] += new_data
        
    def add_data(self, **kwargs):
        
        self.set_data(clear=False, **kwargs)
    
    def clear_data(self):
        
        self.set_data()

    def set_primary(self, Phi0, clear=True, run=False):

        self.set_models(clear=clear, **{self.stages[0]: Phi0})
        
        if run:
            self.calculate(start=0, end=0)
        
        return

    def load_primary(self, p_models=None, clear=True, run=False): # primary CR intensities
        
        if p_models is None:
            p_models = self.p_models
            
        Phi0 = F.load_primary(self, p_models)
        
        self.set_models(clear=clear, **{self.stages[0]: Phi0})

        if run:
            self.calculate(start=0, end=0)
        return

    def set_primary_identity(self, run=False):

        # sets up primary flux matrix to test each energy of p+ and n0 individually

        Phi0 = F.set_primary_identity(self)
        
        self.set_models(**{self.stages[0]: Phi0})

        if run:
            self.calculate(start=0, end=0)
        return
    
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

    def calculate(self, start=0, end=-1, models=None, output=False, clear=True, **kwargs):
            
        """
        
        
        Parameters
        ------------------------
        start - 
        
        end - 
        
        output - bool
            
            
        Returns
        -----------------------
        if output, returns:
        Phi - 
            
        """
        if end == -1:
            end = len(self.stages)-1
        if models is None:
            models = self.models
        
        for s in self.stages[start:end+1]:
            print('Running {} stage...'.format(s))
            if clear:
                self.Phi[s] = np.concatenate([m.run(self) for m in models[s]])
                self.model_names[s] = sum([sum([['{}{}'.format(i,n) for i in self.model_names.get(m.input,[''])] for n in m.names], []) for m in models[s]], [])
            else:
                self.Phi[s] = np.append(self.Phi[s], np.concatenate([m.run(self) for m in models[s]]), axis=0)
                self.model_names[s] += sum([sum([['{}{}'.format(i,n) for i in self.model_names.get(m.input,[''])] for n in m.names], []) for m in models[s]], [])
            print('{} stage complete'.format(s))
        
        if output:
            return self.Phi # should have an option to return intermediate steps as well
        return
    
    # def set_primary_data
    
    # def set_atm_data
    
    # def set_ice_data
    
    # def set_prod_data
    
    # def set_CO_data
    
    # def set_array
    
    # def plot_primary
    
    # def plot_atm
    
    # def plot_ice
    
    # def plot_prod
    
    # def plot_CO
    
    # def save_primary_to_csv
    
    # def save_atm_to_csv
    
    # def save_ice_to_csv
    
    def save_prod_to_csv(self, folder=''):
        if folder != '':
            folder = folder+'/'
            
        """
        
        
        Parameters
        -----------------
        folder - 
            
        """
        
        # iterate this
        np.savetxt('{}P_fast_p+_{}_{}m.csv'.format(folder, self.atmice_labels[0], self.elev), self.P_14C[0,:120,0], delimiter=',')
        np.savetxt('{}P_fast_n0_{}_{}m.csv'.format(folder, self.atmice_labels[0], self.elev), self.P_14C[0,120:,0], delimiter=',')
        np.savetxt('{}P_neg_p+_{}_{}m.csv'.format(folder, self.atmice_labels[0], self.elev), self.P_14C[0,:120,1], delimiter=',')
        np.savetxt('{}P_neg_n0_{}_{}m.csv'.format(folder, self.atmice_labels[0], self.elev), self.P_14C[0,120:,1], delimiter=',')

        np.savetxt('{}Depth.csv'.format(folder), self.z, delimiter=',')
        np.savetxt('{}Energy.csv'.format(folder), self.E, delimiter=',')
        
        return
    
    # def save_CO_to_csv