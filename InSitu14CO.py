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

#UNITS : cgs
#(in progress)

# could split this into:
# Sites
# Calculation Steps (Models)
# Data points

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
    def __init__(self, pressure = 65800, elev=3120, rho_ice = 0.9239, f_factors = [0.072, 0.066], ice_eq_depth_file = 'Real_vs_ice_eq_depth.csv', age_scale_file = 'DomeC_age_scale_Apr2023.csv', z_min = 0, z_deep = 300, z_start = 96.5, sample_length = 20, N_ang = 10, logE_min = -1, logE_max = 11, dlogE = 0.1):
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
        self.set_energy_bins(logE_min, logE_max, dlogE)
        
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

        self.set_models()
            
        self.p_models = [(pm.GlobalSplineFitBeta, None), (pm.HillasGaisser2012, "H3a"), (pm.HillasGaisser2012, "H4a"), (pm.PolyGonato, False),
                   (pm.GaisserStanevTilav, "3-gen"), (pm.GaisserStanevTilav, "4-gen"), (pm.CombinedGHandHG, "H3a"),
                   (pm.ZatsepinSokolskaya, "pamela"), (pm.ZatsepinSokolskaya, "default"), (pm.GaisserHonda, None),
                   (pm.Thunman, None), (pm.SimplePowerlaw27, None)]
        self.p_names = ['GlobalSplineFitBeta', 'HillasGaisser2012 H3a', 'HillasGaisser2012 H4a', 'PolyGonato',
                      'GaisserStanevTilav 3-gen', 'GaisserStanevTilav 4-gen', 'CombinedGHandHG H3a',
                      'ZatsepinSokolskaya pamela', 'ZatsepinSokolskaya default', 'GaisserHonda',
                      'Thunman', 'SimplePowerlaw27']
        
        self.Phi0 = np.zeros((0,2,len(self.E)))
        #Phi0
        #axis0 - Primary Model
        #axis1 - Particle Species (proton, neutron)
        #axis2 - Primary Energy
        
        self.Phi_atm = np.zeros((0,0,len(self.cosTH),2,len(self.E)))
        #Phi_atm
        #axis0 - Atmospheric Model
        #axis1 - Primary Model
        #axis2 - Zenith Angle
        #axis3 - Muon Charge (positive, negative)
        #axis4 - Muon Energy
        
        self.Phi_ice = np.zeros((0,0,len(self.cosTH),2,len(self.E),len(self.z_bins)))
        #Phi_ice
        #axis0 - Atmospheric & Ice Models
        #axis1 - Primary Model
        #axis2 - Muon Charge (positive, negative)
        #axis3 - Muon Energy
        #axis4 - depth bin EDGES (top -> bottom)
        
        self.P_14C = np.zeros((0,0,2,len(self.z)))
        #P_14C
        #axis0 - Production, Atmospheric & Ice Models
        #axis1 - Primary Model
        #axis2 - Production Mode (fast, neg)
        #axis3 - depth (top -> bottom)
        
        self.CO = np.zeros((0,0,len(self.z)))
        #CO
        #axis0 - Flow, Production, Atmospheric & Ice Models
        #axis1 - Primary Model
        #axis2 - depth (top -> bottom)
        

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
    
    def set_energy_bins(self, logE_min = -1, logE_max = 11, dlogE = 0.1):
        
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

        # Define energy bins
        self.logE_bins = np.arange(self.logE_min, self.logE_max+self.dlogE, self.dlogE) # log10 GeV
        self.logE = (self.logE_bins[:-1]+self.logE_bins[1:])/2 # log10 GeV
        self.E_bins = 10.**(self.logE_bins) # GeV
        self.E = 10.**(self.logE) # bin-average of E (GeV)
        self.dE = np.diff(self.E_bins) # bin-width of E (GeV)

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
    
    def add_mceq_models(self, atm = False, ice = False, atmice=False, interaction_models = ["SIBYLL-2.3c","SIBYLL-2.3","SIBYLL-2.1","EPOS-LHC","QGSJET-II-04","DPMJET-III",'DPMJETIII191'], density_models = [('CORSIKA', ('USStd', None)), ('CORSIKA',('SouthPole', 'December'))], density_names = ['CORSIKA_USStd', 'CORSIKA_SP_Dec']):
        
        """
        
        
        Parameters
        ---------------
        
        """
        
        # default functions & parameters
        #elevs = [0,3120]
        #self.interaction_models = ["SIBYLL-2.3c","SIBYLL-2.3","SIBYLL-2.1","EPOS-LHC","QGSJET-II-04","DPMJET-III",'DPMJETIII191']
        #self.density_models = [('CORSIKA', ('USStd', None)), ('CORSIKA',('SouthPole', 'December'))]#,('CORSIKA',('SouthPole', 'June'))]
        #self.density_names = ['CORSIKA_USStd', 'CORSIKA_SP_Dec']#,'CORSIKA_SP_Jun']
        
        if atm:
            for j,inter in enumerate(self.interaction_models):
                for k,d in enumerate(self.density_models):
                    self.atm.append([self.MCEq_atm, (inter, d)])
                    self.atm_labels.append('MCEq-{}-{}'.format(inter, self.density_names[k]))
            
        if ice:
            for j,inter in enumerate(self.interaction_models):
                self.ice.append([self.MCEq_ice, (inter)])
                self.ice_labels.append('_MCEq-{}'.format(inter))

        if atmice:
            for j,inter in enumerate(self.interaction_models):
                for k,d in enumerate(self.density_models):
                    self.atmice.append([self.MCEq_atmice, (inter, d)])
                    self.atmice_labels.append('MCEq-{}-{}'.format(inter, self.density_names[k]))
                    
        self.setup_model_names()
                
        return
    
    def set_models(self, atm = None, atm_labels = None, ice = None, ice_labels = None, atmice = None, atmice_labels = None, prod = None, prod_labels = None, prodfull = None, prodfull_labels = None, flow = None, flow_labels = None, flowfull = None, flowfull_labels = None):
        if atm is None:
            if hasattr(self, 'atm'):
                atm = self.atm
            else:
                atm = [(self.judge_nash, ()), (self.MCEq_atm, ()), (self.daemonflux_atm, ())]
        if atm_labels is None:
            if hasattr(self, 'atm_labels'):
                atm_labels = self.atm_labels
            else:
                atm_labels = ['JN-fit', 'MCEq-atm', 'daemonflux']
                
        if ice is None:
            if hasattr(self, 'ice'):
                ice = self.ice
            else:
                ice = [(self.Heisinger_ice, ()), (self.MCEq_ice, ())]
        if ice_labels is None:
            if hasattr(self, 'ice_labels'):
                ice_labels = self.ice_labels
            else:
                ice_labels = ['_H-ice-norm', '_MCEq-ice']
                
        if atmice is None:
            if hasattr(self, 'atmice'):
                atmice = self.atmice
            else:
                atmice = [(self.MCEq_atmice, ())]
        if atmice_labels is None:
            if hasattr(self, 'atmice_labels'):
                atmice_labels = self.atmice_labels
            else:
                atmice_labels = ['MCEq']
                
        if prod is None:
            if hasattr(self, 'prod'):
                prod = self.prod
            else:
                prod = [(self.Dyonisius_prod, ())]
        if prod_labels is None:
            if hasattr(self, 'prod_labels'):
                prod_labels = self.prod_labels
            else:
                prod_labels = ['']
                
        if prodfull is None:
            if hasattr(self, 'prodfull'):
                prodfull = self.prodfull
            else:
                prodfull = [(self.Heisinger_full, ())]
        if prodfull_labels is None:
            if hasattr(self, 'prodfull_labels'):
                prodfull_labels = self.prodfull_labels
            else:
                prodfull_labels = ['Heisinger']
                
        if flow is None:
            if hasattr(self, 'flow'):
                flow = self.flow
            else:
                flow = [(self.Basic_flow, ())]
        if flow_labels is None:
            if hasattr(self, 'flow_labels'):
                flow_labels = self.flow_labels
            else:
                flow_labels = ['']
                
        if flowfull is None:
            if hasattr(self, 'flowfull'):
                flowfull = self.flowfull
            else:
                flowfull = [(self.load_profile, ())]
        if flowfull_labels is None:
            if hasattr(self, 'flowfull_labels'):
                flowfull_labels = self.flowfull_labels
            else:
                flowfull_labels = ['Balco-Matlab']
                
        """
        
        
        Parameters
        ---------------
        
        """
        
        
        self.atm = atm
        self.atm_labels = atm_labels

        self.ice = ice
        self.ice_labels = ice_labels

        self.atmice = atmice
        self.atmice_labels = atmice_labels

        self.prod = prod
        self.prod_labels = prod_labels

        self.prodfull = prodfull
        self.prodfull_labels = prodfull_labels

        self.flow = flow
        self.flow_labels = flow_labels

        self.flowfull = flowfull
        self.flowfull_labels = flowfull_labels
        
        self.setup_model_names()
        
        return
    
    def clear_models(self):
        # 
        
        self.atm = []
        self.atm_labels = []

        self.ice = []
        self.ice_labels = []

        self.atmice = []
        self.atmice_labels = []

        self.prod = []
        self.prod_labels = []

        self.prodfull = []
        self.prodfull_labels = []

        self.flow = []
        self.flow_labels = []

        self.flowfull = []
        self.flowfull_labels = []
        
        self.setup_model_names()
    
    def setup_model_names(self):
        # 
        
        self.model_names = []

        for l in self.flow_labels:
            for k in self.prod_labels:
                for j in self.ice_labels:
                    for i in self.atm_labels:
                        self.model_names.append('{}{}{}{}'.format(i,j,k,l))
                for i in self.atmice_labels:
                    self.model_names.append('{}{}{}'.format(i,k,l))
            for i in self.prodfull_labels:
                self.model_names.append('{}{}'.format(i,l))
        for i in self.flowfull_labels:
            self.model_names.append(i)
            
        return
    
    def judge_nash(self, Phi0, K_mu = 1.268):
        
        """
        
        
        Parameters
        ----------------
        Phi0 - 
            
        K_mu - 
            
        """
        
        #Phi0
        #axis0 - Primary Model (Energy spectrum & Time dependence)
        #axis1 - Particle Species (proton, neutron)
        #axis2 - Primary Energy
        
        E0 = np.reshape(self.E, (1,1,-1))
        cosTH = np.reshape(self.cosTH, (-1,1,1))
        
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
        
        Phi_mu = np.expand_dims(Phi_mu, 0)
        rescale = np.ones((np.shape(Phi0)[0], 1,1,1))
        Phi_atm = Phi_mu * rescale
        
        print(np.shape(Phi_atm))

        return np.nan_to_num(Phi_atm)

    def bugaev_reyna(self, Phi0, K_mu = 1.268):
        
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
        
        E0 = np.reshape(self.E, (1,1,-1))
        cosTH = np.reshape(self.cosTH, (-1,1,1))
        
        A_B = 0.00253
        a0 = 0.2455
        a1 = 1.288
        a2 = -0.2555
        a3 = 0.0209

        y = np.log10(E0*cosTH)

        Phi_R = cosTH**3 * A_B * (E0*cosTH)**(-(a3*y**3 + a2*y**2 + a1*y + a0))
        
        Phi_mu = Phi_R * np.reshape([K_mu/(K_mu+1), 1/(K_mu+1)], (1,-1,1))
        
        Phi_mu = np.expand_dims(Phi_mu, 0)
        rescale = np.ones((np.shape(Phi0)[0], 1,1,1))
        Phi_atm = Phi_mu * rescale
        
        print(np.shape(Phi_atm))

        return Phi_atm
    
    def SDC(self, Phi0, K_mu = 1.268):
        
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
        
        E0 = np.reshape(self.E, (1,1,-1))
        cosTH = np.reshape(self.cosTH, (-1,1,1))

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
        
        Phi_mu = np.expand_dims(Phi_mu, 0)
        rescale = np.ones((np.shape(Phi0)[0], 1,1,1))
        Phi_atm = Phi_mu * rescale
        
        print(np.shape(Phi_atm))

        return Phi_atm
    
    def att_L(self, h=None):
        if h is None:
            h = self.h
        
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
    
    def phi_vert_slhl(self, h=None):
        if h is None:
            h = self.h
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
    
    def R_vert_slhl(self, h=None):
        if h is None:
            h = self.h
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
    
    def phi_vert_site(self, h=None, dh=None, H=None, h_end=2e3):
        if h is None:
            h = self.h
        if dh is None:
            dh = self.dh
        if H is None:
            H = self.H
            
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
    
        Phi_v= self.phi_vert_slhl(h)

        R_v = self.R_vert_slhl(h)

        R_site = R_v * np.exp(H/self.att_L(h))

        Phi_end = self.phi_vert_slhl(h_end)

        dh_ext = 1
        h_ext = np.arange(h[-1]+dh_ext, h_end+dh_ext, dh_ext)

        h_int = np.append(h, h_ext)
        dh_int = np.append(dh, dh_ext + 0*h_ext)

        R_int = self.R_vert_slhl(h_int) * np.exp(H/self.att_L(h_int))

        Phi_site = np.flip(np.cumsum(np.flip(R_int * dh_int))) + (1-np.exp(H/self.att_L(h_end)))*Phi_end

        Phi_site = Phi_site[:len(h)]

        return Phi_site, R_site
    
    def cos_pow(self, h, H=None):
        if H is None:
            H = self.H
            
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
    
    def f_mu_neg(self, h = None):
        K_mu = 1.268 # +/- 0.008 + 0.002 * E[GeV]

        return 1/(K_mu+1)
    
    def phi_all(self, h=None, dh=None, H=None, cos_pow_func=None, f_func=None):
        if h is None:
            h = self.h
        if dh is None:
            dh = self.dh
        if H is None:
            H = self.H
        if cos_pow_func is None:
            cos_pow_func = self.cos_pow
        if f_func is None:
            f_func = self.f_mu_neg
            
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
    
        n, dndh = cos_pow_func(h, H)

        Phi_v, R_v = self.phi_vert_site(h, dh, H)

        Phi = 2*np.pi/(n+1) * Phi_v

        R = f_func(h) * (2*np.pi * R_v + Phi*dndh) / (n+1)

        return Phi, R # cm^-2 s^-1
    
    def Heisinger(self, h=None):
        if h is None:
            h = self.h
            
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
    
    def E_surf(self, E_d, X, a=None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
            
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
    
        return ((E_d + a/b)*np.exp(X*b)-a/b).clip(min=self.E_bins[0])
    
    def Heisinger_ice(self, Phi_atm, a=None, b=None, norm=True, H=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if H is None:
            H = self.H
            
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

        #Phi_atm
        #axis0 - Atmospheric Model
        #axis1 - Primary Model
        #axis2 - Zenith Angle
        #axis3 - Muon Charge (positive, negative)
        #axis4 - Muon Energy

        X = np.reshape(self.h_bins,(1,1,-1))/np.reshape(self.cosTH,(1,-1,1))

        E_bounds = self.E_surf(np.reshape(self.E_bins,(-1,1,1)), X, a, b) # Energy bins at depth projected back to their surface energies
        
        Phi_proj = np.zeros((*np.shape(Phi_atm), len(self.h_bins)))

        # Now, this is going to look incomprehensible, but...
        for i in tqdm(range(len(self.cosTH))):
            for j in range(len(self.h_bins)):
                E_proj = E_bounds[:,i,j]
                deep = True # Starting from a Phi_proj energy bin edge?  False means Phi_atm
                #print(i,j,E_proj[0])
                k=np.arange(len(self.E_bins))[self.E_bins<=E_proj[0]][-1] # current Phi_atm energy bin
                l=0 # current Phi_proj energy bin
                while k < len(self.E_bins)-1 and l < len(E_proj)-1: # step through energy bin edge, one by one, putting muons from Phi_atm into the Phi_proj bin corresponding to their projected underground energy
                    if self.E_bins[k+1]<=E_proj[l+1]: # if the next bin edge is from Phi_atm
                        Phi_proj[:, :, i, :, l, j] += Phi_atm[:,:,i,:,k] * (self.E_bins[k+1]-(E_proj[l] if deep else self.E_bins[k]))
                        k += 1 # Start the next step from Phi_atm's bin edge
                        deep = False
                    else: # if the next bin boundary is from Phi_proj
                        Phi_proj[:, :, i, :, l, j] += Phi_atm[:,:,i,:,k] * (E_proj[l+1]-(E_proj[l] if deep else self.E_bins[k]))
                        l += 1 # Start the next step from Phi_proj's bin edge
                        deep = True
                # Hate to use a While loop, but it should have to stop before 2*len(E_bins) steps

        Phi_proj = np.sum(Phi_proj / np.reshape(self.dE,(1,1,1,1,-1,1)) * np.reshape(self.dcosTH, (1,1,-1,1,1,1)), axis=2) * 2 * np.pi
        #axis0 - Atmospheric Model
        #axis1 - Primary Model
        #axis2 - Muon Charge (positive, negative)
        #axis3 - Muon Energy
        #axis4 - depth (top -> bottom)

        # Normalize the result to Heisinger's total flux fit (elevation adjusted by Balco)
        if norm:
            return Phi_proj / np.sum(Phi_proj * np.reshape(self.dE,(1,1,1,-1,1)), axis=(2,3), keepdims=True) * np.reshape(self.phi_all(self.h_bins,np.append(self.dh,self.dh[-1]),self.H)[0], (1,1,1,1,-1))
        return Phi_proj
        
    def Heisinger_full(self, Phi0, H=None):
        if H is None:
            H = self.H
            
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
        
        E_pred, Beta_pred = self.Heisinger()

        Phi, R = self.phi_all(self.h, self.dh, H) # Total Muon Flux, Negative Muon Stopping Rate

        P_neg = R * self.f_tot

        P_fast = self.sigma_0 * Phi * E_pred**self.alpha * Beta_pred * self.N

        P_14C = np.reshape([P_fast, P_neg], (1,2,-1)) /100 * 60 * 60 * 24 * 365.25 # g^-1, a^-1

        #rescale = np.ones((np.shape(Phi0)[0], 1))
        
        #P_14C
        #axis1 - Primary Model
        #axis2 - Production Mode (fast, neg)
        #axis3 - depth (top -> bottom)

        return P_14C * np.ones((np.shape(Phi0)[0],1,1))
    
    def get_mceq_path(self, mceq, cTH):
        
        """
        
        
        Parameters
        ---------------------
        mceq - 
            
        cTH - 
            
        
        Returns 
        ----------------------
        dX - 
            
        dz - 
            
        """
        
        mceq.set_theta_deg(180*np.arccos(cTH)/np.pi)
        mceq._calculate_integration_path(None,'X')
        nsteps, dX, rho_inv, grid_idcs = mceq.integration_path
        return dX, dX*rho_inv
    
    # WHERE ARE YOUR PARENTS??
    # Here we're trying to assemble a list of all particle species and energies which could ever lead
    # to the production of a muon
    # Any particles not falling under that category get cut; we don't need to track them

    def get_parents(self, children, mceq=None):
        if mceq is None:
            mceq = self.mceq
            
        """
        
        
        Parameters
        ---------------
        children - 
        
        mceq - 
        
        
        Returns
        ----------------
        parents - 
            
        """
            
        #rows = products
        #columns = sources
        row0, col0, val0 = find(mceq.int_m)
        row1, col1, val1 = find(mceq.dec_m)
        row, col = np.append(row0[val0!=0],row1[val1!=0]), np.append(col0[val0!=0],col1[val1!=0])

        # make sure the identity exists
        row = np.append(row, np.arange(len(mceq._phi0)))
        col = np.append(col, np.arange(len(mceq._phi0)))

        parents = np.copy(children)

        i=0
        while len(parents) != len(np.unique(col[np.isin(row, parents)])) and i<100:
            parents = np.unique(col[np.isin(row, parents)])
            i+=1

        if i>=100:
            print('failed to converge')

        return parents
    
    def i_to_pname(self, i, mceq=None):
        if mceq is None:
            mceq = self.mceq
            
        """
        
        
        Parameters
        ---------------
        i - 
        
        mceq - 
        
        
        Returns
        -----------------
        pnames - 
            
        """
            
        return np.array(list(mceq.pman.pname2pref.keys()))[np.unique(i//len(mceq._energy_grid.c))]
    
    def mceq_integrate(self, phi0, dX, dz, int_m, dec_m, grid=False):
        
        """
        
        
        Parameters
        ----------------
        phi0 - numpy array
            
        dX - numpy array
            
        dz - numpy array
            
        int_m - numpy array
            
        dec_m - numpy array
            
        grid - bool
            
        
        Returns
        ----------------
        grid_sol - numpy array
            
        
        or
        
        phc - numpy array
            
        """
        
        phc = np.copy(phi0)

        if grid:
            grid_sol = np.zeros((len(dX)+1, *np.shape(phi0))) # grid_sol begins with the right shape, to avoid restructuring
            grid_sol[0] = np.copy(phi0)

        for step in tqdm(range(len(dX))): # added option for tqdm progress bar
            phc += int_m.dot(phc)*dX[step] + dec_m.dot(phc)*dz[step]
            phc[phc<1e-250] = 0. # exreme low values set to 0, improving efficiency for large slant depths

            if grid:
                grid_sol[step+1] = np.copy(phc) # grid_sol no longer appends

        if grid:
            return grid_sol

        return phc
    
    def solve_mceq(self, mceq, int_grid=None, grid_var='X', use_tqdm=False):
        
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
    
    def MCEq_atm(self, Phi0, interaction_model="SIBYLL-2.3c", density_model=('CORSIKA', ('USStd', None)), elev=None, solver='default'):
        if elev is None:
            elev = self.elev
            
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
        #Phi0
        #axis0 - Primary Model (Energy spectrum & Time dependence)
        #axis1 - Particle Species (proton, neutron)
        #axis2 - Primary Energy

        import mceq_config as config
        config.debug_level = 0
        config.h_obs = elev # elevation in (m) of Dome-C
        config.enable_default_tracking = False
        config.e_min = self.E_bins[0]*10.**0.1
        config.e_max = self.E_bins[-1]
        config.max_density = 0.001225
        config.dedx_material = 'air'

        #mceq_air.set_interaction_model(interaction_model)
        #config.h_obs = elev
        #mceq_air.set_density_model(density_model)
        #int_m_air, dec_m_air = mceq_air.int_m[p_cut][:,p_cut], mceq_air.dec_m[p_cut][:,p_cut]

        mceq = MCEqRun(
            interaction_model=interaction_model,
            theta_deg = 0,
            density_model = density_model,
            #medium=medium,
            primary_model = (pm.GaisserHonda, None),
        )

        pname = mceq.pman.pname2pref
        phi0 = np.zeros((np.shape(Phi0)[0], len(mceq._phi0)))
        phi0[:,pname['p+'].lidx:pname['p+'].uidx] = Phi0[:,0]
        phi0[:,pname['n0'].lidx:pname['n0'].uidx] = Phi0[:,1]
        #phi0 = phi0[:,p_cut]
        phi0 = np.moveaxis(phi0, 1, 0)
        #phi0 = phi0.reshape((len(p_cut),-1))
        phi0 = phi0.reshape((len(mceq._phi0),-1))

        #phi_mu = np.zeros((np.shape(Phi0)[0],len(cosTH),2,len(mu_pos_cut),np.shape(Phi0)[-1]))
        phi_mu = np.zeros((np.shape(Phi0)[0],len(self.cosTH),2,len(self.E)))

        for i in tqdm(range(len(self.cosTH))):
            #dX_air, dz_air = get_mceq_path(mceq_air, cosTH[i])
            #phi_surf = mceq_integrate(phi0, dX_air, dz_air, int_m_air, dec_m_air)
            #phi_surf = np.moveaxis(phi_surf.reshape(len(p_cut),np.shape(Phi0)[0],np.shape(Phi0)[-1]), 0, 1)

            mceq.set_theta_deg(180*np.arccos(self.cosTH[i])/np.pi)
            mceq._phi0 = phi0
            
            if solver == 'default':
                self.solve_mceq(mceq)
            else:
                # 'numpy', 'cuda', or 'mkl'
                config.kernel_config = solver
                mceq.solve()
            
            phi_surf = np.moveaxis(mceq._solution.reshape(len(phi0),np.shape(Phi0)[0]), 0, 1)

            phi_mu[:,i,0] += phi_surf[:,pname['mu+'].lidx:pname['mu+'].uidx]
            phi_mu[:,i,0] += phi_surf[:,pname['mu+_l'].lidx:pname['mu+_l'].uidx]
            phi_mu[:,i,0] += phi_surf[:,pname['mu+_r'].lidx:pname['mu+_r'].uidx]

            phi_mu[:,i,1] += phi_surf[:,pname['mu-'].lidx:pname['mu-'].uidx]
            phi_mu[:,i,1] += phi_surf[:,pname['mu-_l'].lidx:pname['mu-_l'].uidx]
            phi_mu[:,i,1] += phi_surf[:,pname['mu-_r'].lidx:pname['mu-_r'].uidx]
        
        print(np.shape(phi_mu))

        return phi_mu
    
    def MCEq_ice(self, Phi_atm, interaction_model="SIBYLL-2.3c", solver='default'):
        
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
        #Phi_atm
        #axis0 - Atmospheric Model
        #axis1 - Primary Model
        #axis2 - Zenith Angle
        #axis3 - Muon Charge (positive, negative)
        #axis4 - Muon Energy

        import mceq_config as config
        config.debug_level = 0
        config.enable_default_tracking = False
        config.e_min = self.E_bins[0]*10.**0.1
        config.e_max = self.E_bins[-1]
        config.max_density = self.rho_ice
        config.dedx_material='ice'
        medium = 'ice'

        target = GeneralizedTarget(len_target=self.z_bins[-1]*100, env_density = self.rho_ice, env_name = 'ice')

        mceq = MCEqRun(
            interaction_model=interaction_model,
            theta_deg = 0,
            density_model = target,
            medium=medium,
            primary_model = (pm.GaisserHonda, None),
        )

        #mceq_ice.set_interaction_model(interaction_model)
        #int_m_ice, dec_m_ice = mceq_ice.int_m[p_cut][:,p_cut], mceq_ice.dec_m[p_cut][:,p_cut]

        pname = mceq.pman.pname2pref
        phi0 = np.zeros((*np.shape(Phi_atm)[:3], len(mceq._phi0)))
        phi0[:,:,:,pname['mu+'].lidx:pname['mu+'].uidx] = Phi_atm[:,:,:,0]
        phi0[:,:,:,pname['mu-'].lidx:pname['mu-'].uidx] = Phi_atm[:,:,:,1]
        phiF = np.copy(phi0)
        #phi0 = phi0[:,:,:,p_cut]
        phi0 = np.moveaxis(phi0, (2,3), (0,1))
        phi0 = phi0.reshape((len(self.cosTH),len(mceq._phi0),-1))

        phi_mu = np.zeros((*np.shape(Phi_atm)[:2],2,np.shape(Phi_atm)[-1],len(self.h_bins)))

        for i in tqdm(range(len(self.cosTH))):
            #phi_deep = mceq_integrate(phi0[i], dh/cosTH[i]*100., dz/cosTH[i]*100., int_m_ice, dec_m_ice, grid=True)
            #phi_deep = np.moveaxis(phi_deep.reshape(len(h_bins), len(p_cut), *np.shape(Phi_atm)[:2], np.shape(Phi_atm)[-1]), (0,1), (-2,2))

            target = GeneralizedTarget(len_target=self.z_bins[-1]*100/self.cosTH[i], env_density = self.rho_ice, env_name = 'ice')
            target.mat_list = [[self.z_bins[j]*100/self.cosTH[i], self.z_bins[j+1]*100/self.cosTH[i], self.rho[j], 'ice'] for j in range(len(self.z_bins)-1)]
            target._update_variables()

            mceq.set_density_model(target)
            mceq._phi0 = phi0[i]
            
            if solver == 'default':
                self.solve_mceq(mceq, int_grid=self.h_bins/self.cosTH[i]*100)
            else:
                # 'numpy', 'cuda', or 'mkl'
                config.kernel_config = solver
                mceq.solve(int_grid=h_bins/cosTH[i]*100)
            
            #mceq.grid_sol = np.expand_dims(mceq._phi0, axis=0) * np.ones((len(z_bins),1,1))
            phi_deep = np.moveaxis(mceq.grid_sol.reshape(len(self.h_bins), len(phi0[i]), *np.shape(Phi_atm)[:2]), (0,1), (-1,2))

            #phi_deep = np.expand_dims(phiF[:,:,i], axis=3) * np.ones((1,1,1,len(z_bins),1))

            phi_mu[:,:,0] += phi_deep[:,:,pname['mu+'].lidx:pname['mu+'].uidx]*self.dcosTH[i]
            phi_mu[:,:,0] += phi_deep[:,:,pname['mu+_l'].lidx:pname['mu+_l'].uidx]*self.dcosTH[i]
            phi_mu[:,:,0] += phi_deep[:,:,pname['mu+_r'].lidx:pname['mu+_r'].uidx]*self.dcosTH[i]

            phi_mu[:,:,1] += phi_deep[:,:,pname['mu-'].lidx:pname['mu-'].uidx]*self.dcosTH[i]
            phi_mu[:,:,1] += phi_deep[:,:,pname['mu-_l'].lidx:pname['mu-_l'].uidx]*self.dcosTH[i]
            phi_mu[:,:,1] += phi_deep[:,:,pname['mu-_r'].lidx:pname['mu-_r'].uidx]*self.dcosTH[i]
            #phi_mu += np.expand_dims(Phi_atm[:,:,i], axis=4) * np.ones((1,1,1,1,len(z_bins),1)) * dcosTH[i]

        return phi_mu * 2 * np.pi
    
    def MCEq_atmice(self, Phi0, interaction_model="SIBYLL-2.3c", density_model=('CORSIKA', ('USStd', None)), elev=None, solver='default'):
        
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
            elev = self.elev
            
        #Phi0
        #axis0 - Primary Model (Energy spectrum & Time dependence)
        #axis1 - Particle Species (proton, neutron)
        #axis2 - Primary Energy

        import mceq_config as config
        config.debug_level = 0
        config.h_obs = elev # elevation in (m) of Dome-C
        config.enable_default_tracking = False
        config.e_min = self.E_bins[0]*10.**0.1
        config.e_max = self.E_bins[-1]
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
        config.e_min = self.E_bins[0]*10.**0.1
        config.e_max = self.E_bins[-1]
        config.max_density = self.rho_ice
        config.dedx_material='ice'
        medium = 'ice'

        target = GeneralizedTarget(len_target=self.z_bins[-1]*100, env_density = self.rho_ice, env_name = 'ice')

        mceq_ice = MCEqRun(
            interaction_model=interaction_model,
            theta_deg = 0,
            density_model = target,
            medium=medium,
            primary_model = (pm.GaisserHonda, None),
        )

        #mceq_air.set_interaction_model(interaction_model)
        #config.h_obs = elev
        #mceq_air.set_density_model(density_model)
        #int_m_air, dec_m_air = mceq_air.int_m[p_cut][:,p_cut], mceq_air.dec_m[p_cut][:,p_cut]

        #mceq_ice.set_interaction_model(interaction_model)
        #int_m_ice, dec_m_ice = mceq_ice.int_m[p_cut][:,p_cut], mceq_ice.dec_m[p_cut][:,p_cut]

        pname = mceq_air.pman.pname2pref
        phi0 = np.zeros((np.shape(Phi0)[0], len(mceq_air._phi0)))
        phi0[:,pname['p+'].lidx:pname['p+'].uidx] = Phi0[:,0]
        phi0[:,pname['n0'].lidx:pname['n0'].uidx] = Phi0[:,1]
        #phi0 = phi0[:,p_cut]
        phi0 = np.moveaxis(phi0, 1, 0)
        #phi0 = phi0.reshape((len(p_cut),-1))
        phi0 = phi0.reshape((len(mceq_air._phi0),-1))

        phi_mu = np.zeros((np.shape(Phi0)[0],2,len(self.E),len(self.h_bins)))

        for i in tqdm(range(len(self.cosTH))):
            #dX_air, dz_air = get_mceq_path(mceq_air, cosTH[i])
            #phi_surf = mceq_integrate(phi0, dX_air, dz_air, int_m_air, dec_m_air)
            mceq_air.set_theta_deg(180*np.arccos(self.cosTH[i])/np.pi)
            mceq_air._phi0 = phi0
            
            if solver == 'default':
                self.solve_mceq(mceq_air)
            else:
                # 'numpy', 'cuda', or 'mkl'
                config.kernel_config = solver
                mceq_air.solve()

            #phi_deep = mceq_integrate(phi_surf, dh/cosTH[i]*100., dz/cosTH[i]*100., int_m_ice, dec_m_ice, grid=True)target = GeneralizedTarget(len_target=z_bins[-1]*100, env_density = rho_ice, env_name = 'ice')
            target = GeneralizedTarget(len_target=self.z_bins[-1]*100/self.cosTH[i], env_density = self.rho_ice, env_name = 'ice')
            target.mat_list = [[self.z_bins[j]*100/self.cosTH[i], self.z_bins[j+1]*100/self.cosTH[i], self.rho[j], 'ice'] for j in range(len(self.z_bins)-1)]
            target._update_variables()

            mceq_ice.set_density_model(target)
            mceq_ice._phi0 = mceq_air._solution
            
            if solver == 'default':
                self.solve_mceq(mceq_ice, int_grid=self.h_bins/self.cosTH[i]*100)
            else:
                # 'numpy', 'cuda', or 'mkl'
                config.kernel_config = solver
                mceq_ice.solve(int_grid=h_bins/cosTH[i]*100)
            
            phi_deep = np.moveaxis(mceq_ice.grid_sol.reshape(len(self.h_bins), len(phi0), np.shape(Phi0)[0]), (0,1), (-1,1))

            phi_mu[:,0] += phi_deep[:,pname['mu+'].lidx:pname['mu+'].uidx]*self.dcosTH[i]
            phi_mu[:,0] += phi_deep[:,pname['mu+_l'].lidx:pname['mu+_l'].uidx]*self.dcosTH[i]
            phi_mu[:,0] += phi_deep[:,pname['mu+_r'].lidx:pname['mu+_r'].uidx]*self.dcosTH[i]

            phi_mu[:,1] += phi_deep[:,pname['mu-'].lidx:pname['mu-'].uidx]*self.dcosTH[i]
            phi_mu[:,1] += phi_deep[:,pname['mu-_l'].lidx:pname['mu-_l'].uidx]*self.dcosTH[i]
            phi_mu[:,1] += phi_deep[:,pname['mu-_r'].lidx:pname['mu-_r'].uidx]*self.dcosTH[i]

        return phi_mu * 2 * np.pi
    
    def daemonflux_atm(self, Phi0):
        
        """
        
        
        Parameters
        --------------------
        Phi0 - 
        
        
        Returns
        --------------------
        Phi_atm - 
            
        """
        
        #Phi0
        #axis0 - Primary Model (Energy spectrum & Time dependence)
        #axis1 - Particle Species (proton, neutron)
        #axis2 - Primary Energy
    
        df_cut = (self.E <= 1e9)
        daemon_flux_pos = daemonflux.Flux(location='generic').flux(self.E[df_cut], np.arccos(self.cosTH)*180/np.pi, 'mu+')/np.reshape(self.E[df_cut]**3, (-1,1))
        daemon_flux_neg = daemonflux.Flux(location='generic').flux(self.E[df_cut], np.arccos(self.cosTH)*180/np.pi, 'mu-')/np.reshape(self.E[df_cut]**3, (-1,1))

        Phi_atm = np.zeros((np.shape(Phi0)[0], len(self.cosTH), 2, len(self.E)))
        pname = self.mceq.pman.pname2pref
        Phi_atm[:, :, :, df_cut] = np.reshape(daemon_flux_pos.T, (1,len(self.cosTH), 1, -1)) # positive muons
        Phi_atm[:, :, :, df_cut] = np.reshape(daemon_flux_neg.T, (1,len(self.cosTH), 1, -1)) # negative muons

        #Phi_atm
        #axis0 - Primary Model
        #axis1 - Zenith Angle
        #axis2 - Muon Charge (positive, negative)
        #axis3 - Muon Energy
        
        print(np.shape(Phi_atm))

        return Phi_atm
    
    def Dyonisius_prod(self, Phi_ice, sigma_E = None, E_sigma = None, alpha = None, N = None, f_tot = None):
        if sigma_E is None:
            sigma_E = self.sigma_E
        if E_sigma is None:
            E_sigma = self.E_sigma
        if alpha is None:
            alpha=self.alpha
        if N is None:
            N = self.N
        if f_tot is None:
            f_tot = self.f_tot
            
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

        #Phi_ice
        #axis0 - Atmospheric & Ice Models
        #axis1 - Primary Model
        #axis2 - Muon Charge (positive, negative)
        #axis3 - Muon Energy
        #axis4 - depth (top -> bottom)

        # NOTE: depth starts measured on the bin EDGES and is returned on the bin CENTERS
        # (This is because we need to take a derivative)

        sigma_0 = sigma_E / E_sigma**alpha

        P_neg = f_tot * -np.diff(np.sum(Phi_ice[:,:,1] * np.reshape(self.dE, (1,1,-1,1)), axis=2), axis=-1)/np.reshape(self.dh, (1,1,-1))

        P_fast = sigma_0 * N * np.sum((Phi_ice[:,:,:,:,1:]+Phi_ice[:,:,:,:,:-1])/2 * np.reshape(self.E**alpha * self.dE, (1,1,1,-1,1)), axis=(2,3))

        return np.moveaxis([P_fast, P_neg], 0, 2) /100 * 60 * 60 * 24 * 365.25 # g^-1, a^-1
    
    def diag_sum(self, A, off=None, axis1=-2, axis2=-1):
        # sums along the upper diagonals of two axes in an array
        # the new axis replaces axis1; axis2 is eliminated.
        if off is None:
            off = np.flip(range(np.shape(A)[axis2]))
        return np.moveaxis(np.array([np.trace(A, offset=i, axis1=axis1, axis2=axis2) for i in off]), 0, axis1 if axis1>=0 else axis1+1)
    
    def Basic_flow(self, P_14C, f_factors = None, f_t = None, lambd=None):
        if f_factors is None:
            f_factors = self.f_factors
        if f_t is None:
            f_t = np.ones(len(self.t[self.i_start:]))
        if lambd is None:
            lambd=self.lambd
            
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

        #P_14C - 14C Production Rate
        #axis0 - Production, Atmospheric & Ice Models
        #axis1 - Primary Model
        #axis2 - Production Mode (fast, neg)
        #axis3 - depth (top -> bottom)
        
        lambda_dt = np.reshape((1-lambd)**(self.t[self.i_start:][-1]-self.t[self.i_start:]) * self.dt[self.i_start:] * f_t, (1,1,1,-1))

        return self.diag_sum(np.expand_dims(np.sum(P_14C[:,:,:,self.i_start:]*np.reshape(f_factors, (1,1,-1,1)), axis=2), axis=-1) * lambda_dt)
    
    def load_profile(self, Phi0, file='balco_14co_const_models.fits', i=68):
        
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
        return np.reshape(hdus['CO14'].data[i][1:], (1,-1)) * np.ones((len(Phi0),1))

    def calculate(self, Phi0=None, atm=None, ice=None, atmice=None, prod=None, prodfull=None, flow=None, flowfull=None, output=False):
        if Phi0 is None:
            Phi0 = self.Phi0
        if atm is None:
            atm = self.atm
        if ice is None:
            ice = self.ice
        if atmice is None:
            atmice = self.atmice
        if prod is None:
            prod = self.prod
        if prodfull is None:
            prodfull = self.prodfull
        if flow is None:
            flow = self.flow
        if flowfull is None:
            flowfull = self.flowfull
            
        """
        
        
        Parameters
        ------------------------
        Phi0 - 
        
        atm - 
        
        ice - 
        
        atmice - 
        
        prod - 
        
        prodfull - 
        
        flow - 
        
        flowfull - 
        
        output - bool
            
            
        Returns
        -----------------------
        if output, returns:
        Phi_atm - 
            
        Phi_ice - 
            
        P_14C - 
            
        CO - 
            
        """
        
        # Add models that directly load in values
        
        self.Phi0 = Phi0
        #Phi0
        #axis0 - Primary Model (Energy spectrum & Time dependence)
        #axis1 - Particle Species (proton, neutron)
        #axis2 - Primary Energy

        self.Phi_atm = np.array([a[0](self.Phi0, *a[1]) for a in atm])
        #Phi_atm
        #axis0 - Atmospheric Model
        #axis1 - Primary Model
        #axis2 - Zenith Angle
        #axis3 - Muon Charge (positive, negative)
        #axis4 - Muon Energy

        print('Atmosphere complete')

        if len(self.Phi_atm)>0 and len(ice)>0:
            self.Phi_ice = np.array([i[0](self.Phi_atm, *i[1]) for i in ice])
            self.Phi_ice = np.reshape(self.Phi_ice, (-1, *np.shape(self.Phi_ice)[2:]))
            if len(atmice)>0:
                self.Phi_ice = np.concatenate((self.Phi_ice, [ai[0](self.Phi0, *ai[1]) for ai in atmice]))
        else:
            self.Phi_ice = np.array([ai[0](self.Phi0, *ai[1]) for ai in atmice])
        #Phi_ice
        #axis0 - Atmospheric & Ice Models
        #axis1 - Primary Model
        #axis2 - Muon Charge (positive, negative)
        #axis3 - Muon Energy
        #axis4 - depth (top -> bottom)

        print('Ice complete')

        if len(self.Phi_ice)>0 and len(prod)>0:
            self.P_14C = np.array([p[0](self.Phi_ice, *p[1]) for p in prod])
            self.P_14C = np.reshape(self.P_14C, (-1, *np.shape(self.P_14C)[2:]))
            if len(prodfull)>0:
                self.P_14C = np.concatenate((self.P_14C, [pf[0](self.Phi0, *pf[1]) for pf in prodfull]))
        else:
            self.P_14C = np.array([pf[0](self.Phi0, *pf[1]) for pf in prodfull])
        #P_14C
        #axis0 - Production, Atmospheric & Ice Models
        #axis1 - Primary Model
        #axis2 - Production Mode (fast, neg)
        #axis3 - depth (top -> bottom)

        print('Production Rates complete')

        if len(self.P_14C)>0 and len(flow)>0:
            self.CO = np.array([f[0](self.P_14C, *f[1]) for f in flow])
            self.CO = np.reshape(self.CO, (-1, *np.shape(self.CO)[2:]))
            if len(flowfull)>0:
                self.CO = np.concatenate((self.CO, [ff[0](self.Phi0, *ff[1]) for ff in flowfull]))
        else:
            self.CO = np.array([ff[0](self.Phi0, *ff[1]) for ff in flowfull])
        #CO
        #axis0 - Flow, Production, Atmospheric & Ice Models
        #axis1 - Primary Model
        #axis2 - depth (top -> bottom)

        print('14CO profile calculated')
        
        if output:
            return self.Phi_atm, self.Phi_ice, self.P_14C, self.CO # should have an option to return intermediate steps as well
        return
    
    def get_primary(self, primary_model = (pm.GlobalSplineFitBeta, None)):
        
        """
        
        
        Parameters
        ---------------------
        primary_model - 
            
        
        Returns
        ----------------------
        Phi0 - 
            
        """
        
        self.mceq.set_primary_model(*primary_model)
        pname = self.mceq.pman.pname2pref
        return np.array([self.mceq._phi0[pname['p+'].lidx:pname['p+'].uidx], self.mceq._phi0[pname['n0'].lidx:pname['n0'].uidx]])

    def load_primary(self, p_models=None, output=False): # primary CR intensities
        """
        
        
        Parameters
        --------------------
        p_models - 
            
        output - 
            
        
        Returns
        ------------------
        if output, returns:
        Phi0 - 
            
        """
        if p_models is None:
            p_models = self.p_models
        self.Phi0 = np.array([self.get_primary(p) for p in p_models])
        
        if output:
            return self.Phi0
        return
    
    def set_primary(self, Phi0, output=False):
        
        """
        
        
        Parameters
        ----------------------
        Phi0 - 
        
        output - 
        
        
        Returns
        --------------------
        if output, returns:
        Phi0 - 
            
        """
        
        self.Phi0 = Phi0
        if output:
            return self.Phi0
        return
    
    def set_primary_identity(self, output=False):
        
        """
        
        
        Parameters
        ------------------
        output - bool
        
        
        Returns
        ------------------
        if output, returns:
        Phi0 - 
            
        """
        
        # sets up primary flux matrix to test each energy of p+ and n0 individually
        
        self.Phi0 = np.zeros((240, 2, len(self.E)))
        for i in range(2):
            for j in range(len(self.E)):
                self.Phi0[len(self.E)*i+j, i, j] = 1.
        
        if output:
            return self.Phi0
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