############################################################################################
############################         Imports & Setup         ###############################
############################################################################################

import matplotlib.pyplot as plt
import numpy as np
import scipy

from scipy.interpolate import interp1d

import os.path

from MCEq.core import MCEqRun
from MCEq import config
import crflux.models as pm

from tqdm import tqdm

import pandas as pd

import Functions_14CO as F

from time import time

############################################################################################
############################        Default Variables        ###############################
############################################################################################

# Site Variables
lambd = 1.216e-4 # 14C differential decay rate [yr^-1]
ice_eq_file = 'Real_vs_ice_eq_depth.csv'
elev = 3233 #Elevation above sea level [m]
pressure = 65800 # air pressure (Pa)

# Firn Variables
L = [1, 7e-4] # Ice Grain Leakage Rates (yr^-1)
Acc = 0.033 # Accumulation rate (m.w.e./yr), if None takes average accumulation rate up to close-off depth
rho_co = 0.833 # density at close off depth? idk
R = 8.314472 # gas constant (J/mol K)
g = 9.82 # gravity (m/s^2)
T = -54 # Temperature (Celsius)
M_air = 28.96e-3 # molar mass of air (kg/mol)
D_0eddy = 2.55e-5 # convective diffusivity const.
H_eddy = 3 #1/e depth of convective layer
age_file = 'DomeC_age_scale_Apr2023.csv' # sets depth-ice age conversion
tort_file = 'Firn_Model_Tortuosity_DomeC.csv' # inverse tortuosity & diff_m profiles
dt_A = 0.5 # Accumulation time resolution [years]
dt_D = 0.01 # Diffusion time resolution [years]
dz_D = 0.25 # Diffusion depth bin width [meters]
M_trace = 30e-3 # molar weight [kg/mol]
D_trace_scale = 1.25 * 0.9829926 # free air diffusivity relative to CO2, w/ “sqrt of ratio of reduced masses”

############################################################################################
############################        General Functions        ###############################
############################################################################################

def smooth_diff(f):
    return np.concatenate(( [(f[1]-f[0])], (f[2:]-f[:-2])/2, [(f[-1]-f[-2])] ), axis=0)

def get_bins(x):
    # returns centers and widths of bins
    x = np.array(x)
    if x.shape[0]==2: # if first axis is length 2, takes index 0 as lower bounds, index 1 as upper bounds
        return (x[0]+x[1])/2, x[1]-x[0]
    else:
        return (x[:-1]+x[1:])/2, np.abs(np.diff(x,axis=0))
    
def make_bins(x):
    # returns centers and widths of bins
    x = np.array(x)
    if x.shape[0]==2: # if first axis is length 2, takes index 0 as lower bounds, index 1 as upper bounds
        return (x[0]+x[1])/2, x[1]-x[0]
    else:
        return x, (x[:-1]+x[1:])/2, np.abs(np.diff(x,axis=0))

def interp_mat(out, grid_in, left=False, right=False): # interpolates from grid_in to out
    # input arrays must have broadcastable shape
    # grid's primary index should be along axis -1
    # left/right control outer bounds : True = 
    
    # slope from 0 to 1 as x passes from lower to upper bound
    slope = ((out[...,:-1 if out.shape[-1]>1 else None]-grid_in[...,:-1])/np.diff(grid_in,axis=-1)).clip(0,1)
    return np.concatenate(
        [(1-left)*(out[...,:1]>=grid_in[...,:1])-slope[...,:1],
         -np.diff(slope, axis=-1),
         slope[...,-1:]-(1-right)*(out[...,-1:]>grid_in[...,-1:])],
        axis=-1)

def bin_average_mat(out_up, out_low, in_low, din):
    S = (out_up-in_low).clip(0,din) - (out_low-in_low).clip(0,din)
    return S/np.sum(S, axis=-1, keepdims=True)

###############################################################################################
#################################                ##############################################
#################################  Generic Site  ##############################################
#################################                ##############################################
###############################################################################################

"""

"""


def load_ice_eq_depth(file, rho_ice=0.9239):
    # read ice-eq-depth file
    ice_eq_depth = pd.read_csv(file)
    real_z = np.array(ice_eq_depth['z']) # meters
    ice_eq_z = np.array(ice_eq_depth['ice_eq_z']) # meters (ice-eq) aka mass-depth / ice density

    # ice-eq-depth file assumes a rho_ice that isn't necessarily the same one we're using.
    # Check if this is the right density to use for this file
    h_z = ice_eq_z*rho_ice
    rho = smooth_diff(ice_eq_z)/smooth_diff(real_z) * rho_ice
    return real_z, h_z, rho
    
def load_densities(file):
    density_scale = pd.read_csv(file)
    real_z = np.array(density_scale['z']) # meters
    rho = np.array(density_scale['rho']) # 
    h_z = scipy.integrate.cumulative_trapezoid(rho, dx=np.diff(real_z)[0], initial=0) # assumes dz constant
    return real_z, h_z, rho

# Site Properties
class Site:
    def __init__(
        self,
        z_grid = None,
        h_grid = None,
        rho = None,
        elev = elev, #Elevation above sea level [m]
        pressure = pressure, # air pressure (Pa)
        z_samp = None,
        dz_samp = 0,
        CO = None,
        CO_err = None,
        Sigma = None,
        lambd = lambd,
    ):
        self.elev = elev
        self.pressure = pressure
        self.lambd = lambd
        
        if h_grid is None and rho is None:
            h_grid = ice_eq_file
        self.set_grids(z_grid, h_grid, rho)
        
        self.sample_bins(z_samp, dz_samp)
        
        self.load_sample(CO, CO_err, Sigma)
        
    def set_grids(self, z_grid=None, h_grid=None, rho=None):
        if z_grid is None:
            if isinstance(h_grid, str):
                h_file = h_grid
                z_grid, h_grid, rho = load_ice_eq_depth(h_file)
            elif isinstance(rho, str):
                rho_file = rho
                z_grid, h_grid, rho = load_densities(rho_file)
            else:
                print('No data for real depths')
        else:
            z_grid = np.array(z_grid)
            if isinstance(h_grid, str):
                h_file = h_grid
                z, h, rho1 = load_ice_eq_depth(h_file)
                h_grid = np.interp(z_grid, z, h)
                rho1 = np.interp(z_grid, z, rho1)
            elif not (h_grid is None):
                h_grid = np.array(h_grid)
                rho1 = smooth_diff(h_grid)/smooth_diff(z_grid)
                
            if isinstance(rho, str):
                rho_file = rho
                z, h, rho1 = load_densities(rho_file)
                h = np.interp(z_grid, z, h)
                rho = np.interp(z_grid, z, rho1)
            elif not (rho is None):
                rho = np.array(rho) * np.ones(len(z_grid))
                h = scipy.integrate.cumulative_trapezoid(rho, dx=np.diff(z_grid)[0], initial=0) # assumes dz constant
                
            if h_grid is None and not (rho is None):
                h_grid = h
            elif rho is None:
                rho = rho1
        self.z_grid = z_grid
        self.h_grid = h_grid
        self.rho = rho
        self.rho_ice = max(rho)
        
    def sample_bins(self, z_samp, dz_samp=0):
        if z_samp is None:
            self.z_samp_bins = None
            self.z_samp = None
            self.dz_samp = None
            self.h_samp_bins = None
            self.h_samp = None
            self.dh_samp = None
        z_samp = np.array(z_samp)
        if len(z_samp.shape)==2 and len(z_samp)==2:
            z_samp_bins = z_samp
            z_samp, dz_samp = get_bins(z_samp_bins)
        else:
            dz_samp = np.array(dz_samp) * np.ones(len(z_samp))
            z_samp_bins = z_samp[None,:] + dz_samp[None,:]*np.array([-0.5,0.5])[:,None]
        
        h_samp_bins = self.z_to_h(z_samp_bins)
        h_samp, dh_samp = get_bins(h_samp_bins)
        
        self.z_samp_bins = z_samp_bins
        self.z_samp = z_samp
        self.dz_samp = dz_samp
        self.h_samp_bins = h_samp_bins
        self.h_samp = h_samp
        self.dh_samp = dh_samp
        
    def load_sample(self, CO, CO_err=None, Sigma=None):
        if CO is None:
            self.CO_exp = None
            self.dCO_exp = None
            self.Sigma_exp = None
            return
        self.CO_exp = np.array(CO)
        self.dCO_exp = np.sqrt(np.diag(Sigma)) if CO_err is None and not (Sigma is None) else np.array(CO_err) if not (CO_err is None) else None
        self.Sigma_exp = np.diag(CO_err**2) if Sigma is None and not (CO_err is None) else np.array(Sigma) if not (Sigma is None) else None

    def h_to_z(self, h):
        return np.interp(h, self.h_grid, self.z_grid)
    def z_to_h(self, z):
        return np.interp(z, self.z_grid, self.h_grid)
    
    def accumulation_tensor(self, h_past, h_prod, t_int, const=False):
        dt = np.diff(t_int)[0]
        
        if const:
            A_ice = np.zeros(len(h_past), len(h_prod))
            for i in tqdm(range(len(t_int))):
                A_ice += interp_mat(h_past[:,i,None], h_prod[None,:]) * np.exp(self.lambd * t_int[i]) * dt
        else:
            A_ice = interp_mat(h_past[:,:,None], h_prod[None,None,:]) * np.exp(self.lambd * t_int[None,:,None]) * dt
        return A_ice
    
    
###############################################################################################
#################################                ##############################################
#################################    Firn Site   ##############################################
#################################                ##############################################
###############################################################################################

"""

"""


def efilter(x, lambd): # x cannot grow or decay faster than an exponential rate
    
    for i in range(1,len(x)):
        if x[i]/x[i-1] > np.exp(1/lambd[i-1]):
            x[i] = x[i-1]*np.exp(1/lambd[i-1])
        elif x[i]/x[i-1] < np.exp(-1/lambd[i-1]):
            x[i] = x[i-1]*np.exp(-1/lambd[i-1])
    return x

def cum_mat_prod(A,N): # return the first N powers [0:N) of matrix A
    # A can have any shape, as long as the last 2 axes are square
    A_cum = np.ones((N,*A.shape))
    A_cum[0] = np.identity(A.shape[-1])
    for i in tqdm(range(1,N)):
        A_cum[i] = A @ A_cum[i-1]
    return A_cum

# Gas Properties
class Gas:
    def __init__( # default: 14CO
        self,
        M_trace = M_trace, # molar weight [kg/mol]
        D_trace_scale = D_trace_scale, # free air diffusivity relative to CO2, w/ “sqrt of ratio of reduced masses”
        lambd = lambd, # differential decay rate [yr^-1]
    ):
        self.M_trace = M_trace
        self.D_trace_scale = D_trace_scale
        self.lambd = lambd
        self.L_trace = lambd/(60*60*24*365.25)

def load_age_scale(age_file):
    age_scale = pd.read_csv(age_file)
    ages = np.array(age_scale['ages']) # years
    depths_real = np.array(age_scale['depths_real']) # meters
    return depths_real, ages

# Firn Site Properties
class Firn_Site(Site):
    def __init__(
        self,
        L = L, # Ice Grain Leakage Rates (yr^-1)
        Acc = Acc, # Accumulation rate (m.w.e./yr), if None takes average accumulation rate up to close-off depth
        rho_co = rho_co, # density at close off depth? idk
        R = R, # gas constant (J/mol K)
        g = g, # gravity (m/s^2)
        T = T, # Temperature (Celsius)
        M_air = M_air, # molar mass of air (kg/mol)
        D_0eddy = D_0eddy, # convective diffusivity const.
        H_eddy = H_eddy, #1/e depth of convective layer
        gas = Gas(), # trace gas properties
        age_grid = age_file, # sets depth-ice age conversion
        tort_file = tort_file, # inverse tortuosity & diff_m profiles
        dt_A = dt_A,
        dt_D = dt_D,
        dz_D = dz_D,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.L = np.array(L)
        self.Acc = Acc
        if not (age_grid is None): # this will be set later
            self.Acc = None
        self.R = R
        self.g = g
        self.T = T + 273.15 # Temperature (Kelvin)
        self.M_air = M_air
        self.D_0eddy = D_0eddy
        self.H_eddy = H_eddy
        self.dt_A = dt_A
        self.dt_D = dt_D
        self.dz_D = dz_D
        
        self.set_age_scale(age_grid)
        
        self.set_accumulation_bins(dt_A)
        
        self.set_diffusion_bins(dz_D)
        
        self.calc_porosity(rho_co)
        
        self.calc_bubble_pressure()
        
        self.calc_unit_convert()
        
        self.calc_trapping_rates()
        
        self.load_tortuosity(tort_file)
        
        self.set_gas(gas)
        
        self.Crank_abc()
        
    def set_age_scale(self, age_grid):
        if isinstance(age_grid, str):
            age_file = age_grid
            z,age = load_age_scale(age_file)
            age_grid = np.interp(self.z_grid, z, age)
        else:
            age_grid = np.array(age_grid)
        self.age_grid = age_grid
        
    def set_accumulation_bins(self, dt_A):
        t_A_bins = np.arange(int(self.age_grid[-1]/dt_A) + 1) * dt_A
        t_A_bins = t_A_bins - t_A_bins[-1]
        t_A = get_bins(t_A_bins)[0]
        
        age_A_bins = np.arange(int(self.age_grid[-1]/dt_A) + 1) * dt_A
        age_A, dage_A = get_bins(age_A_bins)
        #age_A = age_A_bins[:-1]

        z_A_bins = self.age_to_z(age_A_bins)
        z_A, dz_A = get_bins(z_A_bins)
        #z_A = z_A_bins[:-1]

        h_A_bins = self.age_to_h(age_A_bins)
        h_A, dh_A = get_bins(h_A_bins)
        #h_A = h_A_bins[:-1]
        
        self.t_A_bins = t_A_bins
        self.t_A = t_A
        self.dt_A = dt_A
        self.z_A_bins = z_A_bins
        self.z_A = z_A
        self.dz_A = dz_A
        self.h_A_bins = h_A_bins
        self.h_A = h_A
        self.dh_A = dh_A
        self.age_A_bins = age_A_bins
        self.age_A = age_A
        self.dage_A = dage_A
        
    def set_diffusion_bins(self, dz_D):
        z_D_bins = np.arange(int(self.z_grid[-1]/dz_D)+1)*dz_D
        z_D = z_D_bins
        #z_D = z_D_bins[:-1]

        h_D_bins = self.z_to_h(z_D_bins)
        h_D = h_D_bins
        dh_D = smooth_diff(h_D_bins)
        #h_D = h_D_bins[:-1]

        age_D_bins = self.z_to_age(z_D_bins)
        age_D = age_D_bins
        dage_D = smooth_diff(age_D_bins)
        #age_D = age_D_bins[:-1]
        
        self.z_D_bins = z_D_bins
        self.z_D = z_D
        self.dz_D = dz_D
        self.h_D_bins = h_D_bins
        self.h_D = h_D
        self.dh_D = dh_D
        self.age_D_bins = age_D_bins
        self.age_D = age_D
        self.dage_D = dage_D
    
    def calc_porosity(self, rho_co):
        rho_ice = max(0.9165 - (self.T-273.15)*1.4438e-4 - (self.T-273.15)**2 * 1.5175e-7, max(self.rho)+1e-6) # Schwander et al.

        s = 1 - self.rho/rho_ice # porosity of ice by depth
        s_closed = (0.37 * s * (s/(1-rho_co/rho_ice))**(-7.6)).clip(max=s) # closed porosity
        s_open = s - s_closed # open porosity

        teller_co = np.argmax(s_closed) # depth index of maximum closed porosity (close off depth??)
        
        if self.Acc is None:
            self.Acc = self.h_grid[teller_co]/self.age_grid[teller_co]
        
        self.rho_co = rho_co
        self.rho_ice = rho_ice
        self.s = s
        self.s_closed = s_closed
        self.s_open = s_open
        self.teller_co = teller_co
        
    def calc_bubble_pressure(self):
        dz_grid = smooth_diff(self.z_grid)

        #dscl = smooth_diff(s_closed)/dz # ds_closed / dz
        dscl = np.diff(self.s_closed, prepend=self.s_closed[0])/dz_grid # ds_closed / dz

        C_air = np.exp(self.M_air * self.g * self.z_grid / (self.R * self.T) )
        v_ice = self.Acc / self.rho # downwards ice velocity

        #strain = smooth_diff( np.log(v_ice))/dz # d log(v_ice) / dz
        #strainsum = scipy.integrate.cumulative_trapezoid(strain, dx=self.dz, initial=0)
        strainsum = np.log(v_ice)
        i = np.arange(len(self.z_grid))
        bubble_pres = np.sum((dscl * C_air * self.s * dz_grid)[None,:] / (1+strainsum[:,None]-strainsum[None,:]) * (i[None,:]<=i[:,None]), axis=-1)/(self.s*(self.s_closed-self.s_closed[0]).clip(min=1e-20))
        # Divide by zero error at the first index, so we set that to 1
        bubble_pres[0] = 1 # what are the units here??
        # Answer: it's relative to surface pressure

        # After teller_co:
        bubble_pres[self.teller_co+1:] = bubble_pres[self.teller_co] * self.s[self.teller_co] / self.s[self.teller_co+1:] * v_ice[self.teller_co] / v_ice[self.teller_co+1:]
        # bubble pres is directly inverse to s and v_ice below close-off depth

        flux = v_ice[self.teller_co+1] * bubble_pres[self.teller_co+1] * self.s_closed[self.teller_co+1]

        velocity = np.minimum(v_ice, (flux + 1e-10 - v_ice * bubble_pres * self.s_closed) / ((self.s_open + 1e-10) * C_air) )

        #never used:
        #air_content = 1000 * bubble_pres[teller_co+1] * s_closed[teller_co+1] * pressure / 101325 * 273.15 / T / rho[teller_co+1]
        
        # Calcluate trapping rates. Both formulas give the same values. (if
        # everything is implemented correctly. CHECK THIS)
        
        #icevel = Site.Acc*Site.rho_ice/Site.rho
        # trapping = -1*(dz./icevel).*smoothdiff(velocity.*s_open.*C_air,dz);
        #trapping_t = (1/icevel)*Acc*rho_ice*smoothdiff(s_closed*bubble_pres/rho,dz)
        trapping_t = self.rho*smooth_diff(self.s_closed*bubble_pres/self.rho)/smooth_diff(self.z_grid) # Acc*rho_ice/(Acc*rho_ice/rho) = rho
        #trapping_z = (dz./icevel).*Acc.*rho_ice.*smoothdiff(s_closed.*bubble_pres./rho,dz);
        
        self.v_ice = v_ice
        self.flux = flux
        self.bubble_pres = bubble_pres
        self.velocity = velocity
        self.trapping_t = trapping_t
        
    def calc_unit_convert(self):
        gram_to_ppm = 1e18*self.R*self.T / (6.022140857e23*self.pressure*self.s_open/self.rho*np.exp(self.M_air*self.g*self.z_grid/(self.R*self.T))*1e-6).clip(min=1e-20)
        gram_to_ppm[self.teller_co:] = 0
        ppm_to_gram = (6.022140857e23 * self.pressure * self.s_open / self.rho * np.exp(self.M_air*self.g*self.z_grid /(self.R*self.T))*1e-6)/(1e18*self.R*self.T)
        ppm_to_gram_cl = (6.022140857e23 * self.pressure * self.s_closed / self.rho * self.bubble_pres * 1e-6)/(1e18*self.R*self.T)
        
        self.gram_to_ppm = gram_to_ppm
        self.ppm_to_gram = ppm_to_gram
        self.ppm_to_gram_cl = ppm_to_gram_cl
        
    def calc_trapping_rates(self):
        trapping_t = self.rho*smooth_diff(self.s_closed*self.bubble_pres/self.rho)/smooth_diff(self.z_grid) # Acc*rho_ice/v_ice = Acc*rho_ice/(Acc*rho_ice/rho) = rho
        self.trapping_t = trapping_t
    
    def load_tortuosity(self, tort_file):
        self.tort_file = tort_file
        
        Profiles = np.array(pd.read_csv(tort_file))
        self.InvTort = np.interp(self.z_grid, Profiles[:,0], Profiles[:,1]) # inverse Tortuosity D(z) = D^0 * Inv tort
        self.Diff_m = np.interp(self.z_grid, Profiles[:,0], Profiles[:,2]) # dispersive mixing in lock-in zone

    def set_gas(self, gas):
        self.gas = gas
        self.M_gas = gas.M_trace
        self.D_ref_CO2 = 5.75e-10 * self.T**1.81 * 101325 / self.pressure # CO2 diffusivity
        self.D_gas = self.D_ref_CO2 * gas.D_trace_scale
        self.lambd = gas.lambd
        self.L_gas = gas.L_trace
    
    def z_to_age(self, z):
        return np.interp(z, self.z_grid, self.age_grid)
    def h_to_age(self, h):
        return np.interp(h, self.h_grid, self.age_grid)
    def age_to_z(self, age):
        return np.interp(age, self.age_grid, self.z_grid)
    def age_to_h(self, age):
        return np.interp(age, self.age_grid, self.h_grid)
    
    
    def Crank_abc(self):
        
        Q = np.arange(len(self.z_D))[self.z_D<=self.z_grid[self.teller_co+20]][-1]
        self.Q = Q

        Diff = self.InvTort * self.D_gas

        # Eddy diffusivity part:
        Diff_e = efilter(self.D_0eddy*np.exp(-self.z_grid/self.H_eddy),0.5/np.diff(self.z_grid))*(self.z_grid<55) # Convection following Kawamura et al
        Diff_e = np.maximum(Diff_e, self.Diff_m)    # Add dispersion in the LIZ
        
        C_air = np.exp(self.M_air * self.g * self.z_grid / (self.R * self.T) )

        # Take the derivatives
        porecl = smooth_diff(Diff*C_air*(self.s_open+1e-9))/smooth_diff(self.z_grid)/((self.s_open+1e-9)*C_air)#.clip(min=1e-20)
        porecl_e = smooth_diff(Diff_e*C_air*(self.s_open+1e-9))/smooth_diff(self.z_grid)/((self.s_open+1e-9)*C_air)#.clip(min=1e-20)

        porecl[self.teller_co]   = porecl[self.teller_co+1]
        porecl_e[self.teller_co] = porecl_e[self.teller_co+1]

        # c_t = alpha*c_zz + beta*c_z + gamma*c

        alpha =  (self.dt_D*60*60*24*365.25/2/self.dz_D**2) * np.interp(self.z_D, self.z_grid, Diff+Diff_e)
        beta  =  (self.dt_D*60*60*24*365.25/4/self.dz_D) * np.interp(self.z_D, self.z_grid, Diff*(self.M_air-self.gas.M_trace)*self.g/(self.R*self.T) + (Diff_e)*(self.M_air)*self.g/(self.R*self.T) + porecl + porecl_e -self.velocity/(60*60*24*365.25))
        gamma =  (self.dt_D*60*60*24*365.25/2) * np.interp(self.z_D, self.z_grid, porecl*(self.M_air-self.gas.M_trace)*self.g/(self.R*self.T) +  -self.gas.L_trace)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        A_for = scipy.sparse.diags_array(
            [
                np.concatenate((alpha[1:Q-1] - beta[1:Q-1], [0])),
                np.concatenate(([1], 1-2*alpha[1:Q-1]+gamma[1:Q-1], [0])),
                np.concatenate(([0], alpha[1:Q-1] + beta[1:Q-1] )),
            ],
            offsets=np.arange(-1,2), shape=(Q,Q)
        ).toarray()

        A_back = scipy.sparse.diags_array(
            [
                np.concatenate((-1*(alpha[1:Q-1] - beta[1:Q-1]), [-1])),
                np.concatenate(([1], 1+2*alpha[1:Q-1]-gamma[1:Q-1], [1])),
                np.concatenate(([0], -1*(alpha[1:Q-1] + beta[1:Q-1]) )),
            ],
            offsets=np.arange(-1,2), shape=(Q,Q)
        ).toarray()

        A_inv = np.linalg.inv(A_back)

        Site.D = A_inv @ A_for
        Site.D[0] = 0
        
        return
    
    def COS_Production(Site, h, scaling_site = [16.95], P0_tuning = np.ones(3)):
        Pn0_SLHL = 12.76 # Borchers ‘16
        QtzToIce = 60.08 / 2 / 18.02 # conversion for O-atom density difference between quartz and ice

        Pn = Pn0_SLHL * QtzToIce * scaling_site[0] * P0_tuning[0] * np.exp(-100 * h * Site.rho_ice/150)

        Balco_mu_neg, Balco_mu_fast = F.Balco_P_mu_total(100*h*Site.rho_ice, Site.pressure/100)

        Pmu = Balco_mu_neg * P0_tuning[1]
        Pmuf = Balco_mu_fast * P0_tuning[2]

        return Pn, Pmu, Pmuf


    def Calc_14C_ice(Site,
                     L = None, # Leakage rates by Reservoir
                     P = None, # input production rates
                     h_p = None, # input production rates mass depth grid
                     op_cl_prob_ratio = 1, # open-closed leak probability ratio
                    ):
        if L is None:
            L = Site.L
        else:
            L = np.array(L)
            
        if h_p is None:
            h_p = Site.h_A
        else:
            h_p = np.array(h_p)
            
        if P is None:
            P = np.array(Site.COS_Production(h=h_p)).T
        else:
            P = np.array(P)
            if len(P.shape)==1:
                P = P[:,None]
        
        if h_p is None:
            P_A = P
        else:
            P_A = interp1d(h_p, P, axis=0, assume_sorted=True, bounds_error=False, fill_value=(P[0],P[-1]))(Site.h_A)

        # Run the calculation
        
        # integrate total leakage over time
        L_safe = L * (L!=1) #reduce L=100% to 0% to avoid nan when integrating
        Lrate = -np.log(1-L_safe)
        # cumsum to integrate : new 14C production - loss to decay & leakage
        Ctemp = np.exp(-(Site.lambd+Lrate)[None,None,:]*(Site.age_A+Site.dt_A)[:,None,None]) * np.cumsum(P_A[:,:,None] * np.exp((Site.lambd+Lrate)[None,None,:]*Site.age_A[:,None,None]) * Site.dt_A, axis=0) * (L!=1)[None,None,:]
        #Ctemp[~np.isfinite(Ctemp)] = 0
        # extra step of accumulation (so min. accumulation time is dt_A)
        Ctemp[1:] = Ctemp[:-1]
        Ctemp[0] = 0
        #axis0 - Depth (=Ice Age)
        #axis1 - Production Mode (n0, mu-, muf)
        #axis2 - Reservoir

        #14C Concentration in ice grains
        grains = Ctemp

        o = np.interp(Site.z_A, Site.z_grid, Site.s_open)
        c = np.interp(Site.z_A, Site.z_grid, Site.s_closed)

        l_o = o * op_cl_prob_ratio/(o * op_cl_prob_ratio + c)
        l_c = c/(o * op_cl_prob_ratio + c)

        #14C leaked into open porosity per timestep
        leak_open = (Ctemp/Site.dt_A + P_A[:,:,None]) * (1-(1-L)**Site.dt_A)[None,None,:] * l_o[:,None,None]
        #14C leaked into closed porosity per timestep
        leak_closed = (Ctemp/Site.dt_A + P_A[:,:,None]) * (1-(1-L)**Site.dt_A)[None,None,:] * l_c[:,None,None]

        #14C stored in open porosity (no bubble tracking)
        cumuleak_open = np.exp(-Site.lambd*Site.age_A)[:,None,None] * np.cumsum(leak_open * np.exp(Site.lambd*Site.age_A)[:,None,None] * Site.dt_A, axis=0)
        # extra step of accumulation (so min. accumulation time is dt_A)
        cumuleak_open[1:] = cumuleak_open[:-1]
        cumuleak_open[0] = 0
        
        #14C stored in closed porosity (no bubble tracking)
        cumuleak_closed = np.exp(-Site.lambd*Site.age_A)[:,None,None] * np.cumsum(leak_closed * np.exp(Site.lambd*Site.age_A)[:,None,None] * Site.dt_A, axis=0)
        # extra step of accumulation (so min. accumulation time is dt_A)
        cumuleak_closed[1:] = cumuleak_closed[:-1]
        cumuleak_closed[0] = 0
        #axis0 - Depth (=Ice Age)
        #axis1 - Production Mode (n0, mu-, muf)
        #axis2 - Reservoir

        grains_z = interp1d([0,*Site.z_A], np.concatenate((np.zeros((1,*grains.shape[1:])),grains), axis=0), axis=0, assume_sorted=True, bounds_error=False, fill_value=(0,grains[-1]))(Site.z_grid)
        leak_open_z = interp1d([0,*Site.z_A], np.concatenate((np.zeros((1,*grains.shape[1:])),leak_open), axis=0), axis=0, assume_sorted=True, bounds_error=False, fill_value=(0,leak_open[-1]))(Site.z_grid)
        leak_closed_z = interp1d([0,*Site.z_A], np.concatenate((np.zeros((1,*grains.shape[1:])),leak_closed), axis=0), axis=0, assume_sorted=True, bounds_error=False, fill_value=(0,leak_closed[-1]))(Site.z_grid)
        cumuleak_open_z = interp1d([0,*Site.z_A], np.concatenate((np.zeros((1,*grains.shape[1:])),cumuleak_open), axis=0), axis=0, assume_sorted=True, bounds_error=False, fill_value=(0,cumuleak_open[-1]))(Site.z_grid)
        cumuleak_closed_z = interp1d([0,*Site.z_A], np.concatenate((np.zeros((1,*grains.shape[1:])),cumuleak_closed), axis=0), axis=0, assume_sorted=True, bounds_error=False, fill_value=(0,cumuleak_closed[-1]))(Site.z_grid)

        return grains_z, leak_open_z, cumuleak_closed_z

    def CrankNic(Site, C14_add, analytic=True, t_end=3000):
        # this solves the diffusion equation c_t = alpha*c_xx + beta*c_x + gamma*c 
        # using finite-differences in space and Crank-Nicolson time-stepping.  
        # t_0 is the initial time, t_end is the final time, N is the number of mesh-points, 
        # and M is the number of time steps.

        Q = Site.Q

        if analytic:
            c_temp = np.linalg.inv(np.identity(Q) - Site.D) @ C14_add[:Q]

        else:
            # Initialise the firn with constant mixing ratio throughout
            c_temp = np.zeros((Q,*C14_add.shape[1:]))
            c_temp[:,0] = C14_add[0,0]

            for j in tqdm(range(int(t_end/Site.dt_D))):
                c_temp = Site.D @ c_temp + C14_add[:Q]

        c_gases = np.zeros(C14_add.shape)
        c_gases[:Q] = c_temp
        c_gases[Q:] = c_temp[-1]

        return c_gases


    def Bubbletracking_all(Site, c_open):
        # Calculate the concentrations in the closed bubbles for all the time
        # steps using a 2-D interpolation
        C_air = np.exp(Site.M_air*Site.g*Site.z_grid/(Site.R*Site.T))

        # interpolate past ice parcel depths, with negative depths for unformed ice.
        #M_age - ice parcel age at time t
        M_age = Site.age_grid[:,None]+Site.t_A[None,:]
        #axis0 - ice parcel final depth
        #axis1 - time

        M_position = np.interp(M_age, Site.age_grid, Site.z_grid, left=-10)
        M_position[:,-1] = Site.z_grid
        #M_position - ice parcel depth history
        #axis0 - ice parcel final depth
        #axis1 - time

        #M_position[M_position < 0] = -10
        M_c_ones = (1*(M_position >=0))[:,:,None] # = 1 if ice has formed, else 0

        # interpolate gas trapping rates (open -> closed porosity) at past ice parcel depths
        #M_trapping = np.interp(M_position, [-10,-Site.dz,*Site.z], [0,trapping_t[0],*trapping_t])
        M_trapping = np.interp(M_position, Site.z_grid, Site.trapping_t, left=0)[:,:,None]

        print('interpolating...')
        c_o = c_open.reshape((len(c_open),-1))
        M_c_open = np.moveaxis([np.interp(M_position, Site.z_D, c, left=0) for c in tqdm(c_o.T)], 0, -1) * np.exp(Site.t_A*Site.lambd)[None,:,None]
        #M_c_open[:,-1] = c_o
        # axis0 - ice parcel final depth
        # axis1 - time
        # axis2 - Extra dimesnions of c_open ...

        # integrate trapped 14C from open porosity over time
        print('integrating...')
        c_closed = np.sum(M_c_open * M_trapping, axis=1) / np.sum(M_c_ones * M_trapping, axis=1)

        return c_open, c_closed.reshape((-1,*c_open.shape[1:]))



    def Calc_profiles(Site, C14_leak_open, gas_history = None, analytic=True):

        if gas_history is None:
            gas_history = np.array([0])

        t = time()

        # Calculate the nr of air molecules per gram of ice, to convert the leakage rate to ppm
        # To keep numbers reasonable, we use units of 1E-12 ppm, as was done in Buizert et al. 2012
        # Vas note: increased coefficients at start of expression to
        # 1e18, as ppm*10^-12 requires. Also added division by rho in
        # the denominators - this is needed to go from grams to cm^3
        gram_to_ppm = Site.gram_to_ppm

        C14_add = np.concatenate((np.zeros((len(Site.z_D),1)),
                                  Site.dt_D * interp1d(Site.z_grid,
                                      C14_leak_open.reshape((len(C14_leak_open),-1)) * Site.gram_to_ppm[:,None],
                                      axis=0,
                                  )(Site.z_D)
                                 ), axis=1)
        C14_add[0,0] = gas_history[0]

        c_gases = Site.CrankNic(C14_add, analytic)

        print(time()-t)
        t=time()

        # now calculate the closed pores as well;
        c_open, c_closed = Site.Bubbletracking_all(c_gases)

        C14_open = c_open[:,1:].reshape((len(Site.z_D),*C14_leak_open.shape[1:])) * np.interp(Site.z_D, Site.z_grid, Site.ppm_to_gram)[:,None,None]
        C14_open[np.isnan(C14_open)] = 0
        C14_closed = c_closed[:,1:].reshape((-1,*C14_leak_open.shape[1:])) * Site.ppm_to_gram_cl[:,None,None]

        C14_open_atm = c_open[:,0] * np.interp(Site.z_D, Site.z_grid, Site.ppm_to_gram)
        C14_open_atm[np.isnan(C14_open_atm)] = 0
        C14_closed_atm = c_closed[:,0] * Site.ppm_to_gram_cl

        print('  Calculated open and closed profiles for gas')
        print(time()-t)
        return C14_open, C14_closed, C14_open_atm, C14_closed_atm, c_gases

    def firn_tensor(self, h_f, h_prod, L, const=False, separate=False):
        
        if const:
            print('Calculating Direct Accumulation...')
            
            A_grains = np.zeros((len(h_f), len(h_prod), len(L)))
            A_closed = np.zeros((len(h_f), len(h_prod), len(L)))
            
            for i in tqdm(range(len(t_int))):
                x = interp_mat(h_past[:,i,None], Prop_DC.h[None,:])[:,:,None] * np.exp(lambd * t_int[i]) * DC.dt_A
                A_grains += x * (1-L[None,None,:])**(-t_int[i]+DC.dt_A)
                A_closed += x * f_c_int[:,i,None,:]
            
            print('Calculating Leakage Rates...')
            Leak_mat = np.zeros((DC.Q, len(Prop_DC.h), len(L)))

            t_int_leak = np.arange(len(DC.t_A[t_D_cut]))*DC.dt_A
            t_int_leak = t_int_leak - t_int_leak[-1]

            for i in tqdm(range(len(t_int_leak))):
                Leak_mat += interp_mat(h_D_past[:,i,None], Prop_DC.h[None,:])[:,:,None] * np.exp(lambd * t_int_leak[i]) * (1-L[None,None,:])**(-t_int_leak[i]) * (1-(1-L[None,None,:])**DC.dt_A)
            Leak_mat *= (f_o_D * gram_to_ppm)[:,None,None]
            
            print('Calculating Diffusion...')
            
            Diff_mat = np.linalg.inv(np.identity(len(DC.D))-DC.D) * DC.dt_D
            
            print('Calculating Trapping Rates...')
            
            Trap_mat = np.zeros((len(z_f), DC.Q))

            for i in tqdm(range(len(t_int))):
                Trap_mat += interp_mat(z_past[:,i,None], DC.z_D[None,:DC.Q]) * np.exp(lambd * t_int[i]) * trap_frac[:,i,None] * DC.dt_A * ppm_to_gram_cl[:,None]

            A_trap = np.einsum('fd, dpr -> fpr', Trap_mat @ Diff_mat, Leak_mat, optimize=True)

        else:
            print('Calculating Direct Accumulation...')
            
            A_grains = interp_mat(h_past[:,:,None], Prop_DC.h[None,None,:])[:,:,:,None] * np.exp(lambd * t_int[None,:,None,None]) * DC.dt_A * (1-L[None,None,None,:])**(-t_int[None,:,None,None]+DC.dt_A)
            A_closed = interp_mat(h_past[:,:,None], Prop_DC.h[None,None,:])[:,:,:,None] * np.exp(lambd * t_int[None,:,None,None]) * DC.dt_A * f_c_int[:,:,None,:]
            
            print('Calculating Leakage Rates...')
            
            print('Calculating Diffusion...')
            
            print('Calculating Trapping Rates...')
            
        
        if separate:
            return A_grains, A_closed, A_trap
        else:
            return A_grains + A_closed + A_trap
    
###############################################################################################
#################################                ##############################################
#################################   Propagator   ##############################################
#################################                ##############################################
###############################################################################################

"""

"""


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
        
        return np.concatenate([self.run_solo(Prop, p, i) for i,p in enumerate(self.params)])
    
    def run_solo(self, Prop, p, i=0):
        print(self.names[i])
        
        if type(p) is tuple:
            return self.function(Prop, *p)
        elif type(p) is dict:
            return self.function(Prop, **p)
        elif p is None:
            return self.function(Prop)
        else:
            return self.function(Prop, p)

class Propagator:
    def __init__(
        self,
        site,
        h = None,
        cosTH_bins = 10,
        logE_min = -1,
        logE_mu_max = 7.5,
        logE_max = 11,
        dlogE = 0.1,
    ):
        self.Site = site
        
        self.set_propagation_depth(h)
        
        self.set_zenith_bins(cosTH_bins)
        
        self.set_energy_bins(logE_min, logE_mu_max, logE_max, dlogE)
        
        # parameters for Gaisser-Stanev Energy loss
        self.a = 0.227 #energy loss due to ionization (GeV cm^2/hg)
        self.b = 2.44e-4 #sum of fractional radiation losses (cm^2/hg)
        self.b_ice = 2.04e-4 #ice value
        
        self.set_cross_sections()
        
        self.calc_H()
        
        # Muon mass
        self.mu_mass = 1.056583745e-1 # GeV
        
        self.stages = ['primary',
                       'atm',
                       'ice',
                       'prod'
                      ]
        
        self.Phi = dict()
        self.clear_Phi()
        
        self.models = dict()
        self.model_names = dict()
        self.set_models()
    
    def set_propagation_depth(self, h = None):
        if h is None:
            if hasattr(self, 'h'):
                h = self.h
            else:
                dh = 1.
                h = np.arange(int(self.Site.h_grid[-1]/dh)+1)*dh
        h = np.array(h)
            
        self.h, self.dh = h, smooth_diff(h)
        self.z = self.Site.h_to_z(h)
        self.dz = smooth_diff(self.z)
        
        self.rho = self.dh/self.dz
        
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
    def set_energy_bins(self, logE_min = -1, logE_mu_max = 7.5, logE_max = 11, dlogE = 0.1):
        
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
    def setup_mceq(self):
        from MCEq import config

        interaction_model = "SIBYLL-2.3d"

        density_model, density_name = ('CORSIKA', ('USStd', None)), 'CORSIKA_USStd'
        
        config.h_obs = self.Site.elev
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
        
        #f_star, df_star = 4.4e-3/f_C/f_D, 2.6e-3/f_C/f_D #<- where did I get this from?? This is wrong!
        self.f_tot = f_tot
        
        return
    
    # Sets up Balco elevation adjustment factors, starting from pressure
    def calc_H(self):

        # figure the difference in atmospheric depth from sea level in g/cm2
        self.H = (1013.25 - self.Site.pressure/100)*1.019716 # the 1.019716 number is basically just 1/g accounting for needed unit conversions
        
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
        
        self.Phi[self.stages[2]] = np.zeros((0,2,len(self.E_mu),len(self.h)))
        #Phi_ice
        #axis0 - Underice Model
        #axis1 - Muon Charge (positive, negative)
        #axis2 - Muon Energy (E_mu)
        #axis3 - depth bin EDGES (top -> bottom)
        
        self.Phi[self.stages[3]] = np.zeros((0,2,len(self.h)))
        #P_14C
        #axis0 - Production Model
        #axis1 - Production Mode (fast, neg)
        #axis2 - depth (top -> bottom)
        
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
                    #self.Phi[s] = np.concatenate([m.run(self) for m in models[s]], axis = 0)
                    self.Phi[s] = self.Phi[s][0:0]
                    self.model_names[s] = []
                #else:
                    #self.Phi[s] =  np.concatenate([self.Phi[s], *[m.run(self) for m in models[s]]], axis = 0)
                #self.model_names[s] += ['{}{}'.format(i,n) for m in models[s] for n in m.names for i in self.model_names.get(m.input,[''])]
                for m in models[s]: # for loop method allows models to draw on output of previous calculations in same stage
                    self.Phi[s] = np.concatenate([self.Phi[s], m.run(self)], axis=0)
                    self.model_names[s] += ['{}{}'.format(i,n) for n in m.names for i in self.model_names.get(m.input,[''])]
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
                         tag='', # Label for Production Rates (usually location, such as DomeC)
                         overwrite=False, # Overwrite existing files?
                        ):
        if tag != '':
            tag = '_'+tag
        
        P_fast = dict()
        P_neg = dict()
        for i,n in enumerate(self.model_names['prod']):
            P_fast[n] = self.Phi['prod'][i,0]
            P_neg[n] = self.Phi['prod'][i,1]
            
        df_fast = pd.DataFrame(P_fast)
        df_neg = pd.DataFrame(P_neg)
        
        i=0
        done=False
        while i<100 and not done:
            num = ' ({})'.format(i) if i>0 else ''
            fast_file = 'Production Rates/P_fast{}_{}m{}.csv'.format(tag, self.Site.elev, num)
            neg_file = 'Production Rates/P_neg{}_{}m{}.csv'.format(tag, self.Site.elev, num)
            if not (os.path.exists(fast_file) or os.path.exists(neg_file)) or overwrite:
                print('Saving fast production rates...')
                df_fast.to_csv(fast_file, index=False)
                print('Saved to:  {}'.format(fast_file))
                print()
                print('Saving neg production rates...')
                df_neg.to_csv(neg_file, index=False)
                print('Saved to:  {}'.format(neg_file))
                done=True
            i+=1
        if not done:
            print('Failed to save.')
        
        return
    
    def load_prod_from_csv(self,
                          tag='',
                          i=0,
                          output = True):
        if tag != '':
            tag = '_'+tag
        if i<0:
            i=0
        i = int(i)
            
        num = ' ({})'.format(i) if i>0 else ''
        fast_file = 'Production Rates/P_fast{}_{}m{}.csv'.format(tag, self.Site.elev, num)
        neg_file = 'Production Rates/P_neg{}_{}m{}.csv'.format(tag, self.Site.elev, num)
        
        if os.path.exists(fast_file) and os.path.exists(neg_file):
            print('Loading fast production rates...')
            df_Pfast_DC = pd.read_csv(fast_file)
            print('Loading neg production rates...')
            df_Pneg_DC = pd.read_csv(neg_file)
            
            self.Phi['prod'] = np.swapaxes([df_Pfast_DC.T, df_Pneg_DC.T], 0, 1)
            self.model_names['prod'] = list(df_Pfast_DC.columns)
            print('Production rates loaded.')
            if output:
                return np.copy(self.Phi['prod']), list(self.model_names['prod'])
        else:
            print('Files do not exist.')
        
        return
    
    # def save_CO_to_csv
