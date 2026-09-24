#basic imports and ipython setup
import numpy as np

from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.io import loadmat

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
    
    if f_t is None:
        f_t = 1.
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
        C += (P_interp(parcel_depth[:,:,i]) - C*lambd)*dt
    
    C = np.moveaxis(C, -1,0)
    # C : final 14CO concentration at sample depth [molecules /g]
    #axis0 - flowpath
    #axis1 - Production Model
    #axis2 - Production Mode (fast, neg)
    #axis3 - sample depth
    
    return C