# In-Situ-14C-Production
Calculates in-situ 14CO produced by cosmic ray muons in ice, starting from primary spectra at the top of the atmosphere

## Installation Guide

Setup python

Install required libraries (below)

Run the Notebook

### Required Python Libraries

numpy

scipy

matplotlib

tqdm

pandas

MCEq

crflux

daemonflux


## File Guide

### Python Files

**FirnModel.py** - 

**Functions_14CO.py** - Defines functions used to propagate primary cosmic rays to 14CO concentrations

**InSitu14CO.py** - Defines the Propagator class, which manages the calculations involved in turning primary CR spectra into 14CO profiles

### Notebooks

**Firn Notebook.ipynb** - 

**InSitu14CO Notebook.ipynb** - Notebook demonstrating how to use the Propagator class defined in InSitu14CO.py

**Technical Doc Notebook.ipynb** - 

### Datasets

**Muon Spectrum Tables/** - Muon surface spectrum measurements taken from various papers and recorded roughly in csv files.  The formatting is not ideal, so each one requires different code to read.  If needed, they could be cleaned up using that code and recorded somewhere else.

**Production Rates/** - Calculations of the fast and negative production rates with depth, labeled by the methods used to calculate them

**14CO_f_all_ChiSq.csv** - Chi square values of f-factors taken from Taylor Glacier analysis (old)

**DomeC 14CO data analysis current.xlsx** - Current version of Dome C measurements

**DomeC_age_scale_Apr2023.csv** - Conversion between depth and ice age at Dome C, taken from Matlab

**factors_2sigma_hull.csv** - Grid of f-factors in the 2sigma range of the Summit, Greenland analysis

**Firn_Model_Density_DomeC.csv** - Density profile at Dome C, used in Firn Model

**Firn_Model_Tortuosity_DomeC.csv** - Tortuosity profile at Dome C, used in Firn Model

**Real_vs_ice_eq_depth.csv** - Conversion between real depth and ice-equivalent depth at Dome C, taken from Matlab

**Summit_Densities.csv** - Density of ice at Summit, Greenland over depth, taken from Matlab
