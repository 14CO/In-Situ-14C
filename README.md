# In-Situ-14C-Production
Calculates in-situ 14CO produced by cosmic ray muons in ice, starting from primary spectra at the top of the atmosphere

## Installation Guide

(WIP)

### Required Python Libraries

numpy

matplotlib

MCEq

crflux

tqdm

pandas

astropy.io

daemonflux

## File Guide

### Python Files

**InSitu14CO.py** - Defines the Propagator class, which manages the calculations involved in turning primary CR spectra into 14CO profiles

### Notebooks

**InSitu14CO Notebook.ipynb** - Notebook demonstrating how to use the Propagator class defined in InSitu14CO.py

### Datasets

**Muon Spectrum Tables/** - Muon surface spectrum measurements taken from various papers and recorded roughly in csv files.  The formatting is not ideal, so each one requires different code to read.  If needed, they could be cleaned up using that code and recorded somewhere else.

**balco_14co_const_models.fits** - 14CO profiles made with the Balco Matlab calculation, stored as an astropy fits file

**DomeC_age_scale_Apr2023.csv** - Conversion between depth and ice age at Dome C, taken from Matlab

**Real_vs_ice_eq_depth.csv** - Conversion between real depth and ice-equivalent depth at Dome C, taken from Matlab

**Summit_Densities.csv** - Density of ice at Summit, Greenland over depth, taken from Matlab
