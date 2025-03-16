import xarray as xr
import numpy as np
import netCDF4 as nc
import os

def build_data_structure(filepath):
    data = xr.open_dataset(filepath)
    