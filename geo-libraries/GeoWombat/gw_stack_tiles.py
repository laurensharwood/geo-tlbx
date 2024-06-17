#!/bin/bash

import os
import sys
import numpy as np
import pandas as pd
import geowombat as gw
import rasterio as rio
import xarray as xr
import rioxarray
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sys import argv

###########################

gridnum = argv[1]
gridfolder = "UNQ_" + str(gridnum).zfill(2)
gridname = "UNQ" + str(gridnum).zfill(2)

in_dir = argv[2]
input_dir = os.path.join(in_dir, gridfolder)
print(input_dir)

out_dir = argv[3]
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
out_file  = os.path.join(out_dir, "RF_stack_" + str(gridname) + ".tif")
features = argv[4]
print(features)
files = [(i + "_" + gridname + ".tif") for i in features]
print(files)
filepaths = [os.path.join(input_dir, i) for i in files]
print('stacking the following files: ', filepaths)

with gw.open(filepaths, band_names=features, stack_dim='band', chunks=1024) as src:
    attrs = src.attrs.copy()
    # Apply operations on the DataArray
    src = src.assign_attrs(**attrs, band_names=features)
    print(src)
    src.gw.to_raster(out_file, band_names=features,
                         verbose=1,
                         n_workers=4,    # number of process workers sent to ``concurrent.futures``
                         n_threads=2,    # number of thread workers sent to ``dask.compute``
                         n_chunks=200)
    
    print('output file: ', out_file)

###########################

sys.stdout.flush()
