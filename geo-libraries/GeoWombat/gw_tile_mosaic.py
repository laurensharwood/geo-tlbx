#!/bin/bash

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import geowombat as gw
from datetime import date 
import json
import time
from shapely.geometry import Point
import rasterio as rio
import xarray as xr
import rioxarray
import tqdm
from itertools import chain
import csv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sys import argv

start = time.time()

###########################

folder_num = argv[1]
input_dir = argv[2]
keyword = argv[3]
output_dir = argv[4]

print(folder_num)

base_image = '/home/downspout-cel/DL/raster/grids2/'+ str(folder_num).zfill(6) + '/brdf_ts/ms/ndmi/2017020.tif'
out_folder1 = "UNQ" + str(folder_num).zfill(2)
out_folder = "UNQ_" + str(folder_num).zfill(2)
out_dir = os.path.join(output_dir, out_folder) 

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

###########################
rasters = [r for r in sorted(os.listdir(input_dir)) if (str(keyword) in r and r.endswith(".tif"))] 
with gw.open(base_image) as tar_res:
    new_res = tar_res.res
with gw.open(base_image) as target:
    new_crs = target.crs
    new_extent = target.gw.bounds
    for rast in rasters:
        print(os.path.join(input_dir, rast))
        with gw.config.update(nodata=0, ref_crs=new_crs,  ref_res=new_res, ref_bounds=new_extent):
            with gw.open(os.path.join(input_dir, rast), chunks=1024) as ref:
                attrs = ref.attrs.copy()
                ref = (ref).assign_attrs(**attrs, count=1, nodata=0, verbose=1)
                ref = (ref).assign_attrs(res=new_res, extent=new_extent, count=1)
                if keyword == "NASA_dem" or keyword == "Palsar":
                    out_rast = rast.split(".")[0] + "_" + out_folder1 + ".tif" 
                    ref.gw.to_raster(os.path.join(out_dir, out_rast), indexes=1, indexes=1, n_workers=4, n_threads=1)                
                    print(os.path.join(out_dir, out_rast))                        
                else:
                    out_rast_VH = rast.split(".")[0] + "VH_" + out_folder1 + ".tif" 
                    out_rast_VV = rast.split(".")[0] + "VV_"  + out_folder1 + ".tif"                 
                    ref[0].gw.to_raster(os.path.join(out_dir, out_rast_VH), indexes=1, indexes=1, n_workers=4, n_threads=1)                
                    print(os.path.join(out_dir, out_rast_VH))   
                    ref[1].gw.to_raster(os.path.join(out_dir, out_rast_VV), indexes=1, indexes=1, n_workers=4, n_threads=1)
                    print(os.path.join(out_dir, out_rast_VV))   
                #ref.gw.save(os.path.join(out_dir, out_rast))

###########################
end = time.time()
minutes = (end - start)/60
print('time this took ' , str(minutes), ' minutes to complete')
sys.stdout.flush()
