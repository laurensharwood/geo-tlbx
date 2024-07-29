#!/usr/bin/ python

import os, sys
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio.merge import merge
from osgeo import gdal
import xarray as xr
import geowombat as gw
import geopandas as gpd

## user input params 
grid_id=int(sys.argv[1]) 
keyword =sys.argv[2] 
input_dir=sys.argv[3] 
output_dir=sys.argv[4] 
##

## test python script
#grid_id = int("022")
#keyword = "S1"
#input_dir = "/home/sandbox-cel/DL/raster/S1_GRD/02_Preprocessed/2022"
#output_dir = "/home/sandbox-cel/DL/raster/S1_GRD/03_Tiled/2022"


## create virtual mosaic of all takes in the AOI per date 
MDT_ids = list(set([i.split("_")[-2].split(".")[0] for i in sorted(os.listdir(input_dir)) if i.endswith(".tif")]))
for take in MDT_ids:
    mos_list =  [os.path.join(input_dir, i) for i in sorted(os.listdir(input_dir)) if (i.endswith(".tif") and i.split("_")[-2]== take)]
    print(mos_list)
    mos_name = mos_list[0].split("/")[-1][:25]+"_"+str(take)+"_mosaic.vrt"
    print(mos_name)
    if not os.path.exists(os.path.join(input_dir, mos_name)):
        vrt_options = gdal.BuildVRTOptions(srcNodata=0, VRTNodata=-9999)
        gdal.BuildVRT(os.path.join(input_dir, mos_name), sorted(mos_list), options=vrt_options)
        print(os.path.join(input_dir, mos_name)+ ' done')
    else:
        print(os.path.join(input_dir, mos_name)+ ' exists')
        
## project parameters 
mosaics = [r for r in sorted(os.listdir(input_dir)) if (str(keyword) in r and r.endswith(".vrt"))] 
proj_epsg = 32632
img_res = 10
grid = gpd.read_file("/home/sandbox-cel/DL/vector/NigerGrid_RCT2_UTM32.gpkg")

## tile mosaics to 20km processing grids
for rast in mosaics:
    this_grid = grid[grid.UNQ == grid_id].iloc[0]
    folder_num = int(this_grid.UNQ)
    folder_name = "UNQ" + str(folder_num).zfill(3)
    new_extent = this_grid.geometry.bounds
    out_dir = os.path.join(str(output_dir), str(folder_num).zfill(3))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(os.path.join(input_dir, rast))
    with gw.config.update(ref_crs=proj_epsg,  ref_res=img_res, ref_bounds=new_extent, nodata=-9999):
        with gw.open(os.path.join(input_dir, rast), chunks=1024) as ref:
            attrs = ref.attrs.copy()
            ref = (ref).assign_attrs(**attrs, verbose=1, nodata=-9999)
            if keyword == "NASA_dem" or keyword == "Palsar":
                out_rast = rast.split(".")[0] + "_" + folder_name + ".tif" 
                ref.gw.to_raster(os.path.join(out_dir, out_rast), indexes=1, n_workers=4, n_threads=1)                
                print(os.path.join(out_dir, out_rast))                        
            elif keyword == "S1":
                out_rast_VH = rast.split(".")[0] + "VH_" + folder_name + ".tif" 
                out_rast_VV = rast.split(".")[0] + "VV_"  + folder_name + ".tif"  
                H, V = ref
                VH=H.gw.mask_nodata()
                VV=V.gw.mask_nodata()
                VH.gw.to_raster(os.path.join(out_dir, out_rast_VH), indexes=1, n_workers=4, n_threads=1)                
                print(os.path.join(out_dir, out_rast_VH))   
                VV.gw.to_raster(os.path.join(out_dir, out_rast_VV), indexes=1, n_workers=4, n_threads=1)
                print(os.path.join(out_dir, out_rast_VV))   
                #ref.gw.save(os.path.join(out_dir, out_rast))
            else:
                print("choose a different keyword")
