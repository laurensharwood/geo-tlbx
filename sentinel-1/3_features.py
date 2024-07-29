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
from pathlib import Path
from IPython.display import Image
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import xarray as xr
import dask.array as da
import re 
import time
from datetime import date 
from shapely.geometry import box
from geowombat.core import dask_to_xarray
from geowombat.moving import moving_window


def polarize_sta(in_dir, out_dir, date, stats_list):
    '''
    stats_list options  ['add', 'subtract', 'divide', 'normalized_diff'] add=VV+VH. subtract=VV-VH. divide=VV/VH.
    '''
    grid_folder = "UNQ" + str(in_dir.split("/")[-1]).zfill(3)   
    ignore_done = ["mon", "foc", "var", "add", "subtract", "divide"]
    full_year = sorted([fi for fi in os.listdir(in_dir) if not any(re.match(f'^.*({s}).*$', fi) for s in ignore_done)])
    VVimg=[i for i in full_year if (str(date) in i and "VV" in i)][0]  
    VHimg=[i for i in full_year if (str(date) in i and "VH" in i)][0]  
    neighborhood_MeanVar(in_file=os.path.join(in_dir, VVimg), window_size=(5,9))
    neighborhood_MeanVar(in_file=os.path.join(in_dir, VHimg), window_size=(5,9))
    
    with rio.open(os.path.join(in_dir, VVimg)) as srcV:
        VV_array = srcV.read(1)  
    with rio.open(os.path.join(in_dir, VHimg)) as srcH:
        VH_array = srcH.read(1)
    out_meta = srcH.meta          
    for stat in stats_list:
        if stat=="divide":
            array = np.divide(VV_array, VH_array)
        if stat=="subtract":
            array = np.subtract(VV_array, VH_array) 
        if stat=="add":
            array = np.add(VV_array.astype(np.float32), VH_array.astype(np.float32)) / 2 
        if stat=="normDiff":
            array = np.subtract(VV_array, VH_array) / np.add(VV_array, VH_array)  
        #array = array*100
        #array=array.astype(np.int16)
       # out_meta.update({"dtype":np.int16})
        out_name = "S1A_VHVV_"+str(date)+"_"+str(stat)+"_"+str(grid_folder)+".tif"  
        with rio.open(os.path.join(out_dir, out_name), "w", **out_meta) as dst:
            dst.write(array, indexes=1)      
            print(os.path.join(out_dir, out_name))  
        stats_file = os.path.join(out_dir, out_name)
        neighborhood_MeanVar(in_file=stats_file, window_size=(5,9))
    
    
def neighborhood_MeanVar(in_file, window_size):      
    grid_folder = "UNQ"+in_file.split(".")[-2][-3:]
    out_name_foc = in_file.replace(grid_folder+".tif", "") +"foc"+str(window_size[0])+"_"+str(grid_folder)+".tif"
    out_name_Var = in_file.replace(grid_folder+".tif", "") +"var"+str(window_size[1])+"_"+str(grid_folder)+".tif"
    with gw.open(in_file, chunks=1024) as src:
        res = src.gw.moving(stat='mean', w=int(window_size[0]), n_jobs = 4, nodata = 0)
        res = dask_to_xarray(src, da.from_array(res.data.compute(num_workers=4), chunks=src.data.chunksize), src.band.values.tolist()) 
        res.gw.to_raster(out_name_foc, n_workers=4, n_threads=1)           
        if ("divide" not in in_file) and ("subtract" not in in_file): ## variance of divide looked like nothing. var of subtract looks like just speckle/noise 
            res_Var = src.gw.moving(stat='var', w=int(window_size[1]), n_jobs = 4, nodata = 0)
            res_Var = dask_to_xarray(src, da.from_array(res_Var.data.compute(num_workers=4), chunks=src.data.chunksize), src.band.values.tolist()) 
            res_Var.gw.to_raster(out_name_Var, n_workers=4, n_threads=1)  
        else:
            pass 
        
        

        
def temporal_stats(in_dir, out_dir, grid_folder, polarizations, stats_list):
    for polar in polarizations:
        ignore_done = ["mon", "foc", "var", "add", "subtract", "divide"]
        full_year = sorted([fi for fi in os.listdir(in_dir) if (str(polar) in fi and not any(re.match(f'^.*({s}).*$', fi) for s in ignore_done))]) # if (str(polar) in fi and not any(re.match(f'^.*({s}).*$', fi) for s in ignore_done))
        Jan=[i for i in full_year if str(i.split("_")[2][4:6]) == str(1).zfill(2)]
        Feb=[i for i in full_year if str(i.split("_")[2][4:6]) == str(2).zfill(2)]
        Mar=[i for i in full_year if str(i.split("_")[2][4:6]) == str(3).zfill(2)]
        Apr=[i for i in full_year if str(i.split("_")[2][4:6]) == str(4).zfill(2)]
        May=[i for i in full_year if str(i.split("_")[2][4:6]) == str(5).zfill(2)]
        Jun=[i for i in full_year if str(i.split("_")[2][4:6]) == str(6).zfill(2)]
        Jul=[i for i in full_year if str(i.split("_")[2][4:6]) == str(7).zfill(2)]
        Aug=[i for i in full_year if str(i.split("_")[2][4:6]) == str(8).zfill(2)]
        Sep=[i for i in full_year if str(i.split("_")[2][4:6]) == str(9).zfill(2)]
        Oct=[i for i in full_year if str(i.split("_")[2][4:6]) == str(10).zfill(2)]
        Nov=[i for i in full_year if str(i.split("_")[2][4:6]) == str(11).zfill(2)]
        Dec=[i for i in full_year if str(i.split("_")[2][4:6]) == str(12).zfill(2)]   
        months=[Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]

        month_num=1
        if in_dir.split("/")[-2] == "2023":
            months = months[:3]
        for mon in months: ## add [:3] for 2023
            big_list = []
            for img in mon:
                with rio.open(os.path.join(in_dir, img)) as src:
                    array = src.read(1)
                big_list.append(array)
                out_transform=src.transform
                out_meta=src.meta
                width=src.width
                height=src.height
            out_rast = np.nanmean(big_list, axis=0)
            out_rast.reshape(1, width, height)
            out_raster = out_rast*100
            out_raster=out_raster.astype(np.int16)
            year = full_year[-1].split("_")[2][:4]
            out_name = "S1A_"+str(polar)+"_"+str(year)+"_mon"+str(month_num).zfill(2)+"_avg_"+str(grid_folder)+".tif"  
            out_meta.update({"driver": "GTiff", "count": 1, "dtype": out_raster.dtype,
                             "width": width,  "height": height, "transform": out_transform})
            month_num+=1
            with rio.open(os.path.join(out_dir, out_name), "w", **out_meta) as dst:
                dst.write(out_raster, indexes=1)
                print(os.path.join(out_dir, out_name))  
                
                
        BonusDry = [fi for fi in Jan+Feb+Mar if "2023" in fi]    
        EarlyWet = [fi for fi in Apr+May+Jun if "2022" in fi]
        LateWet = [fi for fi in Jul+Aug+Sep if "2022" in fi]
        PostDry = [fi for fi in Oct+Nov+Dec if "2022" in fi]          

        Seasons = {"BonusDry":BonusDry, "EarlyWet": EarlyWet , "LateWet":LateWet, "PostDry":PostDry}
        for szn_pair in Seasons.items():
            name = szn_pair[0]
            szn = szn_pair[1]
            if len(szn) > 1:
                for i in szn:
                    with rio.open(os.path.join(in_dir, i)) as src:
                        array = src.read(1)
                        big_list.append(array)
                    out_transform=src.transform
                    out_meta=src.meta
                    width=src.width
                    height=src.height
                for stat in stats_list:
                    print(stat)
                    if str(stat) == "Mean":
                        out_rast = np.nanmean(big_list, axis=0)  
                    if str(stat) == "Max":
                        out_rast = np.max(big_list, axis=0) 
                    if str(stat) == "Min":
                        out_rast = np.min(big_list, axis=0)  
                    if str(stat) == "Median":
                        out_rast = np.nanmedian(big_list, axis=0)          
                    elif str(stat) == "StDev":
                        out_rast = np.nanstd(big_list, axis=0)  
                    elif str(stat) == "Variance":
                        out_rast = np.nanvar(big_list, axis=0)  
                    elif str(stat) == "Covariance":
                        out_ras = np.nanmean(big_list, axis=0)  
                        out_rast = np.cov(out_ras)
                    out_rast.reshape(1, width, height) # (1band, width, height)
                    out_raster = out_rast*100
                    out_raster=out_raster.astype(np.int16)
                    grid_folder="UNQ"+in_dir.split("/")[-1]
                    out_name = "S1A_"+str(polar)+"_"+ str(name)+"_"+str(stat)+"_"+str(grid_folder)+".tif"   
                    out_meta.update({"dtype": np.int16, "driver": "GTiff", "nodata": 0, "count": 1, "height": height, "width": width, "transform": out_transform})
                    with rio.open(os.path.join(out_dir, out_name), "w", **out_meta) as dst:
                        dst.write(out_raster, indexes=1)
                        print(os.path.join(out_dir, out_name))       

def main():
    grid_arg = str(sys.argv[1]).zfill(3)
    grid_folder = "UNQ"+str(grid_arg) ##in_dir.split("/")[-1]
    for in_mainDir in ["/home/sandbox-cel/DL/raster/S1_GRD/03_Tiled/2022", "/home/sandbox-cel/DL/raster/S1_GRD/03_Tiled/2023"]:
        in_dir = os.path.join(in_mainDir, str(grid_arg))
        out_dir = os.path.join("/home/sandbox-cel/DL/raster/S1_GRD/04_Features", str(grid_arg))                
        ## POLARIZE STATS (VH+VV AND NEIGHBORHOOD MEAN AND VARIANCE)

        ignore_done = ["mon", "foc", "var", "add", "subtract", "divide"]
        full_year = sorted([fi for fi in os.listdir(in_dir) if not any(re.match(f'^.*({s}).*$', fi) for s in ignore_done)])
        for dat in [im.split("_")[2] for im in full_year]:
            if not os.path.exists(out_dir):
                os.system("mkdir "+out_dir)
            polarize_sta(in_dir, out_dir, date=dat, stats_list=["add", "subtract", "divide"])                    

    ## TEMPORAL STATS (MEAN MONTHLY AND SEASONAL VARIATION)
    polarizations = ["VH", "VV"]
    stats_list = ["Median", "Max", "Variance"]
    temporal_stats(in_dir, out_dir, grid_folder, polarizations, stats_list)

        

if __name__ == "__main__":
    main()
