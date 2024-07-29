#!/usr/bin/ python

import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import xarray as xr
import glob
import datetime
    
def peakSzn(szn_name, year, VI, grid_dir, out_dir):
    month_list = season_dict.get(szn_name)
    months = sorted([str(i).zfill(2) for i in month_list])
    start_date = ".".join([str(year), str(int(months[0])).zfill(2), str(1).zfill(2)])
    end_date = ".".join([str(year), str(int(months[-1])+1).zfill(2), str(1).zfill(2)])
    start_time = datetime.datetime.strptime(start_date, '%Y.%m.%d')
    end_time = datetime.datetime.strptime(end_date, '%Y.%m.%d')
    print(start_time,end_time)

    monthly_grids = sorted([os.path.join(grid_dir, g) for g in os.listdir(grid_dir) if len(g) == 3]) ##  and g in [str(i).zfill(3) for i in RCT1_ONLY]
    for path in monthly_grids:
        grid = os.path.basename(path)
        out_full_dir = os.path.join(out_dir, grid)
        out_name=str(year)[-2:]+os.path.basename(grid_dir).replace("S2_dailyGreen", "_peak_")+szn_name+"_"+VI.upper()+"_"+grid+".tif"
        images = sorted([os.path.join(grid_dir, grid, fi) for fi in os.listdir(path) if ((VI in fi and fi.endswith(".tif")) and (datetime.datetime.strptime(os.path.basename(fi).split("_")[3], '%Y%m%d') > start_time and datetime.datetime.strptime(os.path.basename(fi).split("_")[3], '%Y%m%d') < end_time))])
        first = rio.open(images[0], 'r')
        out_meta = first.meta.copy()
        season = []
        for file in images:
            with rio.open(file) as src:
                day = src.read(1)
                gt = src.transform
                season.append(day)
        smoothed_peak = np.nanpercentile(season, 90, axis=0)
        print(smoothed_peak)
        out_meta.update({"width":smoothed_peak.shape[1], "height":smoothed_peak.shape[0], "count":1, "dtype":np.float32, "crs":32632, "transform":gt })       
        with rio.open(os.path.join(out_dir, out_name), "w", **out_meta) as dst:
            dst.write(smoothed_peak, indexes=1)
        print(out_name)
        
        
def main():

      grid_dir=sys.argv[1] #"/home/sandbox-cel/DL/raster/S2_dailyGreen"
      out_dir1 =sys.argv[2] # "/home/l_sharwood/outcomes/dailyVI_vars"
      year=int(sys.argv[3])
      global season_dict
      season_dict = {"Dry":[1,2,3], "Grow":[8,9,10]}
      for VI in ["ndvi", "evi", "gcvi", "ndmi1", "ndmi2"]:
          for szn_name in season_dict:
              out_dir = os.path.join(out_dir1, str(year)[-2:]+"_"+"peak_"+szn_name+"_"+VI)
              if not os.path.exists(out_dir):
                  os.makedirs(out_dir)
              peakSzn(szn_name, year, VI, grid_dir, out_dir)

if __name__ == '__main__':
    main()





