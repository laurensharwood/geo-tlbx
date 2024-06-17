#!/usr/bin/ python

import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import xarray as xr
import glob
                                
def main():

    in_dir=sys.argv[1] #="/home/l_sharwood/outcomes/monthlyVIvars/"
    fName=sys.argv[2] #="/home/l_sharwood/outcomes/new/RCT1_fldPts_inBounds.shp"

    mosaics = glob.glob(in_dir+'**/*.vrt', recursive = True)
    mosaics = [i for i in mosaics if "_Dry_" not in i]

    data_orig = gpd.read_file(fName)
    data = data_orig.copy()
    coords_list = list(zip(data['geometry'].x, data['geometry'].y))
    crs_str = data.crs.name.split(" / ")[-1].replace(" zone ", "")
    data[crs_str+'_XY'] = coords_list
    data.set_index([crs_str+'_XY'], inplace=True)
    for mosaic in sorted(mosaics):
        mosName = os.path.basename(mosaic).replace(".vrt", "")
        print(mosName)
        with rio.open(mosaic) as src:
            ## create column for raster, row is raster value at each coordinate
            data[mosName] = [x[0] for x in src.sample(coords_list)]
    out_df=data.drop(columns=['geometry'])
    newName = os.path.join(fName.replace(".shp", "_"+os.path.basename(in_dir[:-1])+".csv"))
    out_df.to_csv(newName)
    print(newName)
    
if __name__ == '__main__':
    main()





