#!/usr/bin/ python

### daily VI for S2 brdf-corrected .nc files



import os, sys
import numpy as np
import xarray as xr
import rasterio as rio
import geopandas as gpd


## project parameters

def main():
    UNQ=int(sys.argv[1])
    grid = str(UNQ).zfill(3)
    VI=sys.argv[2] ##'evi'
    download_dir=sys.argv[3] ## '/home/downspout-cel/DL/raster/S2_stac/'
    mid_out_dir=sys.argv[4] ## '/home/downspout-cel/DL/raster/S2_dailyGreen/'

    if not os.path.exists(mid_out_dir):
        os.makedirs(mid_out_dir)           

    full_outDir = os.path.join(mid_out_dir, grid)
    if not os.path.exists(full_outDir):
        os.makedirs(full_outDir)

    full_inDir = os.path.join(download_dir, grid, 'brdf')
    for daily_fi in sorted([i for i in os.listdir(full_inDir) if (i.startswith('L3A_S2') and  i.endswith('.nc'))]):
        out_fi = os.path.join(full_outDir, 'L3A_S2_'+VI+'_'+os.path.basename(daily_fi).split("_")[3]+'_UNQ'+grid+'.tif')
        print(out_fi)
        ds = xr.open_dataset(os.path.join(full_inDir, daily_fi))
        datatype = ds.blue.dtype
        epsg = int(ds.crs.split(":")[-1])
        ds = ds.assign_coords(ds.coords)
        ds = ds.rio.write_crs(epsg)
        gt = [i for i in ds.transform]
        noData = ds.nodatavals[0]
        if VI=='ndmi1':
            band_arr = (ds.data_vars.get('nir')-ds.data_vars.get('swir1'))/(ds.data_vars.get('nir')+ds.data_vars.get('swir1'))
        elif VI=='ndmi2':
            band_arr = (ds.data_vars.get('nir')-ds.data_vars.get('swir2'))/(ds.data_vars.get('nir')+ds.data_vars.get('swir2'))
        elif VI=='ndwi':
            band_arr = (ds.data_vars.get('green')-ds.data_vars.get('nir'))/(ds.data_vars.get('green')+ds.data_vars.get('nir'))
        elif VI=='ndwi_rev':
            band_arr = (ds.data_vars.get('nir')-ds.data_vars.get('green'))/(ds.data_vars.get('green')+ds.data_vars.get('nir')) 
        elif VI=='ndvi':
            band_arr = (ds.data_vars.get('nir')-ds.data_vars.get('red'))/(ds.data_vars.get('nir')+ds.data_vars.get('red'))
        elif VI=='evi':
            band_arr = 2.5*((ds.data_vars.get('nir')/10000-ds.data_vars.get('red')/10000)/(ds.data_vars.get('nir')/10000+2.4*ds.data_vars.get('red')/10000+1))
        elif VI=='savi':
            band_arr = ((ds.data_vars.get('nir')/10000-ds.data_vars.get('red')/10000)/(ds.data_vars.get('nir')/10000+ds.data_vars.get('red')/10000+0.5))*(1.5)   
        elif VI=='gcvi':
            band_arr = (((ds.data_vars.get('nir')/10000)/(ds.data_vars.get('green')/10000))-1)

        with rio.open(out_fi,'w',  driver='GTiff', 
                    width=band_arr.shape[1], height=band_arr.shape[0], count=1, 
                    crs=epsg, transform=gt,
                    dtype=datatype, nodata = noData) as dst:
            dst.write(band_arr, indexes=1)

    
if __name__ == '__main__':
    main()





