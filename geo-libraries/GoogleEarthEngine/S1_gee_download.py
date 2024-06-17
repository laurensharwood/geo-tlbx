#!/usr/bin/ python

import os, sys, glob
import io
import requests
import ee
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import shapely
from shapely.geometry import Polygon, mapping
import geemap
import pyproj
from pyproj import Proj, transform, CRS, Transformer
import os, sys
import rasterio as rio
from osgeo import gdal 
import geowombat as gw
from geowombat.core import dask_to_xarray
from geowombat.moving import moving_window
import xarray as xr
import dask.array as da


try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate() # do NOT select read only scopes
    ee.Initialize()
    
###################################################

ksize=3
enl=7
def toNatural(img):
    return ee.Image(10.0).pow(img.select(0).divide(10.0))

def speckleFilter(image):
    """ apply the speckle filter """
    # Convert image from dB to natural values
    nat_img = toNatural(image)

    # Square kernel, ksize should be odd (typically 3, 5 or 7)
    weights = ee.List.repeat(ee.List.repeat(1,ksize),ksize)
    print(weights)

    # ~~(ksize/2) does integer division in JavaScript
    kernel = ee.Kernel.fixed(ksize, ksize, **{'weights': weights})

    # Get mean and variance
    mean = nat_img.reduceNeighborhood(ee.Reducer.mean(), kernel)
    variance = nat_img.reduceNeighborhood(ee.Reducer.variance(), kernel)

    # "Pure speckle" threshold
    ci = variance.sqrt().divide(mean)# square root of inverse of enl

    # If ci <= cu, the kernel lies in a "pure speckle" area -> return simple mean
    cu = 1.0/math.sqrt(enl)

    # If cu < ci < cmax the kernel lies in the low textured speckle area
    # -> return the filtered value
    cmax = math.sqrt(2.0) * cu

    alpha = ee.Image(1.0 + cu*cu).divide(ci.multiply(ci).subtract(cu*cu))
    b = alpha.subtract(enl + 1.0)
    d = mean.multiply(mean).multiply(b).multiply(b).add(alpha.multiply(mean).multiply(nat_img).multiply(4.0*enl))
    f = b.multiply(mean).add(d.sqrt()).divide(alpha.multiply(2.0))

    # If ci > cmax do not filter at all (i.e. we don't do anything, other then masking)

    # Compose a 3 band image with the mean filtered "pure speckle", 
    # the "low textured" filtered and the unfiltered portions
    out = ee.Image.cat(mean.updateMask(ci.lte(cu)),
                       f.updateMask(ci.gt(cu)).updateMask(ci.lt(cmax)),
                       image.updateMask(ci.gte(cmax)))

    return out.reduce(ee.Reducer.sum())
    
#############################################
        
import rasterio as rio
from rasterio.windows import Window
from rasterio.transform import Affine

## HELPER FUNC FOR RETILING
def windowed_read(gt, bbox): 
    """
    helper function for rasterio windowed reading of chip within grid to save into cnet time_series_vars folder
    gt = main raster's geotransformation (src.transform)
    bbox = bounding box polygon as subset from raster to read in
    """
    origin_x = gt[2]
    origin_y = gt[5]
    pixel_width = gt[0]
    pixel_height = gt[4]
    x1_window_offset = int(round((bbox[0] - origin_x) / pixel_width))
    x2_window_offset = int(round((bbox[1] - origin_x) / pixel_width))
    y1_window_offset = int(round((bbox[3] - origin_y) / pixel_height))
    y2_window_offset = int(round((bbox[2] - origin_y) / pixel_height))
    x_window_size = x2_window_offset - x1_window_offset
    y_window_size = y2_window_offset - y1_window_offset
    return [x1_window_offset, y1_window_offset, x_window_size, y_window_size]


#############################################

def download_monthly_S1(boundary_shape, grid_num, out_path, YYYYdMMdDD):
    start_date=YYYYdMMdDD
    year=YYYYdMMdDD[:4]
    month=YYYYdMMdDD[5:7]
    day=YYYYdMMdDD[9:]
    if int(month) < 12:
        end_date=year+"-"+str(int(month)+1).zfill(2)+"-"+day.zfill(2)
    elif int(month) == 12:
        end_date=(str(int(year)+1))+"-"+str(1).zfill(2)+"-"+day.zfill(2)
    print(start_date)
    print(end_date)
    gdf = gpd.read_file(boundary_shape)
    
    #for grid in grid_list:

    full_out_dir = os.path.join(out_path, str(grid_num).zfill(3))
    UNQ_gdf = gdf[gdf['UNQ'] == int(grid_num)]
    gdf_web = UNQ_gdf.to_crs('EPSG:4326')
    aoi = ee.Geometry.Rectangle([gdf_web.bounds.minx.min(), gdf_web.bounds.miny.min(), gdf_web.bounds.maxx.max(), gdf_web.bounds.maxy.max()])
    for polarization in ["VV", "VH"]:
        ## get Sentinel-1 image from EE catalog 

        s1_grd = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi).filterDate(start_date, end_date)\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization))\
            .filter(ee.Filter.eq('instrumentMode', 'IW'))\
            .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))\
            .select(polarization).first().clip(aoi) 

        ## get crs transformation/extent info for saving image 
        projection = s1_grd.select(polarization).projection().getInfo()

        s1_col = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi).filterDate(start_date, end_date)\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization))\
            .filter(ee.Filter.eq('instrumentMode', 'IW'))\
            .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))\
            .select(polarization)

        s1_median = ee.Image([s1_col.median()]).select(polarization).clip(aoi)
        ## apply speckle filter
        #s1_speck = ee.Image(speckleFilter(s1_grd))
        ## rename band to polarization
        #image = ee.Image(s1_speck).select(['sum']).rename([polarization])
        date=start_date.replace("-", "")[:-2]
        out_name = os.path.join(full_out_dir, "S1A_"+str(polarization)+"_"+str(date)+"_Med_UNQ"+str(grid_num).zfill(3)+"_")
        grid_mosaic=os.path.join(full_out_dir, out_name[:-1]+"_mos.vrt")
        out_rast = grid_mosaic.replace("_mos.vrt", ".tif")            
        download_tiles = geemap.fishnet(aoi, rows=2, cols=2, delta=0)
        if not os.path.exists(out_rast):
            geemap.download_ee_image_tiles(image=s1_median, 
                                            features=download_tiles,
                                            crs=projection.get('crs'), 
                                            crs_transform=projection.get('transform'),  
                                            scale=10,
                                            prefix=out_name)


#############################################

def main():
    grid_file=sys.argv[1] ## "/home/sandbox-cel/DL/vector/AOIs/NigerGrid_RCT2_UTM32.gpkg"
    main_dir=sys.argv[2] ## "/home/downspout-cel/DL/raster/grids/"
    start_month=sys.argv[3] ##  MONTHLY_START_DATE=("2017-04-01" "2017-08-01" "2018-01-01" "2018-04-01" "2018-08-01" "2019-01-01")
    grid_num=int(sys.argv[4]) ## single value from list  [21,22,23,24,35,36,45,46,58,59,60,61,64,65,66,72,73,74,75,76,77,78,79,86,87,88,89,91,92,93,100,101,105,106,107,108,119,120,121,122,123,124,133,134,135,136,137,138,160,174,175,188,189,201]

    download_monthly_S1(boundary_shape=grid_file,
                        grid_num=grid_num, 
                        out_path=main_dir, 
                        YYYYdMMdDD=start_month)   


if __name__ == "__main__":
    main()