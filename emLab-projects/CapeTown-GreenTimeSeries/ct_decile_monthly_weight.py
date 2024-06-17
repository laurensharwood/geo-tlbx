#!/usr/bin/ python

import os, sys
import pandas as pd
import geopandas as gpd
import time
import rasterio as rio
from rasterio.windows import Window
from rasterio.features import shapes, coords, bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterstats import zonal_stats

def raster_to_vector(input_raster):
    with rio.open(input_raster, 'r') as src:
        rast = src.read(1)
        rast_crs = src.crs
    instance_shapes = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes(rast, mask=None, transform=src.transform)))
    vectorized  = gpd.GeoDataFrame.from_features(list(instance_shapes), crs=rast_crs)
    return vectorized



def main():

    inShp=sys.argv[1] ## '/home/sandbox-cel/capeTown/vector/deciDisolv_Mask2017_utm32S.shp'
    month_dir=sys.argv[2] ## '/home/sandbox-cel/capeTown/monthly/landsat/vrts' ## '/home/sandbox-cel/capeTown/monthly/landsat'
    VI=sys.argv[3] ## 'evi' #, 'ndvi', 'ndmi1', 'ndmi2', 'ndwi','ndwi_rev'
    startYr=int(sys.argv[4]) ## 2016
    endYr=int(sys.argv[5]) ## 2021
    
    ###################
    

    merged_deciles = gpd.read_file(inShp)
    merged_deciles['deciArea'] = merged_deciles.area
    deciYr = int(inShp.split(".")[-2][-4:])     
    IDcol = 'd'+str(deciYr)
    sensor_version = month_dir.split("/")[-2]
    outFi=inShp.replace(".shp", "_"+sensor_version+"_"+VI+"_w.csv")
    print(outFi)
    dfs=[]
    if not os.path.exists(outFi):
        monthly_deciles = {}
        for year in list(range(startYr, endYr+1, 1)):
            for month in list(range(1, 13, 1)):
                date = str(year)+str(month).zfill(2)
                monthRast = sorted([os.path.join(month_dir, file) for file in os.listdir(month_dir) if VI+"_"+date in file])[0]
                col_name = '-'.join(os.path.basename(monthRast).replace(".vrt", "").split("_")[1:])   
                print(col_name)
    #                 if PixWgtAvg==True:
                vector_pixels = raster_to_vector(input_raster=monthRast)
                print(vector_pixels)
                pixIntersect = vector_pixels.overlay(merged_deciles, how='intersection')
                print(pixIntersect)
                pixIntersect[col_name] = (pixIntersect.area/pixIntersect['deciArea'])*pixIntersect['raster_val']
                sums = pixIntersect.groupby(by=[IDcol]).sum(numeric_only=True)[col_name]
                decilePolys = decilePolys.set_index(IDcol).join(sums, on=IDcol)
                decilePolys = decilePolys.reset_index()                    
                print(decilePolys)
                dfs.append(decilePolys)
    #                 else:
                # decile_means = zonal_stats(inShp, monthRast, stats="mean")
                # ## dictionary w/ date as key, list of means as value
                # monthly_deciles.update({date:[i['mean'] for i in decile_means]})
                # df = pd.DataFrame.from_dict(monthly_deciles)
                # dfs.append(df)
            print(dfs[-1])
    dfs_comb=pd.concat(dfs, axis=0)
    print(dfs_comb)
    dfs_comb.to_csv(outFi)



if __name__ == '__main__':
    main()




