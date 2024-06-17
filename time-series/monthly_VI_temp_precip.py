#!/usr/bin/ python

import os, sys
import numpy as np
import xarray as xr
import rasterio as rio
import geopandas as gpd

def monthlyVI(stat, VI, startYr, endYr, inDir, sensor_full, outDir, overwrite=False):
    
    ## project params 
    epsg=32732

        
    if len([i for i in os.listdir(inDir) if i.endswith(".tif")]) > 1:
        proc_level = 'refl'
    else: ## in .nc format 
        proc_level = 'brdf'
        
    if sensor_full == "sentinel2":
        sensor = "S2"
    elif sensor_ful == "landsat":
        sensor = "LC"
        
    grid =  inDir.split("/")[-2]

    if not os.path.exists(inDir):
        os.makedirs(inDir)
        
    for year in list(range(startYr, endYr, 1)):
        for month in list(range(1, 13, 1)):
            start_date = str(year)+str(month).zfill(2)
            out_fi = os.path.join(outDir, sensor+'_'+VI+'_'+start_date+'_'+stat+'_UNQ'+grid+'.tif')
            print(out_fi)
            #  if not os.path.exists(out_fi):
            if proc_level == 'brdf':
                monthly_fis = sorted([i for i in os.listdir(inDir) if ((i.startswith('L3A_'+sensor) and '_'+start_date in i) and i.endswith('.nc'))])
                if len(monthly_fis) > 0:
                    print(monthly_fis)
                    stack = []
                    for file in [os.path.join(inDir, fi) for fi in monthly_fis]:
                        ds = xr.open_dataset(file)
                        ds = ds.assign_coords(ds.coords)
                        ds = ds.rio.write_crs(epsg)
                        gt = [i for i in ds.transform]
                        noData = ds.nodatavals[0]
                        if VI=='ndmi1':
                            band_arr = (ds.data_vars.get('nir')-ds.data_vars.get('swir1'))/(ds.data_vars.get('nir')+ds.data_vars.get('swir1'))
                        elif VI=='ndmi2':
                             band_arr = (ds.data_vars.get('nir')-ds.data_vars.get('swir2'))/(ds.data_vars.get('nir')+ds.data_vars.get('swir2'))
                        elif VI=='gcvi':
                             band_arr = (ds.data_vars.get('nir')/ds.data_vars.get('green'))-1
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
                        stack.append(band_arr)
                            
                    if len(stack) >= 1:
                        if stat == 'Mean':
                            monthly = np.nanmean(stack, axis=0)
                        elif stat == 'Med':
                            monthly = np.nanmedian(stack, axis=0)
                        elif stat == 'Max':
                            monthly = np.nanmax(stack, axis=0)
                        elif stat == 'Min':
                            monthly = np.nanmin(stack, axis=0)
                            
                        with rio.open(out_fi,'w',  driver='GTiff', width=monthly.shape[1], height=monthly.shape[0], count=1, crs=epsg, transform=gt, dtype=np.float32, nodata = noData) as dst:
                              dst.write(monthly, indexes=1)
  
            elif proc_level == 'refl': ## currently no brdf corrections, so in .tif format, not .nc 
                monthly_fis = sorted([i for i in os.listdir(inDir) if ((sensor in i and '_'+start_date in i) and i.endswith('.tif'))])
                if len(monthly_fis) > 0:
                    stack = []
                    for file in [os.path.join(inDir, fi) for fi in monthly_fis]:               
                        with rio.open(file) as src:
                            gt = src.transform
                            noData = src.nodata
                            blue, green, red, nir, swir1, swir2 = src.read()
                        if VI=='ndvi':
                            band_arr = (nir-red)/(nir+red)
                        elif VI=='evi':
                            band_arr = 2.5*(nir-red)/(nir + (2.4*red) + 1)
                        elif VI=='ndmi1':
                            band_arr = (nir-swir1)/(nir+swir1)
                        elif VI=='ndmi2':
                            band_arr = (nir-swir2)/(nir+swir2)
                        elif VI=='ndwi':
                            band_arr = (green-nir)/(green+nir)
                        elif VI=='ndwi_rev':
                            band_arr = (green-nir)/(green+nir)
                            band_arr = band_arr*-1
                        elif VI=='gcvi':
                            band_arr = (nir/green)-1
                        stack.append(band_arr)                                
  
                    if len(stack) >= 1:
                        if stat == 'Mean':
                            monthly = np.nanmean(stack, axis=0)
                        elif stat == 'Med':
                            monthly = np.nanmedian(stack, axis=0)
                        elif stat == 'Max':
                            monthly = np.nanmax(stack, axis=0)
                        elif stat == 'Min':
                            monthly = np.nanmin(stack, axis=0)
                                                        
                        with rio.open(out_fi,'w',  driver='GTiff', width=monthly.shape[1], height=monthly.shape[0], count=1, crs=epsg, transform=gt, dtype=np.float32, nodata = noData) as dst:
                            dst.write(monthly, indexes=1)
                           



def monthly_temp(in_dir, grid_shape, reproject=True):
    
    out_dir = os.path.join(in_dir, "monthly")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    RCT_fi = gpd.read_file(grid_shape)
    RCT_UTM = RCT_fi.set_crs(32632)
    RCT_WGS = RCT_UTM.to_crs(4326)
    bounds = RCT_WGS.total_bounds ## bounds returns (minx, miny, maxx, maxy)
    boundary = (float(bounds[0]), float(bounds[2]), float(bounds[1] ), float(bounds[3]))
    for stat in ["mean", "min", "max", "std"]:
        all_files = sorted([i for i in os.listdir(in_dir) if i.endswith(".tif")])
        years = sorted(list(range(int(start_yr), int(end_yr), 1)))
        months = sorted([str(i).zfill(2) for i in list(range(1, 13, 1))])
        for year in years:
            for month in months:
                ## find files for ... ERA5: "CHIRTS-ERA5.daily_"+ os.path.basename(in_dir)+"."+str(year)+"."+str(month) ## precip: str(year)+str(month)
                current =  "CHIRTS-ERA5.daily_"+os.path.basename(in_dir)+"."+str(year)+"."+str(month) 
                files = [os.path.join(in_dir, i) for i in all_files if current in i] 
                out_name = os.path.join(out_dir, os.path.basename(in_dir)+"_"+str(year)+str(month)+"_"+stat+".tif")
                if not os.path.exists(out_name):
                    monthly_arr = []
                    for file in files:
                        with rio.open(file) as src:
                            gt = src.transform
                            offset = img_to_bbox_offsets(gt, boundary)
                            daily_arr = src.read(window=Window(offset[0], offset[1], offset[2], offset[3]))
                            new_gt = rio.Affine(gt[0], gt[1], (gt[2] + (offset[0] * gt[0])), 0.0, gt[4], (gt[5] + (offset[1] * gt[4])))
                            out_meta = src.meta.copy()
                            monthly_arr.append(daily_arr)
                    if stat=="mean":
                        stat_arr = np.nanmean(monthly_arr, axis=0)
                    elif stat=="min":
                        stat_arr = np.nanmin(monthly_arr, axis=0)
                    elif stat=="max":
                        stat_arr = np.nanmax(monthly_arr, axis=0)
                    elif stat=="std":
                        stat_arr = np.nanstd(monthly_arr, axis=0)
                    out_meta.update({"transform":new_gt, "width":stat_arr.shape[2], "height":stat_arr.shape[1]})
                    with rio.open(out_name, 'w+', **out_meta) as dst:
                        dst.write(stat_arr)
                        print(out_name)

    ## reproject to UTM32
    for file in sorted([os.path.join(out_dir, i) for i in os.listdir(out_dir) if i.endswith(".tif") and "UTM" not in i]):
        reproject_raster(out_crs="EPSG:32632", in_path=os.path.join(out_dir, file), overwrite=True)





def monthly_precip(in_dir, grid_shape, reproject=True):

    out_dir = os.path.join(in_dir, "monthly")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    RCT_fi = gpd.read_file(grid_shape)
    RCT_UTM = RCT_fi.set_crs(32632)
    RCT_WGS = RCT_UTM.to_crs(4326)
    bounds = RCT_WGS.total_bounds ## bounds returns (minx, miny, maxx, maxy)
    boundary = (float(bounds[0]), float(bounds[2]), float(bounds[1] ), float(bounds[3]))
    for stat in ["cumsum", "sum", "mean", "min", "max", "std"]: 
        all_files = sorted([i for i in os.listdir(in_dir) if i.endswith(".tif")])
        years = sorted(list(range(int(start_yr), int(end_yr), 1))) ## sorted(list(range(2015, 2017, 1)))
        months = sorted([str(i).zfill(2) for i in list(range(1, 13, 1))])
        for year in years:
            for month in months:
                current = str(year)+str(month)
                files = [os.path.join(in_dir, i) for i in all_files if i.startswith(current)] 
                out_name = os.path.join(out_dir, os.path.basename(in_dir)+"_"+str(year)+str(month)+"_"+stat+".tif")
                if not os.path.exists(out_name):
                    monthly_arr = []
                    for file in files:
                        with rio.open(file) as src:
                            gt = src.transform
                            offset = img_to_bbox_offsets(gt, boundary)
                            daily_arr = src.read(window=Window(offset[0], offset[1], offset[2], offset[3]))
                            new_gt = rio.Affine(gt[0], gt[1], (gt[2] + (offset[0] * gt[0])), 0.0, gt[4], (gt[5] + (offset[1] * gt[4])))
                            out_meta = src.meta.copy()
                            monthly_arr.append(daily_arr)
                    if stat=="mean":
                        stat_arr = np.nanmean(monthly_arr, axis=0)
                    elif stat=="min":
                        stat_arr = np.nanmin(monthly_arr, axis=0)
                    elif stat=="max":
                        stat_arr = np.nanmax(monthly_arr, axis=0)
                    elif stat=="std":
                        stat_arr = np.nanstd(monthly_arr, axis=0)
                    elif stat=="sum":
                        stat_arr = np.sum(monthly_arr, axis=0)
                    elif stat=="cumsum": ## precip: check if this works
                        stat_arr += np.sum(monthly_arr, axis=0)
                    out_meta.update({"transform":new_gt, "width":stat_arr.shape[2], "height":stat_arr.shape[1]})
                    with rio.open(out_name, 'w+', **out_meta) as dst:
                        dst.write(stat_arr)
                        print(out_name)

    if reproject==True:
        ## reproject to UTM32
        for file in sorted([os.path.join(out_dir, i) for i in os.listdir(out_dir) if i.endswith(".tif") and "UTM" not in i]):
            reproject_raster(out_crs="EPSG:32632", in_path=os.path.join(out_dir, file), overwrite=True)



def main():

    monthlyStat=sys.argv[1]
    VI=sys.argv[2]
    startYr=int(sys.argv[3])
    endYr=int(sys.argv[4])
    inDir=sys.argv[5]
    sensor_full=sys.argv[6]
    outDir = sys.argv[7]
    overwrite=bool(sys.argv[8])

    monthlyVI(stat=monthlyStat, VI=VI, startYr=startYr, endYr=endYr, inDir=inDir, sensor_full=sensor_full, outDir=outDir, overwrite=overwrite)
    
if __name__ == '__main__':
    main()





