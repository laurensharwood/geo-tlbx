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
import rasterio as rio
from rasterio.windows import Window
from rasterio.transform import Affine

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate() # do NOT select read only scopes
    ee.Initialize()
     

## SENTINEL-2 CALIBRATION 
def get_s2_sr_cld_col(boundary_shape, start_date, end_date):
    gdf = gpd.read_file(boundary_shape)
    gdf_web = gdf.to_crs('EPSG:4326')

    aoi = ee.Geometry.Rectangle([gdf_web.bounds.minx.min(), gdf_web.bounds.miny.min(), gdf_web.bounds.maxx.max(), gdf_web.bounds.maxy.max()])

    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
         )
    s2_sr_col.aggregate_array("system:time_start").getInfo()
    s2_sr_col = s2_sr_col.map(lambda img: img.set({"DATE": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")}))
    print(s2_sr_col.aggregate_array('DATE').getInfo()[0])
    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        )
    s2_cloudless_col.aggregate_array("system:time_start").getInfo()
    s2_cloudless_col = s2_cloudless_col.map(lambda img: img.set({"DATE": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")}))


    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))
    
    
def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 20}).select('distance').mask().rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

#### COMBINE CLOUD+SHADOW MASKS
def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20}).rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
   # return img_cloud_shadow.addBands(is_cld_shdw)
    return img.addBands(is_cld_shdw)

#### APPLY MASK
def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)

#### FEATURE FUNCS 
def add_NDVI(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI') ##.multiply(100).toInt16()
    ndvi = ndvi.multiply(10000).toInt16()
    return image.addBands(ndvi) ##.multiply(10000).double()

def add_EVI(image):
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', 
        { 'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2') }).rename('EVI')
    evi = evi.multiply(10000).toInt16()
    return image.addBands(evi)

def add_MSAVI2re1(image):
    msavi2re1 = (image.select('B6').multiply(2).add(1).subtract(image.select('B6').multiply(2).add(1).pow(2)
    .subtract(image.select('B6').subtract(image.select('B4')).multiply(8)).sqrt()).divide(2)).rename('MSAVIre1')
    msavi2re1 = msavi2re1.multiply(10000).toInt16()
    return image.addBands(msavi2re1)

def add_MSAVI2re2(image):
    msavi2re2 = (image.select('B8A').multiply(2).add(1).subtract(image.select('B8A').multiply(2).add(1).pow(2)
    .subtract(image.select('B8A').subtract(image.select('B4')).multiply(8)).sqrt()).divide(2)).rename('MSAVIre2')
    msavi2re2 = msavi2re2.multiply(10000).toInt16()
    return image.addBands(msavi2re2)

def add_NDWI_openWater(image): ## of open water features
    ndwio = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndwio = ndwio.multiply(10000).toInt16()
    return image.addBands(ndwio)

def add_NDMI_1(image): ## of veg water features (w/ SWIR1)
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI1')
    ndmi = ndmi.multiply(10000).toInt16()
    return image.addBands(ndmi)

def add_NDMI_2(image): ## of open water features (w/ SWIR2)
    ndmi = image.normalizedDifference(['B8', 'B12']).rename('NDMI2')
    ndmi = ndmi.multiply(10000).toInt16()
    return image.addBands(ndmi)


################ helper funcs for brdf correction 
MAX_DISTANCE = 1000000

def create(footprint):
    return (azimuth(footprint), zenith(footprint))


def azimuth(footprint):
    upperCenter = line_from_coords(footprint, UPPER_LEFT, UPPER_RIGHT).centroid().coordinates()
    lowerCenter = line_from_coords(footprint, LOWER_LEFT, LOWER_RIGHT).centroid().coordinates()
    slope = ((y(lowerCenter)).subtract(y(upperCenter))).divide((x(lowerCenter)).subtract(x(upperCenter)))
    slopePerp = ee.Number(-1).divide(slope)
    azimuthLeft = ee.Image(PI().divide(2).subtract((slopePerp).atan()))
    return azimuthLeft.rename(['viewAz'])


def zenith(footprint):
    leftLine = line_from_coords(footprint, UPPER_LEFT, LOWER_LEFT)
    rightLine = line_from_coords(footprint, UPPER_RIGHT, LOWER_RIGHT)
    leftDistance = ee.FeatureCollection(leftLine).distance(MAX_DISTANCE)
    rightDistance = ee.FeatureCollection(rightLine).distance(MAX_DISTANCE)
    viewZenith = rightDistance.multiply(ee.Number(MAX_SATELLITE_ZENITH * 2)) \
        .divide(rightDistance.add(leftDistance)) \
        .subtract(ee.Number(MAX_SATELLITE_ZENITH)) \
        .clip(ee.Geometry.Polygon(footprint)) \
        .rename(['viewZen'])
    return degToRad(viewZenith)


    		
def create(date, footprint):
    jdp = date.getFraction('year')
    seconds_in_hour = 3600
    hourGMT = ee.Number(date.getRelative('second', 'day')) \
        .divide(seconds_in_hour)

    latRad = degToRad(ee.Image.pixelLonLat().select('latitude'))
    longDeg = ee.Image.pixelLonLat().select('longitude')

    # Julian day proportion in radians
    jdpr = jdp.multiply(PI()).multiply(2)

    a = ee.List([0.000075, 0.001868, 0.032077, 0.014615, 0.040849])
    meanSolarTime = longDeg.divide(15.0).add(ee.Number(hourGMT))
    localSolarDiff1 = value(a, 0) \
        .add(value(a, 1).multiply(jdpr.cos())) \
        .subtract(value(a, 2).multiply(jdpr.sin())) \
        .subtract(value(a, 3).multiply(jdpr.multiply(2).cos())) \
        .subtract(value(a, 4).multiply(jdpr.multiply(2).sin()))

    localSolarDiff2 = localSolarDiff1.multiply(12 * 60)

    localSolarDiff = localSolarDiff2.divide(PI())
    trueSolarTime = meanSolarTime \
        .add(localSolarDiff.divide(60)) \
        .subtract(12.0)

    # Hour as an angle
    ah = trueSolarTime.multiply(degToRad(ee.Number(MAX_SATELLITE_ZENITH * 2)))
    b = ee.List([0.006918, 0.399912, 0.070257, 0.006758, 0.000907, 0.002697, 0.001480])
    delta = value(b, 0) \
        .subtract(value(b, 1).multiply(jdpr.cos())) \
        .add(value(b, 2).multiply(jdpr.sin())) \
        .subtract(value(b, 3).multiply(jdpr.multiply(2).cos())) \
        .add(value(b, 4).multiply(jdpr.multiply(2).sin())) \
        .subtract(value(b, 5).multiply(jdpr.multiply(3).cos())) \
        .add(value(b, 6).multiply(jdpr.multiply(3).sin()))
    cosSunZen = latRad.sin().multiply(delta.sin()) \
        .add(latRad.cos().multiply(ah.cos()).multiply(delta.cos()))
    sunZen = cosSunZen.acos()

    # sun azimuth from south, turning west
    sinSunAzSW = ah.sin().multiply(delta.cos()).divide(sunZen.sin())
    sinSunAzSW = sinSunAzSW.clamp(-1.0, 1.0)

    cosSunAzSW = (latRad.cos().multiply(-1).multiply(delta.sin())
                  .add(latRad.sin().multiply(delta.cos()).multiply(ah.cos()))) \
        .divide(sunZen.sin())
    sunAzSW = sinSunAzSW.asin()

    sunAzSW = where(cosSunAzSW.lte(0), sunAzSW.multiply(-1).add(PI()), sunAzSW)
    sunAzSW = where(cosSunAzSW.gt(0).And(sinSunAzSW.lte(0)), sunAzSW.add(PI().multiply(2)), sunAzSW)

    sunAz = sunAzSW.add(PI())
    # Keep within [0, 2pi] range
    sunAz = where(sunAz.gt(PI().multiply(2)), sunAz.subtract(PI().multiply(2)), sunAz)

    footprint_polygon = ee.Geometry.Polygon(footprint)
    sunAz = sunAz.clip(footprint_polygon)
    sunAz = sunAz.rename(['sunAz'])
    sunZen = sunZen.clip(footprint_polygon).rename(['sunZen'])

    return (sunAz, sunZen)


################ tiling to 20km x 20km 
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


def main():

    startdate = sys.argv[2] ## "202210" 
    grid_file =  sys.argv[1] ## "/home/sandbox-cel/DL/vector/AOIs/NigerGrid_RCT2_UTM32.gpkg" 
    gridCells_TODO1= list(str(sys.argv[3]).replace("]","").replace("[","").split(","))
    gridCells_TODO = [int(i) for i in gridCells_TODO1]
    band_list = list(str(sys.argv[4]).replace("]","").replace("[","").split(","))


    
    start_date = startdate[:4]+"-"+startdate[4:] ## "2022-10" 
    year = start_date.split("-")[0] ## '2022'
    month = start_date.split("-")[1].zfill(2) ## '10'
    START_DATE = year+"-"+month+"-01"
    if month == "02":
        END_DATE=year+"-"+month+"-28"   
    elif month in {'04', '06', '09', '11'}:
        END_DATE=year+"-"+month+"-30"   
    else:
        END_DATE=year+"-"+month+"-31"     
    print(START_DATE)
    print(END_DATE)
    
    ####  VIs or BANDS
    ####['EVI','NDVI','NDMI1','NDMI2','MSAVIre1','MSAVIre2','NDWI']
    ###['B2','B3','B4','B6','B8','B8A','B11','B12']
    
    UNQ_CELLS=[87] ####[45,46,59,60,61,73,74,75,88,89]
    
    
    ### CLOUD and SHADOW MASK PARAMS  
    # CLOUD_FILTER: Maximum image cloud cover percent allowed in image collection
    global CLOUD_FILTER
    CLOUD_FILTER = 70 ## int(sys.argv[3]) 
    # CLD_PRB_THRESH: Cloud probability (%); values greater than are considered cloud
    global CLD_PRB_THRESH
    CLD_PRB_THRESH = 40 ## int(sys.argv[4]) 
    # NIR_DRK_THRESH: values less than are considered potential cloud shadow
    global NIR_DRK_THRESH
    NIR_DRK_THRESH = 0.15 ## float(sys.argv[5]) 
    # CLD_PRJ_DIST: Maximum distance (km) to search for cloud shadows from cloud edges
    global CLD_PRJ_DIST
    CLD_PRJ_DIST = 2 ## int(sys.argv[6]) 
    # BUFFER: Distance (m) to dilate the edge of cloud-identified objects
    global BUFFER
    BUFFER = 50 ## int(sys.argv[7]) ## BUFFER = 50 

    
    ## project arguments 
    gee_dir = sys.argv[10] ## "/home/sandbox-cel/DL/raster/S2_GEE/"
    grid_dir = sys.argv[11] ## "/home/downspout-cel/DL/raster/grids/" 

    
        
   #################################################################     
        
    ## SENTINEL-2 DOWNLOAD 
    gdf = gpd.read_file(grid_file)
    gdf = gdf.set_crs(gdf.crs)
    gdf_web = gdf.to_crs('EPSG:4326')
    gdf_web =  gdf_web[gdf_web['UNQ'].isin(gridCells_TODO)] # subset UNQ rows that are in gridCells_TODO list 
    for k, cell in gdf_web.iterrows():
        aoi = ee.Geometry.Rectangle([cell.geometry.bounds[0], cell.geometry.bounds[1], cell.geometry.bounds[2], cell.geometry.bounds[3]])
        UNQ = int(cell['UNQ'])
        
        # add date to band name 
        w_dates =  [str(i)+"_"+str(startdate) for i in band_list]
        band_date_dict = dict(zip(band_list, w_dates))
        print(band_date_dict)
        
        ## get S2 and S2cloudless
        s2_sr_cld_col = get_s2_sr_cld_col(grid_file, START_DATE, END_DATE)
        s2_sr_cld_col = s2_sr_cld_col.map(lambda img: img.set({"DATE": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")}))
        first_date = s2_sr_cld_col.aggregate_array('DATE').getInfo()[0] ## save first date for filename
        projection = s2_sr_cld_col.first().select(['B2']).projection().getInfo() ## arbitrary band, to grab CRS extent info
        s2_sr_cld_col = s2_sr_cld_col.map(lambda image: image.clip(aoi)) ## clip image collection to grid extent 
        
        ## create cloud mask from s2 collection, apply cloud mask, then create cloudless image from median value 
        s2_median = ee.Image([s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median()]) 
        
        
        out_date = str(first_date).replace("-", "")[:-2]
        month_dir = os.path.join(gee_dir, out_date)
        if not os.path.exists(month_dir):
            os.makedirs(month_dir)
            
        out_dir = os.path.join(grid_dir, str(UNQ).zfill(3))
        if not os.path.exists(out_dir):
            os.system("mkdir "+out_dir)
        
        ## FOR BAND or VIs
        if band_list[0].startswith("B"):
            ## create copy of cloud-free image
            image = ee.Image(s2_median)
        else:
            ## create image of vegetation indices 
            image = ee.Image(add_NDMI_2(add_NDMI_1(add_NDWI_openWater(add_MSAVI2re2(add_MSAVI2re1(add_NDVI(add_EVI(s2_median))))))))

        for band in band_list: # save one band at a time 
            new_name=band_date_dict.get(band)
            out_img = image.select(band).rename(new_name)

            gee_Fname = os.path.join(month_dir, "S2_"+str(new_name)+"_UNQ"+str(UNQ).zfill(3)+".tif" )
            out_Fname = os.path.join(out_dir, "S2_"+new_name+"_UNQ"+str(UNQ).zfill(3)+".tif")

            if not os.path.exists(gee_Fname.replace(".tif", "_4.tif")) and "EVI" in new_name:
                download_tiles = geemap.fishnet(aoi, rows=2, cols=2, delta=0)
                geemap.download_ee_image_tiles(image=out_img, features=download_tiles,crs=projection.get('crs'), crs_transform=projection.get('transform'),  scale=10, prefix=gee_Fname.replace(".tif", "_"))
            elif not os.path.exists(gee_Fname) and "EVI" not in new_name:
                    geemap.ee_export_image(out_img, 
                                        filename=gee_Fname, 
                                        crs=projection.get('crs'), crs_transform=projection.get('transform'),
                                        scale=10, region=aoi, file_per_band=True)


            
            if not os.path.exists(out_Fname):
                ## RETILE MINI FISHNETS (output images have a couple extra rows and columns, should be 2000 x 2000 exactly)
                mosaic_list = [os.path.join(month_dir, i) for i in os.listdir(month_dir) if (str(UNQ).zfill(3) in i and band in i) and (".vrt" not in i)]
                print(mosaic_list)
                chip_clip_shape = gdf[gdf['UNQ'] == UNQ]
                bounds = (float(chip_clip_shape.bounds['minx']), float(chip_clip_shape.bounds['maxx']), float(chip_clip_shape.bounds['miny'] ), float(chip_clip_shape.bounds['maxy']))

                grid_mosaic=out_Fname.replace(".tif", "_tmp.vrt")
                # specify no data value (np.NaN for float, -32768 for int16, -2147483648 for int32)
                vrt_options = gdal.BuildVRTOptions(srcNodata=-32768, VRTNodata=-32768)
                gdal.BuildVRT(grid_mosaic, mosaic_list, options=vrt_options)
                ## read in window of chip bounds 
                with rio.open(grid_mosaic) as src:
                    crs=src.crs
                    gt = src.transform
                    offset = windowed_read(gt, bounds)
                    grid_arr = src.read(1, window=Window(offset[0], offset[1], offset[2], offset[3]))
                    new_gt = rio.Affine(gt[0], gt[1], (gt[2] + (offset[0] * gt[0])), 0.0, gt[4], (gt[5] + (offset[1] * gt[4])))
                    with rio.open(out_Fname, "w", driver='GTiff', dtype=np.int16, nodata=-32768, height=offset[3], width=offset[2], count=1, crs=crs, transform=new_gt) as dst:  
                        dst.write(grid_arr, indexes=1)   
                        print(out_Fname)
                os.remove(grid_mosaic)
            
if __name__ == "__main__":
    main()
