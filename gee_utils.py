#!/usr/bin/ python

import os, sys, glob, time
import numpy as np
import pandas as pd
import io
import requests
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import ee
import geemap
import geemap.geemap as geemap
import localtileserver 
import jupyter_contrib_nbextensions 
import ipyleaflet
import ipywidgets as widgets

################################
## convert geopandas geodataframe to gee feature collection
################################

import shapely
from shapely.geometry import Polygon, MultiPolygon

def gdf_poly_to_ee(gdf):
    ee_shapes = []
    for k, v in gdf.iterrows():
        if type(v.geometry)==shapely.Polygon:
            ee_shapes.append(ee.Feature(ee.Geometry.Polygon(list(zip(v.geometry.exterior.xy[1], v.geometry.exterior.xy[0])))))
        elif type(v.geometry)==shapely.MultiPolygon:
            geom_list = [list(x.exterior.coords) for x in v.geometry.geoms]
            poly_area_dict = dict(zip([Polygon(i) for i in geom_list], [Polygon(i).area for i in geom_list]))
            new_geom = max(poly_area_dict, key=poly_area_dict.get)
            ee_shapes.append(ee.Feature(ee.Geometry.Polygon(tuple(zip(new_geom.exterior.xy[0], new_geom.exterior.xy[1])))))
    ee_feat_col = ee.FeatureCollection(ee_shapes)
    return ee_feat_col

################################
## GENERAL GEE IMAGE COLLECTION
################################

def gee_image_col(img_col, band_name, aoi_shape, region=None):
    if type(aoi_shape)==str:
        gdf = gpd.read_file(aoi_shape)
    gdf = gdf.set_crs(gdf.crs)
    gdf_web = gdf.to_crs("EPSG:4326")
    if "grid" in aoi_shape:
        gdf_web = gdf_web[gdf_web["region"] == str(region)]
    aoi = ee.Geometry.Rectangle([gdf_web.bounds.minx.min(), gdf_web.bounds.miny.min(), gdf_web.bounds.maxx.max(), gdf_web.bounds.maxy.max()])
    collection = ee.ImageCollection(img_col).filterBounds(aoi).select(band_name)
    clip_col = collection.map(lambda col : col.clip(aoi))
    return clip_col

def gee_image_time_series(img_col, start, end, aoi_shape, region=None):
    if type(aoi_shape)==str:
        gdf = gpd.read_file(aoi_shape)
    gdf = gdf.set_crs(gdf.crs)
    gdf_web = gdf.to_crs("EPSG:4326")
    if "grid" in aoi_shape:
        gdf_web = gdf_web[gdf_web["region"] == str(region)]
    aoi = ee.Geometry.Rectangle([gdf_web.bounds.minx.min(), gdf_web.bounds.miny.min(), gdf_web.bounds.maxx.max(), gdf_web.bounds.maxy.max()])
    collection = ee.ImageCollection(img_col).filterBounds(aoi).filterDate(start, end)
    clip_col = collection.map(lambda col : col.clip(aoi))
    return clip_col

################################
## PLANETSCOPE TIME SERIES
################################

def planet_monthly_timeseries(region_num, chip_shape, poly_shape, start_date, end_date, plot_YYYYMM, NICFI_prod="projects/planet-nicfi/assets/basemaps/americas", export_composite=False):
    '''
    region_num = chip region number
    chip_shape = full file path to user_train chip region shapefile 
    poly_shape = full file path to user_train poly region shapefile 
    start_date = planet monthly time series start date 
    end_date = planet monthly time series end date 
    plot_YYYYMM = display 3 dates(YYYY-MM) in a list as RGB 
    export_composite = folder to export planet monthly NDVI timeseries 
    '''
    start = time.time()
    poly_gdf = gpd.read_file(poly_shape)
    poly_gdf.crs = "EPSG:8858"
    web_polys = poly_gdf.to_crs("EPSG:4326")
    web_polys = web_polys[web_polys["region"] == str(region_num)]
    gdf = gpd.read_file(chip_shape)
    gdf.crs = "EPSG:8858"
    gdf_web = gdf.to_crs("EPSG:4326")
    gdf_web = gdf_web[gdf_web["region"] == str(region_num)]
    #for Polygon geo data type
    geoms = [i for i in web_polys.geometry]
    geomss = [i for i in gdf_web.geometry]
    features=[]
    for i in range(len(geoms)):
        g = [i for i in web_polys.geometry]
        x,y = g[i].exterior.coords.xy
        cords = np.dstack((x,y)).tolist()
        g=ee.Geometry.Polygon(cords)
        feature = ee.Feature(g)
        features.append(feature)
        ee_object = ee.FeatureCollection(features)
    featuress=[]
    for ii in range(len(geomss)):
        ge = [i for i in gdf_web.geometry]
        xx,yy = ge[ii].exterior.coords.xy
        cordss = np.dstack((xx,yy)).tolist()
        ge=ee.Geometry.Polygon(cordss)
        featuree = ee.Feature(ge)
        featuress.append(featuree)
        chip_ee_object = ee.FeatureCollection(featuress)
    planet_NDVI = gee_image_time_series(gee_name=NICFI_prod, 
                 start=start_date, end=end_date, 
                 aoi_shape=chip_shape, 
                 region=region_num).map(get_ndvi).toBands()
    YYYYMM_bands = ['planet_medres_normalized_analytic_'+str(i)+'_mosaic_NDVI' for i in plot_YYYYMM]
    band_list = [i['id'] for i in planet_NDVI.getInfo()['bands']]
    plotNDVIbands = [i for i in band_list if i in YYYYMM_bands]
    planet_NDVI=ee.Image(planet_NDVI).rename(band_list)
    if not export_composite == False:
        if not os.path.exists(export_composite):
            os.makedirs(export_composite)
        out_file = os.path.join(export_composite, "PS_monthly_"+str(region_num)+".tif")
        planet_NDVI_export = planet_NDVI.resample('bilinear').reproject(crs='EPSG:8858', scale=4.7)
        geemap.ee_export_image(planet_NDVI, out_file, scale=4.7) 
    Map = geemap.Map(center=(float(gdf_web.geometry.centroid.y.values), float(gdf_web.geometry.centroid.x.values)), zoom=15)
    #Map.add_basemap('HYBRID')
    Map.add_tile_layer('http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}', name='Google Satellite', attribution='Google')
    Map.addLayer(planet_NDVI, {'bands':plotNDVIbands, 'min': 0.0, 'max': 1.0, 'gamma':[1,1,1]}, name="Planet monthly NDVI", shown=True, opacity=1.0)
    Map.addLayer(chip_ee_object.style(**{'color': '000000', 'fillColor': '00000000'}), {},"Training chips")
    Map.addLayer(ee_object.style(**{'color': 'maroon', 'fillColor': '00000000'}), {},"Training digitizations")
    end = time.time()
    print('this took '+str(float(end)-float(start))+' seconds')
    Map
    return Map

def planet_ndvi_metrics(region, aoi_shape, plot_YYYYMM, out_dir):
    gdf = gpd.read_file(aoi_shape)
    gdf_web = gdf.to_crs("EPSG:4326")
    gdf_web = gdf_web[gdf_web["UNQ"] == region]
    geom = ee.Geometry.Rectangle([gdf_web.bounds.minx.min(), gdf_web.bounds.miny.min(), gdf_web.bounds.maxx.max(), gdf_web.bounds.maxy.max()])
    Map = geemap.Map(center=(float(gdf_web.geometry.centroid.y), float(gdf_web.geometry.centroid.x)), zoom=15)
    #Map.add_basemap('HYBRID')
    Map.add_tile_layer('http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}', name='Google Satellite', attribution='Google')
    RGB = gee_download(gee_name=NICFI_prod,
                 start=start_date, end=end_date,
                 aoi_shape=chip_shape,
                 region=region)   #.map(add_ndvi)
    planet_VI = gee_download(gee_name=NICFI_prod,
                 start=start_date, end=end_date,
                 aoi_shape=chip_shape,
                 region=region).map(get_ndvi).toBands()
    YYYYMM_bands = ['planet_medres_normalized_analytic_'+str(i)+'_mosaic_NDVI' for i in plot_YYYYMM]
    band_list = [i['id'] for i in planet_VI.getInfo()['bands']]
    plotNDVIbands = [i for i in band_list if i in YYYYMM_bands]
    print(plotNDVIbands)
    Map.addLayer(planet_VI, {'bands':plotNDVIbands, 'min': 0.0, 'max': 1.0, 'gamma': [0.5, 0.5, 0.5]}, name="Planet monthly NDVI", shown=True, opacity=1.0)
    Map.add_time_slider(RGB, {"min": 10, "max": 5000, 'bands':['R', 'G', 'B']}, name="Planet Time Series", shown=True)
    ## single month
    planet1 = ee.Image(planet_VI.select(YYYYMM_bands[2]).rename(YYYYMM_bands[2]).multiply(10000))
    ## max value
    planet2 = ee.Image(planet_VI.reduce(ee.Reducer.percentile([90])).multiply(10000)).subtract(ee.Image(planet_VI.reduce(ee.Reducer.percentile([10]))).multiply(10000))
    ## normalized difference btwn two months (first two plotted months)
    planet3 = planet_VI.normalizedDifference([YYYYMM_bands[0],YYYYMM_bands[1]]).multiply(10000)
    ## planet4 = ee.Image(planet_VI.select(period_bands).reduce(ee.Reducer.kurtosis()))
    planet_feats = planet1.addBands(planet2).addBands(planet3).clip(geom).int16()
    out_fi = os.path.join(out_dir, "PSfeats_"+str(endYR)+"_"+str(region)+".tif")
    ## planet_export = planet_feats.resample('bilinear').reproject(crs="EPSG:8858", scale=4.7)
    geemap.ee_export_image(planet_feats, out_fi, crs="EPSG:3857") #, scale=4.7
    return Map, out_fi

## JUST DOWNLOAD CHIPS 
def get_monthly_timeseries(region_num, chip_shape, start_date, end_date, plot_YYYYMM, export_composite=False):
    start = time.time()
    gdf = gpd.read_file(chip_shape)
    gdf.crs = "EPSG:8858"
    gdf_web = gdf.to_crs("EPSG:4326")
    gdf_web = gdf_web[gdf_web["region"] == str(region_num)]
    #for Polygon geo data type
    geoms = [i for i in gdf_web.geometry]
    features=[]
    for ii in range(len(geoms)):
        ge = [i for i in gdf_web.geometry]
        x,y = ge[ii].exterior.coords.xy
        cords = np.dstack((x,y)).tolist()
        ge=ee.Geometry.Polygon(cords)
        feature = ee.Feature(ge)
        features.append(feature)
        chip_ee_object = ee.FeatureCollection(features)
    planet_NDVI = get_gee(gee_name=NICFI_prod, 
                 start=start_date, end=end_date, 
                 aoi_shape=chip_shape, 
                 region=region_num).map(get_ndvi).toBands()
    YYYYMM_bands = ['planet_medres_normalized_analytic_'+str(i)+'_mosaic_NDVI' for i in plot_YYYYMM]
    band_list = [i['id'] for i in planet_NDVI.getInfo()['bands']]
    plotNDVIbands = [i for i in band_list if i in YYYYMM_bands]
    planet_NDVI=ee.Image(planet_NDVI).rename(band_list)
    if not os.path.exists(export_composite):
        os.makedirs(export_composite)
    out_file = os.path.join(export_composite, "PS_monthly_"+str(region_num)+".tif")
    planet_NDVI_export = planet_NDVI.resample('bilinear').reproject(crs='EPSG:8858', scale=4.7)
    geemap.ee_export_image(planet_NDVI, out_file, scale=4.7) 
    print(out_file)

def export_digitizations(region, feature_collection, class_list):
    '''
    region = chip region number (GGGGNN)
    feature_collection = feature collection from list of new polygon digitizations (in order drawn)
    class_list = list of classes for new polygon digitizations (in order drawn)
    lookup_table = py class lookup table for recoding
    '''
    new_polys  = ee.FeatureCollection(region_map.draw_features)
    if (len(list(set(class_list))) == 1) and (1 in class_list) :
        print('only mono crops')
        new_polys = new_polys.set("class", 1)
    return Map


################################
## SENTINEL2 CLOUD / SHADOW MASKING 
################################

def get_s2_sr_cld_col(boundary_shape, start_date, end_date):
    gdf = gpd.read_file(boundary_shape)
    gdf_web = gdf.to_crs('EPSG:4326')
    aoi = ee.Geometry.Rectangle([gdf_web.bounds.minx.min(), gdf_web.bounds.miny.min(), gdf_web.bounds.maxx.max(), gdf_web.bounds.maxy.max()])
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(aoi).filterDate(start_date, end_date).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
    s2_sr_col.aggregate_array("system:time_start").getInfo()
    s2_sr_col = s2_sr_col.map(lambda img: img.set({"DATE": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")}))
    print(s2_sr_col.aggregate_array('DATE').getInfo()[0])
    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterBounds(aoi).filterDate(start_date, end_date))
    s2_cloudless_col.aggregate_array("system:time_start").getInfo()
    s2_cloudless_col = s2_cloudless_col.map(lambda img: img.set({"DATE": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")}))
    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{'primary': s2_sr_col,
                                                                        'secondary': s2_cloudless_col,
                                                                        'condition': ee.Filter.equals(**{'leftField': 'system:index', 'rightField': 'system:index' })}))

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

def add_cld_shdw_mask(img):
    # Add cloud component bands
    img_cloud = add_cloud_bands(img) 
    # Add cloud shadow component bands
    img_cloud_shadow = add_shadow_bands(img_cloud) 
    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0) 
    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input. 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20).reproject(**{'crs': img.select([0]).projection(), 'scale': 20}).rename('cloudmask'))
    # Add the final cloud-shadow mask to the image and return img_cloud_shadow.addBands(is_cld_shdw)
    return img.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()
    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


################################
## CREATE VEGETATION INDICES
################################

def get_ndvi(image):
    """returns just the new band--NDVI-- a measure of vegetation greenness"""
    ndvi = image.normalizedDifference(['N','R']).rename("NDVI")
    return(ndvi)

def add_ndvi(image):
    """returns image with new band--NDVI-- a measure of vegetation greenness"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndvi = ndvi.multiply(10000).toInt16()
    return image.addBands(ndvi)

def add_evi(image):
    """returns image with new band--EVI-- a measure of vegetation greenness"""
    evi = image.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', { 'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2') }).rename('EVI')
    evi = evi.multiply(10000).toInt16()
    return image.addBands(evi)

def add_msavi_i(image):
    """returns image with new band--MSAVI(from the B6 red-edge band)-- a measure of soil moisture"""
    msavi2re1 = (image.select('B6').multiply(2).add(1).subtract(image.select('B6').multiply(2).add(1).pow(2).subtract(image.select('B6').subtract(image.select('B4')).multiply(8)).sqrt()).divide(2)).rename('MSAVIre1')
    msavi2re1 = msavi2re1.multiply(10000).toInt16()
    return image.addBands(msavi2re1)

def add_msavi_ii(image):
    """returns image with new band--MSAVI(from the B8A red-edge band)-- a measure of soil moisture"""
    msavi2re2 = (image.select('B8A').multiply(2).add(1).subtract(image.select('B8A').multiply(2).add(1).pow(2).subtract(image.select('B8A').subtract(image.select('B4')).multiply(8)).sqrt()).divide(2)).rename('MSAVIre2')
    msavi2re2 = msavi2re2.multiply(10000).toInt16()
    return image.addBands(msavi2re2)

def add_ndwi_openwater(image):
    """returns image with new band--NDWI-- a measure of wetness of open water bodies"""
    ndwio = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndwio = ndwio.multiply(10000).toInt16()
    return image.addBands(ndwio)

def add_ndmi_i(image): 
    """returns image with new band--NDMI(w/ SWIR1)-- which measures ___  """
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI1')
    ndmi = ndmi.multiply(10000).toInt16()
    return image.addBands(ndmi)

def add_ndmi_ii(image): ## 
    """returns image with new band--NDMI(w/ SWIR2)-- which measures ___  """
    ndmi = image.normalizedDifference(['B8', 'B12']).rename('NDMI2')
    ndmi = ndmi.multiply(10000).toInt16()
    return image.addBands(ndmi)

