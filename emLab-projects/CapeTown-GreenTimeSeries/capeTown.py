#!/usr/bin/ python

import os, sys
import time
import numpy as np
import xarray as xr
import rasterio as rio
from rasterio.features import shapes
import pandas as pd
import geopandas as gpd
import shapely
from shapely import geometry
from shapely.geometry import Polygon, Point
from pyproj import Proj, transform
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from leafmap import leafmap
import folium
from folium.plugins import MousePosition, MarkerCluster
from folium.map import Marker
import geopandas as gpd


def monthly_landsat(inDir, VI, overwrite=False):
    
    ## project params 
    epsg=32734
    stat='mean'
    out_dir='/home/sandbox-cel/capeTown/monthly/landsat'
    if os.path.basename(out_dir) == 'landsat':
        sensor = 'LC'
    if os.path.basename(out_dir) == 'sentinel2':
        sensor = 'S2'        
            
    grid = inDir.split('/')[5]
    print(grid)
    out_full_dir = os.path.join(out_dir, grid)
    if not os.path.exists(out_full_dir):
        os.makedirs(out_full_dir)

    for year in list(range(2016, 2022, 1)):
        for month in list(range(1, 13, 1)):
            start_date = str(year)+str(month).zfill(2)

            out_fi = os.path.join(out_full_dir, sensor+'_'+VI+'_'+start_date+'_UNQ'+grid+'.tif')
            if (not os.path.exists(out_fi)) | (os.path.exists(out_fi) and overwrite==True):
                monthly_fis = sorted([i for i in os.listdir(inDir) if i.startswith('L3A_'+sensor[:-1]) and '_'+start_date in i])
                stack = []
                for file in [os.path.join(inDir, fi) for fi in monthly_fis]:
                    ds = xr.open_dataset(file)
                    ds = ds.assign_coords(ds.coords)
                    ds = ds.rio.write_crs(epsg)
                    gt = [i for i in ds.transform]
                    noData = ds.nodatavals[0]
                    if VI=='ndvi':
                        band_arr = (ds.data_vars.get('nir')-ds.data_vars.get('red'))/(ds.data_vars.get('nir')+ds.data_vars.get('red'))
                    elif VI=='evi':
                        band_arr = 2.5*(ds.data_vars.get('nir')-ds.data_vars.get('red'))/(ds.data_vars.get('nir') + (2.4*ds.data_vars.get('red')) + 1)
                    elif VI=='ndmi1':
                        band_arr = (ds.data_vars.get('nir')-ds.data_vars.get('swir1'))/(ds.data_vars.get('nir')+ds.data_vars.get('swir1'))
                    elif VI=='ndmi2':
                         band_arr = (ds.data_vars.get('nir')-ds.data_vars.get('swir2'))/(ds.data_vars.get('nir')+ds.data_vars.get('swir2'))
                    elif VI=='ndwi':
                        band_arr = (ds.data_vars.get('green')-ds.data_vars.get('nir'))/(ds.data_vars.get('green')+ds.data_vars.get('nir'))
                    stack.append(band_arr)

                    if len(stack) >= 1:
                        if stat == 'mean':
                            monthly = np.nanmean(stack, axis=0)
                        elif stat == 'median':
                            monthly = np.nanmedian(stack, axis=0)
                            
                    if np.max(monthly) >= 1:
                        monthly[(monthly >= 1) | (monthly == np.inf) ] = 1
                    if np.min(monthly) <= -1:
                        monthly[(monthly <= -1) | (monthly == np.inf*-1) ] = -1
                    with rio.open(out_fi,'w',  driver='GTiff', width=monthly.shape[0], height=monthly.shape[1], count=1, crs=epsg, transform=gt, dtype=np.float32, nodata = noData) as dst:
                            dst.write(monthly, indexes=1)
                    print(out_fi)
                    
def raster_to_vector(input_raster):
    with rio.open(input_raster, 'r') as tmp:
        rast = tmp.read(1)
        rast_crs = tmp.crs
    mask = None
    instance_shapes = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes(rast, mask=mask, transform=tmp.transform)))
    vectorized  = gpd.GeoDataFrame.from_features(list(instance_shapes))
    vectorized = vectorized.set_crs(rast_crs)
    return vectorized

def weighted_avg(gdf, IDcol, inRast):
    UNQ = os.path.basename(inRast).split("_")[-1].replace(".tif", "")
    col_name = '-'.join(os.path.basename(inRast).split("_")[:3])    
    print(col_name)
    vector_pixels = raster_to_vector(inRast)
    with rio.open(inRast) as tmp:
        UNQbounds = tmp.bounds
    gdf_sub = gdf.within(UNQbounds)
    gdf_sub['area_sqm'] = gdf_sub.area
    intrsect = vector_pixels.overlay(gdf_sub, how='intersection')
    intrsect[col_name] = (intrsect.area/intrsect['area_sqm'])*intrsect['raster_val']
    sums = intersections.groupby(by=[IDcol]).sum(numeric_only=True)[col_name]
    gdf_join = gdf_sub.set_index(IDcol).join(sums, on=IDcol)
    gdf_join.reset_index().to_csv(os.path.join(outDir, col_name+"_"+str(UNQ).zfill(2)+".csv"))
    return gdf_join

def extract_pts(gdfPts, inRast):
    col_name = '-'.join(os.path.basename(inRast).split("_")[:3])
    gdf = gdfPts.copy()
    coords_list = list(zip([i.x for i in gdf.geometry.to_list()],  [i.y for i in gdf.geometry.to_list()]))
    with rio.open(inRast) as src:
        gdf[col_name] = [x[0] for x in src.sample(coords_list)]
        return gdf

################### plot time series 

def webmap_of_grid(grid_file):
    '''
    creates leaflet webmap with lat-lon coordinates in bottom-right corner 
    turn on grid layer to find what grid number an area is in. to add marker, toggle off grid layer
    '''
    grid = gpd.read_file(grid_file)

    if grid_file.endswith("LUCinLA_grid_8858.gpkg"):
        grid = grid[grid.CEL_projec.str.contains(str("PARAGUAY"))]
        grid = grid.set_crs(8858)
        grid = grid.to_crs({'init': 'epsg:4326'}) 
    elif grid_file.endswith("AI4B_grid_UTM31N.gpkg"):
        grid = grid.set_crs(32631)
        grid = grid.to_crs({'init': 'epsg:4326'})
    elif grid_file.endswith("cape_grid_utm32S.gpkg"):
        grid = grid.set_crs(32734)
        grid = grid.to_crs({'init': 'epsg:4326'})
                
    ## folium map
    m = folium.Map(location=(grid.iloc[0].geometry.centroid.y, grid.iloc[0].geometry.centroid.x), 
                   zoom_start=5, width="%100", height="%100", epsg="EPSG4326")

    # add basemap tiles 
    tile = folium.TileLayer(tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                            attr = 'ESRI', name = 'ESRI Satellite', overlay = False, control = True, show = True).add_to(m)
    ## add lucinLA paraguay grids
    fg = folium.map.FeatureGroup(name='grid', show=False).add_to(m)
    for k, v in grid.iterrows():
        poly=folium.GeoJson(v.geometry, style_function=lambda x: { 'color': 'black' , 'fillOpacity':0} )
        popup=folium.Popup("UNQ: "+str(v.UNQ)) 
        poly.add_child(popup)
        fg.add_child(poly) 

    folium.LatLngPopup().add_to(m)
    m.add_child(folium.ClickForMarker(popup="user added point- highlight coords in bottom right corner, then hover over point and CTRL+C / CTRL+V into for next function"))
    folium.LayerControl(collapsed=False, autoZIndex=False).add_to(m)
    return m


def click_to_coords(raw_coords, class_name):
    split_coords = raw_coords.split(":")
    new_coords = float(split_coords[0]), float(split_coords[1])
    return new_coords, class_name

def transform_point_coords(inepsg, outepsg, XYcoords):
    x2,y2 = transform(Proj(init=inepsg), Proj(init=outepsg), XYcoords[1],XYcoords[0])
    return (x2,y2)

def XY_marker(feature, marker_dict):
    return marker_dict.get(feature.marker)

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    
def random_color_feats(features):
    feat_colors=[]
    for i in range(len(features)):
        hex_color = '#%02x%02x%02x' % tuple(np.random.choice(range(256), size=3))
        feat_colors.append(hex_color)
    FeatColor_dict = dict(zip(features, feat_colors))
    return FeatColor_dict


# plot list of coords on same plot
def plot_TS_filled(grid_file, web_coord_list, VI):
    grid = gpd.read_file(grid_file)

    sandbox_dir="/home/sandbox-cel/capeTown/stac_grids"
    vrt_dir="/home/sandbox-cel/capeTown/monthly/landsat/vrts/"     
    crs="EPSG:32734"        
    # matplotlib figure parameters
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes()    
    # color by each coordinate pair 
    color_list = ["goldenrod", "green", "purple", "gray", "pink"]
    # initialize list to append smoothed TS max's 
    y_maxs=[]
    y_mins=[]     
    item_num = 0    
    # transform web mercator XY coordinates to coordinates in projection rasters are in
    rast_list = sorted([os.path.join(vrt_dir, img) for img in os.listdir(vrt_dir) if img.endswith(".vrt") and VI+"_20" in img])
    date_ts=[i[:-4].split("_")[-1] for i in rast_list]   

    for i in web_coord_list:
        TS_list=[]
        xy_coords= transform_point_coords(inepsg="EPSG:4326", outepsg=crs, XYcoords=i[0])
        for rast in rast_list:
            with xr.open_dataset(rast ) as xrimg:
                point = xrimg.sel(x=xy_coords[0], y=xy_coords[1], method="nearest")
                TS_list.append(point.band_data.values[0])
            y_maxs.append(max(TS_list))
            y_mins.append(min(TS_list))      
        
        ax.plot(date_ts, TS_list, color=color_list[item_num], label='_nolegend_')   ##
        item_num+=1
        
    y_ax_max=max(y_maxs)
    y_ax_min=min(y_mins)   
    ax.set_ylim([y_ax_min, y_ax_max])   
    ax.set_xticklabels(date_ts, rotation=45, ha='right', fontsize= 6 )
    ax.set_xlabel(date_ts[0] + " to " + date_ts[-1], fontsize = 10)   
    ax.set_ylabel(VI, fontsize = 10)      
    ax.legend(loc='upper right')
    return fig



def main():
    
    startYr = sys.argv[1] ## 2016
    endYr = sys.argv[2] ##  2021
    inShp = sys.argv[3] ## '/home/sandbox-cel/vector/___.json"
    IDcol = sys.argv[4] ## 'liskey'
    month_dir = sys.argv[5] ## '/home/sandbox-cel/capeTown/monthly/landsat'
    VI = sys.argv[6]
    
    gdf = gpd.read_file(inShp)

    for year in list(range(startYr, endYr+1, 1)):
        for month in list(range(1, 13, 1)):
            date = str(year)+str(month).zfill(2)
            grids = sorted([os.path.join(month_dir, g) for g in os.listdir(month_dir) ])
            for grid_dir in grids:
                monthlyVI = sorted([os.path.join(grid_dir, file) for file in os.listdir(grid_dir) if date in file and VI in file])
                if len(monthlyVI) == 1:
                    inRast = monthlyVI[0]
                    print(inRast)
                    start = time.time()
                   ## prop_green = weighted_avg(gdf=gdf, IDcol=IDcol, inRast=inRast)
                   ## prop_green = extract_pts(gdfPts=gdf, inRast=inRast)
                    end = time.time()
                    print(end - start)

        
if __name__ == "__main__":
    main()




