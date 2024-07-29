#!/usr/bin/ python

import sys, os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import xarray as xr
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

#################################################################################
#################################################################################

def webmap_w_grid(project_name="PARAGUAY"):
    '''
    creates leaflet webmap with lat-lon coordinates in bottom-right corner 
    turn on grid layer to find what grid number an area is in. to add marker, toggle off grid layer
    
    1. project_name = grids to plot. subsets LUCinLA_grid_8858.gpkg where the project_name string is in grid's CEL_proj column
        ex) project_name="PARAGUAY" default
        other options: "CHILE", "WSA", "BRAZIL"
    '''
    LUCinLA_grid = gpd.read_file("/home/sandbox-cel/LUCinLA_grid_8858.gpkg")
    pry_grid = LUCinLA_grid[LUCinLA_grid.CEL_projec.str.contains(str(project_name))]
    pry_grid = pry_grid.set_crs(8858)
    pry_grid = pry_grid.to_crs({'init': 'epsg:4326'}) 

    ## folium map
    m = folium.Map(location=(-23.65558,-57.1495), 
                   zoom_start=10, width="%100", height="%100", epsg="EPSG4326")

    # add basemap tiles 
    tile = folium.TileLayer(tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                            attr = 'ESRI', name = 'ESRI Satellite', overlay = False, control = True, show = True).add_to(m)
    ## add lucinLA paraguay grids
    fg = folium.map.FeatureGroup(name='grid', show=False).add_to(m)
    for k, v in pry_grid.iterrows():
        poly=folium.GeoJson(v.geometry, style_function=lambda x: { 'color': 'black' , 'fillOpacity':0} )
        popup=folium.Popup("UNQ: "+str(v.UNQ)) 
        poly.add_child(popup)
        fg.add_child(poly) 

    folium.LatLngPopup().add_to(m)
    m.add_child(folium.ClickForMarker(popup="user added point- highlight coords in bottom right corner, then hover over point, then CTRL+C coords and CTRL+V into first argument of next function"))
    MousePosition().add_to(m)
    folium.plugins.Geocoder().add_to(m)
    folium.LayerControl(collapsed=False, autoZIndex=False).add_to(m)

    return m

#################################################################################

def click_to_coords(raw_coords, class_name):
    '''
    reformats lat-lon coordinates by replacing : with , 
    1. raw_coords: in quotes, "lat:lon", copied from webmap bottom-right corner. CTRL+C only when mouse is over intended pixel 
        ex) raw_coords="-26.55087 : -54.9663"
    2. in quotes, "name" of the pixel's class
        ex) class_name="mandioca"
    save click_to_coords as object, and use that object within a list as input to plot_raw_smoothed_TS()
       ex) pt1 = click_to_coords("-26.55087 : -54.9663", "mandioca")
    '''
    split_coords = raw_coords.split(":")
    new_coords = float(split_coords[0]), float(split_coords[1])
    return new_coords, class_name

#################################################################################

def transform_point_coords(inepsg, outepsg, XYcoords):
    x2,y2 = transform(Proj(init=inepsg), Proj(init=outepsg), XYcoords[1],XYcoords[0])
    return (x2,y2)

def XY_marker(feature, marker_dict):
    return marker_dict.get(feature.marker)

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    
# plot list of coords on same plot
def plot_coords_TS(web_coord_list, start_date=2020202, end_date=2021202, sandbox_dir = "/home/sandbox-cel/paraguay_lc/stac/grid/", downspout_dir="/home/downspout-cel/paraguay_lc/stac/grids/"):
    '''
    for each XYcoordinate in web_coords_list, plot_raw_smoothed_TS()
    - plots line-graph of brdf-corrected, coregistered, SMOOTHED evi2
    - plots scatter-graph of evi2 calculated from brdf-corrected, coregistered individual images
    - outputs .txt file with list of images from each point where point values are 0 or NA 
      saves file to sandbox_dir base directory. filename = "class_name" from each XYcoord separated by "_" followed by "_brdfZeroNA_check.txt"
    
    1. web_coord_list = output from click_to_coords 
        ex: [ pt1 , pt2, pt3 ]
            where pt1=click_to_coords("-26.55087 : -54.9663", "grass")
    start_date & end_date = yyyyjjj (year+juliandate)
    '''
    
    LUCinLA_grid = gpd.read_file("/home/sandbox-cel/LUCinLA_grid_8858.gpkg")
    
    qc_file = os.path.join(sandbox_dir,  "_".join([i[1] for i in web_coord_list])+"_brdfZeroNA_check.txt")
    text_file = open(qc_file, "wt")
    # matplotlib figure parameters
    fig = plt.figure()
    plt.figure(figsize=(10,5))
    ax = plt.axes()    
    plt.xticks(fontsize=10, rotation=90)        
    fig2 = plt.figure()
    plt.figure(figsize=(10,5))
    ax2 = plt.axes() 
    plt.xticks(fontsize=10, rotation=90)   
    
    # color by each coordinate pair 
    color_list = ["goldenrod", "green", "purple", "gray", "pink"]
    # initialize list to append smoothed TS max's 
    y_maxs=[]
    y_mins=[]     
    item_num = 0    
    # transform web mercator XY coordinates to coordinates in projection rasters are in
    for i in web_coord_list:
        xy_coords= transform_point_coords(inepsg="EPSG:4326", outepsg="EPSG:8858", XYcoords=i[0])
        # find grid that those coordinates are in 
        grid_number = int(LUCinLA_grid['UNQ'][LUCinLA_grid.contains( Point(xy_coords[0], xy_coords[1]) )==True])
        # combine filepath for that grid's VI rasters 
        full_unsmoothed_dir = os.path.join(sandbox_dir, str(grid_number).zfill(6), "brdf")
        full_grid_dir = os.path.join(downspout_dir, str(grid_number).zfill(6), "brdf_ts", "ms", "evi2")
        # include image if it's in the input directory, ends with .tif, & is w/in time range
        rast_list = [os.path.join(full_grid_dir, img) for img in os.listdir(full_grid_dir) if img.endswith(".tif") and (int(img.split(".")[0]) < end_date and int(img.split(".")[0]) > start_date)]
        pt_ts=[]
        date_ts=[]   
        julian_dates=[]
        for rast in sorted(rast_list):
            # grab smoothed time-series vals 
            with xr.open_dataset(os.path.join(full_grid_dir, rast) ) as xrimg:
                point = xrimg.sel(x=xy_coords[0], y=xy_coords[1], method="nearest")
                pt_ts.append(point.band_data.values[0])
            date_ts.append(rast.split("/")[-1].split(".")[0])        
        y_maxs.append(max(pt_ts))
        y_mins.append(min(pt_ts))      
        
        # grab Landsat images (end with _C01.nc) and Sentinel-2 images (end with _C01_coreg.nc)
        brdf_images = [fi for fi in os.listdir(full_unsmoothed_dir) if (fi.endswith("_C01.nc") and "S2" not in fi) or (fi.endswith("_C01_coreg.nc") and "S2" in fi)]
        # convert yyyymmdd brdf images to yyyyjjj to match smoothed images and user input start-end time
        unsmoothed_ts = []
        for img in brdf_images:
            yyyymmdd = img.split("_")[3]
            real_date = datetime.datetime.strptime(yyyymmdd, '%Y%m%d').date()
            jd = real_date.timetuple().tm_yday
            julian_date = str(img.split("_")[3][:4])+str(jd)
            if (int(julian_date) < end_date and int(julian_date) > start_date):
                unsmoothed_ts.append(os.path.join(full_unsmoothed_dir, img))
                julian_dates.append(julian_date)
        file_jd = list(zip(unsmoothed_ts, julian_dates))
        fileJD_df = pd.DataFrame(file_jd, columns = ['fname', 'jDate']).sort_values('jDate')
        
        brdf_names = [] # don't use unsmoothed_ts -- brdf_names removes points with NA
        brdf_vals = []
        brdf_maxs = []
        zero_list = []
        NAs_list = []
        JDs = []
        for k, brdf in fileJD_df.iterrows():
            # grab unsmoothed brdf time-series vals 
            with xr.open_dataset(brdf.fname) as bimg:
                point_brdf = bimg.sel(x=xy_coords[0], y=xy_coords[1], method="nearest")
                nir_val = point_brdf[str("nir")].data
                red_val =  point_brdf[str("red")].data
                if np.isnan(red_val)==True:
                    brdf_names.append(brdf.fname)   
                    brdf_vals.append(-100)
                    JDs.append(fileJD_df.jDate[k])
                else:
                    brdf_names.append(brdf.fname)
                    EVI2 = 2.5*( int(nir_val) - int(red_val)) / int((nir_val) + 2.4 * int(red_val) + 1.0 )
                    brdf_vals.append(int(EVI2*1000))
                    JDs.append(fileJD_df.jDate[k])
                    if int(nir_val) == 0 and int(red_val) == 0:
                        zero_list.append(brdf.fname)
        brdf_maxs.append(max(brdf_vals))
        marker_dict = {"LT05":"*", "LE07":"X", "LC08":"x", "S2A": "1", "S2B":"+"}
        unsmoothed_df = pd.DataFrame(  {'fname': [i.replace(full_unsmoothed_dir, "") for i in brdf_names],
                                        'jDate': [i for i in JDs],
                                        'value': brdf_vals,
                                        'sensor': [i.split("/")[-1].split("_")[1] for i in brdf_names] })   
        unsmoothed_df["marker"] = unsmoothed_df.apply(lambda x: marker_dict[x["sensor"]], axis=1)
        s = unsmoothed_df["sensor"]
        for d in range(len(unsmoothed_df)):
            ax2.scatter(unsmoothed_df.jDate[d], unsmoothed_df.value[d], color=color_list[item_num], alpha=0.5,
                       marker=marker_dict[s[d]], label=str(s[d]))
        ax.plot(date_ts, pt_ts, color=color_list[item_num], label=str(i[1]))       
        ax.legend(bbox_to_anchor=(1, 1), title="class")
        ax2.legend(bbox_to_anchor=(1, 1))
        legend_without_duplicate_labels(ax2)
        item_num+=1
        
        text_file.write(str(i[1])+" point -- images with 0 values at XYcoords "+str(i[0])+": \n")
        for li in zero_list:
            text_file.write(li)
            text_file.write("\n")
        text_file.write(str(i[1])+" point -- images with NAN values at XYcoords "+str(i[0])+": \n")
        for lin in NAs_list:
            text_file.write(lin)
            text_file.write("\n")
        text_file.write("\n")
    text_file.close()
    print(qc_file)
  
    y_ax_max=max(y_maxs)+500
    y_ax_min=min(y_mins)-500    
    ax.set_ylim([y_ax_min, y_ax_max])        
    y_ax2_max = max(brdf_maxs)+200
    ax2.set_ylim([-200, y_ax2_max])    
    ax.set_xlabel(str(start_date)[:4] + " to " + str(end_date)[:4] + ' Julian Dates', fontsize = 10)   
    ax2.set_xlabel(str(start_date)[:4] + " to " + str(end_date)[:4] + ' Julian Dates', fontsize = 10)   
    ax.set_ylabel("EVI2", fontsize = 10)      
    ax2.set_ylabel("EVI2", fontsize = 10)   

    plt.show()
    
    
    
#################################################################################
#################################################################################

def color_points(feature, CropColor_dict):
    return CropColor_dict.get(feature.CULTIVO)


def map_RAFA_per_grid(grid_number, VI="EVI2", RAFA_file="/home/lsharwood/RAFA/RAFA_2022_RUBROS_AGRICOLAS.gpkg", project_dir="/home/downspout-cel/paraguay_lc/stac/grids/"):
    
    full_grid_dir = os.path.join(project_dir, str(grid_number).zfill(6), "brdf_ts", "ms", VI.lower())
    rasters=[]
    for img in os.listdir(full_grid_dir): # include image if it's in the input directory, ends with .tif, & is w/in time range
        if img.endswith('.tif'):
            rasters.append(os.path.join(full_grid_dir,img)) 

    # get bounds and crs of first image in timeseries rast_list
    with rio.open(rasters[0]) as tmp:
        img_bounds=tmp.bounds
        img_crs=tmp.crs


    RAFA_pts_all = gpd.read_file(RAFA_file)
    all_crops = tuple(RAFA_pts_all.CULTIVO.unique())
    RAFA_pts = RAFA_pts_all.loc[(RAFA_pts_all['grid_UNQ'] == int(grid_number))]
    RAFA_pts = RAFA_pts.set_crs(RAFA_pts.crs)
    crops = tuple(RAFA_pts.CULTIVO.unique())
    crop_count = RAFA_pts.CULTIVO.value_counts()
    print('cultivo count in this grid')
    print(crop_count)        
    
    
    #RAFA_pts = RAFA_pts.to_crs(img_crs)
    RAFA_pts_web = RAFA_pts.to_crs(4326)

    colors=cm.get_cmap('Set1', len(all_crops))
    crop_colors=[]
    for i in range(colors.N):
        rgba = colors(i)
        crop_colors.append(matplotlib.colors.rgb2hex(rgba))
    CropColor_dict = dict(zip(all_crops, crop_colors))
    
    ## folium map
    mapp = folium.Map(location=(RAFA_pts_web.iloc[1,-1].y,RAFA_pts_web.iloc[1,-1].x), 
                   zoom_start=15, width="%100", height="%100", epsg="EPSG4326")
    # add basemap tiles 
    tile = folium.TileLayer(tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                            attr = 'ESRI', name = 'ESRI Satellite', overlay = False, control = True, show = True).add_to(mapp)

    marker_cluster = MarkerCluster().add_to(mapp)
    for k, r in RAFA_pts_web.iterrows():
        folium.Circle(location=[r.geometry.y, r.geometry.x], radius=15, fill=True, fill_color=color_points(r, CropColor_dict), color=color_points(r, CropColor_dict), popup='CULTIVO: <br>'+ str(r['CULTIVO']) ).add_to(marker_cluster)

    folium.LatLngPopup().add_to(mapp)
    mapp.add_child(folium.ClickForMarker(popup="user added point- highlight coords in bottom right corner, then hover over point and copy coords for next function"))
    MousePosition().add_to(mapp)

    folium.LayerControl(collapsed=False, autoZIndex=False).add_to(mapp)
    return mapp

#################################################################################
#################################################################################

## helper funcs

def random_color_feats(features):
    feat_colors=[]
    for i in range(len(features)):
        hex_color = '#%02x%02x%02x' % tuple(np.random.choice(range(256), size=3))
        feat_colors.append(hex_color)
    FeatColor_dict = dict(zip(features, feat_colors))
    return FeatColor_dict
