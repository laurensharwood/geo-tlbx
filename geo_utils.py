import os
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, Point, LineString
import rasterio as rio
import datetime
from datetime import datetime, timedelta, date, timezone
## under RASTER TO VECTOR 
from rasterio.windows import Window
from rasterio.features import shapes
## under GEO, for transform_point_coords
from pyproj import Proj, transform
## under GEO, for address_to_yx
import pygris
## under EXTRACT RASTER, for zonal_time_series function
from rasterstats import zonal_stats
from haversine import haversine
## parsing TCX/GPX functions
import gpxpy
from tcxreader.tcxreader import TCXReader
## under TCX/GPX, adding to postgreSQL db
import psycopg2
## under CENSUS
from pygris import counties, tracts, block_groups, blocks, validate_state
from pygris.utils import erase_water
from pygris.data import get_census
## under OPENSTREETMAP
import overpy
## routes
import openrouteservice
from openrouteservice import convert
## geo-tlbx functions/modules
import raster_utils as ru


################################
## RASTER TO VECTOR
################################

def raster_to_vector(inRast):
    """converts 'inRast' raster to vector format where pixels of the same value are grouped to a single polygon and have an attribute of that raster value; returns vector polygon geodataframe"""
    with rio.open(inRast, 'r') as tmp:
        rast = tmp.read(1)
        rast_crs = tmp.crs
    img_polys = gpd.GeoDataFrame.from_features(
        list({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes(rast, mask=None, transform=tmp.transform))), crs = rast_crs)
    return img_polys

def rast_bounds_to_vector(inRast):
    """helper function used in extract_TV_polygons function"""
    geo = rio.open(inRast)
    w, s, e, n = geo.bounds
    img_poly = Polygon([(w, n), (e, n), (e, s), (w, s)])
    return img_poly

################################
## EXTRACT RASTER
################################
    
def extract_pts(gdfPts, inRast):
    """adds a new column onto 'gdfPts', matching'inRast' file name and has the value of inRast at each point/row location"""
    col_name = '-'.join(os.path.basename(inRast).split("_")[:3])
    gdf = gdfPts.copy()
    coords_list = list(zip([i.x for i in gdf.geometry.to_list()],  [i.y for i in gdf.geometry.to_list()]))
    with rio.open(inRast) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        gdf[col_name] = [x[0] for x in src.sample(coords_list)]
    return gdf

def line_pofile(inRast, coords):
    """uses 'inRast':elevation raster/DEM to calculate the profile (distance vs. altitude) of a path segment from 'coords': the list of coordinates. 
    returns two items: the total distance (in miles) and elevation (in feet)"""
    # coordinates are [lon, lat], flip for rasterio
    coords = [[c[1], c[0]] for c in coords]
    with rio.open(inRast) as src:
        elev = [x[0]*3.28084  for x in src.sample(coords)]
    dist = [0.0]
    for j in range(len(coords) - 1):
        # use haversine distance
        dist.append(dist[j] + haversine((coords[j][1], coords[j][0]), (coords[j + 1][1], coords[j + 1][0]), Unit.MILES))
    return dist, elev

def zonal_stat(instance_shape, in_rast, stat="mean"): 
    """
    instance_shape = polygon shapefile to calculate raster stat within. saves output shape in that directory with _TS appended 
    rast_dir = input raster directory, where each raster's name must be called YYYYJD.tif 
    start_date & end_date format YYYYJD
    stat options: mean, median, std, var"""
    with rio.open(in_rast) as tmp:
        rast_crs = tmp.crs
    polys = gpd.read_file(instance_shape) 
    if polys.crs == rast_crs:
        polys_stat = zonal_stats(instance_shape, in_rast, stats=[stat], geojson_out=True)
        for p in polys_stat: 
            p['properties'][str(os.path.basename(in_rast).split(".")[0])+stat] = p['properties'].pop(stat)
        feat = {'type': 'FeatureCollection', 'features':pd.DataFrame(polys_stat)}
        polys_stat_gdf = gpd.GeoDataFrame.from_features(feat).set_crs(rast_crs, allow_override=True).dropna()
    return polys_stat_gdf

def zonal_time_series(instance_shape, rast_list, stat="mean"): 
    """
    instance_shape = polygon shapefile to calculate raster stat within
    rast_dir = input raster directory, where each raster's name must be called YYYYJD.tif 
    start_date & end_date format YYYYJD (year-julian date)
    stat options: mean, median, std, var"""
    stat_dates = []
    with rio.open(rast_list[0]) as tmp:
        rast_crs = tmp.crs
    polys = gpd.read_file(instance_shape)
    if polys.crs == rast_crs:
        for in_rast in rast_list:
            stat_dates.append(zonal_stat(instance_shape, in_rast, stat))
        stats_dates_gdf = pd.concat(stat_dates)

        return stats_dates_gdf
    else:
        return 'reproject polygons'

def poly_area_weight_avg(gdf, IDcol, inRast):
    '''
    raster - polygon area weighted average 
    args:
        gdf = polygon geodataframe
        IDcol = 
        inRast = input raster filename
    returns geodataframe with ____
    '''
    col_name = '-'.join(os.path.basename(inRast).split("_")[:3])    
    print(col_name)
    vector_pixels = raster_to_vector(inRast)
    with rio.open(inRast) as tmp:
        UNQbounds = tmp.bounds
    gdf_sub = gdf.within(UNQbounds)
    gdf_sub['area_sqm'] = gdf_sub.area
    intrsect = vector_pixels.overlay(gdf_sub, how='intersection')
    intrsect[col_name] = (intrsect.area/intrsect['area_sqm'])*intrsect['raster_val']
    sums = intrsect.groupby(by=[IDcol]).sum(numeric_only=True)[col_name]
    gdf_join = gdf_sub.set_index(IDcol).join(sums, on=IDcol)
    return gdf_join

################################
## CENSUS 
################################
## cb = True to obtain a cartographic boundary shapefile pre-clipped to the US shoreline


## get primary roads 
def get_roads_by_county(county, state):
    roads = pygris.roads(state = state, county = county, cache = True)
    ## roads.explore()
    return roads

def get_census_var_gdf(acs_var_id, acs_yr, hierarch_region, county_name, state_abr):
    '''
    hierarch_region options: tract or block group for region
    '''
    shape = pygris.tracts(county = county_name, state = state_abr, cb = True)
    census_var = get_census(dataset = "acs/acs5", variables = acs_var_id, year = acs_yr, 
                            params = {"for":hierarch_region+":*", "in": f"state:{validate_state(state_abr)}"},
                            guess_dtypes = True, return_geoid = True)
    census_join = shape.set_index("GEOID").join(census_var.set_index("GEOID")).reset_index()
    return census_join

## by county
def get_county_blocks(county, state_abr):
    county_shape = pygris.blocks(state = state_abr, county = county)
    return county_shape

def get_county_block_groups(county, state_abr):
    county_shape = pygris.block_groups(state = state_abr, county = county, cb = True)
    return county_shape

def get_county_tracts(county, state_abr):
    county_shape = pygris.tracts(state = state_abr, county = county, cb = True)
    return county_shape
    
## by state
def get_state_counties(state_abr):
    shapes = counties(state = validate_state(state_abr), cb = True, cache = True)
    return shapes
    
def get_state_tracts(state_abr):
    shapes = tracts(state = validate_state(state_abr), cb = True, cache = True)
    return shapes

def get_state_block_groups(state_abr):
    shapes = tracts(state = validate_state(state_abr), cb = True, cache = True)
    return shapes

def get_state_blocks(state_abr):
    shapes = blocks(state = validate_state(state_abr), cb = True, cache = True)
    return shapes

## by buffered distance of address 
def get_tracts_groups_by_addr_buff(addr, buff_dist):
    shapes = tracts(state = validate_state(str(addr[-8:-6])), cb = True, subset_by = {addr: buff_dist}, cache = True)
    return shapes

def get_block_groups_by_addr_buff(addr, buff_dist):
    shapes = block_groups(state = validate_state(str(addr[-8:-6])), cb = True, subset_by = {addr: buff_dist}, cache = True)
    return shapes

def get_blocks_by_addr_buff(addr, buff_dist):
    shapes = blocks(state = validate_state(str(addr[-8:-6])), subset_by = {addr: buff_dist}, cache = True)
    return shapes


################################
## OPENSTREETMAP 
#########################s#######

def osm_poi(bbox, amenity):
    '''
    bbox = (x_min, y_min, x_max, y_max) *tuple in 4326
    ex) x_min, y_min, x_max, y_max = tuple([i for i in county_shp.to_crs(4326).total_bounds])
    amenity="drinking_water"
    returns geopandas geodataframe of all OpenStreetMap amenities within bounding box tuple 
    '''

    api = overpy.Overpass()
    # fetch all ways and nodes
    result = api.query("""node(%f,%f,%f,%f)""" % bbox + "['amenity'='"+amenity+"']; (._;>;); out body; ")
    pois=[]
    for pt in result.nodes:
        pois.append((pt.lat, pt.lon))
    df = pd.DataFrame([pois]).T
    df.columns=[amenity]
    gdf = gpd.GeoDataFrame(pd.DataFrame(list(range(0,len(df),1))), geometry=gpd.points_from_xy([float(i[1]) for i in df[amenity]], [float(i[0]) for i in df[amenity]], crs=4326))
    gdf.columns = ['index', 'geometry']
    return gdf 

def osm_lines(bbox, waytype):
    '''
    bbox = (x_min, y_min, x_max, y_max) *tuple in 4326
    ex) x_min, y_min, x_max, y_max = tuple([i for i in county_shp.to_crs(4326).total_bounds])
    waytype = highway, waterway
    returns geopandas geodataframe of all OpenStreetMap roads within bounding box tuple 
    '''
    api = overpy.Overpass()
    # fetch all ways and nodes
    result = api.query("""way(%s)["%s"]; (._;>;); out body; """ % (str(bbox)[1:-1], waytype))
    names=[]
    types=[]
    line_nodes=[]
    for way in result.ways:
        names.append(way.tags.get("name", "n/a"))
        types.append(way.tags.get(waytype, "n/a"))
        nodes=[]
        for node in way.nodes:
            nodes.append(Point(node.lon, node.lat))
        line_nodes.append(LineString(nodes))
    lines_gdf = gpd.GeoDataFrame(pd.DataFrame([names, types]).T, geometry=line_nodes, crs=4326)
    lines_gdf.columns = ["name", "type", "geometry"]
    return lines_gdf

    
def osm_roads(bbox):
    '''
    bbox = (x_min, y_min, x_max, y_max) *tuple in 4326
    ex) x_min, y_min, x_max, y_max = tuple([i for i in county_shp.to_crs(4326).total_bounds])
    returns geopandas geodataframe of all OpenStreetMap roads within bounding box tuple 
    '''
    api = overpy.Overpass()
    # fetch all ways and nodes
    result = api.query("""way(%f,%f,%f,%f)["highway"]; (._;>;); out body; """ % bbox)
    names=[]
    roadtypes=[]
    roads_nodes=[]
    for way in result.ways:
        names.append(way.tags.get("name", "n/a"))
        roadtypes.append(way.tags.get("highway", "n/a"))
        nodes=[]
        for node in way.nodes:
            nodes.append(Point(node.lon, node.lat))
        roads_nodes.append(LineString(nodes))
    roads_gdf = gpd.GeoDataFrame(pd.DataFrame([names, roadtypes]).T, geometry=roads_nodes, crs=4326)
    roads_gdf.columns = ["name", "type", "geometry"]
    return roads_gdf



def open_route(start_xy, end_xy, client_key):
    coords = ((start_xy[0], start_xy[1]),(end_xy[0], end_xy[1]))
    client = openrouteservice.Client(key=client_key) # Specify your personal API key
    geometry = client.directions(coords)['routes'][0]['geometry']
    decoded = convert.decode_polyline(geometry)['coordinates']
    route = LineString([Point(i[0], i[1]) for i in decoded])
    return route
