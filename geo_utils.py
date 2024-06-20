import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio as rio
import datetime
from datetime import datetime, timedelta, date, timezone
## under RASTER TO VECTOR 
from rasterio.features import shapes
## under GEO, for transform_point_coords
from pyproj import Proj, transform
## under GEO, for address_to_yx
import pygris
from pygris.geocode import geocode
## under CENSUS
from pygris import counties, tracts, block_groups, blocks
from pygris import validate_state
from pygris.data import get_census
## under EXTRACT RASTER, for zonal_time_series function
from rasterstats import zonal_stats 
## parsing TCX/GPX functions
import gpxpy
import tcxreader 
from tcxreader.tcxreader import TCXReader, TCXTrackPoint
## under TCX/GPX, adding to postgreSQL db
import psycopg2

import general_utils as gu
import raster_utils as ru
import vector_utils as vu
import plot_utils as pu

################################
## GEO 
################################

def address_to_yx(address):
    df = geocode(address = address)
    long = df.iloc[0].longitude
    lat = df.iloc[0].latitude
    return (lat, long)

def transform_point_coords(inepsg, outepsg, XYcoords):
    """takes 'XYcoords': (lon,lat) coordinate pair as a tuple or list, in their 'inepsg': coordinate reference system EPGS (as an integer), and returns that (lon,lat) pair as a tuple or list in 'outepsg':output EPSG (as an integer)"""
    lon,lat = transform(
        Proj(init="EPSG:"+str(inepsg)), 
        Proj(init="EPSG:"+str(outepsg)), 
        XYcoords[1],
        XYcoords[0])
    return (lon,lat)

################################
## RASTER TO VECTOR
################################

def raster_to_vector(inRast):
    """converts 'inRast' raster to vector format where pixels of the same value are grouped to a single polygon and have an attribute of that raster value; returns vector polygon geodataframe"""
    with rio.open(inRast, 'r') as tmp:
        rast = tmp.read(1)
        rast_crs = tmp.crs
    mask = None
    instance_shapes = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes(rast, mask=mask, transform=tmp.transform)))
    img_polys  = gpd.GeoDataFrame.from_features(list(instance_shapes))
    img_polys = img_polys.set_crs(rast_crs)
    return img_polys

def rast_bounds_to_vector(inRast):
    """helper function used in extract_TV_polygons function"""
    geo = rio.open(inRast)
    w, s, e, n = geo.bounds
    corners = [(w, n), (e, n), (e, s), (w, s)]
    img_poly = Polygon(corners)
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

def pts_to_lines(pts_df, group_col, x_col="lon", y_col="lat", crs=4326):
    gdf = gpd.GeoDataFrame(pts_df, geometry=[Point(xy) for xy in zip(pts_df.x_col, pts_df.y_col)], crs=crs)
    lines = gdf.groupby([group_col])['geometry'].apply(lambda x: LineString(x.tolist()))
    lines_gdf = gpd.GeoDataFrame(lines, geometry='geometry', crs=crs)
    return lines_gdf
    
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
        polys_stat = zonal_stats(field_instance_shp, in_rast, stats=[stat], geojson_out=True)
        for p in polys_stat: 
            p['properties'][str(os.path.basename(in_rast).split(".")[0])+stat] = p['properties'].pop(stat)
        feat = {'type': 'FeatureCollection', 'features':pd.DataFrame(polys_stat)}
        polys_stat_gdf = gpd.GeoDataFrame.from_features(feat).set_crs(rast_crs, allow_override=True).dropna()
    return polys_stat_gdf

def zonal_time_series(instance_shape, rast_dir, start_date, end_date, stat="mean"): 
    """
    instance_shape = polygon shapefile to calculate raster stat within
    rast_dir = input raster directory, where each raster's name must be called YYYYJD.tif 
    start_date & end_date format YYYYJD (year-julian date)
    stat options: mean, median, std, var"""
    stat_dates = []
    rast_list = [os.path.join(rast_dir, i) for i in sorted(os.listdir(rast_dir) if (img.endswith('.tif') & (int(img[:-4]) <= end_date) & (int(img[:-4]) >= start_date))]
    with rio.open(rast_list[0]) as tmp:
        rast_crs = tmp.crs
    polys = gpd.read_file(instance_shape) 
    if polys.crs == rast_crs:
        for in_rast in rast_list:
            stat_dates.append(zonal_stat(instance_shape, in_rast, stat))
        stats_df_merged = pd.concat(stat_dates)
    return stats_dates_gdf

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
    sums = intersections.groupby(by=[IDcol]).sum(numeric_only=True)[col_name]
    gdf_join = gdf_sub.set_index(IDcol).join(sums, on=IDcol)
    gdf_join.reset_index().to_csv(os.path.join(outDir, col_name+".csv"))
    return gdf_join

def extract_train_val_polys(inRast, input_shape, class_field, filter_string, TV_field, polygon_id, band_list="all"):
    TV_data = []  # initialize training_data, class_code labels, XYcoords, and unique field ID 
    TV_labels = []
    TV_xy = []
    TV_fieldID = []
    TV_split = []
    with rio.open(inRast, 'r') as geo:
        columns = [str(polygon_id), str(class_field), str(TV_field),  "Xcoord", "Ycoord"]
        bands=list(geo.descriptions)
        column_list = columns + bands
        rgt = geo.transform 
        # create bounding box shape of input image
        im_poly = rastBounds_to_vector(inRast) 
        vds = ogr.Open(input_shape, GA_Update)
        assert vds
        # read data into dataframe
        vlyr = vds.GetLayer(0)  
        # number of polygons
        feature_count = vlyr.GetFeatureCount()  
        for fid in range(feature_count):
            feature = vlyr.GetFeature(fid)  # creating instance of single polygon
            # exports xy info in terms of spatial ref
            shppoly = shapely.wkt.loads(feature.geometry().ExportToWkt()) ## APPLY NEGATIVE BUFFER OF -0.5 TO NOT EXTRACT BOUNDARY PIXELS: .buffer(-0.5)  
            # if the input image bounding box contains the Training/Validation polygon shape
            if im_poly.contains(shppoly) is True:
                # if it is a Training polygon (tv_field = T) and should NOT be ignored (class_field does not equal -99)
                if feature.GetField(class_field) != -99:
                    tv_val = feature.GetField(class_field)
                    tv_uid = feature.GetField(polygon_id)
                    TV = feature.GetField(TV_field)
                    ## find the size of the polygon bbox (in pixels) and the location of its UL corner within the image
                    src_offset = img_to_bbox_offsets(rgt, feature.geometry().GetEnvelope())
                    # if the width or height are 0, the polygon is too small, print out warning.
                    if src_offset[2] == 0 or src_offset[3] == 0:
                        print("Warning: Polygon {} too small to rasterize, "
                                "skipping".format(feature.GetFID()))
                        continue
                    else:
                        if isinstance(band_list, list):
                            # initialize array for horizontally stacking bands, the number of rows of bbox pixels
                            im_sub1 = np.zeros((src_offset[2] * src_offset[3], 1))  
                            for i in band_list:  # for each item in the list of bands
                                # read window of raster band(i), stack pixels below one another in one column
                                tmp = geo.read(i+1, window=Window(src_offset[0], src_offset[1], src_offset[2], src_offset[3])).reshape(-1, 1)
                                # stack pixels in each band as a new column, horizontally
                                im_sub1 = np.hstack((im_sub1, tmp))
                            # delete first column of zeros
                            im_sub1 = im_sub1[:, 1:]
                            # stack pixels for each new polygon vertically
                            im_sub = np.vstack((im_sub1))
                        elif band_list.lower() == 'all':
                            im_sub1 = np.zeros((src_offset[2] * src_offset[3], 1))
                            for i in range(1, geo.count+1, 1):  # for all bands in the raster
                                tmp = geo.read(i, window=Window(src_offset[0], src_offset[1], src_offset[2], src_offset[3])).reshape(-1, 1)
                                im_sub1 = np.hstack((im_sub1, tmp))
                            im_sub1 = im_sub1[:, 1:]
                            im_sub = np.vstack((im_sub1)) ##ADDED FLOAT DATATYPE HERE ##, dtype=np.float32
                        else:
                            print("impropper band list... exiting")
                            sys.exit()
                        msk_sub = np.ones((im_sub.shape[0], 1))
                        mem_drv = ogr.GetDriverByName('Memory')
                        driver = gdal.GetDriverByName('MEM')
                        # create new geotransformation for rasterized shape
                        new_gt = ((rgt[2] + (src_offset[0] * rgt[0])), rgt[0], 0.0, (rgt[5] + (src_offset[1] * rgt[4])), 0.0, rgt[4])
                        # Get XY Arrays from polygon subset
                        mgx, mgy = np.meshgrid(np.arange(0, src_offset[2], 1), np.arange(0, src_offset[3], 1))
                        xarr_sub = new_gt[0] + new_gt[1] * mgx + new_gt[2] * mgy
                        yarr_sub = new_gt[3] + new_gt[4] * mgx + new_gt[5] * mgy
                        xys = np.vstack((np.ravel(xarr_sub), np.ravel(yarr_sub))).T
                        # Create Memory Drivers
                        mem_ds = mem_drv.CreateDataSource('out')
                        mem_layer = mem_ds.CreateLayer('poly', vlyr.GetSpatialRef(), ogr.wkbPolygon)
                        mem_layer.CreateFeature(feature.Clone())
                        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
                        rvds.SetGeoTransform(new_gt)
                        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
                        rv_array = rvds.ReadAsArray()
                        rshp = rv_array.reshape(-1, 1)
                        full_mask = rshp * msk_sub
                        im_sub = full_mask * im_sub
                        xys = full_mask * xys
                        im_sub = im_sub[np.ravel(full_mask) == 1]
                        xys = xys[np.ravel(full_mask) == 1]
                        TV_fieldID += [tv_uid] * im_sub.shape[0]
                        TV_xy += [list(xy) for xy in xys]
                        TV_labels += [tv_val] * im_sub.shape[0]
                        TV_split += [TV] * im_sub.shape[0]
                        TV_data += [list(px) for px in im_sub]
    X=np.array(TV_data) 
    Y=np.array(TV_labels).reshape(-1, 1)
    groups=np.array(TV_fieldID).reshape(-1, 1)
    TV_fields=np.array(TV_split).reshape(-1, 1)
    XY_coords=np.array(TV_xy) 
    grid_name = input_image.split(".")[0][-6:] ## UNQ000
    train_holdout_csv = input_shape[:-4]+"_"+filter_string+"_"+str(len(bands))+"b_"+grid_name+"_Val.csv"
    csvarr = np.hstack((groups, Y, TV_fields, XY_coords, X))
    TV_df = pd.DataFrame(csvarr, columns = column_list)
    TV_df.to_csv(train_holdout_csv)
    return train_holdout_csv 


################################
## TCX / GPX ACTIVITY PARSING 
################################

def parse_tcx(data_dir):
    ''' 
    parse all TCX files in data directory -- make directory name the date of last activity -- writes csv with that date 
    function is quick so okay to reparse TCX files -- for next run, add new files to folder
    '''
    date = os.path.basename(data_dir)
    tcx_files = [i for i in sorted(os.listdir(data_dir)) if i.endswith(".tcx")]
    
    df = pd.DataFrame(columns=['file','date', 'duration',  'distance', 'ascent', 'hr_max', 'cadence', 'avg_speed'])
    df_bike = pd.DataFrame(columns=['file','date', 'duration',  'distance', 'ascent', 'hr_max',  'avg_speed'])
    for tf in tcx_files:
        xy_per_run = []
        file = open(os.path.join(data_dir, tf), 'r')
        tcx_reader = TCXReader()
        TCXTrackPoint = tcx_reader.read(file)
        if TCXTrackPoint.activity_type == "Running":
            df.loc[len(df.index)] = [file.name, TCXTrackPoint.start_time, TCXTrackPoint.duration, TCXTrackPoint.distance, TCXTrackPoint.ascent, TCXTrackPoint.hr_max, TCXTrackPoint.tpx_ext_stats.get('RunCadence'), TCXTrackPoint.avg_speed]
        elif TCXTrackPoint.activity_type == "Biking":
            df_bike.loc[len(df_bike.index)] = [file.name, TCXTrackPoint.start_time, TCXTrackPoint.duration, TCXTrackPoint.distance, TCXTrackPoint.ascent, TCXTrackPoint.hr_max,  TCXTrackPoint.avg_speed]
    ## clean running activity attributes
    df['file'] = [os.path.basename(i) for i in df['file']]
    df.set_index('file', inplace=True)
    df['name'] = [os.path.basename(f) for f in df.index]
    df['miles'] = [f/1609.34 for f in df['distance']]
    df['seconds'] = df['duration']
    df['minutes'] = [f/60 for f in df['seconds']]
    df['meters'] = df['distance']
    df['avg_mph'] = df['miles']/df['minutes']*60
    df['min_per_mi'] = df['minutes']/df['miles']
    df['cad_avg'] = [str(i).split(":")[-1].replace("}", "").replace(" ", "") for i in df.cadence]
    df['cad_max'] = [int(str(i).split(": ")[-3].split(", ")[0]) if len(str(i).split(": ")) == 4 else 0 for i in df['cadence']]
    df['PST'] = df['date'].astype('datetime64[ns]') - timedelta(hours=6)
    df.sort_values('PST').dropna().to_csv(os.path.join(data_dir,'runTCX_'+date+'.csv'))
    df['PST']=[str(i) for i in df['PST']]
    return df, df_bike

def parse_gpx(data_dir):
    date = os.path.basename(data_dir)
    pts_df = pd.DataFrame(columns=['file', 'lat', 'lon', 'ele', 'speed', 'time'])
    gpx_files = [i for i in sorted(os.listdir(data_dir)) if i.endswith(".gpx")]
    for fi in gpx_files:
        gpx_file = open(os.path.join(data_dir, fi), 'r')
        gpx = gpxpy.parse(gpx_file,version="1.1")
        for track in gpx.tracks:
            for seg in track.segments:
                for point_no, pt in enumerate(seg.points):
                    if pt.speed != None:
                        speed = pt
                    elif point_no > 0:
                        speed = pt.speed_between(seg.points[point_no - 1])
                    elif point_no == 0:
                        speed = 0     
                    pts_df.loc[len(pts_df.index)] = [fi, pt.latitude, pt.longitude, pt.elevation, speed, pt.time]
    pts_df['PST'] = pts_df['time'] - timedelta(hours=6)
    pts_df['PST']=[str(i).split("+")[0] for i in pts_df['PST']]
   # pts_df.set_index('file', inplace=True)
    new_pt_csv = os.path.join(data_dir, "allGPX_"+date+".csv")
    pts_df.sort_values('time').to_csv(new_pt_csv)
    return pts_df, new_pt_csv


################################
## POSTGRESQL
################################

def update_gcx_table(df, db, usr, pwd, localhost, port):
    conn = psycopg2.connect(database = db, user = usr, password = pwd, host = localhost, port = port)
    cur = conn.cursor()
    for i in range(0 ,len(df)):
        values = (df['file'][i], df['time'][i], df['lat'][i], df['lon'][i], df['ele'][i], df['speed'][i])
        cur.execute("INSERT INTO gpx_runs (gpx_file, time, lat, lon, ele, speed) VALUES (%s, %s, %s, %s, %s, %s)", values)
    conn.commit()
    print("Records created successfully")
    conn.close()

def update_tcx_table(df, db, usr, pwd, localhost, port):
    conn = psycopg2.connect(database = db, user = usr, password = pwd, host = localhost, port = port)
    cur = conn.cursor()
    for i in range(0 ,len(df)):
        values = (df['file'][i], df['PST'][i], df['minutes'][i], df['miles'][i], df['ascent'][i], int(df['hr_max'][i]), df['cad_avg'][i])
        cur.execute("INSERT INTO tcx_runs (tcx_file, date, minutes, miles, vert_m, hr_max, cad_avg) VALUES (%s, %s, %s, %s, %s, %s, %s)", values)
    conn.commit()
    print("Records created successfully")
    conn.close()


################################
## CENSUS 
################################

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
    shapes = tracts(state = validate_state(str(addr[-8:-6])), cb = True, subset_by = {addr: subset_buff_dist_m}, cache = True)
    return shapes

def get_block_groups_by_addr_buff(addr, buff_dist):
    shapes = block_groups(state = validate_state(str(addr[-8:-6])), cb = True, subset_by = {addr: subset_buff_dist_m}, cache = True)
    return shapes

def get_blocks_by_addr_buff(addr, buff_dist):
    shapes = blocks(state = validate_state(str(addr[-8:-6])), subset_by = {addr: subset_buff_dist_m}, cache = True)
    return shapes

## get primary roads 
def get_roads_by_county(county, state):
    roads = pygris.roads(state = state, county = county, cache = True)
    ## roads.explore()
    return roads
    
## get acs census variable 
def get_census_var(region, state_abr, acs_var_id, acs_yr):
    census_var = get_census(dataset = "acs/acs5", variables = acs_var_id, year = acs_yr, 
                            params = {"for": region+":*", "in": f"state:{validate_state(state_abr)}"},
                            guess_dtypes = True, return_geoid = True)
    # census_dict = dict(zip(census_var['GEOID'], census_var[acs_var_id]))
    return census_var
    
