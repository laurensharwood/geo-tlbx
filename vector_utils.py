import os, math
import pandas as pd
import numpy as np
import geopandas as gpd
## under PROJECT PREP, for create_proc_grid function
import shapely
from shapely.geometry import LineString, Point, Polygon, MultiPolygon, box
from pyproj import Proj, transform
## under GEO PROCESSING
import fiona
from shapely.ops import nearest_points
from pygris.geocode import geocode
## for functions under POINT_PATTERN_ANALYSIS
from pointpats import centrography
from sklearn.neighbors import KernelDensity
import rasterio as rio
from rasterio.transform import Affine
from scipy.spatial import distance
## under POST PROCESSING, for split_hangpolys function

################################
## PROJECT PREP
################################

def create_proc_grid(AOI_bounds, cell_size, out_epsg, grid_file, sampPts=False):
    """creates 'grid_file': a tiled/mesh grid for parallel processing, which covers 'AOI_bounds': the area of interest (AOI) bounding box as a https://geojson.io/, shapefile, or geopackage; and each unique, UNQ, grid cell has the 'cell_size': width and height of  in the units (m or ft) of 'out_epsg': the coordinate reference system EPSG code (as an integer). optionally, if 'SampPts' is True, each grid cell's centroid is saved as a point shapefile, 'grid_file'_sampPts.shp. returns proc grid as a geodataframe"""
    gdf_bounds = gpd.read_file(AOI_bounds)
    gdf_proj = gdf_bounds.to_crs(out_epsg)
    xmin, ymin, xmax, ymax = gdf_proj.total_bounds
    n_cells=(xmax-xmin) / cell_size
    grid_cells = []
    for x in np.arange(xmin, xmax+cell_size, cell_size):
        for y in np.arange(ymin, ymax+cell_size, cell_size):
            grid_cells.append(shapely.geometry.box(x, y, x-cell_size, y+cell_size))
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=out_epsg)
    grid.loc[:, 'UNQ'] = [str(i).zfill(len(str(len(grid)))) for i in range(1, grid.shape[0]+1)]
    grid.to_file(grid_file)
    if sampPts == True:
        pts = gpd.GeoDataFrame(grid.drop(columns=["geometry"]), geometry=grid.centroid, crs=out_epsg) 
        pts.to_file(grid_file.replace(".", "_sampPts."))
    return grid

def wkt_bounds(grid_file):
    aoi = gpd.read_file(grid_file)
    aoi_4326 = aoi.to_crs(4326)
    ## Extract the Bounding Box Coordinates
    bounds = aoi_4326.total_bounds
    ## Create GeoDataFrame of the Bounding Box 
    gdf_bounds = gpd.GeoSeries([box(*bounds)])
    ## Get WKT Coordinates
    wkt_aoi = gdf_bounds.to_wkt().values.tolist()[0]
    return wkt_aoi


################################
## GEO PROCESSING
################################



    
def lon_lat_to_gdf(points_df):
    pts_gdf = gpd.GeoDataFrame(points_df, geometry=gpd.points_from_xy(points_df.lon, points_df.lat, crs=4326))
    return pts_gdf

def transform_point_coords(inepsg, outepsg, XYcoords):
    """takes 'XYcoords': (lon,lat) coordinate pair as a tuple or list, in their 'inepsg': coordinate reference system EPGS (as an integer), and returns that (lon,lat) pair as a tuple or list in 'outepsg':output EPSG (as an integer)"""
    lon,lat = transform(
        Proj(str(inepsg)), 
        Proj(str(outepsg)), 
        XYcoords[1],
        XYcoords[0])
    return (lon,lat)

def address_to_yx(address):
    df = geocode(address = address) ## uses pygris 
    long = df.iloc[0].longitude
    lat = df.iloc[0].latitude
    return (lat, long)
    
def merge_vectors(in_files):
    """combines 'in_files': a list of vector shapes (as long as they have the same column names).
    returns the full merged geodataframe"""
    gdf_list=[]
    for in_fi in in_files:
        gdf_list.append(gpd.read_file(in_fi))
    full_gdf = pd.concat(gdf_list)
    return full_gdf
    
def xyz_to_gdf(input_xyz):
    """reads 'input_xyz': a point cloud .xyz filewith ['x', 'y', 'z_m'] -- Z in meters; and returns a geodataframe with x(lon), y(lat), and z(elevation in meters and feet) columns for each row (point)"""
    df  = pd.read_table(input_xyz, skiprows=1, delim_whitespace=True, names=['x', 'y', 'z_m'])
    df['x'] = df['x'].str.strip(',')
    df[['y', 'z']] = df['y'].str.split(',', expand=True)
    df['z_m'] = np.where(df['z_m'] == '', df['z'], df['z_m'])
    df['z_ft'] = df['z_m'].astype(float)*3.28084
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'].astype(float), df['y'].astype(float)))
    return gdf
    
def unnest(gdf):
    return gdf.explode(index_parts=True)

def pts_to_lines(pts_df, group_col, x_col="lon", y_col="lat", crs=4326):
    gdf = gpd.GeoDataFrame(pts_df, geometry=[Point(xy) for xy in zip(pts_df.x_col, pts_df.y_col)], crs=crs)
    lines = gdf.groupby([group_col])['geometry'].apply(lambda x: LineString(x.tolist()))
    lines_gdf = gpd.GeoDataFrame(lines, geometry='geometry', crs=crs)
    return lines_gdf

def snap_pt_to_line_xy(pts, lines):
    if pts.crs != lines.crs:
        pts = pts.to_crs(lines.crs)
    pts["pt_geom"] = pts.geometry #Save the point geometry or it is lost in spatial join
    #Join the points to the lines
    points_join = gpd.sjoin_nearest(left_df=lines, right_df=pts, how="inner", max_distance=0.00005, distance_col="line_dist")
    #Duplicate points/rows are created when there are multiple join features within max_distance. Drop the duplicates, keep the nearest
    points_join = points_join.sort_values(by=["road_dist", "time"], ascending=True).drop_duplicates(subset="time", keep="first")
    # Calculate nearest points on line and the joined point. Keep index [0], which is the point on the line nearest to the joined point
    points_join["line_point"] = points_join.apply(lambda x: nearest_points(g1=x["geometry"], g2=x["pt_geom"])[0], axis=1)
    return gpd.GeoDataFrame(points_join, geometry=points_join["line_point"], crs=pts.crs)
    
################################
## POINT PATTERN ANALYSIS
################################

def euclid_dist(pointOne, pointTwo):
    """calculates the Euclidean distance between two points, 'pointOne' & 'pointTwo': each as a (lon,lat) coordinate pair tuple or list; returns the Euclidean distance between the two input points""" 
    dist = math.sqrt((pointTwo[0] - pointOne[0])**2 + (pointTwo[1] - pointOne[1])**2)
    return dist
    
def convex_hull(gdf):
    """from a 'gdf': geodataframe point shapefile, calculates their convex hull, or minimum bounding geometry and returns it as a polygon geodataframe"""
    all_pts=gdf['geometry']
    convex_hulls = [i.convex_hull for i in [shapely.Point(i.to_list()) for i in all_pts]]
    convex_hull_gdf = gpd.GeoDataFrame(
        data = all_pts[[i for i in all_pts.columns.to_list() if "geometry" not in i]], 
        geometry = convex_hulls, 
        columns = [i for i in all_pts.columns.to_list() if "geometry" not in i])
    return convex_hull_gdf

def convex_hull_grouped(gdf, group_col):
    """from a 'gdf': geodataframe of points, calculates the convex hull, or minimum bounding geometry for all points in 'gdf' that have the same 'group_col' column/attribute value. returns a polygon geodataframe for each unique group_col's bounding geometry"""
    group_IDs = sorted(list(set(gdf[group_col].to_list())))
    all_pts=[]
    for i in group_IDs:
        current = gdf[gdf[group_col] == i]
        all_pts.append(current['geometry'])
    multipoints = [shapely.MultiPoint(i.to_list()) for i in all_pts]
    convex_hulls = [i.convex_hull for i in multipoints]
    convex_hull_gdf = gpd.GeoDataFrame(data=group_IDs, geometry=convex_hulls, columns=[group_col])
    return convex_hull_gdf

def central_point(pt_list):
    """calculates and returns the central point from 'pt_list': a list of coordinate pairs as a list or tuple-- this is generally NOT the feature closest to the mean center, meaning that the central feature in most circumstances has no obvious relationship to the mean center; one needs to actually calculate the cumulative distances for all points and pick the one point with the smallest cumulative distance"""
    sumDist = []
    for elem in pt_list:
        dist_list = []
        for item in pt_list:
            dist_list.append(distance(elem, item))
        sumDist.append([sum(dist_list), elem[0]])
    sumDist.sort()
    return sumDist[0][1]

def mean_center(pts_gdf):
    """returns each point's distance to mean center of all points in 'pts_gdf'"""
    pts_gdf['x'] = pts_gdf.geometry.x.to_list()
    pts_gdf['y'] = pts_gdf.geometry.y.to_list()   
    pts_gdf['uX'] = [centrography.mean_center(pts_gdf[["x", "y"]])[0]]*len(pts_gdf)
    pts_gdf['uY'] = [centrography.mean_center(pts_gdf[["x", "y"]])[1]]*len(pts_gdf)
    ## calculate euclidean distance of each point location (xy coord pair) to the mean center location
    dist2mean = [distance.euclidean(i[0], i[1]) for i in list(zip(list(zip(pts_gdf['x'], pts_gdf['y'])), list(zip(pts_gdf['uX'], pts_gdf['uY'])))) ]
    return dist2mean
    
def median_center(pts_gdf):
    """returns each point's distance to median center of all points in 'pts_gdf'"""
    pts_gdf['x'] = pts_gdf.geometry.x.to_list()
    pts_gdf['y'] = pts_gdf.geometry.y.to_list()   
    pts_gdf['medX'] = [centrography.euclidean_median(pts_gdf[["x", "y"]])[0]]*len(pts_gdf)
    pts_gdf['medY'] = [centrography.euclidean_median(pts_gdf[["x", "y"]])[1]]*len(pts_gdf)
    ## calculate euclidean distance of each point location (xy coord pair) to the median center location
    dist2med = [distance.euclidean(i[0], i[1]) for i in list(zip(list(zip(pts_gdf['x'], pts_gdf['y'])), list(zip(pts_gdf['medX'], pts_gdf['medY'])))) ]
    return dist2med

def dispersion(pts_gdf):
    """returns the standard distance of a point geodataframe, 'pts_gdf'"""
    pts_gdf['x'] = pts_gdf.geometry.x.to_list()
    pts_gdf['y'] = pts_gdf.geometry.y.to_list()   
    dispersion = centrography.std_distance(pts_gdf[["x", "y"]])
    return dispersion

def kernel_density_estimation(gdf, bandWidth, out_dir, kde_kernel='gaussian', kde_metric='euclidean', kde_algorithm='auto'):
    """saves kernel density raster for each gdf (subset by a group by--village column)"""
    # Get X and Y coordinates
    x_sk = gdf["geometry"].x
    y_sk = gdf["geometry"].y
    reg_bounds = gdf.total_bounds
    # Get minimum and maximum coordinate values of buffered points
    min_x_sk, min_y_sk, max_x_sk, max_y_sk = reg_bounds
    min_x_sk=min_x_sk-100
    min_y_sk=min_y_sk-100
    max_x_sk=max_x_sk+100
    max_y_sk=max_y_sk+100

    # Create a cell mesh grid
    # Horizontal and vertical cell counts should be the same
    XX_sk, YY_sk = np.mgrid[min_x_sk:max_x_sk:100j, min_y_sk:max_y_sk:100j]

    xres = (max_x_sk - min_x_sk) / len(XX_sk)
    yres = (max_y_sk - min_y_sk) / len(YY_sk)

    # Create 2-D array of the coordinates (paired) of each cell in the mesh grid
    positions_sk = np.vstack([XX_sk.ravel(), YY_sk.ravel()]).T
    # Create 2-D array of the coordinate values of the well points
    Xtrain_sk = np.vstack([x_sk, y_sk]).T
    # Get kernel density estimator (can change parameters as desired)
    kde_sk = KernelDensity(bandwidth = bandWidth, kde_kernel = kde_kernel, metric = kde_metric, algorithm = kde_algorithm)
    # Fit kernel density estimator to wells coordinates
    kde_sk.fit(Xtrain_sk)
    # Evaluate the estimator on coordinate pairs
    Z_sk = np.exp(kde_sk.score_samples(positions_sk))
    # Reshape the data to fit mesh grid
    Z_sk1 = Z_sk.reshape(XX_sk.shape)
    Z_sk=Z_sk1*100000000
    # Flip array vertically and rotate 270 degrees
    Z_exp = np.rot90(np.flip(Z_sk, 0), 3)
    ##Export raster
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with rio.open(os.path.join(out_dir, "KernelDensity_"+str(bandWidth)+".tif"), mode = "w", driver = "GTiff", dtype = Z_exp.dtype, height = Z_exp.shape[0], width = Z_exp.shape[1], count = 1, crs = gdf.crs, transform = Affine.translation(min_x_sk-xres/2,min_y_sk-yres/2)*Affine.scale(xres, yres)) as dst:
        dst.write(Z_exp, 1)

################################
## POST PROCESSING
################################

def largest_multipolypart(gdf):
    new_geoms = []
    for i in gdf.geometry:
        if type(i) == MultiPolygon:
            geom_list = [list(x.exterior.coords) for x in i.geoms]
            poly_area_dict = dict(zip([Polygon(i) for i in geom_list], [Polygon(i).area for i in geom_list]))
            new_geoms.append(max(poly_area_dict, key=poly_area_dict.get))
        elif type(i) == Polygon:
            new_geoms.append(i)
        else:
            new_geoms.append(i)
    gdf['geometry']  = new_geoms
    return gdf

def line_intersections(geom1, geom2):
    '''
    find intersection of two line geometries with the same CRS. 
    ex) 
    geom1 = trails.dissolve().to_crs(4326).geometry
    geom2 = streams.dissolve().to_crs(4326).geometry
    returns two item tuple: (intersections points, intersection lines) 
    '''
    line_xing = geom1.intersection(geom2)
    line_xing=line_xing[line_xing!=None].explode()
    ## separate by lines and points
    xing_pts = line_xing[line_xing.geometry.apply(lambda x : x.geom_type=="Point")]
    xing_lines = line_xing[line_xing.geometry.apply(lambda x : x.geom_type=="LineString")]
    return (xing_pts, xing_lines)

def self_intersection(gdf):
    cross = pd.merge(left=gdf, right=gdf, how='cross') #Cross join the dataframe to itself
    ##cross = cross.loc[cross['villID_x'] != cross['villID_y']] #Remove self joins < or !=    !!! *******************
    cross = cross.loc[cross.geometry_x.intersects(cross.geometry_y)] #Select only polygons intersecting
    cross['inter'] = cross.geometry_x.intersection(cross.geometry_y) #Intersect them
    cross = cross.set_geometry('inter') 
    cross = cross[[column for column in cross.columns if 'geom' not in column]] #Drop old geometries
    return cross

def split_hangpolys(vectorized_gdf):
    proj_crs = "EPSG:8858"
    int_epsg=int(proj_crs.replace("EPSG:", ""))
    vectorized_gdf['pred_id'] = list(range(0,len(vectorized_gdf), 1))
    ##  cut: negative buffer by X. if it's a polygon, rebuffer. if it's a multipolygon, split parts then rebuffer 
    eroded_geom = vectorized_gdf.buffer(distance=-10, resolution=1,  join_style=1)
    erd_clean = gpd.GeoDataFrame.from_features(eroded_geom.buffer(0))
    ## if it's an empty polygon (eroded away into nothing), take old geometry
    if not len(erd_clean) > 0:
        return erd_clean
    emptyTF = erd_clean[erd_clean['geometry'].isna()].reset_index()
    emptyTF.columns=['pred_id','empty']
    empty_old = vectorized_gdf.set_index('pred_id').join(emptyTF.set_index('pred_id'), how='inner')
    empty_old = empty_old[ empty_old['geometry'] !=  None]
    empty_old = empty_old.reset_index()
    empty_old = empty_old.drop(columns=['empty'])
    ## if it has a real geometry remaining but is a polygon or multipolygon 
    multi_polygons=[]
    polygons=[]
    not_empty = erd_clean[~erd_clean['geometry'].isna()].reset_index()
    for i, row in not_empty.iterrows():
        if row.geometry.geom_type.startswith("Multi"):
            multi_polygons.append(row.geometry)
        if row.geometry.geom_type.startswith("Polygon"):
            polygons.append(row.geometry)
    ## if it's a polygon (didn't cut), use old geom (orig shape) 
    polys_gdf = gpd.GeoDataFrame(gpd.geoseries.GeoSeries(polygons), columns=['geometry'], crs=int_epsg)
    old_polys = vectorized_gdf.sjoin(polys_gdf, how="inner", predicate="intersects")
    ## if it's a multipolygon, split parts (explode), then rebuffer to old geom 
    multi_geoS = gpd.geoseries.GeoSeries(multi_polygons).explode(index_parts=True)
    multi_geoS=multi_geoS.reset_index()
    multi_geoS=multi_geoS.drop(columns=["level_0", "level_1"]) ## to-do: if level is in a column drop that column
    multi_explode = gpd.GeoDataFrame(geometry=multi_geoS[0], crs=int_epsg)#
    multi_explode_reBuff = multi_explode.buffer(10, join_style=1)
    multi_explode_reBuff = gpd.GeoDataFrame(geometry=multi_explode_reBuff)#
    ## combine old and new parts
    new_cut_geom = pd.concat([empty_old, old_polys, multi_explode_reBuff], axis=0) 
    ## dissolve shapes that touch 
    dissolved_geom = gpd.geoseries.GeoSeries([geom for geom in new_cut_geom.unary_union.geoms])
    dissolved_gdf = gpd.GeoDataFrame(pd.DataFrame(list(range(0,len(dissolved_geom), 1)), columns=['pred_id']), geometry=dissolved_geom, crs=int_epsg)   
    return dissolved_gdf


################################
## ID ACCURACY
################################

def largest_overlap(ref_df, pred_df):        
    ## find all Pred fields that intersect (spatial join), as a list (1:many)
    intersecting = pd.DataFrame(pred_df.sjoin(ref_df, how='inner')['Rindex']) #Find the polygons that intersect. Keep savedindex as a series
    pred_val_matches = intersecting.reset_index()
    pred_val_matches.columns = ["PredIndex", "RefIndex"]
    pred_ref_intersecting_index = pd.DataFrame(pred_val_matches.groupby(['RefIndex'])['PredIndex'].apply(list)).reset_index()
    ## find the polygon w/ largest overlap w/ each Ref field (1:1)
    overlap_areas_all_fields=[]
    for k,v in pred_ref_intersecting_index.iterrows():
        ref_index = v[0]
        pred_matches = v[1]
        Rdf=ref_df[ref_df["Rindex"]==ref_index]
        overlap_areas_per_ref_field=[]
        pred_indices_per_ref_field=[]
        for pred_index in pred_matches:
            Pdf = pred_df[pred_df["Pindex"]==pred_index]
            Rdf['area'] = Rdf.geometry.area
            Pdf['area'] = Pdf.geometry.area 
            intersect_df = gpd.overlay(Rdf, Pdf, how="intersection")
            if not len(intersect_df) == 0:
                interction_area = intersect_df['geometry'].area
                pred_indices_per_ref_field.append(pred_index)
                overlap_areas_per_ref_field.append(interction_area[0])
            else:  
                pred_indices_per_ref_field.append(0)
                overlap_areas_per_ref_field.append(0)                
        overlap_areas_all_fields.append(dict(zip(pred_indices_per_ref_field, overlap_areas_per_ref_field)))
    ## find largest overlap from list 
    largest_overlapping_pred = []
    for r in overlap_areas_all_fields:
        max_index = max(r, key=r.get)
        largest_overlapping_pred.append(max_index)
    RP_index_matches = list(zip(pred_ref_intersecting_index['RefIndex'].to_list(), largest_overlapping_pred))
    return RP_index_matches
 
def calc_metrics(PredVector, ref_df, pred_df, RP_index_matches):
    IoUs=[]
    overSeg_rates=[]
    underSeg_rates=[]
    location_similarities=[]
    for i in RP_index_matches:
        ref_gdf = ref_df[ref_df['Rindex'] == i[0]]
        pred_gdf = pred_df[pred_df['Pindex'] == i[1]]
        intersect_df = gpd.overlay(ref_gdf, pred_gdf, how="intersection")
        ref_area = ref_gdf['geometry'].iloc[0].area
        pred_area = pred_gdf['geometry'].iloc[0].area    
        intersect_area = intersect_df['geometry'].loc[0].area
        union_area = ref_area+pred_area-intersect_area
        ## IoU
        IoUs.append(intersect_area/union_area)
        ## overseg rates 
        overSeg_rates.append(1-(intersect_area/ref_area))
        ## underseg rates 
        underSeg_rates.append(1-(intersect_area/pred_area))    
        ## location similarity 
        pred_centroid = pred_gdf.geometry.centroid.iloc[0]
        ref_centroid = ref_gdf.geometry.centroid.iloc[0]
        centr_dist=ref_centroid.distance(pred_centroid) 
        circRadius=2*np.sqrt(union_area/np.pi)
        location_similarities.append(1-centr_dist/circRadius)
    filename = os.path.basename(PredVector)
    Region = (list(set(ref_df['region'])))[0] ### UNQ or region
    print(Region)
    match_metrics = [os.path.basename(filename), Region, RP_index_matches,IoUs,overSeg_rates,underSeg_rates,location_similarities]
    match_metrics=pd.DataFrame(match_metrics).T
    ## show accuracy metrics by chip 
    match_metrics_per_grid = [os.path.basename(filename), Region, np.mean(IoUs), np.mean(overSeg_rates), np.mean(underSeg_rates), np.mean(location_similarities)]
    metrics_per_grid = pd.DataFrame(match_metrics_per_grid).T
    metrics_per_grid.columns=["version", "region", "IoU", "overseg", "underseg", "location_sim"]
    metrics_per_grid.fillna(0, inplace=True)
    return match_metrics_per_grid
