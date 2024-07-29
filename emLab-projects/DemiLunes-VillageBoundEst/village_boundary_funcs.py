#!/usr/bin/ python


import os, sys, glob
import io

import pandas as pd
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from seaborn import lineplot
import numpy as np
import geopandas as gpd
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from scipy import stats
from shapely.geometry import Polygon, box
from sklearn.datasets import fetch_species_distributions
from sklearn.neighbors import KernelDensity
import shapely
from shapely.geometry import Polygon, LineString, Point, MultiPoint
from scipy.spatial import distance
from pointpats import centrography
from itertools import chain

def TC_histo(gdf, field):
    C_vils = gdf[gdf['treated'] == 0]
    T_vils = gdf[gdf['treated'] == 1]
    
    x = C_vils[field]
    y = T_vils[field]

    plt.hist(y, 10, alpha=0.5, label='Treatment village overlap %')
    plt.hist(x, 10, alpha=0.5,  label='Control village overlap %')
    plt.legend(loc='upper right')
    plt.show()
    
    
def export_kde_raster(Z, XX, YY, min_x, max_x, min_y, max_y, proj, filename):
    # Flip array vertically and rotate 270 degrees
    Z_exp = np.rot90(np.flip(Z, 0), 3)

    xres = (max_x - min_x) / len(XX)
    yres = (max_y - min_y) / len(YY)

    transform = Affine.translation(min_x - xres / 2, min_y - yres / 2) * Affine.scale(xres, yres)

    with rasterio.open(filename, mode = "w", driver = "GTiff",  height = Z_exp.shape[0], width = Z_exp.shape[1], count = 1, dtype = Z_exp.dtype, crs = proj, transform = transform) as dst:
            dst.write(Z_exp, 1)
            
def convex_hull(gdf, group_col):
    villIDs = sorted(list(set(gdf[group_col].to_list())))
    all_pts=[]
    for i in villIDs:
        current = gdf[gdf[group_col] == i]
        all_pts.append(current['geometry'])
    multipoints = [shapely.MultiPoint(i.to_list()) for i in all_pts]
    convex_hulls = [i.convex_hull for i in multipoints]
    new_geoms = gpd.GeoDataFrame(villIDs, geometry=convex_hulls, columns=[group_col])

    
    return new_geoms

import warnings
warnings.filterwarnings("ignore")


def KDE_dropFAR(points_fi, gdf, IDs, group_col, bandWidth, Kdens_drop, KDE_dir):
    
    numVils=0
    numPts=0    
    remaining_pts=[]
    KDE_dict={}
    KDEs=[]
    for i in IDs:
        # Get dataframe for all points with a given villID
        current = gdf[gdf[group_col] == i]
        
        # Get X and Y coordinates of well points
        x_sk = current["geometry"].x
        y_sk = current["geometry"].y
        reg_bounds = current.total_bounds
        # Get minimum and maximum coordinate values of buffered points
        min_x_sk, min_y_sk, max_x_sk, max_y_sk = reg_bounds
        min_x_sk=min_x_sk-100
        min_y_sk=min_y_sk-100
        max_x_sk=max_x_sk+100
        max_y_sk=max_y_sk+100
        # Create a cell mesh grid
        # Horizontal and vertical cell counts should be the same
        XX_sk, YY_sk = np.mgrid[min_x_sk:max_x_sk:100j, min_y_sk:max_y_sk:100j]
        # Create 2-D array of the coordinates (paired) of each cell in the mesh grid
        positions_sk = np.vstack([XX_sk.ravel(), YY_sk.ravel()]).T
        # Create 2-D array of the coordinate values of the well points
        Xtrain_sk = np.vstack([x_sk, y_sk]).T
        # Get kernel density estimator (can change parameters as desired)
        kde_sk = KernelDensity(bandwidth = bandWidth, kernel = 'gaussian', metric = 'euclidean', algorithm = 'auto')
        # Fit kernel density estimator to wells coordinates
        kde_sk.fit(Xtrain_sk)
        # Evaluate the estimator on coordinate pairs
        Z_sk = np.exp(kde_sk.score_samples(positions_sk))
        # Reshape the data to fit mesh grid
        Z_sk1 = Z_sk.reshape(XX_sk.shape)
        Z_sk=Z_sk1*100000000
        
        fiName=os.path.basename(points_fi)
        if not os.path.exists(KDE_dir):
            os.makedirs(KDE_dir)
        out_rast=os.path.join(KDE_dir, "KernelDensity_"+str(i)+"_"+str(bandWidth)+".tif")
        ##Export raster
        export_kde_raster(Z = Z_sk, XX = XX_sk, YY = YY_sk,
                          min_x = min_x_sk, max_x = max_x_sk, min_y = min_y_sk, max_y = max_y_sk,
                          proj = gdf.crs, filename = out_rast)

        coord_list = [(x, y) for x, y in zip(current["geometry"].x, current["geometry"].y)]
        with rasterio.open(out_rast) as src:
            current["KerDen"] = [x for x in src.sample(coord_list)]
            
        ## new
        for coord, kern in list(zip(coord_list, current["KerDen"].to_list())):
            KDE_dict.update({coord: kern[0]})
            
        current["KerDens"] = [i[0] for i in current["KerDen"]]
        
        rmOutliers=current[current["KerDens"] > Kdens_drop]
        
        if len(current) != len(rmOutliers):
            numVils+=1
            numPts+=len(current)-len(rmOutliers)
        remaining_pts.append(rmOutliers['geometry'])
            
    
    print(str(numVils)+" villages lost "+ str(numPts)+ " points")

    return remaining_pts, KDE_dict


def intersecting_area(gdf):
    cross = pd.merge(left=gdf, right=gdf, how='cross') #Cross join the dataframe to itself
    cross = cross.loc[cross['villID_x'] != cross['villID_y']] #Remove self joins < or !=!!! *******************
    cross = cross.loc[cross.geometry_x.intersects(cross.geometry_y)] #Select only polygons intersecting
    cross['inter'] = cross.geometry_x.intersection(cross.geometry_y) #Intersect them
    cross = cross.set_geometry('inter') 
    cross = cross[[column for column in cross.columns if 'geom' not in column]] #Drop old geometries
    return cross


def TC_overlap(gdf):

    ## find overlapping area with other fields, flag if T <-> C overlap

    ## for each field, find the total area of overlap -- don't double count the same location where for instance two T fields overlap, so merge the intersecting field shape into one multipolygon
    ## then find the area this polygon takes up in each village (that's olapArea)

    gdf['Area'] = gdf.area
    ## shape for all village boundary intersections (poly per village intersection) 
    intersection = intersecting_area(gdf=gdf)

    ## dissolve shape to single multipolygon
    single_intersect = intersection.dissolve()
    single_intersect.drop(columns=[i for i in single_intersect.columns.to_list() if "inter" not in i], inplace=True)

    ## shape for each village boundary's intersections (poly per village's intersections) -- grab overlap area from this 
    olap_poly_per_village = gdf.overlay(single_intersect) 
    olap_poly_per_village['olapArea'] = olap_poly_per_village.area
    olap_poly_per_village['pctOlap'] = olap_poly_per_village['olapArea']/olap_poly_per_village['Area']
    origGeoms_wPctOlap = gdf.set_index("villID").join(olap_poly_per_village.set_index("villID"), how="left", rsuffix="_int")
    origGeoms_wPctOlap.drop(columns=[i for i in origGeoms_wPctOlap.columns.to_list() if "_int" in i], inplace=True)


    ## FLAG HIGH % of T-C OVERLAP 
    intersection['TCflag'] = np.where(intersection['treated_x'] != intersection['treated_y'], 1, 0)
    bad_intersections = intersection[intersection['TCflag'] == 1]

    single_bad_intersect = bad_intersections.dissolve()
    single_bad_intersect.drop(columns=[i for i in single_intersect.columns.to_list() if "inter" not in i], inplace=True)

    bad_olap_poly_per_village = gdf.overlay(single_bad_intersect) 
    bad_olap_poly_per_village['BadArea'] = bad_olap_poly_per_village.area
    bad_olap_poly_per_village['BadOlp'] = bad_olap_poly_per_village['BadArea']/bad_olap_poly_per_village['Area']
    geom_wBadOLAP = gdf.set_index("villID").join(bad_olap_poly_per_village.set_index("villID"), how="left", rsuffix="_int")
    geom_wBadOLAP.drop(columns=[i for i in geom_wBadOLAP.columns.to_list() if "_int" in i], inplace=True)


    geom_wOLAP = origGeoms_wPctOlap.join(geom_wBadOLAP, how="left", rsuffix="_b")
    ##geom_wOLAP = geom_wOLAP.drop(columns=[i for i in geom_wOLAP.columns.to_list() if "_b" in i or "_y" in i or "_x" in i])
    geom_wOLAP = geom_wOLAP.drop(columns=[i for i in geom_wOLAP.columns.to_list() if "_b" in i])
    geom_wOLAP = geom_wOLAP.drop(columns=[i for i in geom_wOLAP.columns.to_list() if "_y" in i])
    geom_wOLAP = geom_wOLAP.drop(columns=[i for i in geom_wOLAP.columns.to_list() if "_x" in i])
    geom_wOLAP=geom_wOLAP.replace(np.nan, 0)

    return geom_wOLAP

## point pattern analysis
def pt_patterns(pts_df, poly_shape, group_col, buffNeighb=400, CntVilRat_Thresh_pct=0.25, dst2cntr_Thresh_m=5000):
    all_points = pts_df.copy()
    out_dfs = []
    for group in sorted(list(set(all_points[group_col]))):
        df = all_points[all_points[group_col] == group]
        df['x'] = df.geometry.x.to_list()
        df['y'] = df.geometry.y.to_list()   
        df['medCenLocX'] = [centrography.euclidean_median(df[["x", "y"]])[0]]*len(df)
        df['medCenLocY'] = [centrography.euclidean_median(df[["x", "y"]])[1]]*len(df)
        df['dispers'] = centrography.std_distance(df[["x", "y"]])
        pt_locs = list(zip(df['x'], df['y']))
        med_cen_locs = list(zip(df['medCenLocX'], df['medCenLocY'])) 
        df['dst2cntr'] = [distance.euclidean(i[0], i[1]) for i in list(zip(pt_locs, med_cen_locs)) ]
        out_dfs.append(df)
    out_full = pd.concat(out_dfs)
    
    out_full = out_full.set_index('villID').join(poly_shape[['villArea']], rsuffix='_r')
    out_full = out_full.drop(columns=[i for i in out_full.columns.to_list() if i.endswith("_r")])

    ## standardize by area AND ADD NUM FIELDS / VIL
    out_full['DstCenAreNrm'] = out_full['dst2cntr']/out_full['villArea']*1000
    out_full['FldVilCount'] = out_full.reset_index().groupby(['villID'])['villID'].count()
    out_full['DstCenCntNrm'] = out_full['dst2cntr']/out_full['FldVilCount']

    neighbor_bounds = []
    count_all_pts = []
    count_thisVil_pts = []
    for villID, v in out_full.iterrows():
        neighbGeom = v.geometry.buffer(buffNeighb)
        count_all_pts.append(len(out_full[out_full.geometry.within(neighbGeom) ]))
        thisVill = out_full[out_full.index == villID]
        count_thisVil_pts.append(len(thisVill[ thisVill.geometry.within(neighbGeom) ]))

    out_full['CntVil'] = count_thisVil_pts
    out_full['CntAll'] = count_all_pts
    out_full['CntVilRat'] = out_full['CntVil'] / out_full['CntAll'] 
    out_full=out_full[out_full['CntVilRat'] > CntVilRat_Thresh_pct] 
    out_full=out_full[out_full['dst2cntr'] < dst2cntr_Thresh_m] 

    return out_full


    
def rm_bad_olap(villages, thresh, outDir):
    badOlap = villages[villages['BadOlp'] > thresh] ## [['treated']]
    bad_olaps = []
    for v in badOlap.index.to_list():
        bad_vil = villages[villages.index == v]
        bad = bad_vil.index.to_list()
        t= gpd.sjoin(bad_vil, villages, predicate='intersects')
        badWolap = t['index_right'].to_list()
        bad_olaps.append(badWolap)
    bad_olap_vills = list(chain(*bad_olaps))

    bad_vil = villages[villages.index.isin(bad_olap_vills)]
    bad_vil_T = bad_vil[bad_vil['treated'] == 1]
    bad_vil_C = bad_vil[bad_vil['treated'] == 0]

    new_badC = bad_vil_C.overlay(bad_vil_T, how='symmetric_difference')
    new_badC = gpd.sjoin(new_badC, bad_vil_C, how="inner")
    new_badC = new_badC[new_badC['villName_1'].notna()]
    new_badT = bad_vil_T.overlay(bad_vil_C, how='symmetric_difference')
    new_badT = gpd.sjoin(new_badT, bad_vil_T, how="inner")
    new_badT = new_badT[new_badT['villName_1'].notna()]
    new_bad = pd.concat([new_badT, new_badC])
    new_bad.columns = [i.replace("villName_1", "villName2") for i in new_bad.columns.to_list()]
    new_bad = new_bad[[i for i in new_bad.columns.to_list() if "_1" not in i and "_2" not in i]]
    new_bad.columns = [i.replace("index_right", "villID") for i in new_bad.columns.to_list()]
    new_bad=new_bad[new_bad['villName2'] == new_bad['villName']]
    new_bad= new_bad.drop_duplicates()
    good_vil = villages[~villages.index.isin(bad_olap_vills)]
    
    all_new = pd.concat([new_bad.set_index('villID'), good_vil])
    all_new = all_new[[i for i in all_new.columns.to_list() if "villName2" not in i ]]
    print(len(all_new))
    return all_new    


########################

def est_bounds(points_fi, og_olap_pct, og_size_pctile, KDE_BW, KDE_DropVal, PPA_neighb_buff, PPA_same_neighb_ratio, PPA_dist2cntr, out_dir, KDE_dir, plot=False):
    version_suffix='KDE_'+str(og_olap_pct).replace(".", "pt")+"_"+str(og_size_pctile).replace(".", "pt")+"_"+str(KDE_BW)+"_"+str(KDE_DropVal).replace(".", "pt")
    version = version_suffix+'_PPA_'+str(PPA_neighb_buff)+'_'+str(PPA_same_neighb_ratio).replace(".", "pt")+'_'+str(PPA_dist2cntr).replace(".", "pt")

    if not os.path.exists(os.path.join(out_dir, "NewBounds"+version+"wOlap_UTM32.gpkg")):
        print('villages w/ > '+str(og_olap_pct*100)+'% area overlapping a different treatment group village')
        print('OR is > '+str(og_size_pctile*100)+' percentile for area') 
        print('will be considered in the kernel density estimation to drop distant points')
        print('with a bandwith of '+str(KDE_BW)+' and threshold of '+str(KDE_DropVal))
        print('--')
        all_points = gpd.read_file(points_fi)
        ## create dict to add village name 
        village_code_dict = dict(zip(all_points['villID'].to_list(), (all_points['villName'].to_list())))
        ## create dict to add treated value 
        village_treatment_dict = dict(zip(all_points['villID'].to_list(), (all_points['treated'].to_list()))) 

        global villIDs
        villIDs='villID'

        ## 1) calculate convex hull, find % overlap
        orig_bounding_geo = convex_hull(gdf=all_points, group_col=villIDs)
        ## add treatment and village name from village ID dict
        orig_bounding_geo['villName'] = [village_code_dict.get(i)for i in orig_bounding_geo[villIDs]]
        orig_bounding_geo['treated'] = [village_treatment_dict.get(i)for i in orig_bounding_geo[villIDs]]
        orig_bounding_geo['Area'] = orig_bounding_geo.area
        ## calc cross-treatment overlap statistics to find bad overlap area 
        orig_bounding_geo = TC_overlap(orig_bounding_geo)
        orig_bounding_geo.crs = all_points.crs
        orig_bounding_geo.to_file(os.path.join(out_dir, "0_OrigBounds_UTM32.gpkg"))
        if plot==True:
            TC_histo(orig_bounding_geo, 'BadOlp')

        ## 2) run KDE on villages 
        global origGeoms_wPctOlaps1
        origGeoms_wPctOlaps1=orig_bounding_geo.reset_index()
        pctile = origGeoms_wPctOlaps1['Area'].quantile(q=og_size_pctile)
        print('original: average % overlap between T and C villages: '+str(np.mean(origGeoms_wPctOlaps1['BadOlp'])))

        ## 3) drop points below certain kernel density value if it has large overlap or the polygon is large 
        ## gdf of points to CHANGE
        point4KDE = origGeoms_wPctOlaps1[(origGeoms_wPctOlaps1['pctOlap'] > og_olap_pct) | (origGeoms_wPctOlaps1['Area'] > pctile)]
        ## list of village IDs to CHANGE 
        villIDs_4KDE = point4KDE['villID']
        ## gdf of points to LEAVE 
        points2leave = origGeoms_wPctOlaps1[~origGeoms_wPctOlaps1['villID'].isin(villIDs_4KDE)]
        ## list of village IDs to LEAVE 
        leave_villIDs = points2leave['villID']
        ## convert points to multipoint per village
        leave_pts=[]
        for i in leave_villIDs:
            current = all_points[all_points['villID'] == i]
            leave_pts.append(current['geometry'])
        leave_points_multi_gdf = gpd.GeoDataFrame(leave_villIDs, geometry=[shapely.MultiPoint(i.to_list()) for i in leave_pts])
        ## drop far points
        KDE_points, KDEs = KDE_dropFAR(points_fi=points_fi, gdf=all_points, IDs=villIDs_4KDE, group_col='villID', bandWidth=KDE_BW, Kdens_drop=KDE_DropVal, KDE_dir=KDE_dir)

        KDE_points_multi = gpd.GeoDataFrame(villIDs_4KDE, 
                                            geometry=[shapely.MultiPoint(i.to_list()) for i in KDE_points], 
                                           crs=all_points.crs)

        ## 4) convex hulls from new and orig points
        new_pts_for_hulls = pd.concat([KDE_points_multi, leave_points_multi_gdf], axis=0)
        ##new_pts_for_hulls.crs = all_points.crs

        ## create convex hulls 
        convex_hulls = [i.convex_hull for i in new_pts_for_hulls['geometry']]
        new_geoms = gpd.GeoDataFrame(new_pts_for_hulls['villID'], geometry=convex_hulls, columns=['villID'])
        new_geoms['treated'] =[village_treatment_dict.get(i) for i in new_geoms['villID']]
        new_geoms['villName'] =[village_code_dict.get(i) for i in new_geoms['villID']]
        new_geoms1 = gpd.GeoDataFrame(new_geoms, 
                                      geometry=new_geoms.geometry.buffer(100), 
                                      crs=all_points.crs)

        new_geoms_OLAP = TC_overlap(new_geoms1)
        ### _1 comes from orig geoms... so can see previous version's olap %s.. TreatFlag from 1->0 but not 0-> 1 (means got 'worse')
        new_geoms_wOLAP1 = new_geoms_OLAP.join(orig_bounding_geo, rsuffix="_1")
        new_geoms_wOLAP1 = new_geoms_wOLAP1.drop(columns=[i for i in new_geoms_wOLAP1.columns.to_list() if i.endswith("_1")])
        new_geoms_wOLAP1['villArea'] = new_geoms_wOLAP1.area
        ## new_geoms_wOLAP1.to_file(os.path.join(out_dir, "1_NewBounds_"+version_suffix+"_UTM32.gpkg"), crs=all_points.crs)
        print('after KDE drop points: average % overlap between T and C villages: '+ str(np.mean(new_geoms_wOLAP1['BadOlp'])))

        ## multipoint to point to manually delete some clear outlier points
        multi2pt = new_pts_for_hulls.explode(index_parts=True).reset_index()    
        multi2pt = multi2pt[[i for i in multi2pt.columns.to_list() if "level_" not in i]]
        ## add kernel density value from KDEs dictionary (geom:KDE)
        multi2pt['KDE'] = [KDEs.get(i) for i in list(zip(multi2pt.geometry.x, multi2pt.geometry.y))]
        ## join with orig shapes for other attributes
        multi2pt_wFld = gpd.sjoin(multi2pt, all_points, predicate="intersects", how="inner", lsuffix="", rsuffix="_1")
        multi2pt_wFld = multi2pt_wFld[[i for i in multi2pt_wFld.columns.to_list() if "_1" not in i]]
        multi2pt_wFld.columns = [i.replace("villID_", "villID") for i in multi2pt_wFld.columns.to_list()]

        ## point pattern analysis 
        multi2pt_wFldStats = pt_patterns(pts_df=multi2pt_wFld, 
                                         poly_shape=new_geoms_wOLAP1, 
                                         group_col="villID", 
                                         buffNeighb=PPA_neighb_buff, 
                                         CntVilRat_Thresh_pct=PPA_same_neighb_ratio, 
                                         dst2cntr_Thresh_m=PPA_dist2cntr)
        ## visual removal
        pts_to_remove = gpd.read_file("/home/l_sharwood/code/demilunes/vilBoundEst/rmOutlierPts.gpkg")
        HHFLD_2rm = pts_to_remove['HH_fld'].to_list()
        newPts = multi2pt_wFldStats[~multi2pt_wFldStats['HH_fld'].isin(HHFLD_2rm)]

       ## newPts.to_file(os.path.join(out_dir, "2_NewPoints_"+version+"_UTM32.gpkg"))

        ## new points to polys: convex hull 
        newBounds = convex_hull(gdf=newPts.reset_index(), group_col="villID")
        newBounds['villName'] = [village_code_dict.get(i) for i in newBounds[villIDs]]
        newBounds['treated'] = [village_treatment_dict.get(i) for i in newBounds[villIDs]]
        newBounds['newArea'] = newBounds.area
        newBounds = gpd.GeoDataFrame(newBounds, 
                                     geometry=newBounds.geometry.buffer(100), 
                                     crs=newPts.crs)
        newBoundsGeo = TC_overlap(newBounds)
        print('after points analysis & vis inspec... average % overlap between T and C villages--'+ str(version)+': '+ str(np.mean(newBoundsGeo['BadOlp'])))

       ## newBoundsGeo.to_file(os.path.join(out_dir, "2_NewBounds_"+version+"_UTM32.gpkg"))
        if plot==True:
            TC_histo(newBoundsGeo, 'BadOlp')

        newBounds_wOlap = rm_bad_olap(villages=newBoundsGeo, thresh=0.75, outDir=out_dir)
       ## newBounds_wOlap.to_file(os.path.join(out_dir, "NewBounds"+version+"wOlap_UTM32.gpkg"))
        return newBounds_wOlap
    
    
import shapely
from shapely import Polygon, MultiPolygon


def find_largest(gdf):
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


########################

def param_assess(assess_fname, noLap_shp_fname, og_gdf, orig_overlap_pctile_4KDE, orig_size_pctile_4KDE, KDE_bandwidth, KDE_rmBlThresh, ppa_neighb_buff, ppa_same_neighb_ratio, ppa_dist2cntr):

    no_lap_shp = gpd.read_file(noLap_shp_fname)

    PTwithin = gpd.sjoin(og_gdf[['villID', 'geometry', 'treated']], no_lap_shp, how="inner", lsuffix="pt", rsuffix="ply")

    Cwithin = PTwithin[PTwithin['treated_pt'] == 0]
    Twithin = PTwithin[PTwithin['treated_pt'] == 1] 

    ## find number of C within T and T within T 
    CNOLAPvills = no_lap_shp[no_lap_shp['treated'] == 0]
    PTwithinC = gpd.sjoin(og_gdf[['villID', 'geometry', 'treated']], CNOLAPvills, how="inner", lsuffix="pt", rsuffix="ply")
    CwithinC = PTwithinC[PTwithinC['treated_pt'] == 0]
    TwithinC = PTwithinC[PTwithinC['treated_pt'] == 1] 

    ## find number of T within C and C within C 
    TNOLAPvills = no_lap_shp[no_lap_shp['treated'] == 1]
    PTwithinT = gpd.sjoin(og_gdf[['villID', 'geometry', 'treated']], TNOLAPvills, how="inner", lsuffix="pt", rsuffix="ply")
    CwithinT = PTwithinT[PTwithinT['treated_pt'] == 0]
    TwithinT = PTwithinT[PTwithinT['treated_pt'] == 1] 

    num_orig_pts = len(og_gdf)
    num_olap_pts = len(PTwithin)
    numCwithin = len(Cwithin)
    numTwithin = len(Twithin)
    numCwithinC = len(CwithinC)
    numTwithinC = len(TwithinC)
    numCwithinT = len(CwithinT)
    numTwithinT = len(TwithinT)

    version_suffix='KDE_'+str(orig_overlap_pctile_4KDE).replace(".", "pt")+"_"+str(orig_size_pctile_4KDE).replace(".", "pt")+"_"+str(KDE_BW)+"_"+str(KDE_DropVal).replace(".", "pt")
    version_suff=version_suffix+'_PPA_'+str(ppa_neighb_buff)+'_'+str(ppa_same_neighb_ratio).replace(".", "pt")+'_'+str(ppa_dist2cntr).replace(".", "pt")
    params = [
        ["TOTAL corect points", numCwithinC+numTwithinT],
        ["filename", "NewBounds_"+version_suff+".gpkg"],
        ["orig overlap percentile to consider for kernel density estimation (KDE)", orig_overlap_pctile_4KDE],
        ["orig size percentile to consider for kernel density estimation (KDE)", orig_size_pctile_4KDE],
        ["KDE bandwith", KDE_bandwidth],
        ["remove values below X KDE value", KDE_rmBlThresh],
        ["point pattern analysis (PPA) buffer size to consider as neighbor", ppa_neighb_buff],
        ["same village neighbor fields / all neighbor fields-- drop ratios below", ppa_same_neighb_ratio],
        ["distance to cluster center of pts in that village", ppa_dist2cntr],
        ["num orig pts within orig bounds", num_orig_pts],
        ["num pts within new bounds", num_olap_pts],
        ["num C pts within new bounds", numCwithin],
        ["num T pts within new bounds", numTwithin],
        ["num T pts w/in C vills", numCwithinC],
        ["num T pts w/in T vills", numTwithinT] ,   
        ["num T pts w/in C vills", numTwithinC],
        ["num C pts w/in T vills", numCwithinT],
    ]

    df=pd.DataFrame(params).T
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.reset_index()
    df = df.set_index("filename")
    df = df.drop(columns=["index"])

    hdr = False  if os.path.isfile(assess_fname) else True
    df.to_csv(assess_fname, mode='a', header=hdr)

    return df.sort_values("TOTAL corect points")


def main():
    
    ogPts=sys.argv[1]
    orig_overlap_pctile_4KDE=sys.argv[2]
    orig_size_pctile_4KDE=sys.argv[3]
    out_dir=sys.argv[4]
    
    outDir=os.path.join(out_dir, "bounds_op"+orig_overlap_pctile_4KDE.replace(".", "pt")+"_sp"+orig_size_pctile_4KDE.replace(".", "pt"))
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        
    KDE_tmp_dir=os.path.join( out_dir, "KDE_tmp", "op"+orig_overlap_pctile_4KDE.replace(".", "pt")+"_sp"+orig_size_pctile_4KDE.replace(".", "pt"))
    if not os.path.exists(KDE_tmp_dir):
        os.makedirs(KDE_tmp_dir)
        
    for KDE_BW in [800, 1000, 1200]: ## 800 ##1000, 1000,
        for KDE_rmBlThresh in [1.2, 1.4]: ## 1.0, 1.2, 1.4
            for ppa_neighb_buff in [200, 400, 600, 800]:
                for ppa_same_neighb_ratio in [0.1, 0.2, 0.3, 0.4]:
                    for ppa_dist2cntr in [5000, 6000, 7000]: ## 4000
                            
                        outDir=os.path.join(out_dir, "bounds_op"+str(orig_overlap_pctile_4KDE).replace(".", "pt")+"_sp"+str(orig_size_pctile_4KDE).replace(".", "pt"))
                        if not os.path.exists(outDir):
                            os.makedirs(outDir)

                        KDE_tmp_dir=os.path.join( out_dir, "KDE_tmp", "op"+str(orig_overlap_pctile_4KDE).replace(".", "pt")+"_sp"+str(orig_size_pctile_4KDE).replace(".", "pt"))
                        if not os.path.exists(KDE_tmp_dir):
                            os.makedirs(KDE_tmp_dir)
                        version_suffix='KDE_'+str(orig_overlap_pctile_4KDE).replace(".", "pt")+"_"+str(orig_size_pctile_4KDE).replace(".", "pt")+"_"+str(KDE_BW)+"_"+str(KDE_rmBlThresh).replace(".", "pt")
                        version = version_suffix+'_PPA_'+str(ppa_neighb_buff)+'_'+str(ppa_same_neighb_ratio).replace(".", "pt")+'_'+str(ppa_dist2cntr).replace(".", "pt")
                        if not os.path.exists(os.path.join(outDir, "NewBounds"+version+"wOlap_UTM32.shp")):
                            try:
                                villages = est_bounds(points_fi=ogPts, og_olap_pct=np.float64(str(orig_overlap_pctile_4KDE)), og_size_pctile=np.float64(orig_size_pctile_4KDE), KDE_BW=KDE_BW, KDE_DropVal=KDE_rmBlThresh, PPA_neighb_buff=ppa_neighb_buff, PPA_same_neighb_ratio=ppa_same_neighb_ratio, PPA_dist2cntr=ppa_dist2cntr, out_dir=outDir, KDE_dir=KDE_tmp_dir)
                            except:
                                print('error on '+version)
                            villages.to_file(os.path.join(outDir, "NewBounds"+version+"wOlap_UTM32.shp"))
                            print('----------')


if __name__ == "__main__":
    main()



