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

    version_suffix='KDE_'+f'{orig_overlap_pctile_4KDE:.2f}'[-2:]+"_"+f'{orig_size_pctile_4KDE:.2f}'[-2:]+"_"+str(KDE_bandwidth)+"_"+str(KDE_rmBlThresh).replace(".", "pt")
    version_suff=version_suffix+'_PPA_'+str(ppa_neighb_buff)+'_'+str(ppa_same_neighb_ratio).replace(".", "pt")+'_'+str(ppa_dist2cntr).replace(".", "pt")
    params = [
        ["TOTAL corect points", numCwithinC+numTwithinT],
        ["filename", "2_NewBounds_"+version_suff+"_UTM32.gpkg"],
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
    = "/home/l_sharwood/code/demilunes/vilBoundEst/all_wOlap/"    


if __name__ == "__main__":
    main()




