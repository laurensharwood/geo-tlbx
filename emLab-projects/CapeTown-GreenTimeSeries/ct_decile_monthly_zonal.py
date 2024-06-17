#!/usr/bin/ python

import os, sys
import pandas as pd
import geopandas as gpd
import time
from rasterstats import zonal_stats

def main():

    inShp=sys.argv[1] ## '/home/sandbox-cel/capeTown/vector/deciDisolv_Mask2017_utm32S.shp'
    month_dir=sys.argv[2] ## '/home/sandbox-cel/capeTown/monthly/landsat/vrts' ## '/home/sandbox-cel/capeTown/monthly/landsat'
    VI=sys.argv[3] ## 'evi' #, 'ndvi', 'ndmi1', 'ndmi2', 'ndwi','ndwi_rev'
    startYr=int(sys.argv[4]) ## 2016
    endYr=int(sys.argv[5]) ## 2021
    
    ###################

    deciYr = int(inShp.split(".")[-2][-4:])     
    IDcol = 'd'+str(deciYr)
    sensor_version = month_dir.split("/")[-2]
    outFi=inShp.replace(".shp", "_"+sensor_version+"_"+VI+"_z.csv")
    print(outFi)
    dfs=[]
    if not os.path.exists(outFi):
        monthly_deciles = {}
        for year in list(range(startYr, endYr+1, 1)):
            for month in list(range(1, 13, 1)):
                date = str(year)+str(month).zfill(2)
                monthRast = sorted([os.path.join(month_dir, file) for file in os.listdir(month_dir) if VI+"_"+date in file])[0]
                col_name = '-'.join(os.path.basename(monthRast).replace(".vrt", "").split("_")[1:])   
                print(col_name)

                decile_means = zonal_stats(inShp, monthRast, stats="mean")
                ## dictionary w/ date as key, list of means as value
                monthly_deciles.update({date:[i['mean'] for i in decile_means]})
                df = pd.DataFrame.from_dict(monthly_deciles)
                dfs.append(df)
            print(dfs[-1])
    dfs_comb=pd.concat(dfs, axis=0)
    print(dfs_comb.tail(10))
    dfs_comb.tail(10).to_csv(outFi)

if __name__ == '__main__':
    main()




