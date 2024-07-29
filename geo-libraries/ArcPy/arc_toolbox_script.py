# 1. enter file path to geodatabase where mosaics will be created. makes gdb if it doesn't exist
# gdb: G:\Shared drives\emlab\projects\current-projects\cel-Paraguay-crops\Data\01b_Remote_sensing_images\Mosaic_Datasets\caazapa_images.gdb

# 2. single or list of mosaics to create. for Sentinel2, mosaic type(MMA or JJN), - to separate the year (2017 or 2021). for Planet, month with / to separate the year 
# mosaic: JJN-2021,MMA-2021,01/2021,02/2021,03/2021,04/2021,05/2021,06/2021,07/2021,08/2021,09/2021,10/2021,11/2021,12/2021
# mosaic: JJN-2017,MMA-2017,01/2017,02/2017,03/2017,04/2017,05/2017,06/2017,07/2017,08/2017,09/2017,10/2017,11/2017,12/2017


import arcpy
import sys

PS_img_path = r"G:\Shared drives\emlab\projects\current-projects\cel-Paraguay-crops\Data\01b_Remote_sensing_images\Py_PlanetBasemaps"
epsg_3857 = r"G:\Shared drives\emlab\projects\current-projects\cel-Paraguay-crops\Data\01b_Remote_sensing_images\Mosaic_Datasets\epsg\EPSG3857.shp"
S2_img_path = r"G:\Shared drives\emlab\projects\current-projects\cel-Paraguay-crops\Data\01b_Remote_sensing_images\Py_Composites"
epsg_8858 = r"G:\Shared drives\emlab\projects\current-projects\cel-Paraguay-crops\Data\01b_Remote_sensing_images\Mosaic_Datasets\epsg\EPSG8858.shp"

def add_rasters(env_workspace_plus_gdb, mos):
    
    if '/20' in mos:
        month = mos.split("/")[0]
        year = mos.split("/")[-1]
        mosaic_name = "PS_" + str(month) + "_" + str(year)
        gdb_and_mosaic_name = str(gdb_name) + '/' + str(mosaic_name)
        filterdates = "*" + str(year) + "_" + str(month) + "_*"
        spatial_ref = arcpy.Describe(epsg_3857).spatialReference
        if arcpy.Exists(gdb_and_mosaic_name):
            arcpy.AddMessage('mosaic exists... removing rasters from '+ str(gdb_and_mosaic_name)+ ' dataset and recreating')
            arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%_%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
        else:
            arcpy.CreateMosaicDataset_management(gdb_name, mosaic_name, spatial_ref, "5")
        arcpy.AddRastersToMosaicDataset_management(gdb_and_mosaic_name,  "Raster Dataset", PS_img_path, "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "UPDATE_OVERVIEWS", "2", "#", "#", "#",  filterdates, "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
        arcpy.AddMessage('created and added images to '+ str(gdb_and_mosaic_name))
    name_delim = arcpy.AddFieldDelimiters(S2_img_path, "Name")
        
    if "MMA" in mos:
        year = mos.split('-')[-1]
        mosaic_name = "S2_" + str(year) + "_" +  mos.split('-')[0]
        gdb_and_mosaic_name = str(gdb_name) + '/' + str(mosaic_name)
        year_filter ="*" + str(year) + "*" ##### FIX
        spatial_ref = arcpy.Describe(epsg_8858).spatialReference
        if arcpy.Exists(gdb_and_mosaic_name):
            arcpy.AddMessage('mosaic exists... removing rasters from ' + str(gdb_and_mosaic_name) + ' dataset and recreating')
            arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%_%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
        else:
            arcpy.CreateMosaicDataset_management(gdb_name, mosaic_name, spatial_ref, "3")
        arcpy.AddRastersToMosaicDataset_management(gdb_and_mosaic_name,  "Raster Dataset", S2_img_path, "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "NO_OVERVIEWS", "2", "#", "#", "#", year_filter, "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
        arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%JanJunNov%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
        arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%(1)%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
        arcpy.AddMessage('created and added images to ' + str(gdb_and_mosaic_name))
    if 'JJN' in mos:
        year = mos.split('-')[-1]
        mosaic_name = "S2_" + str(year) + "_" +  mos.split('-')[0]
        gdb_and_mosaic_name = str(gdb_name) + '/' + str(mosaic_name)
        year_filter ="*" + str(year) + "*" ##### FIX
        spatial_ref = arcpy.Describe(epsg_8858).spatialReference
        if arcpy.Exists(gdb_and_mosaic_name):
            arcpy.AddMessage('mosaic exists... removing rasters from '+ str(gdb_and_mosaic_name)+ ' dataset and recreating')
            arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%_%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
        else:
            arcpy.CreateMosaicDataset_management(gdb_name, mosaic_name, spatial_ref, "3")
        arcpy.AddRastersToMosaicDataset_management(gdb_and_mosaic_name,  "Raster Dataset", S2_img_path, "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "NO_OVERVIEWS", "2", "#", "#", "#", year_filter, "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
        arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%MinMax%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")    
        arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%(1)%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
        arcpy.AddMessage('created and added images to ' + str(gdb_and_mosaic_name))
        
    return
    
    
# This is used to execute code if the file was run but not imported
if __name__ == '__main__':
    env_workspace_plus_gdb = arcpy.GetParameterAsText(0)        
    env_workspace_split = env_workspace_plus_gdb.split('\\')
    gdb_name = env_workspace_split[-1]
    arcpy.env.workspace = env_workspace_plus_gdb.replace(gdb_name, '')
    mosaic_type = arcpy.GetParameterAsText(1)
    mosaic_type_list = mosaic_type.split(',')
    if arcpy.Exists(env_workspace_plus_gdb):
        arcpy.AddMessage('gdb exists :) ')
        for m in mosaic_type_list:
            add_rasters(env_workspace_plus_gdb, m)
    else:
        arcpy.CreateFileGDB_management(arcpy.env.workspace, gdb_name)
        arcpy.AddMessage('gdb does not exist, creating one there then adding rasters...')
        for m in mosaic_type_list:
            add_rasters(env_workspace_plus_gdb, m)
    arcpy.AddMessage('script finished')
    
    # Update derived parameter values using arcpy.SetParameter() or arcpy.SetParameterAsText()
