import arcpy
import sys


'''
- find location of python27\ArcGIS10.8\python.exe on Bren computer
- copy Mosaic_Datasets folder with script to create referenced mosaic datasets into an Arc gdb (arcpy_add_PY_images_to_mosaic_datasets.py), EPSG3857, and EPSG 8858 shapefiles for Sentinel-2 and PlanetScope coordinate reference info
open Command Prompt and run... script SPACE path to local gdb SPACE mosaic to create 
EX) 
C:\python27\ArcGIS10.8\python.exe C:\python27\ArcGIS10.8\Mosaic_Datasets\arcpy_add_PY_images_to_mosaic_datasets.py C:\Users\Lauren\Documents\00_Paraguay\segmentations\MyProject6\MyProject6.gdb 
MMA-2017, MMA-2021
JJN-2017, JJN-2021
01/2017, 02/2017, 03/2017, 04/2017, 05/2017, 06/2017, 07/2017, 08/2017, 09/2017, 10/2017, 11/2017, 12/2017
01/2021, 02/2021, 03/2021, 04/2021, 05/2021, 06/2021, 07/2021, 08/2021, 09/2021, 10/2021, 11/2021, 12/2021

'''

# user inputs name of local gdb with path
env_workspace_plus_gdb = sys.argv[1]
# user inputs 'MMA-YYYY' or 'JJN-YYYY' for Sentinel-2 composite, or MM/YYYY for PS monthly images
mosaic_type = sys.argv[2]

env_workspace_split = env_workspace_plus_gdb.split('\\')
gdb_name = env_workspace_split[-1]
arcpy.env.workspace = env_workspace_plus_gdb.replace(gdb_name, '')

PS_img_path = r"G:\Shared drives\emlab/projects\current-projects\cel-Paraguay-crops\Data\01b_Remote_sensing_images\Py_PlanetBasemaps"
S2_img_path = 

if '/20' in mosaic_type:
    month = mosaic_type.split("/")[0]
    year = mosaic_type.split("/")[-1]
    mosaic_name = "PS_" + str(month) + "_" + str(year)
    gdb_and_mosaic_name = str(gdb_name) + '/' + str(mosaic_name)
    filterdates = "*" + str(year) + "_" + str(month) + "_*"
    spatial_ref = arcpy.Describe(r"G:\Shared drives\emlab\projects\current-projects\cel-Paraguay-crops\Data\01b_Remote_sensing_images\Mosaic_Datasets\EPSG3857.shp").spatialReference
    if arcpy.Exists(mosaic_name):
        arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%_%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
    else:
        arcpy.CreateMosaicDataset_management(gdb_name, mosaic_name, spatial_ref, "5")
    arcpy.AddRastersToMosaicDataset_management(gdb_and_mosaic_name,  "Raster Dataset", PS_img_path, "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "UPDATE_OVERVIEWS", "2", "#", "#", "#",  filterdates, "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
    print('created and added images to ', str(gdb_and_mosaic_name))
if "MMA" in mosaic_type:
    year = mosaic_type.split('-')[-1]
    mosaic_name = "S2_" + str(year) + "_" +  mosaic_type.split('-')[0]
    gdb_and_mosaic_name = str(gdb_name) + '/' + str(mosaic_name)
    filterdates = "*" + str(year) + "*"
    spatial_ref = arcpy.Describe(r"G:\Shared drives\emlab\projects\current-projects\cel-Paraguay-crops\Data\01b_Remote_sensing_images\Mosaic_Datasets\EPSG8858.shp").spatialReference
    if arcpy.Exists(mosaic_name):
        arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%_%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
    else:
        arcpy.CreateMosaicDataset_management(gdb_name, mosaic_name, spatial_ref, "3")
    arcpy.AddRastersToMosaicDataset_management(gdb_and_mosaic_name,  "Raster Dataset", "G:/Shared drives/emlab/projects/current-projects/cel-Paraguay-crops/Data/01b_Remote_sensing_images/Py_Composites", "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "UPDATE_OVERVIEWS", "2", "#", "#", "#", filterdates, "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
    arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%MinMax%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
    arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%(1)%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
    print('created and added images to ', str(gdb_and_mosaic_name))
if 'JJN' in mosaic_type:
    year = mosaic_type.split('-')[-1]
    mosaic_name = "S2_" + str(year) + "_" +  mosaic_type.split('-')[0]
    gdb_and_mosaic_name = str(gdb_name) + '/' + str(mosaic_name)
    filterdates = "*" + str(year) + "*"
    spatial_ref = arcpy.Describe(r"G:\Shared drives\emlab\projects\current-projects\cel-Paraguay-crops\Data\01b_Remote_sensing_images\Mosaic_Datasets\EPSG8858.shp").spatialReference
    if arcpy.Exists(mosaic_name):
        arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%_%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
    else:
        arcpy.CreateMosaicDataset_management(gdb_name, mosaic_name, spatial_ref, "3")
    arcpy.AddRastersToMosaicDataset_management(gdb_and_mosaic_name,  "Raster Dataset", "G:/Shared drives/emlab/projects/current-projects/cel-Paraguay-crops/Data/01b_Remote_sensing_images/Py_Composites", "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "UPDATE_OVERVIEWS", "2", "#", "#", "#", filterdates, "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
    arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%JanJunNov%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
    arcpy.RemoveRastersFromMosaicDataset_management(gdb_and_mosaic_name, "Name LIKE '%(1)%'", "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")    
    print('created and added images to ', str(gdb_and_mosaic_name))

else:
    print("after (1) script and (2) gdb, put (3) MM/YYYY for Planet monthly image or (3) 'MMA-YYYY' or 'JJN-YYYY' for Sentinel-2 composite")
    