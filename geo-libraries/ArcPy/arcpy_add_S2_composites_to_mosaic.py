import arcpy
arcpy.env.workspace = "C:/Users/Lauren/Documents/00_Paraguay/segmentations/MyProject6/"

remove_MMA_expresstion = "Name LIKE '%MinMaxAmp%'"
remove_JJN_expresstion = "Name LIKE '%JanJunNov%'"

arcpy.AddRastersToMosaicDataset_management("MyProject6.gdb/S2_2021_MMA",  "Raster Dataset", "G:/Shared drives/emlab/projects/current-projects/cel-Paraguay-crops/Data/01b_Remote_sensing_images/Py_Composites", "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "UPDATE_OVERVIEWS", "2", "#", "#", "#", "*_2021*", "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
arcpy.RemoveRastersFromMosaicDataset_management("MyProject6.gdb/S2_2021_MMA", remove_JJN_expresstion, "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
print('done with removing JJN from S2_2021_MMA')

arcpy.AddRastersToMosaicDataset_management("MyProject6.gdb/S2_2021_JanJunNov",  "Raster Dataset", "G:/Shared drives/emlab/projects/current-projects/cel-Paraguay-crops/Data/01b_Remote_sensing_images/Py_Composites", "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "UPDATE_OVERVIEWS", "2", "#", "#", "#", "*_2021*", "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
arcpy.RemoveRastersFromMosaicDataset_management("MyProject6.gdb/S2_2021_JanJunNov", remove_MMA_expresstion, "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
print('done with removing MMA from S2_2021_JJN')

arcpy.AddRastersToMosaicDataset_management("MyProject6.gdb/S2_2017_MMA",  "Raster Dataset", "G:/Shared drives/emlab/projects/current-projects/cel-Paraguay-crops/Data/01b_Remote_sensing_images/Py_Composites", "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "UPDATE_OVERVIEWS", "2", "#", "#", "#", "*_2017*", "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
arcpy.RemoveRastersFromMosaicDataset_management("MyProject6.gdb/S2_2017_MMA", remove_JJN_expresstion, "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
print('done with S2_2017_MMA')

arcpy.AddRastersToMosaicDataset_management("MyProject6.gdb/S2_2017_JanJunNov",  "Raster Dataset", "G:/Shared drives/emlab/projects/current-projects/cel-Paraguay-crops/Data/01b_Remote_sensing_images/Py_Composites", "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "UPDATE_OVERVIEWS", "2", "#", "#", "#", "*_2017*", "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
arcpy.RemoveRastersFromMosaicDataset_management("MyProject6.gdb/S2_2017_JanJunNov", remove_MMA_expresstion, "UPDATE_BOUNDARY", "MARK_OVERVIEW_ITEMS", "DELETE_OVERVIEW_IMAGES", "DELETE_ITEM_CACHE", "REMOVE_MOSAICDATASET_ITEMS", "UPDATE_CELL_SIZES")
print('done with S2_2017_JanJunNov')