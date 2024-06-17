import arcpy
arcpy.env.workspace = "C:/Users/Lauren/Documents/00_Paraguay/segmentations/MyProject6/"



arcpy.AddRastersToMosaicDataset_management("MyProject6.gdb/PS_2021_Jan",  "Raster Dataset", "G:/Shared drives/emlab/projects/current-projects/cel-Paraguay-crops/Data/01b_Remote_sensing_images/Py_PlanetBasemaps", "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "UPDATE_OVERVIEWS", "2", "#", "#", "#", "*2021_01_*", "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
print('done with Planet Jan 2021')


arcpy.AddRastersToMosaicDataset_management("MyProject6.gdb/PS_2017_Jan",  "Raster Dataset", "G:/Shared drives/emlab/projects/current-projects/cel-Paraguay-crops/Data/01b_Remote_sensing_images/Py_PlanetBasemaps", "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "UPDATE_OVERVIEWS", "2", "#", "#", "#", "*2017_01_*", "SUBFOLDERS", "EXCLUDE_DUPLICATES", "BUILD_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", "Add Raster Datasets", "#")
print('done with Planet Jan 2017')
