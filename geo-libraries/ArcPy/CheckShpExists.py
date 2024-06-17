# lauren sharwood
#3/6/18
# check if a shapefile exists. If it exists (type in name with .shp to check), delete it
# else (if it doesn't exist), create one (out_shp) in that workspace. 
import arcpy, sys

inputWorkspace = sys.argv[1]
arcpy.env.workspace = inputWorkspace
in_shp = sys.argv[2]
out_shp = sys.argv[3]

if arcpy.Exists(in_shp):
    arcpy.AddMessage('shapefile exists')
    arcpy.Delete_management(in_shp)
    arcpy.AddMessage('shapefile deleted')
else: #or if not
    arcpy.AddMessage('shapefile does not exist... creating shapefile...')
    out_path = inputWorkspace
    geometry_type = "POINT"

    arcpy.CreateFeatureclass_management(out_path, out_shp, geometry_type)
        
