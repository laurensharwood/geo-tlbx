# lauren sharwood
#3/6/18
# describe a raster: (1) say whether z values are an integer or float.
# (2) tell the user the 4 corners of the bounding box 


import arcpy, sys

rasterDesc = sys.argv[1]
raster = arcpy.Describe(rasterDesc)
arcpy.AddMessage('\n')
if raster.IsInteger:
    arcpy.AddMessage('Input raster values are integers.')
else:
    arcpy.AddMessage('Input raster values are float.')
arcpy.AddMessage('\n')

extentt = raster.extent
xmin = extentt.XMin
ymin = extentt.YMin
xmax = extentt.XMax
ymax = extentt.YMax

arcpy.AddMessage('Full extent of the raster:')
arcpy.AddMessage(str(extentt))
arcpy.AddMessage('\n')
arcpy.AddMessage('x_min is:')
arcpy.AddMessage(str(xmin))
arcpy.AddMessage('y_min is:')
arcpy.AddMessage(str(ymin))
arcpy.AddMessage('x_max is:')
arcpy.AddMessage(str(xmax))
arcpy.AddMessage('y_max is:')
arcpy.AddMessage(str(ymax))
arcpy.AddMessage('\n')
