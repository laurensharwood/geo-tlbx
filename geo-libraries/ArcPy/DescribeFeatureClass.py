# lauren sharwood
#3/6/18
# describe a feature class.
# if it is a point, add message saying it's a point.
# if it's a polyline, add message saying it's a polyline.
# if it's a polygon, add message saying it's a polygon.
# it it's neither of those things, say it's unknown 

import arcpy, sys

inputVar = sys.argv[1]
desc = arcpy.Describe(inputVar)


if desc.shapeType == "Point":
    fcType = "Shape type is a point"
elif desc.shapeType == "Polyline":
    fcType = "Shape type is a polyline"
elif desc.shapeType == "Polygon":
    fcType = "Shape type is a polygon"
else:
    fcType = "Shape type is unknown"


arcpy.AddMessage(fcType)
