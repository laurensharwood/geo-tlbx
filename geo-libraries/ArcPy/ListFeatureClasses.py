# lauren sharwood
#3/6/18
# lists the feature classes in the user's input workspace
# counts how many feature classes are in this workspace


import arcpy, sys

inputWorkspace = sys.argv[1]
arcpy.env.workspace = inputWorkspace

arcpy.AddMessage('\n')
listt = arcpy.ListFeatureClasses()

arcpy.AddMessage('Here are the feature classes:')
c = 0
for item in listt:
    arcpy.AddMessage(item)
    c += 1
    str(c)
arcpy.AddMessage('\n')
arcpy.AddMessage('Number of feature classes:')
arcpy.AddMessage(c)
arcpy.AddMessage('\n')
