# !/bin/bash

in_dir="/mnt/c/Users/Lauren/Documents/02_Paraguay/composites_probas/all"

extension=".tif"
new="_WGS84"$extension

yourfilenames="ls $in_dir/*.tif"
for eachfile in $yourfilenames
do
   echo $eachfile
   newName=${eachfile//$extension/$new}
   ###gdal_translate -of GTiff -r bilinear $eachfile $newName
   gdalwarp  -of GTiff  -r bilinear -s_srs EPSG:8858 -t_srs EPSG:4326 $eachfile $newName
   echo $newName
   
done

echo "done"


### create new name