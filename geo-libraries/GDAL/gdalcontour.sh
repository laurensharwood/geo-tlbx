#!/bin/bash 


in_dir="/mnt/c/Users/Lauren/Documents/02_Paraguay/composites_probas/EO_7"
new="_contour.gpkg"

for eachfile in "$in_dir"/*.tif
do
   echo $eachfile
   newName=${eachfile//$extension/$new}
   gdal_contour -a elev $eachfile $newName -i 2.0 -f "GPKG"
   echo $newName

   
done

