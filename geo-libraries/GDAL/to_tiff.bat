:: Converts a directory of any gdal-supported raster format to TIFF
:: Before running, make sure the osgeo path is correct.
:: Double click bat file to run, respond to input prompts

@echo off
SET osgeopath=C:\OSGeo4W64\OSGeo4W.bat
SET /p inputdir="Directory of image files: "
SET /p inputextension="File extension of input images (ex: .ecw): "
SET /p outputdir="Output directory "

for %%f in (%inputdir%\*%inputextension%) do %osgeopath% gdal_translate -of GTiff -co ALPHA=UNSPECIFIED -co TFW=YES -co TILED=YES -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 %%f %outputdir%\%%~nf.tif

echo.
echo Processing complete.
echo.

pause
@echo on
