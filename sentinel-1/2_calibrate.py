#!/usr/bin/ python3.6

import sys, os, gc
snappy_install_location='/home/lsharwood/.snap/snap-python'
sys.path.append(snappy_install_location)
import datetime
import time
import snappy
from snappy import ProductIO, GPF, HashMap

## https://docs.terradue.com/better-pipelines/tutorial/step-4.html 

inpath =  sys.argv[1]
outpath = sys.argv[2] 
grid_file = sys.argv[3]
## bash script:
## cd ~/
## cd /home/lsharwood/.snap/snap-python 
## python3.6 ~/code/bash/calibrate_S1.py $inpath $outpath $grid
## cd ~/


def apply_orbit_file(source):
    parameters = HashMap()
    parameters.put('orbitType', 'Sentinel Precise (Auto Download)')
    parameters.put('polyDegree', '3')
    parameters.put('continueOnFail', 'false')
    parameters.put('Apply-Orbit-File', True)
    output = GPF.createProduct('Apply-Orbit-File', parameters, source)
    return output

def border_noise_removal(source):
    parameters = HashMap()
    output = GPF.createProduct('Remove-GRD-Border-Noise', parameters, source)
    return output    

def thermal_noise_removal(source):
    parameters = HashMap()
    parameters.put('removeThermalNoise', True)
    output = GPF.createProduct('ThermalNoiseRemoval', parameters, source)
    return output

def calibration(source, polarization, pols):
    parameters = HashMap()
    parameters.put('outputSigmaBand', True)
    if polarization == 'DH':
        parameters.put('sourceBands', 'Intensity_HH,Intensity_HV')
    elif polarization == 'DV':
        parameters.put('sourceBands', 'Intensity_VH,Intensity_VV')
    elif polarization == 'SH' or polarization == 'HH':
        parameters.put('sourceBands', 'Intensity_HH')
    elif polarization == 'SV':
        parameters.put('sourceBands', 'Intensity_VV')
    else:
        print("different polarization!")
    parameters.put('selectedPolarisations', pols)
    parameters.put('auxFile', 'Product Auxiliary File')
    parameters.put('outputImageScaleInDb', False)
    parameters.put('createBetaBand', 'true')
    parameters.put('outputBetaBand', 'false')
    output = GPF.createProduct("Calibration", parameters, source)
    return output

def speckle_filtering(source):
    parameters = HashMap()
    parameters.put('filter', 'Lee')
    parameters.put('filterSizeX', 5)
    parameters.put('filterSizeY', 5)
    output = GPF.createProduct('Speckle-Filter', parameters, source)
    return output

def terrain_correction(source):
    parameters = HashMap()
    parameters.put('demName', 'SRTM 3Sec')
    parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('pixelSpacingInMeter', '10.0')
    parameters.put('saveProjectedLocalIncidenceAngle', False)
    parameters.put('saveSelectedSourceBand', True)
    parameters.put('auxFile', 'Latest Auxiliary File')
    output = GPF.createProduct('Terrain-Correction', parameters, source)
    return output

def subset(source, wkt):
    parameters = HashMap()
    parameters.put('geoRegion', wkt)
    output = GPF.createProduct('Subset', parameters, source)
    return output


def wkt_bounds(grid_file):
    aoi = gpd.read_file(grid_file)
    aoi_4326 = aoi.to_crs(4326)
    ## Extract the Bounding Box Coordinates
    bounds = aoi_4326.total_bounds
    ## Create GeoDataFrame of the Bounding Box 
    gdf_bounds = gpd.GeoSeries([box(*bounds)])
    ## Get WKT Coordinates
    wkt_aoi = gdf_bounds.to_wkt().values.tolist()[0]
    return wkt_aoi

##########################################################################################################################
start = time.time()
            
## grid shape wkt (from S1_download.sh out file)
wkt = wkt_bounds(grid_file)
## All Sentinel-1 data sub folders are located within a super folder (make sure the data is already unzipped and each sub folder name ends with '.SAFE'):

if not os.path.exists(outpath):
    os.makedirs(outpath)
zipped_folders = [i for i in os.listdir(inpath) if i.endswith(".zip")]
unzipped_folders = [i for i in os.listdir(inpath) if i.endswith(".SAFE")]
if len(unzipped_folders) == 0 and len(zipped_folders) > 0:
    os.chdir(inpath)
    os.system("unzip '*.zip'")

new_unzipped = [i for i in os.listdir(inpath) if i.endswith(".SAFE")]
for folder in sorted(new_unzipped):
    gc.enable()
    gc.collect()

    sentinel_1 = ProductIO.readProduct(os.path.join(inpath, folder, "manifest.safe"))
    print(sentinel_1)

    loopstarttime=str(datetime.datetime.now())
    print('Start time:', loopstarttime)
    start_time = time.time()

    ## Extract mode, product type, and polarizations from filename
    polarization = folder.split("_")[3][2:4]
    if polarization == 'DV':
        pols = 'VH,VV'

        ## Start preprocessing:
        print('Apply orbit file...')
        applyorbit = apply_orbit_file(sentinel_1)

        print('Border noise removal...')
        rmborders = border_noise_removal(applyorbit)
        del applyorbit

        print('Thermal noise removal...')
        thermaremoved = thermal_noise_removal(rmborders)
        del rmborders

        print('Calibration...')
        calibrated = calibration(thermaremoved, polarization, pols)
        del thermaremoved

        print('Speckle filtering...') ## before terrain correction 
        down_filtered = speckle_filtering(calibrated)
        del calibrated

        print('Terrain correction...')
        tercorrected = terrain_correction(down_filtered)
        del down_filtered

        print('Linear to db...')
        lineartodb = GPF.createProduct('linearToFromdB', HashMap(), tercorrected)
        del tercorrected

        print('Subsetting...')
        sub = subset(lineartodb, wkt)
        del lineartodb

        print("Writing...")
        out_file=os.path.join(outpath, folder[:-5]+'.tif')
        ProductIO.writeProduct(sub, out_file, 'GeoTIFF-BigTIFF')
        del sub

        print('Done.')
        sentinel_1.dispose()
        sentinel_1.closeIO()
        print("--- %s seconds ---" % (time.time() - start_time))
        


end = time.time()
minutes = (end - start)/60
print('time this took ' , str(minutes), ' minutes to complete')
