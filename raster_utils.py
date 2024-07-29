import os, sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.features import bounds, shapes
from rasterio.warp import calculate_default_transform, transform, reproject, Resampling
from rasterio.merge import merge
import xarray as xr
from pathlib import Path
from IPython.display import Image
from ipywidgets import interact
import dask.array as da
## under PROCESSING, for mosaic_rasters function
from rasterio.merge import merge
## under PROCESSING, for windowed_read functions
import rasterio as rio
## for functions under QA/QC
import matplotlib.pyplot as plt
## under TRANSFORM, for watershed segmentation function
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
## under TRANSFORM, for kmeans clustering function
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#import geowombat as gw
#from geowombat.core import dask_to_xarray
#from geowombat.moving import moving_window

################################
## PROCESSING
################################

def mosaic_rasters(in_rasters, out_path):
    """uses 'in_rasters': a list of rasters to mosaic or the input directory where all files in that folder are mosaiced; 
    returns 'out_path': directory and name to save mosaic... no extension (will be .tif)"""
    if type(in_rasters) == str:
        rasters = sorted([os.path.join(in_rasters, i) for i in os.listdir(in_rasters) if i.endswith(".tif")])
    elif type(in_rasters) == list:
        rasters = in_rasters
    print(rasters)
    with rio.open(rasters[1]) as src:
        rast_crs=src.crs
        nodata = src.nodata
    # The merge function returns a single array and the affine transform info
    arr, out_trans = merge(rasters)
    out_meta = {'driver': 'GTiff', 'dtype':arr[0][0][0].dtype, 'nodata': nodata, 
    'count': arr.shape[0], 'height': arr.shape[1], 'width': arr.shape[2], 
    'crs': rast_crs, 'transform': out_trans}
    with rio.open(out_path, 'w', **out_meta) as dst:
        dst.write(arr)
    return out_path


def stack_rasters(raster_paths, out_path, overwrite=False):
    """
    stacks rasters
        raster_paths = list of rasters (full file paths)...
        out_path = full file path for output raster stack
    NOTE: all rasters in list must be the same extent, crs, pixel size, dtype, no data value, etc. takes these attributes from first raster in list 
    """
    if Path(out_path).is_file() and overwrite==False:
        print('stack exists, not overwriting')
        return out_path
    else:
        first_band = rio.open(raster_paths[0], 'r')
        out_meta = first_band.meta.copy()
        out_meta.update(count=len(raster_paths))
        big_band_list = []
        for file in raster_paths:
            with rio.open(file, 'r') as src:
                band = src.read(1)
                nodata = src.nodata
                dtype = src.dtype 
                big_band_list.append(band)
        out_meta.update({"count":len(raster_paths), "nodata":nodata, "dtype":dtype})
        with rio.open(out_path, 'w', **out_meta) as dst:
            for i, fi in tqdm(enumerate(raster_paths)):
                dst.write(big_band_list[i], i+1)
            dst.descriptions = tuple([Path(i).stem for i in raster_paths])
            return out_path
            
def create_mask(rast_arr, thresh, remove):
    '''
    Args:
        rast_arr=raster array
        thresh=threshold to remove any values...
        remove=lte or gte
    returns mask, where OFF=0 and ON=1
    '''
    arr_cp = np.copy(rast_arr)
    if remove == "lte":
        mask_arr =np.where(arr_cp > thresh, 1, 0)
    elif remove == "gte":
        mask_arr=np.where(arr_cp < thresh, 1, 0)
    return mask_arr

def mask_raster(rast_arr, thresh, remove):
    '''
    Args:
        rast_arr=raster array
        thresh=threshold to remove any values...
        remove=lte or gte
    returns masked array, where masked/OFF=np.nan and ON=rast_arr pixel values
    '''
    arr_cp = np.copy(rast_arr)
    if remove == "lte":
        masked_arr =np.where(arr_cp > thresh, arr_cp, np.nan)
    elif remove == "gte":
        masked_arr=np.where(arr_cp < thresh, arr_cp, np.nan)
    return masked_arr

def windowed_read(gt, bbox): 
    """ 
    helper function for rasterio windowed reading of chip within grid to save into cnet time_series_vars folder.
    Args:
        gt = main raster's geotransformation (src.transform)
        bbox = bounding box polygon as subset from raster to read in
    returns [xoffset, yoffset, xsize, ysize] for rasterio windowed reading
    """
    origin_x = gt[2]
    origin_y = gt[5]
    pixel_width = gt[0]
    pixel_height = gt[4]
    x1_window_offset = int(round((bbox[0] - origin_x) / pixel_width))
    x2_window_offset = int(round((bbox[1] - origin_x) / pixel_width))
    y1_window_offset = int(round((bbox[3] - origin_y) / pixel_height))
    y2_window_offset = int(round((bbox[2] - origin_y) / pixel_height))
    x_window_size = x2_window_offset - x1_window_offset
    y_window_size = y2_window_offset - y1_window_offset
    return [x1_window_offset, y1_window_offset, x_window_size, y_window_size]

def window_buffered(gt, bbox, bbox_buffer_size): 
    """ 
    helper function for rasterio windowed reading of chip within grid to save into cnet time_series_vars folder.
    Args:
        gt = main raster's geotransformation (raster.transform)
        bbox = bounding box polygon as subset from raster to read in
        bbox_buffer_size = biggest field shouldn't be larger than half this size... 400 adds 200m on four sides
    returns buffered [xoffset, yoffset, xsize, ysize] for rasterio windowed reading
    """
    origin_x = gt[2]
    origin_y = gt[5]
    pixel_width = gt[0]
    pixel_height = gt[4]
    x1_window_offset = int(round((bbox[0] - origin_x) / pixel_width))
    x2_window_offset = int(round((bbox[1] - origin_x) / pixel_width))
    y1_window_offset = int(round((bbox[3] - origin_y) / pixel_height))
    y2_window_offset = int(round((bbox[2] - origin_y) / pixel_height))
    x_window_size = x2_window_offset - x1_window_offset
    y_window_size = y2_window_offset - y1_window_offset
    x1_window_offset_buffered = x1_window_offset - (bbox_buffer_size/2)
    y1_window_offset_buffered = y1_window_offset - (bbox_buffer_size/2)
    x_window_size_buffered = x_window_size + bbox_buffer_size
    y_window_size_buffered = y_window_size + bbox_buffer_size
    return [x1_window_offset_buffered, y1_window_offset_buffered, x_window_size_buffered, y_window_size_buffered]


################################
## TRANSFORM
################################

def reproject_raster(out_crs, in_path):
    """reprojects 'in_path' raster into 'out_crs' EPSG (with "EPSG:") ex: 'EPSG:32632'.
    returns reprojected raster filename."""
    out_path = Path(in_path).stem+"_"+str(out_crs).split(":")[-1]+".tif"
    with rio.open(in_path) as src:
        src_crs = src.crs
        transform, width, height = calculate_default_transform(src_crs, out_crs, src.width, src.height, *src.bounds)
        out_meta = src.meta.copy()
        out_meta.update({'crs': out_crs, 'transform': transform, 'width': width, 'height': height})
        with rio.open(out_path, 'w', **out_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rio.band(src, i), destination=rio.band(dst, i), src_transform=src.transform,
                            src_crs=src.crs, dst_transform=transform, dst_crs=out_crs, resampling=Resampling.nearest)
            dst.descriptions = src.descriptions
    return out_path

def make_reclass_dict(csv_path, old_col, new_col):
    """ 0 stays as 0 """
    reclass_df = pd.read_csv(csv_path)
    old_new_dict = dict(zip(reclass_df.old_col, reclass_df.new_col))
    old_new_dict[0] = 0   
    return old_new_dict

def reclassify_raster(raster_path, old_new, out_dir):
    """
    Args:
        in_dir = input directory of rasters to reclassify
        old_new = dictionary with old:new value as key:value pair. OR csv with 'old' column and 'new' column 
        out_dir = output directory to save reclassed raster, the name as the input raster+'_reclass'
    0 should be no data value 
    """
    if type(old_new) == str:
        reclass_dict = make_reclass_dict(csv_path=old_new, old_col="old", new_col="new")
    elif type(old_new) == dict:
        reclass_dict = old_new
    print(reclass_dict)
    new_name = Path(raster_path).stem+"_reclass.tif"
    with rio.open(raster_path) as src:
        old_arr = src.read(1)
        out_meta = src.meta.copy()
        if len(np.unique(old_arr)) > 1: ## if there are any values other than 0, nodata 
            new_arr = np.vectorize(reclass_dict.get)(old_arr)
            print(raster_path, "old raster vals: ", np.unique(old_arr), "new raster vals: ", np.unique(new_arr))
            out_meta.update({'nodata': 0})
            with rio.open(os.path.join(out_dir, new_name), 'w', **out_meta) as dst:
                dst.write(new_arr, indexes=1)
            print(new_name)

def z_m_to_ft(input_raster, output_raster):
    """
    Args:
        input_raster = input raster (in m) filename
        output_raster = output raster (in ft) filename
    returns xarray raster object where Z-pixel value is converted to ft
    """
    with xr.open_dataset(input_raster) as xrimg:
        B1m = xrimg.Band1
        B1ft = B1m.copy()
        B1ft.values = B1m.values*(3.28084)
        B1ft.rio.to_raster(output_raster)
        return B1ft

def z_ft_to_m(input_raster, output_raster):
    """
    Args:
        input_raster = input raster (in ft) filename
        output_raster = output raster (in m) filename
    returns xarray raster object where Z-pixel value is converted to m
    """
    with xr.open_dataset(input_raster) as xrimg:
        B1ft = xrimg.Band1
        B1m = B1ft.copy()
        B1m.values = B1ft.values/(3.28084)
        B1m.rio.to_raster(output_raster)
        return B1m


################################
## QA/QC
################################

def count_tif_nans(in_file):
    """takes 'in_file' image.tif and returns the number of nan values in that image"""
    with rio.open(in_file) as src:
        arr=src.read(1)
        num_nans = np.count_nonzero(np.isnan(arr))
        return num_nans 

def plot_images(in_dir, sensor_string, out_dir):
    """
    plot all images in 'in_dir' that have 'filter_string'. save image plots in 'out_dir'
    1. scroll through returned images from function
    2. click through images with widget dropdown at the end of function 
    3. run zip command then download/extract to quickly pan through images with filename alongside
    """
    img_list= sorted([im for im in os.listdir(in_dir) if sensor_string in im])
    for img in img_list: 
        with rio.open(os.path.join(in_dir, img)) as src:
            array = src.read(1)
        fig, (ax1) = plt.subplots(1, 1, figsize=(20, 20))
        ax1.imshow(array) 
        out_name = img[:-4]+".jpg"
        plt.savefig(os.path.join(out_dir, out_name))
        print(out_name)
    zip_cmd = "zip "+str(out_dir)+str(sensor_string)+"_jpgs.zip "+str(out_dir)+sensor_string+"*"
    os.system(zip_cmd)
    
    @interact
    def show_images(file=[f for f in sorted(os.listdir(out_dir)) if str(sensor_string) in f]):
        display(Image(out_dir+file))


############################################################################
######## FEATURE EXTRACTION 
############################################################################

def watershed_segmentation(in_arr, extent_mask, seed_size):
    ftp_xy = peak_local_max(in_arr, footprint=np.ones((int(seed_size), int(seed_size))), labels=extent_mask) 
    mask = np.zeros(extent_mask.shape, dtype=bool)
    mask[tuple(ftp_xy.T)] = True
    markers, _ = ndi.label(mask)
    instances = watershed(-in_arr, markers, mask=extent_mask) 
    return instances

def focal_mean_variance(in_file, window_size):      
    grid_folder = "UNQ"+in_file.split(".")[-2][-3:]
    out_name_foc = in_file.replace(grid_folder+".tif", "") +"foc"+str(window_size[0])+"_"+str(grid_folder)+".tif"
    out_name_Var = in_file.replace(grid_folder+".tif", "") +"var"+str(window_size[1])+"_"+str(grid_folder)+".tif"
    with gw.open(in_file, chunks=1024) as src: ## geowombat imported as gw
        res = src.gw.moving(stat='mean', w=int(window_size[0]), n_jobs = 4, nodata = 0)
        res = dask_to_xarray(src, da.from_array(res.data.compute(num_workers=4), chunks=src.data.chunksize), src.band.values.tolist()) 
        res.gw.to_raster(out_name_foc, n_workers=4, n_threads=1)           
        if ("divide" not in in_file) and ("subtract" not in in_file): ## variance of divide looked like nothing. var of subtract looks like just speckle/noise 
            res_Var = src.gw.moving(stat='var', w=int(window_size[1]), n_jobs = 4, nodata = 0)
            res_Var = dask_to_xarray(src, da.from_array(res_Var.data.compute(num_workers=4), chunks=src.data.chunksize), src.band.values.tolist()) 
            res_Var.gw.to_raster(out_name_Var, n_workers=4, n_threads=1)  
        else:
            pass 
 
def norma(X):
    return X / np.sqrt( np.sum((X**2), 0) )
     
def bright_norm(X):  
    return np.apply_along_axis(norma, 0, X)
    
def minimum_noise_fraction(inRast, n_components=5, BrightnessNormalization=False):
    outMNF = inRast.split(".")[:-1]+ "_MNF"+str(n_components)+".tif"
    with rio.open(inRast) as r:
        meta = r.profile
        img = r.read()  
        count = r.count
        width = r.width
        height = r.height
    if BrightnessNormalization==True:
        print('Applying preprocessing...')
        img = bright_norm(img)
        outMNF = outMNF[:-4]+'_prepro.tif'
        print('Done!')
    img = np.transpose(img, [1,2,0]) 
    img = img.reshape((width*height, count))
    if np.any(np.isinf(img))==True:
        img[img == np.inf] = 0
    if np.any(np.isnan(img))==True:
        img[img == np.nan] = 0
    pca = PCA(whiten=True)
    img = pca.fit_transform(img)
    print('Done!')
    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    print("The explained variance per component is:")
    print(pca.explained_variance_ratio_)
    print("The accumulative explained variance per component is:")
    print(var)
    np.savetxt(outMNF[:-4]+'_variance.txt', pca.explained_variance_ratio_)
    np.savetxt(outMNF[:-4]+'_accVariance.txt', var)
    img = img.reshape((height, width, count))
    img = np.transpose(img, [2,0,1])  
    out_img = img[:n_components, :, :]
    meta.update(count=n_components, dtype='float32', driver="GTiff")
    with rio.open(outMNF, "w", **meta) as dst:
        dst.write(out_img) 
    return out_img
       
def kmeans_cluster(train_rast_path, predict_rast_path, num_clusters, bands="all"):
    if bands=="all":
        with rio.open(train_rast_path, "r") as src:
            # create an empty array with same dimension and data type
            imgxyb = np.empty((src.height, src.width, src.count), src.meta['dtype'])
            # loop through the raster's bands to fill the empty array
            for band in range(imgxyb.shape[2]):
                imgxyb[:,:,band] = src.read(band+1)
                #imgxyb[np.isnan(imgxyb)] = 0
            # convert to 1d array
            img1d=imgxyb[:,:,:src.count].reshape((imgxyb.shape[0]*imgxyb.shape[1],imgxyb.shape[2]))
            #img1d[np.isnan(img1d)] = 0
            img1d[img1d<-10000] = 0 # made everything zero     
            # assign number of clusters
            cl=cluster.KMeans(n_clusters=int(num_clusters)) # create an object of the classifier
            param=cl.fit(img1d) # train it
            img_cl=cl.labels_ # get the labels of the classes
            img_cl=img_cl.reshape(imgxyb[:,:,0].shape) # reshape labels to a 3d array (one band only)
            input_pred_path=train_rast_path[:-4] + "_train_img_all_" + str(imgxyb.shape[2]) + "b" + str(num_clusters) + "cl.tif" 
            # export training image
            with rio.open(input_pred_path, 'w', driver='GTiff', height=img_cl.shape[0], width=img_cl.shape[1], count=1, dtype=img_cl.dtype, crs=src.crs, transform=src.transform) as dst:
                print('Exporting training clusters... ' + input_pred_path)
                dst.write(img_cl, 1)
            if train_rast_path != predict_rast_path:
                # open the raster image to predict 
                with rio.open(predict_rast_path, "r") as pred:
                    pred_xyb=np.empty((pred.height, pred.width, pred.count), pred.meta['dtype'])
                    for band in range(pred_xyb.shape[2]):
                        pred_xyb[:,:,band] = pred.read(band+1)
                    pred_1d=pred_xyb[:,:,:pred.count].reshape(pred_xyb.shape[0]*pred_xyb.shape[1], pred_xyb.shape[2])
                    pred_clust=cl.predict(pred_1d)
                    pred_cul=pred_clust
                    pred_cul=pred_cul.reshape(pred_xyb[:,:,0].shape)

                    # export prediction
                    pred_path=predict_rast_path[:-4] + "_pred_img_all" + str(num_clusters) + "cl.tif"
                    with rio.open(pred_path, 'w', driver='GTiff', height=pred_cul.shape[0], width=pred_cul.shape[1], count=1, dtype=pred_cul.dtype, crs=pred.crs, transform=pred.transform) as dst:
                        print('Exporting predicted raster... ' + pred_path)
                        dst.write(pred_cul, 1)
            else:
                 return(print('No new raster to predict'))
    else:
        with rio.open(train_rast_path, "r") as src:
            # create an empty array with same dimension and data type
            imgxyb=np.empty((src.height, src.width, len(bands)), src.meta['dtype']) #src.count > len(bands)
            # loop through the raster's bands to fill the empty array
            for i, band in enumerate(bands):
                imgxyb[:,:,i] = src.read(band)
                #imgxyb[np.isnan(imgxyb)] = 0
            # convert to 1d array
            img1d=imgxyb[:,:,:len(bands)].reshape((imgxyb.shape[0]*imgxyb.shape[1],imgxyb.shape[2])) #src.count > len(bands)
            #img1d[np.isnan(img1d)] = 0
            img1d[img1d<-10000] = 0
            print(img1d)
            # assign number of clusters
            cl=cluster.KMeans(n_clusters=num_clusters) # create an object of the classifier
            param=cl.fit(img1d) # train it
            img_cl=cl.labels_ # get the labels of the classes
            img_cl=img_cl.reshape(imgxyb[:,:,0].shape) # reshape labels to a 3d array (one band only)
            input_pred_path = train_rast_path[:-4] + "_train_img_" + str(len(bands)) + "b_" + str(num_clusters) + "cl.tif" 
            # export training image
            with rio.open(input_pred_path, 'w', driver='GTiff', height=img_cl.shape[0], width=img_cl.shape[1], count=1, dtype=img_cl.dtype, crs=src.crs, transform=src.transform) as dst:
                print('Exporting training clusters to ' + input_pred_path)
                dst.write(img_cl, 1)
                input_sub_path = train_rast_path[:-4] + "subset" + str(len(bands)) + "b.tif"
            if train_rast_path != predict_rast_path:
                # open the raster image to predict 
                with rio.open(predict_rast_path, "r") as pred:
                    pred_xyb=np.empty((pred.height, pred.width, len(bands)), pred.meta['dtype']) #pred.count > len(bands)
                    for band in range(pred_xyb.shape[2]):
                        pred_xyb[:,:,band]=pred.read(band+1)
                    pred_1d=pred_xyb[:,:,:len(bands)].reshape(pred_xyb.shape[0]*pred_xyb.shape[1], pred_xyb.shape[2]) #pred.count > len(bands)
                    pred_clust=cl.predict(pred_1d)
                    pred_cul=pred_clust
                    pred_cul=pred_cul.reshape(pred_xyb[:,:,0].shape)
                    # export prediction
                    pred_path=predict_rast_path[:-4] + "_pred_img_" + str(len(bands)) + "b_" + str(num_clusters) + "cl.tif"
                    with rio.open(pred_path, 'w', driver='GTiff', height=pred_cul.shape[0], width=pred_cul.shape[1], count=1, dtype=pred_cul.dtype, crs=pred.crs, transform=pred.transform) as dst:
                        print('Exporting predicted raster... ' + pred_path)
                        dst.write(pred_cul, 1)
            else:
                return(print('No new raster to predict'))

