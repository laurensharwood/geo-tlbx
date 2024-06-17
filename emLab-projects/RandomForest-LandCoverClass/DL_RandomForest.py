#!/usr/bin/ python

import os, sys, glob
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn import tree
import time
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

################################
## 1: split T/V 

import os
from osgeo import ogr
from osgeo.gdalconst import *
import numpy as np
import random

def assign_TrainingHoldout(vector_path, class_field_name, tv_field):
    '''
    groups polygons by Class, then creates Training/Validation splits where 75% of polygons from each class -> Training, and 25% -> Validation/Holdout
    vector_path = filepath to Training/Validation polygons drawn around different classes
    class_field_name = field for each polygon's class/label 
    tv_field = name to assign field where each polygon will receive a "T" or "V", to separate into T-Training or V-Validation/Holdout 
    '''
    
    vds = ogr.Open(vector_path, GA_Update)
    assert vds
    vlyr = vds.GetLayer(0)
    # Create Dictionary of Classes to OID's
    vals = {}
    feat = vlyr.GetNextFeature()
    while feat is not None:
        if not feat.GetFieldAsInteger(class_field_name) in vals.keys():
            vals[feat.GetFieldAsInteger(class_field_name)] = [feat.GetFID()]
        else:
            vals[feat.GetFieldAsInteger(class_field_name)].append(feat.GetFID())

        feat = vlyr.GetNextFeature()

    vds = None
    tv_dict = {}
    for uni in np.unique(list(vals.keys())):
        # create random list of numbers, that is the length of # of
        # records for each key
        data = [x for x in range(len(vals[uni]))]
        random.shuffle(data)

        for i, d in enumerate(data):
            if d >= (len(data) / 4.0):
                t_v = 'T'
            else:
                t_v = 'V'
            oid = vals[uni][i]
            tv_dict[oid] = t_v

    vds = ogr.Open(vector_path, GA_Update)
    assert vds
    vlyr = vds.GetLayer(0)
    f, d = (str(tv_field), ogr.OFTString)
    ldef = vlyr.GetLayerDefn()
    if ldef.GetFieldIndex(f) != -1:
        print('Values in "%s" will be overwritten.' % f)
    else:
        fd = ogr.FieldDefn(f, d)
        vlyr.CreateField(fd)
    for OID in range(vlyr.GetFeatureCount()):
        feat = vlyr.GetFeature(OID)
        feat.SetField(str(tv_field), tv_dict[OID])
        vlyr.SetFeature(feat)
    vds = None
    return vector_path

################################
## 2: stack rasters

import rasterio as rio
from tqdm import tqdm

def stack_raster_list(full_dir, list_of_rasters,  name_prefix, out_directory, overwrite=False):
    
    raster_paths = sorted([os.path.join(full_dir, rast) for rast in list_of_rasters])
    out_path = os.path.join(out_directory, name_prefix+"_stack_" + str(len(raster_paths)) + "b_UNQ" + str(full_dir).split("/")[-1]+".tif")  
    if os.path.exists(out_path) and overwrite==False:
        print('stack exists, not overwriting')
    else:
        first_band = rio.open(raster_paths[0], 'r')
        out_meta = first_band.meta.copy()
        out_meta.update(count=len(raster_paths))
        big_band_list = []
        for file in raster_paths:
            with rio.open(file, 'r') as src:
                width=src.width
                height=src.height
                name=src.name
                band = src.read(1)
                ## in S1 rasts, need to change nan to -32768.0, then use -32768.0 as no data val 
                band = np.where(np.isnan(band), -32768.0, band)
                if "NASA" in file:
                    band = band*1000
                if "S1A_" in file:
                    band = band*100
                if "S2_" in file:
                    band = np.where(band==0, -32768.0, band)
                big_band_list.append(band)
                
        out_meta.update({"count":len(raster_paths), "dtype": np.int16, "width":width, "height":height, "nodata":-32768.0})

        print(out_path)
        with rio.open(out_path, 'w', **out_meta) as dst:
            for i, fi in tqdm(enumerate(raster_paths)):
                dst.write(big_band_list[i], i+1)
            dst.descriptions = tuple([i.split("/")[-1][:-4] for i in raster_paths])          
       

import os, sys
import pandas as pd 
import numpy as np
from osgeo import ogr, osr, gdal
from osgeo.gdalconst import GA_Update, GA_ReadOnly
import shapely
from shapely.geometry import Polygon
from shapely.wkt import dumps
import time
import fiona
import rasterio as rio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

def image_poly(img_path):
    geo = rio.open(img_path)
    w, s, e, n = geo.bounds
    corners = [(w, n), (e, n), (e, s), (w, s)]
    im_poly = Polygon(corners)
    return im_poly

def img_to_bbox_offsets(gt, bbox):
    origin_x = gt[2]
    origin_y = gt[5]
    pixel_width = gt[0]
    pixel_height = gt[4]
    x1 = int(round((bbox[0] - origin_x) / pixel_width))
    x2 = int(round((bbox[1] - origin_x) / pixel_width))
    y1 = int(round((bbox[3] - origin_y) / pixel_height))
    y2 = int(round((bbox[2] - origin_y) / pixel_height))
    xsize = x2 - x1
    ysize = y2 - y1
    return [x1, y1, xsize, ysize]

def extract_TV_polygons(input_image, input_shape, class_field, filter_string, TV_field, polygon_id, band_list="all"):
    TV_data = []  # initialize training_data, class_code labels, XYcoords, and unique field ID 
    TV_labels = []
    TV_xy = []
    TV_fieldID = []
    TV_split = []
    
    with rio.open(input_image, 'r') as geo:
        columns = [str(polygon_id), str(class_field), str(TV_field),  "Xcoord", "Ycoord"]
        bands=list(geo.descriptions)
        column_list = columns + bands
        rgt = geo.transform 
        im_poly = image_poly(input_image) # create bounding box shape of input image
        vds = ogr.Open(input_shape, GA_Update)
        assert vds
        vlyr = vds.GetLayer(0)  # read data into dataframe
        feature_count = vlyr.GetFeatureCount()  # number of polygons
        for fid in range(feature_count):
            feature = vlyr.GetFeature(fid)  # creating instance of single polygon
            ## APPLY NEGATIVE BUFFER OF -0.5 TO NOT EXTRACT BOUNDARY PIXELS
            shppoly = shapely.wkt.loads(feature.geometry().ExportToWkt()).buffer(-0.5)  # exports xy info in terms of spatial ref
            # if the input image bounding box contains the Training/Validation polygon shape
            if im_poly.contains(shppoly) is True:
                # if it is a Training polygon (tv_field = T) and should NOT be ignored (class_field does not equal -99)
                if feature.GetField(class_field) != -99:
                    tv_val = feature.GetField(class_field)
                    tv_uid = feature.GetField(polygon_id)
                    TV = feature.GetField(TV_field)
                    ## find the size of the polygon bbox (in pixels) and the location of its UL corner within the image
                    src_offset = img_to_bbox_offsets(rgt, feature.geometry().GetEnvelope())
                    # if the width or height are 0, the polygon is too small, print out warning.
                    if src_offset[2] == 0 or src_offset[3] == 0:
                        print("    Warning: Polygon {} too small to rasterize, "
                                "skipping".format(feature.GetFID()))
                        continue

                    else:
                        if isinstance(band_list, list):
                            im_sub1 = np.zeros((src_offset[2] * src_offset[3], 1))  # initialize array for horizontally stacking bands, the number of rows of bbox pixels
                            for i in band_list:  # for each item in the list of bands
                                # read window of raster band(i), stack pixels below one another in one column
                                tmp = geo.read(i+1, window=Window(src_offset[0], src_offset[1], src_offset[2], src_offset[3])).reshape(-1, 1)
                                # stack pixels in each band as a new column, horizontally
                                im_sub1 = np.hstack((im_sub1, tmp))
                            # delete first column of zeros
                            im_sub1 = im_sub1[:, 1:]
                            # stack pixels for each new polygon vertically
                            im_sub = np.vstack((im_sub1))

                        elif band_list.lower() == 'all':
                            im_sub1 = np.zeros((src_offset[2] * src_offset[3], 1))
                            for i in range(1, geo.count+1, 1):  # for all bands in the raster
                                tmp = geo.read(i, window=Window(src_offset[0], src_offset[1], src_offset[2],
                                                                src_offset[3])).reshape(-1, 1)
                                im_sub1 = np.hstack((im_sub1, tmp))
                            im_sub1 = im_sub1[:, 1:]
                            im_sub = np.vstack((im_sub1)) ##ADDED FLOAT DATATYPE HERE ##, dtype=np.float32

                        else:
                            print("impropper band list... exiting")
                            sys.exit()

                        msk_sub = np.ones((im_sub.shape[0], 1))

                        #
                        mem_drv = ogr.GetDriverByName('Memory')
                        driver = gdal.GetDriverByName('MEM')
                        # create new geotransformation for rasterized shape
                        new_gt = ((rgt[2] + (src_offset[0] * rgt[0])), rgt[0], 0.0,
                                    (rgt[5] + (src_offset[1] * rgt[4])), 0.0, rgt[4])

                        # Get XY Arrays from polygon subset
                        mgx, mgy = np.meshgrid(np.arange(0, src_offset[2], 1), np.arange(0, src_offset[3], 1))
                        xarr_sub = new_gt[0] + new_gt[1] * mgx + new_gt[2] * mgy
                        yarr_sub = new_gt[3] + new_gt[4] * mgx + new_gt[5] * mgy
                        xys = np.vstack((np.ravel(xarr_sub), np.ravel(yarr_sub))).T

                        # Create Memory Drivers
                        mem_ds = mem_drv.CreateDataSource('out')
                        mem_layer = mem_ds.CreateLayer('poly', vlyr.GetSpatialRef(),
                                                        ogr.wkbPolygon)

                        mem_layer.CreateFeature(feature.Clone())
                        rvds = driver.Create('', src_offset[2], src_offset[3], 1,
                                                gdal.GDT_Byte)
                        rvds.SetGeoTransform(new_gt)
                        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
                        rv_array = rvds.ReadAsArray()
                        rshp = rv_array.reshape(-1, 1)

                        full_mask = rshp * msk_sub
                        im_sub = full_mask * im_sub
                        xys = full_mask * xys

                        im_sub = im_sub[np.ravel(full_mask) == 1]
                        xys = xys[np.ravel(full_mask) == 1]

                        TV_fieldID += [tv_uid] * im_sub.shape[0]
                        TV_xy += [list(xy) for xy in xys]
                        TV_labels += [tv_val] * im_sub.shape[0]
                        TV_split += [TV] * im_sub.shape[0]
                        TV_data += [list(px) for px in im_sub]
    X=np.array(TV_data) 
    Y=np.array(TV_labels).reshape(-1, 1)
    groups=np.array(TV_fieldID).reshape(-1, 1)
    TV_fields=np.array(TV_split).reshape(-1, 1)
    XY_coords=np.array(TV_xy) 
    grid_name = input_image.split(".")[0][-6:] ## UNQ000
    train_holdout_csv = input_shape[:-4]+"_"+filter_string+"_"+str(len(bands))+"b_"+grid_name+"_Val.csv"
    print(train_holdout_csv)
    csvarr = np.hstack((groups, Y, TV_fields, XY_coords, X))
    TV_df = pd.DataFrame(csvarr, columns = column_list)
    TV_df.to_csv(train_holdout_csv)

    return train_holdout_csv 

################################
# 4. append extracted features from different GRIDS (LONG) 

from itertools import groupby, chain
import pandas as pd

def append_grid_vals(point_file_list, filter_criteria, TV_field, Class_field, output_dir, feats):
    '''
    append points (LONG FORMAT/ADD ROWS/POINTS) from different tiles 
    input directory is where .gpkg files with extracted data for each UNQ grid are
    filter criteria is the NASA or S1 data to combine
    returns combined CSV to be used as input to SENSOR COMBINATION (WIDE/ADD COLS/FEATURES) def initial_trianing_sample()
    '''
    points_list = []
    for i in point_file_list:
        df = pd.read_csv(i, index_col=0)
        cols1=[c for c in df.columns[:5]]  
        ## remove UNQ grid # from column name 
        cols2=[c[:-7].replace("_UNQ", "")  for c in df.columns[5:]]
        df.columns=cols1+cols2
        df['Xcoord'] = df['Xcoord'].round(decimals = 1)
        df['Ycoord'] = df['Ycoord'].round(decimals = 1)
        df['XYcoords'] = [', '.join(str(x) for x in y) for y in map(tuple, df[['Xcoord', 'Ycoord']].values)]
        df['count'] = df.groupby('field_id').cumcount() + 1
        df['pixel_id'] = df['field_id'].astype(str)+"_"+df['count'].astype(str)        
        df = df.drop(columns=['count', 'Xcoord', 'Ycoord'])
        points_list.append(df)
    ## combine points from all dataframes 
    all_df = pd.concat(points_list)
    all_df.set_index(['pixel_id'], inplace=True)  
    
    ## drop NAs
    all_df = all_df.replace(-32768, np.nan)
    all_df = all_df.replace(-32768.0, np.nan)
    all_df.dropna(axis=0, how='any', inplace=True) ## axis=0 -- rows
    all_df = all_df.loc[all_df[Class_field] != -99 ]

    
    ## save FULL csv
    combined_name = point_file_list[0].split("/")[-1].split(".")[0][:-11]+"_Vals.csv"
    out_name=os.path.join(output_dir, combined_name)
    all_df.to_csv(out_name)
    print(out_name)   
    
    
    ## save TRAIN and HOLDOUT csv's (only keep FEATS and LABEL)
    Tdf = all_df[all_df[TV_field] == "T"]
    Vdf = all_df[all_df[TV_field] == "V"]
    #Tdf = Tdf.drop(columns=[TV_field,  "XYcoords", "field_id"])
    TRAIN_NAME = out_name.replace(".csv", "_TRAIN.csv")
    HOLDOUT_NAME = out_name.replace(".csv", "_HOLDOUT.csv")
    Tdf.to_csv(TRAIN_NAME)
    Vdf.to_csv(HOLDOUT_NAME)
    
    TRAIN_X = pd.DataFrame(Tdf[feats])
    TRAIN_Y = pd.DataFrame(Tdf[Class_field])
    VAL_X = pd.DataFrame(Tdf[feats])
    VAL_Y = pd.DataFrame(Vdf[Class_field])    

    return (out_name, TRAIN_X, VAL_X, TRAIN_Y, VAL_Y)


# fit random forest model 
def build_rf_mod(train_csv, class_field, feature_list, n_trees, seed):
    
    training_data = pd.read_csv(train_csv, index_col="pixel_id")
    X_train = training_data.drop(columns=[col for col in training_data if col not in feature_list])
    Y_train = training_data[str(class_field)]
    
    rf = RandomForestClassifier(n_estimators = n_trees, oob_score=False, random_state = seed)
    rf_model = rf.fit(X_train.values, Y_train.values)
    rf_mod_name = train_csv.replace("_Vals_TRAIN.csv", "_Mod.joblib")
    rf_mod_file = joblib.dump(rf_model, rf_mod_name, compress=3)[0]
    return rf_mod_file

from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn import tree

def tune_rf_mod(train_csv, class_field, feature_list, seed):
    
    training_data = pd.read_csv(train_csv, index_col="pixel_id")
    Y_train = training_data[str(class_field)]
    X_train = training_data.drop(columns=[col for col in training_data if col not in feature_list])
    # ## hyperparamer tuning 
    # params = {'n_estimators': np.arange(50,200,50), # np.arange(100,200,15),
    #           'max_features': np.arange(0.1, 1, 0.2), # np.arange(0.1, 1, 0.1),
    #           'max_depth': np.arange(3, 15, 2), #[3, 5, 7],
    #           'max_samples': np.arange(0.1, 0.9, 0.2), # [0.3, 0.5, 0.8]
    #           #'min_samples_split':[5]
    #              }
    params = {'n_estimators': np.arange(100,200,50), # np.arange(100,200,15),
              'max_features': np.arange(0.1, 1, 0.2), # np.arange(0.1, 1, 0.1),
              'max_depth': np.arange(3, 15, 2), #[3, 5, 7],
              'max_samples': np.arange(0.1, 0.9, 0.2), # [0.3, 0.5, 0.8]
              #'min_samples_split':[5]
                 }

    rf_mod = GSCV(RandomForestClassifier(random_state = seed, oob_score=False), 
                  params, 
                 refit = True
                 ).fit(X_train.values, Y_train.values)
    print(rf_mod)
    print(rf_mod.best_params_)
    print(rf_mod.best_estimator_)

    rf_model = rf_mod.best_estimator_
    rf_mod_name = train_csv.replace("_Vals_TRAIN.csv", "_Mod.joblib")
    rf_mod_file = joblib.dump(rf_model, rf_mod_name, compress=3)[0]
    return rf_mod_file

################################
## 6. get predictions and accuracy 

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay



def get_holdout_preds(lookup_csv, holdout_path, class_field, variables, model_file, predORthresh="pred"):
    print(predORthresh)
    model = joblib.load(model_file)

    folders = holdout_path.split("/")[:-1]
    out_folder = "/".join(map(str, folders))
    holdout_pred_name = os.path.join(out_folder, "RFmod_Holdout_preds.csv")
    out_OA_name = os.path.join(out_folder, "RFmod_Holdout_acc.csv")
    
    lut = pd.read_csv(lookup_csv)
    NAME_FIELD=lut.columns[0]
    CLASS_FIELD=lut.columns[1]
    class_names = lut[NAME_FIELD].to_list()
    class_codes = lut[CLASS_FIELD].to_list()
    CodeName_dict = dict(zip(class_names, class_codes))
    glacis_val = CodeName_dict.get('Glacis')
    
    ## Save info for extra columns and drop (model is expecting only variable input columns)
    holdout_fields = pd.read_csv(holdout_path, index_col="pixel_id")
    ## save variable for holdout labels 
    holdout_labels = holdout_fields[class_field]  
    ## Remove all columns not in variable list (i.e. label column)
    holdout_fields = holdout_fields[holdout_fields.columns[holdout_fields.columns.isin(variables)]]

    holdout_preds=[] 
    if predORthresh == "pred":
        ## Calculate scores
        holdout_preds.append(model.predict(holdout_fields.values))  ## predict
        holdout_preds=holdout_preds[0]
    else:
        predORthresh=round(predORthresh,3)
        ##Calculate scores
        holdout_fields_predicted = model.predict_proba(holdout_fields.values)   ## predict_proba 
        for i in holdout_fields_predicted:
            ## prediction is position of largest proba value (position starts at 0 so add 1 for class_code 1, 2 for class_code 2..)  
            pred_code = i.argmax()+1
            ## if prediction is glacis and the proba is < the glacis threshold, assign class of second-highest proba 
            if pred_code == glacis_val and i.max() < predORthresh: 
                pred_code = i.argsort()[-2,]+1
            ## if the max proba isn't glacis, but the glacis proba is > threshold, assign to glacis class 
            if pred_code != glacis_val and i[0] >= predORthresh: 
                pred_code = glacis_val       
            holdout_preds.append(pred_code)
    ## Add column for predictions/probas
    holdout_fields['pred'] = holdout_preds        
    ## Add class_field/label column back 
    holdout_fields[class_field] = holdout_labels
    holdout_fields.to_csv(holdout_pred_name, sep=',', na_rep='NaN', index=True)  
    
    print('overall accuracy:')
    real_OA = accuracy_score(holdout_labels, holdout_preds)    
    print(real_OA)
    
    print('class accuracies / confusion matrix:')
    ##plot = plt.show(ConfusionMatrixDisplay.from_predictions(holdout_labels, holdout_preds))
    cMatrix=confusion_matrix(holdout_labels, holdout_preds)
    cMatrix_fname = os.path.join(out_folder, "RFmod_"+str(predORthresh)+"_cMatrix"+".csv")
    np.savetxt(cMatrix_fname, cMatrix, delimiter=",")    
    
    print('Glacis vs NOT accuracy:')    
    TP = len(holdout_fields[(holdout_fields.pred == glacis_val) & (holdout_fields[class_field] == glacis_val)])
    TN = len(holdout_fields[(holdout_fields.pred != glacis_val) & (holdout_fields[class_field] != glacis_val)])
    FP = len(holdout_fields[(holdout_fields.pred == glacis_val) & (holdout_fields[class_field] != glacis_val)])
    FN = len(holdout_fields[(holdout_fields.pred != glacis_val) & (holdout_fields[class_field] == glacis_val)])
    OA = ((TP+TN)/(TP+TN+FP+FN))*100
    #Error_of_omission = float(FN/(TP+TN+FP+FN))*100
    accuracy_df = pd.DataFrame([OA, TP, TN, FP, FN], 
                               ["GlacisNOT_OA", "TP", "TN", "FP", "FN"]) #"Glacis Error of omission", 
    accuracy_df.to_csv(out_OA_name)    
    print(accuracy_df)
    #return (real_OA, plot, accuracy_df)


    
################################
            
import rasterio as rio
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import joblib

## 6. predict on surface 
def RF_classify(raster_stack, model_file, pred_dir, discrete_thresh_continuous="discrete"):
    '''
    discrete_thresh_continuous = "discrete"for regular prediction or 0-1 float for fuzzy threshold where a pixel is classified as glacis if its val is > the float
    '''
    in_dir = "/".join(raster_stack.split("/")[:-1])
    
        
    rf_model = joblib.load(model_file)
    with rio.open(raster_stack, "r") as src:
        out_meta = src.meta
        ## create an empty array with same dimension and data type
        imgxyb = np.empty((src.height, src.width, src.count), np.float32)
        
        ## loop through the raster's bands to fill the empty array
        for band in range(imgxyb.shape[2]):
            imgxyb[:,:,band] = src.read(band+1)
        ## convert to 2d array (width*height, numBands)
        img2d = imgxyb.reshape((imgxyb.shape[1]*imgxyb.shape[0], imgxyb.shape[2]))


    if discrete_thresh_continuous=="discrete":
        out_image = raster_stack[:-4].replace(in_dir, pred_dir)+"_RFPred.tif"
        class_pred = rf_model.predict(img2d).reshape((1, imgxyb.shape[0], imgxyb.shape[1]))
        out_meta.update({"dtype":np.int8, "count":1, "nodata": 0})

    elif discrete_thresh_continuous=="continuous":
        out_image = raster_stack[:-4].replace(in_dir, pred_dir)+"_RFGlacisProba.tif"
        probas = rf_model.predict_proba(img2d)
        class_pred=probas[:,0].copy()
        class_pred=class_pred.reshape((1, imgxyb.shape[0], imgxyb.shape[1]))
        out_meta.update({"dtype":np.float32, "count":1, "nodata": -32768.0})
        
    elif type(discrete_thresh_continuous) == type(0.5):
        proba_img = raster_stack[:-4].replace(in_dir, pred_dir)+"_RFProba.tif"
        probas = rf_model.predict_proba(img2d)
        class_pred=probas[:,0].copy()
        class_pred[class_pred >= float(discrete_thresh_continuous)] = 1
        class_pred[class_pred < float(discrete_thresh_continuous)] = 2
        class_pred = class_pred.reshape((1, imgxyb.shape[0], imgxyb.shape[1]))
        thresh = str(discrete_thresh_continuous).replace(".","pt")
        out_image = proba_img.replace("_RFProba.tif", "_RFPbaPred"+str(thresh)+".tif")
        out_meta.update({"dtype":np.int8, "count":1, "nodata": 0})
        

    class_pred[class_pred < 0] = 0
    with rio.open(out_image , "w", **out_meta) as dst:
        dst.write(class_pred)          
            

        
    return out_image     
    
################################
## feature importances

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import joblib

def get_feature_importance(model_file, X_data, Y_data): 
    mod_dir="/".join(model_file.split("/")[:-1])

    mod = joblib.load(model_file)
    gini_imp = mod.feature_importances_
    feats = {} # a dict to hold feature_name: feature_importance
    for feature, gini_imp in zip(X_data.columns, gini_imp):
        feats[feature] = gini_imp #add the name/value pair 
    gini_importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    gini_importances_desc = gini_importances.sort_values(by='Gini-importance', ascending=False)
    gini_importances_desc.to_csv( os.path.join(mod_dir, 'varImp_gini.csv'), sep=',', na_rep='NaN', index=True)   

    perm_imp = permutation_importance(mod, X_data.values, Y_data.values, n_repeats=5, random_state=531)
    sorted_Pimp_idx = perm_imp.importances_mean.argsort()
    permutation_importances = pd.DataFrame(perm_imp.importances[sorted_Pimp_idx].T, columns=X_data.columns[sorted_Pimp_idx])
    perm_imp_avg = permutation_importances.mean(axis=0)
    perm_imp_avg_df = pd.DataFrame(perm_imp_avg)
    perm_imp_avg_df.columns = ['Permutation-importance']
    permutation_importances_desc = perm_imp_avg_df.sort_values(by='Permutation-importance', ascending=False)
    permutation_importances_desc.to_csv( os.path.join(mod_dir, 'varImp_permut.csv'), sep=',', na_rep='NaN', index=True)   
    
    gini = pd.read_csv(os.path.join(mod_dir, "varImp_gini.csv"))
    perm = pd.read_csv(os.path.join(mod_dir, "varImp_permut.csv"))
    gini_list = list(gini.set_index(gini.iloc[:,0]).iloc[:,0].index.to_list())
    perm_list = list(perm.set_index(perm.iloc[:,0]).iloc[:,0].index.to_list())
    
    return gini_importances_desc, permutation_importances_desc, gini_list,  perm_list



    
def main():
    
    ### dict of features for ALL YEARS-- year_feat_dict.get(year)
    year_feat_dict = {}
    for year in list(range(2017, 2024, 1)):
        topo_feats = ['NASA_DEM', 'NASA_Slope']
        s1_feats = ['S1A_VH_'+str(year)+'01_Med', 'S1A_VH_'+str(year)+'01_Med_foc5', 
                    'S1A_VV_'+str(year)+'01_Med', 'S1A_VV_'+str(year)+'01_Med_foc5',         
                    'S1A_VH_'+str(year)+'04_Med', 'S1A_VH_'+str(year)+'04_Med_foc5', 
                    'S1A_VV_'+str(year)+'04_Med', 'S1A_VV_'+str(year)+'04_Med_foc5' ]
        even_month_bands = ["B11", "NDMI2", "NDWI"]
        odd_month_bands = ["B12", "NDMI1", "NDVI"]    
        all_month_bands = ["B3", "B4", "B8", "EVI"]
        even_months = [str(i).zfill(2) for i in list(range(6, 13, 2))]
        odd_months = [str(i).zfill(2) for i in list(range(6, 13, 2))]
        all_months = [str(i).zfill(2) for i in list(range(6, 13, 1))]
        s2_feats = []
        for band in even_month_bands:
            for month in even_months:
                s2_feats.append("S2_"+band+"_"+str(year)+str(month))
        for band in odd_month_bands:
            for month in odd_months:
                s2_feats.append("S2_"+band+"_"+str(year)+str(month))
        for band in all_month_bands:
            for month in all_months:
                s2_feats.append("S2_"+band+"_"+str(year)+str(month))
        stac_feats = sorted(s1_feats + topo_feats + s2_feats)
        year_feat_dict.update({year:stac_feats})

    ### list of steps to perform
    TASKS_TO_DO =  list(str(sys.argv[1]).replace("]","").replace("[","").split(",")) 
    
    ### STACK PREFIX NEEDS TO BE "YR##"
    STACK_PREFIX=str(sys.argv[2])
    pred_year = int("20"+STACK_PREFIX[-2:])
    ALL_FEATS = year_feat_dict.get(pred_year)
    
    TV_SHAPEFILE = str(sys.argv[3])
    VALS_DIR = "/".join(TV_SHAPEFILE.split("/")[:-1])
    print(VALS_DIR)
    LUT_FILE=[os.path.join(VALS_DIR,l) for l in os.listdir(VALS_DIR) if ("Lookup" in l or "lookup" in l) and l.endswith(".csv")][0]
    lut=pd.read_csv(LUT_FILE)
    CLASS_FIELD=lut.columns[1]
    class_names = lut[lut.columns[0]].to_list()
    class_codes = lut[CLASS_FIELD].to_list()
    CodeName_dict = dict(zip(class_names, class_codes))
    print(CodeName_dict)   
    
    TV_FIELD = str(sys.argv[4])
    SEED = int(sys.argv[5])
    pred_type = str(sys.argv[6])
    if "." in pred_type:
        GLACIS_THRESH=round(float(int(pred_type.split(".")[-1])*0.1), 2)
    else:
        GLACIS_THRESH=pred_type
    
    gridCells= sys.argv[7].replace("]","").replace("[","").split(",")
    GRID_LIST = [int(i) for i in gridCells]    
    GRID_DIR=str(sys.argv[8])
    STACK_DIR=str(sys.argv[9])
    if not os.path.exists(STACK_DIR):
        os.makedirs(STACK_DIR)
        

    MODEL_VERSION=STACK_PREFIX+"_"+str(len(ALL_FEATS))+"b_"+str(TV_FIELD)
    MOD_VALS_DIR=os.path.join(VALS_DIR, MODEL_VERSION)
    if not os.path.exists(MOD_VALS_DIR):
        os.makedirs(MOD_VALS_DIR)
        
    PRED_RAST_DIR=os.path.join(MOD_VALS_DIR, "surfPreds")    
    if not os.path.exists(PRED_RAST_DIR):
        os.makedirs(PRED_RAST_DIR)
        
######################### steps
    
    if "splitTrainHoldout" in TASKS_TO_DO:
        TV_SHAPE = assign_TrainingHoldout(vector_path=TV_SHAPEFILE,
                               class_field_name=CLASS_FIELD,
                               tv_field=TV_FIELD)      
    
    if "stack" in TASKS_TO_DO:
        for i in sorted(os.listdir(GRID_DIR)):
            grid=i
            FULL_GRID_DIR=os.path.join(GRID_DIR, grid)
            ALL_FEATS_UNQ = [s+"_UNQ"+grid+".tif" for s in ALL_FEATS]
            stack_raster_list(full_dir=FULL_GRID_DIR, 
                              list_of_rasters=ALL_FEATS_UNQ, 
                              name_prefix=STACK_PREFIX, 
                              out_directory=STACK_DIR, 
                              overwrite=False)

    if "extract" in TASKS_TO_DO:
        ## only run extract function on these grids or script will crash 
        grids_wTrain =[21,22,23,24,35,36,58,64,65,66,72,76,77,78,79,86,87,91,92,93,100,101,105,106,107,108,119,120,121,122,123,124,133,134,135,136,137,138,160,174,175,188,189,201]    
    
        vals_csv_list = []
        for stack in sorted([s for s in os.listdir(STACK_DIR) if (int(str(s).split(".")[0][-3:]) in grids_wTrain and STACK_PREFIX in s)]):
            print(stack)
            extract_val = extract_TV_polygons(input_image=os.path.join(STACK_DIR, stack),
                                input_shape=TV_SHAPEFILE, # around 20 polygons were too small
                                class_field=CLASS_FIELD, 
                                filter_string=STACK_PREFIX,
                                TV_field=TV_FIELD, 
                                polygon_id="field_id", 
                                band_list="all")   
            vals_csv_list.append(extract_val)        

    if "append" in TASKS_TO_DO:
        pt_files=sorted([os.path.join(VALS_DIR, i) for i in os.listdir(VALS_DIR) if (str(len(ALL_FEATS))+"b_" in i and STACK_PREFIX in i) and (i.endswith("_Val.csv"))])
        FEATURE_VALS = append_grid_vals(point_file_list=pt_files,  #vals_csv_list
                                        filter_criteria=STACK_PREFIX, 
                                        TV_field=TV_FIELD, 
                                        Class_field=CLASS_FIELD,
                                        output_dir=MOD_VALS_DIR,
                                       feats=ALL_FEATS)  
        FULL_CSV = FEATURE_VALS[0]
        ## objects as dataframes, for feature importance 
        TRAIN_X = FEATURE_VALS[1]
        VAL_X = FEATURE_VALS[2]
        TRAIN_Y = FEATURE_VALS[3] 
        VAL_Y = FEATURE_VALS[4]  
        TRAINING_VALS = FEATURE_VALS[0].replace(".csv", "_TRAIN.csv")
        HOLDOUT_VALS = FEATURE_VALS[0].replace(".csv", "_HOLDOUT.csv")
    else:
        FULL_CSV=[os.path.join(MOD_VALS_DIR,f) for f in os.listdir(MOD_VALS_DIR) if f.endswith("_Vals.csv")][0]
        TRAINING_VALS=FULL_CSV.replace(".csv", "_TRAIN.csv")
        Tdf=pd.read_csv(TRAINING_VALS)
        TRAIN_Y = pd.DataFrame(Tdf[CLASS_FIELD])
        TRAIN_X = pd.DataFrame(Tdf[ALL_FEATS])
        HOLDOUT_VALS=FULL_CSV.replace(".csv", "_HOLDOUT.csv")
        Vdf=pd.read_csv(HOLDOUT_VALS)
        VAL_Y = pd.DataFrame(Vdf[CLASS_FIELD])    
        VAL_X = pd.DataFrame(Vdf[ALL_FEATS])
        
        
    if "tune" in TASKS_TO_DO:
        RF_MOD = tune_rf_mod(train_csv=TRAINING_VALS, 
                             class_field=CLASS_FIELD, 
                             feature_list=ALL_FEATS, 
                             seed=SEED)
    elif "train" in TASKS_TO_DO:
        RF_MOD = build_rf_mod(train_csv=TRAINING_VALS, 
                              class_field=CLASS_FIELD, 
                              feature_list=ALL_FEATS, 
                              n_trees=200, 
                              seed=SEED)
    else:
        RF_MOD=TRAINING_VALS.replace("_Vals_TRAIN.csv", "_Mod.joblib")
    
    if "saveAccuracy" in TASKS_TO_DO:
        get_holdout_preds(lookup_csv=LUT_FILE, 
                          holdout_path=HOLDOUT_VALS, 
                          class_field=CLASS_FIELD, 
                          variables=ALL_FEATS, 
                          model_file=RF_MOD, 
                          predORthresh="pred")
        for T in np.arange(0.3, 0.8, 0.1):
            get_holdout_preds(lookup_csv=LUT_FILE, 
                              holdout_path=HOLDOUT_VALS,
                              class_field=CLASS_FIELD, 
                              variables=ALL_FEATS, 
                              model_file=RF_MOD, 
                              predORthresh=round(T,3))
    
    if "savePredRasts" in TASKS_TO_DO:
        print('RF model: '+RF_MOD)
        print('prediction type: '+str(GLACIS_THRESH))
        for GRID_NUM in GRID_LIST:
            PRED_STACK = os.path.join(STACK_DIR, str(STACK_PREFIX)+"_stack_"+str(len(ALL_FEATS))+"b_UNQ"+str(GRID_NUM).zfill(3)+".tif")
            print('input raster: '+PRED_STACK)
            '''
            RF_classify(raster_stack=PRED_STACK, 
                        model_file=RF_MOD, 
                        pred_dir=PRED_RAST_DIR, 
                        discrete_thresh_continuous="discrete")
            RF_classify(raster_stack=PRED_STACK, 
                        model_file=RF_MOD, 
                        pred_dir=PRED_RAST_DIR,
                        discrete_thresh_continuous="continuous")
            '''
            RF_classify(raster_stack=PRED_STACK, 
                        model_file=RF_MOD, 
                        pred_dir=PRED_RAST_DIR,
                        discrete_thresh_continuous=GLACIS_THRESH)
            
    if "saveFeatImp" in TASKS_TO_DO:
        get_feature_importance(model_file=RF_MOD, 
                               X_data=VAL_X, 
                               Y_data=VAL_Y)


if __name__ == "__main__":
    main()