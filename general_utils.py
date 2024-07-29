import os
import numpy as np
import pandas as pd
import geopandas as gpd
## under FILES, for find_file_diff function
import difflib
## for TIME functions
import datetime
from datetime import date, timedelta
from datetime import datetime
import math
## under POSTGRESQL
import psycopg2
## under MDB
import sqlite3


################################
## FILES
################################

def search_shp_subdir(in_dir, extension):
    """searches 'in_dir'/input directory's subdirectories for files that are the file type/'extension', returns list of matching files""" 
    ## search subdirectories & matches if a file ends with 'extension', save first item of each folder's list 
    files = [[os.path.join(in_dir, i, f) for f in os.listdir(os.path.join(in_dir,i)) if f.endswith(extension)][0] for i in os.listdir(in_dir) if "." not in i]
    ## returns single list of files in in_dir's sub directories that end with .extension 
    return files
    
def find_file_diff(file1, file2, diff_file):
    """finds difference between 'file1' and 'file2' .txt files, returns 'diff_file', a .txt file with each line's differences""" 
    with open(file1) as file_1:
        file_1_text = file_1.readlines()
    with open(file2 ) as file_2:
        file_2_text = file_2.readlines()
    lines=[]
    for line in difflib.unified_diff(file_1_text, file_2_text, fromfile=file2, tofile=file1, lineterm=''):
            lines.append(line)
    ## save diff_file as a .txt file with each line's differences in a new line 
    with open(diff_file) as out:
        out.write(lines)
    ## returns difference file's filepath as a string
    return diff_file 


################################
## SUMMARY STATISTICS 
################################

def count_instances(df, field):
    """
    creates a new column in 'df' based on the number of occurences of that 'field' value in the 'df'
    returns 'df' with new column 'field_count'
    """
    
    instance_count = df.groupby([field])[field].apply(lambda x: len(x.tolist()))
    count_dict = dict(zip(instance_count.index, instance_count.values))
    df[field+"_count"] = [count_dict[i] for i in df[field]]
    return df

def field_percentile(df, field):
    """
    creates a new column in 'df' which is that row's 'field' (column / attribute)  percentile compared to all rows
    returns df with new column 'field_pct'
    """
    
    sz = df[field].size-1
    df[field+'_pct'] = df[field].rank(method='max').apply(lambda x: 100.0*(x-1)/sz)
    return df


################################
## TIME
################################

def subtract_hours(in_fi, date_col, subtr_hrs, new_col):
    """saves 'new_col' column in 'in_fi' with 'subtr_hrs' number of hours before 'date_col'"""
    df = pd.read_csv(in_fi)
    df[new_col] = df[date_col].astype('datetime64[ns]') - timedelta(hours=int(subtr_hrs))
    df.to_csv(in_fi)
    return df

def convert_seconds(seconds):
    """takes user input 'seconds' and returns how many hours, minutes, and seconds that is, as a list where first item is the total number of hours, second item is number of minutes, and third item is number of seconds"""
    mini, sec = divmod(seconds, 60)
    hour, mini = divmod(mini, 60)
    return [hour, mini, sec]

def find_days_btwn(date0, date1):
    """calculates the difference between 'date0' and 'date1', in MM-DD-YYYY format, returns the number of days as an integer. would return negative number if 'date1' was before 'date0' """    
    last_date = date0.replace("-", "", 2)
    current_date = date1.replace("-", "", 2)
    d0 =  date(int(last_date.split("-")[2]), int(last_date.split("-")[0]), int(last_date.split("-")[1]))
    d1 = date(int(current_date.split("-")[2]), int(current_date.split("-")[0]), int(current_date.split("-")[1]))
    numDays = d1 - d0
    return numDays.days

def fill_missing(TS_list):
    """interpolate missing/nan values in 'TS_list' filling in with the average of the previous and next non-nan values"""
    for idx, val in enumerate(TS_list):
        while idx >= 1:
            if math.isnan(val):
                prevval = TS_list[idx-1]
                nextval = TS_list[idx+1]
                ## if only one gap
                if not math.isnan(prevval) and not math.isnan(nextval):
                    newval = np.nanmean([prevval, nextval])                    
                    TS_list[idx] =  newval                        
                ## if next val is nan but previous val is not nan
                elif (not math.isnan(prevval) and math.isnan(nextval)):
                    ## if next2 are nan, take previous and the next3
                    if math.isnan(TS_list[idx+2]):
                        newval = np.nanmean([TS_list[idx-1], TS_list[idx+3] ])          
                        TS_list[idx] =  newval                        
                    ## if only the next1 is nan, take the previous and next2
                    else:
                        newval = np.nanmean([TS_list[idx-1], TS_list[idx+2] ])   
                        TS_list[idx] =  newval                        
                else:
                    print('fix')
    return TS_list