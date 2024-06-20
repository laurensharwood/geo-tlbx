import os
import pandas as pd
## under FILES, for find_file_diff function
import difflib
## for TIME functions
import datetime
from datetime import timedelta
from datetime import datetime
import math
## under POSTGRESQL
import psycopg2
import sqlalchemy
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
## MDB 
################################

def list_mdb_tables(in_db):
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect(in_db)
    cur = con.cursor()
    # reading all table names
    table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    # Be sure to close the connection
    con.close()

def mdb_table_as_df(in_db, table_name):
    sqlite3.connect(in_db)
    cursor = conn.execute("SELECT * from "+table_name)
    entries=[]
    for i in cursor:
        entries.append(i)
    return pd.DataFrame(entries)

################################
## POSTGRESQL
################################

def update_posgres_table(df, table_name, db, usr, pwd, localhost="localhost", port="5432"):
    conn = psycopg2.connect(database = db, user = usr, password = pwd, host = localhost, port = port)
    cur = conn.cursor()
    col_names = df.columns.to_list()
    for i in range(0 ,len(df)):
        values = tuple(df[col][i] for col in col_names)
        cur.execute("INSERT INTO {} ({}) VALUES({})".format(table_name, ", ".join(col_names), ", ".join(["%s"] * len(col_names))
    ))
    conn.commit()
    conn.close()

def query_postgres(SQL_query, db, usr, pwd, localhost, port):
    conn = psycopg2.connect(database = db, user = usr, password = pwd, host = localhost, port = port)
    cur = conn.cursor()
    cur.execute(SQL_query)
    hits=[]
    items = cur.fetchall()
    for i in items:
        hits.append(i)
    hits_df=pd.DataFrame(hits, columns=SQL_query[7:].split(" FROM ")[0].split(","))
    hits_df.rename(columns={ df.columns[:2]:["lon", "lat"]}, inplace = True)
    print(hits_df)
    hits_gdf = gpd.GeoDataFrame(hits_df, geometry=gpd.points_from_xy(hits_df.iloc[:,0],hits_df.iloc[:,1], crs="EPSG:4326"))
    return hits_gdf

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
    last = date0.replace("-", "", 2)
    current = date1.replace("-", "", 2)
    d0 =  date(int(last_date.split("-")[2]), int(last_date.split("-")[0]), int(last_date.split("-")[1]))
    d1 = date(int(current_date.split("-")[2]), int(current_date.split("-")[0]), int(current_date.split("-")[1]))
    numDays = d1 - d0
    return numDays.days

def fill_missing(TS_list):
    """interpolate missing/nan values in 'TS_list' filling in with the average of the previous and next non-nan values"""
    for idx, val in enumerate(TS_list):
        while idx >= 1:
            if math.isnan(val):
                old_val = val
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