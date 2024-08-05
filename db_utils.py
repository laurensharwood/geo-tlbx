import os

import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point, LineString
import datetime
from datetime import timedelta, date
## parsing TCX/GPX functions
import gpxpy
from tcxreader.tcxreader import TCXReader

import psycopg2





################################
## GDB 
################################

def gdb_to_other(in_file, extension):
    layers=fiona.listlayers(in_file)
    if len(layers) > 1:
        for lyr in layers:
            ds=gpd.read_file(in_file, layer=lyr)
            ds.to_file(in_file.replace(os.path.basename(in_file), lyr+"."+extension.replace(".", "")), layer=lyr)
    else:
        ds=gpd.read_file(in_file)
        ds.to_file(in_file.replace(".gdb", "."+extension.replace(".", "")))   




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
    con = sqlite3.connect(in_db)
    cursor = con.execute("SELECT * from "+table_name)
    entries=[]
    for i in cursor:
        entries.append(i)
    return pd.DataFrame(entries)


################################
## POSTGRESQL
################################


def df_to_postgres(df, table_name, db, usr, pwd, localhost="localhost", port="5432"):
    try:
        conn = psycopg2.connect(database = db, user = usr, password = pwd, host = localhost, port = port)
        cur = conn.cursor()
        col_names = df.columns.to_list()
        for i in range(0 ,len(df)):
            values = tuple(df[col][i] for col in col_names)
            cur.execute("INSERT INTO {} ({}) VALUES({})".format(table_name, ", ".join(col_names), ", ".join(["%s"] * len(col_names))))
        conn.commit()

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)
    
    finally:
        if conn:
            cur.close()
            conn.close()
            print("PostgreSQL connection is closed")

def postgres_to_df(SQL_query, db, user="postgres", pwd="", host="localhost", port=5432):
    try:
        conn = psycopg2.connect(database=db, user="postgres", password=pwd, host=host, port=port)
        cur = conn.cursor()
        cur.execute(SQL_query)
        items = cur.fetchall()
        hits=[]
        for row in items:
            hits.append(row)
        col_names = [desc[0] for desc in cur.description] 
        hits_df=pd.DataFrame(hits, columns=col_names)
        if ("lat" in col_names and "lon" in col_names):
            hits_gdf = gpd.GeoDataFrame(hits_df, geometry=gpd.points_from_xy(hits_df.loc[:,'lon'],hits_df.loc[:,'lat'], crs="EPSG:4326"))
            return hits_gdf
        else:
            return hits_df

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)
    
    finally:
        if conn:
            cur.close()
            conn.close()
            print("PostgreSQL connection is closed")

def split_gpx_at(fi, split_min):
    gpx_file = open(fi, 'r')
    gpx = gpxpy.parse(gpx_file, version='1.1')
    trackpoints = []
    trackpoints2 = []
    for track in gpx.tracks:
        for seg in track.segments:
            for point_no, pt in enumerate(seg.points):
                first_part = True
                if point_no == 0:
                    trackpoints.append([pt.time, fi, pt.latitude, pt.longitude, pt.elevation])
                elif point_no > 0:
                    secs_btwn = pt.time - seg.points[point_no - 1].time
                    minutes = secs_btwn.total_seconds() / 60
                    if minutes < split_min:
                        trackpoints.append([pt.time, fi, pt.latitude, pt.longitude, pt.elevation])
                    elif (minutes > split_min or first_part == False):
                        trackpoints2.append([pt.time, fi.replace(".gpx", "_2.gpx"), pt.latitude, pt.longitude, pt.elevation])
                        first_part = False
                    else:
                        print('CHECK')
    return (trackpoints, trackpoints2)

def gpx_to_postgres(data_dir, table_name, db='garmin_activities'):
    '''
    data_dir  = input directory to put GPX waypoints in
    db = database
    table_name = table in db that gpx waypoints will be added to (should already exist in db)
        CREATE TABLE gpx_runs (date TIMESTAMP PRIMARY KEY, filename CHAR(18) NOT NULL, lat FLOAT NOT NULL, lon FLOAT NOT NULL, ele FLOAT NOT NULL, speed FLOAT NOT NULL);
        CREATE TABLE gpx_bikes (date TIMESTAMP PRIMARY KEY, filename CHAR(18) NOT NULL, lat FLOAT NOT NULL, lon FLOAT NOT NULL, ele FLOAT NOT NULL, speed FLOAT NOT NULL);
    returns list of files that were parsed
    '''
    gpx_files = [i for i in sorted(os.listdir(data_dir)) if i.endswith('.gpx')]
    try:
        conn = psycopg2.connect(database=db, user='postgres', password='', host='localhost', port=5432)
        cur = conn.cursor()
        for fi in gpx_files:
            gpx_file = open(os.path.join(data_dir, fi), 'r')
            gpx = gpxpy.parse(gpx_file, version='1.1')
            for track in gpx.tracks:
                for seg in track.segments:
                    for point_no, pt in enumerate(seg.points):
                        if pt.speed != None:
                            speed = pt
                        elif point_no > 0:
                            speed = pt.speed_between(seg.points[point_no - 1])
                        elif point_no == 0:
                            speed = 0
                        else:
                            speed = 0
                        ## add _2 to filename if consecutive trackpoints are more than 60 minutes apart 
                        run_parts = split_gpx_at(fi = os.path.join(data_dir, fi), split_min = 60)
                        for run_part in run_parts:
                            cur.execute('INSERT INTO '+table_name+' (date, filename, lat, lon, ele, speed) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING',
                                       run_part)
        conn.commit()
        print('Records inserted successfully')
        # conn.close()
        return gpx_files

    except (Exception, psycopg2.Error) as error:
        print('Error while fetching data from PostgreSQL', error)

    finally:
        if conn:
            cur.close()
            conn.close()
            print('PostgreSQL connection is closed')

def tcx_to_postgres(data_dir, db='garmin_activities'):
    '''
    data_dir  = input directory to parse all TCX files in
    1) adds basic activity stats to 'run_stats' or 'bike_stats' table in 'tcx_activities' postgres database
    2) stats per waypoint go into 'run_pts' or 'bike_pts' table in 'tcx_activities' postgres database
    returns list of files that were parsed 
    '''

    ## for PST my start times need to be subtracted by six or seven hours depending on daylight savings  
    subtr_hrs = 6
    tcx_files = [i for i in sorted(os.listdir(data_dir)) if i.endswith(".tcx")]
    try:
        conn = psycopg2.connect(database=db, user="postgres", password="", host="localhost", port=5432)
        cur = conn.cursor()
        for tf in tcx_files:
            file = open(os.path.join(data_dir, tf), "r")
            tcx_reader = TCXReader()
            exercise = tcx_reader.read(os.path.join(data_dir, tf))
            if (exercise.activity_type == "Running" and exercise.duration != None):
                cur.execute("INSERT INTO run_stats (filename, start, distance, duration, ascent, avg_speed, hr_avg, hr_max) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING", 
                           [tf, exercise.start_time - timedelta(hours=subtr_hrs), exercise.distance, exercise.duration, exercise.ascent, exercise.avg_speed, exercise.hr_avg, exercise.hr_max])
                for pt_info in exercise.trackpoints:
                    cur.execute("INSERT INTO run_pts (date, filename, lon, lat, distance, elevation, hr, cadence) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING", 
                                [pt_info.time - timedelta(hours=subtr_hrs), tf, pt_info.longitude, pt_info.latitude, pt_info.distance,  pt_info.elevation, pt_info.hr_value, pt_info.cadence])        
            elif (exercise.activity_type == "Biking" and exercise.duration != None):
                cur.execute("INSERT INTO bike_stats (filename, start, distance, duration, ascent, avg_speed, hr_avg, hr_max) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING", 
                           [tf, exercise.start_time - timedelta(hours=subtr_hrs), exercise.distance, exercise.duration, exercise.ascent, exercise.avg_speed, exercise.hr_avg, exercise.hr_max])
                for pt_info in exercise.trackpoints:
                    cur.execute("INSERT INTO bike_pts (date, filename, lon, lat, distance, elevation, hr) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING", 
                                [pt_info.time - timedelta(hours=subtr_hrs), tf, pt_info.longitude, pt_info.latitude, pt_info.distance,  pt_info.elevation, pt_info.hr_value])        
        conn.commit()
        print("Records created successfully")
        conn.close()
        return tcx_files
                    
    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)
        
    finally:
        if conn:
            cur.close()
            conn.close()
            print("PostgreSQL connection is closed")
