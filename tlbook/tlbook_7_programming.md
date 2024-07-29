# Programming

---

## Python packages used in [geo-tlbx repository](https://github.com/laurensharwood/geo-tlbx)

General:  
- ```numpy``` - arrays   
- ```pandas``` - tables   
- ```datetime``` - dates, times  
- ```tqdm```- progress bars   
- ```dask``` -  for parallel computing / reading arrays / tables larger than memory  

Database Translation:  
- ```sqlalchemy``` - ORM: generates SQL statements
- ```psycopg2-binary``` - driver: sends SQL statements to postgreSQL database
- ```sqlglot``` -  SQL parser, transpiler, optimizer, & engine  DuckDB, Spark / Databricks, Snowflake, BigQuery


Vector:
- ```pyproj``` - projection information 
- ```shapely``` - vector geometry 
- ```geopandas``` - pandas table + shapely geometry + pyproj projection

Read in spatial subset of file_name that's within region_gdf bounds to improve file reading speed: 
~~~
gdf = gpd.read_file(file_name, bbox = tuple(region_gdf.total_bounds))
~~~

<u>Geopandas drivers</u>:  

| Driver | Extension    | Name             | 
| ----- |--------------|------------------|
| GPKG  | .gpkg        | geopackage       |
| ESRI Shapefile | .shp         | shapefile        |
| OpenFileGDB    | .gdb         | file geodatabase |
| GeoJSON    | .geojson     | geojson          |
| SQLite | .db, .sqlite | sqlite database  |

Append gdf as a layer to an existing file, file_name, if the file is a .gdb, .gpkg, or .geojoson: 
~~~
gdf.to_file("out_gdb.gdb", layer="in_file", driver="OpenFileGDB", mode="a")
~~~



Create a [STRtree](https://shapely.readthedocs.io/en/stable/strtree.html#) <b>spatial index</b> to improve query / set operation speed: 
~~~
features_sub = features_gdf[features_gdf.index.isin(list(features_gdf.sindex.query(region_gdf.geometry, predicate="contains")[1]))]
~~~

[Spatial Joins](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.sjoin.html#geopandas.GeoDataFrame.sjoin)


Raster:
- ```rasterio``` - for reading/writing .tiff (geoTIFF) files
- ```rasterstats``` - for zonal statistics w/ a raster & vector
- ```xarray``` - dask arrays, for .nc (netCDF) files
- ```rioxarray``` - for saving netCDF files as geoTIFFs
- ```geowombat``` - dask arrays

Analysis:
- ```whitebox``` - WhiteboxTools is an advanced geospatial data analysis platform
- ```scipy``` - math / algorithms 
- ```scikit-learn``` - machine learning
- ```scikit-image``` - image (raster) processing
- ```pysal``` - spatial analysis
- ```pointpats``` - point pattern analysis  
- ```haversine``` - lon-lat distance calculation

Data Visualization:  
- ```matplotlib``` - standard static figures
- ```plotly``` - interactive charts 
- ```seaborn``` - interactive charts   
- ```branca``` - color maps

Mapping:  
- ```localtileserver``` - loading basemap raster tiles 
- ```ipyleaflet```  - Leaflet.js interactive web maps 
- ```leafmap``` - interactive web mapping
- ```folium``` - web mapping
- ```mapclassify``` -  choropleth map classification schemes

Data APIs:   
- ```pygris```: US Census ## ACS & TIGER   
- ```overpy```, ```osmnx```: OpenStreetMap    
- ```openrouteservice```: routes
- ```py3dep```: US 3D Elevation Program (3DEP)

GPX/TCX activity parsing:  
- ```tcxreader```: parses .tcx activity files
- ```gpxpy```: parses .gpx activity files

IMAGERY / EARTH OBSERVATION DATA CATALOGS:       
Microsoft Stac Catalog:
- ```planetary-computer```
- ```pystac```
- ```pystac-client```

ESA - Sentinel:
- ```sentinelhub```  

Planet Labs:  
- https://github.com/planetlabs/planet-client-python   
- [Planet Labs Jupyter Notebooks](https://github.com/planetlabs/notebooks/tree/master/jupyter-notebooks)

Google Earth Engine:
- ```earthengine-api``` - earth engine 
- ```geemap```- earth engine mapping helper

### Google Earth Engine - Python API

[Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets)+ https://gee-community-catalog.org   & Cloud Computing
* quota - download rate limit at 2 at any given time 

~~~
import ee

## do NOT select read only scopes 
ee.Authenticate() 
ee.Initialize()
~~~

#### Visualize Planet Basemaps monthly time series and download composite 
1) Create a Planet account under [Sign Up For Level 1 User Access](https://www.planet.com/nicfi/#sign-up)     
  *Note: Current Planet data users will need to sign up with a different email not associated with their Planet account.*    
2) Follow [setup instructions](https://developers.planet.com/docs/integrations/gee/nicfi/) to access [NICFI Planet Basemaps - Tropical Americas](https://developers.google.com/earth-engine/datasets/catalog/projects_planet-nicfi_assets_basemaps_americas) in Google Earth Engine (GEE).    
  *Note: The email associated with your GEE account might differ from the email used for your Planet account.*  


Jupyter IDE:   
- ```jupyterlab``` (*note to launch: > jupyter lab)   
- ```jupyter_contrib_nbextensions```   
- ```ipykernel```  
- ```ipywidgets```


---


## Virtual Environments
"A virtual environment is a Python environment such that the Python interpreter, libraries and scripts installed into it are isolated from those installed in other virtual environments, and (by default) any libraries installed in a “system” Python, i.e., one which is installed as part of your operating system"

| Virtual Environment Manager                                    | create envt                             | activate envt                   | install packages into envt             | deactivate envt|
|----------------------------------------------------------------|-----------------------------------------|---------------------------------|----------------------------------------|--------------|
| Mac [venv](https://docs.python.org/3/library/venv.html) | py -m venv .tlbx310                  | source .tlbx310/bin/activate | pip install {package}                  | deactivate| 
| Linux [venv](https://docs.python.org/3/library/venv.html)  | python3 -m venv .tlbx310             | source .tlbx310/bin/activate | pip install {package}                  | deactivate|   
| Windows [venv](https://docs.python.org/3/library/venv.html) | python3 -m venv .tlbx310             | .tlbx310\Scripts\activate    | pip install {package}                  | deactivate|   
| [anaconda](https://www.anaconda.com/download/success)          | conda create -n .tlbx310 python=3.11 | conda activate                  | conda install -c conda-forge {package} | conda deactivate |

Move into projects directory w/ all virtual environments (envt):  
~~~
cd {your_venv_dir}
~~~

For Anaconda only: do once after creating & activating environment, BUT before installing packages 
~~~
conda config --add channels conda-forge
conda config --set channel_priority strict
~~~

### Do once to connect virtual environment interpreter to IDE 
 
#### Jupyter:  
~~~
py -m ipykernel install --user --name=.tlbx310   
~~~

From activated venv, in your venv project directory, launch jupyter lab:
(.tlbx310) > C:\Users\laure\projects> jupyter lab


#### PyCharm:  
- Settings > Settings > Project > Python Interpreter > + > Add Local Interpreter > Existing (venv directory)  


---

## Cron Scheduler


1. <b>create bash script</b> ```get_garmin.sh``` <b>to be called by cron</b>-- which have the following lines to activate a virtual environment (venv) with necessary packages installed, then run the python script executable from that active venv:    
~~~
#!/bin/bash

source .tlbx310/bin/activate
python get_garmin.py
deactivate
~~~

2. enter the following commands in the terminal to ensure get_garmin.sh and get_garmin.py have <b>appropriate execute</b> permissions:    
~~~
chmod +x get_garmin.py
chmod +x get_garmin.sh
~~~

3. <b>create task</b>: enter ```crontab -e``` in the terminal 

Add a line specifying how often to execute a script:   
~~~
min hour day-of-month month day-of-week {command}  
~~~

ex) every day at 10:00am run get_garmin.sh & don't send that email  (```*``` == all)    
~~~
00 10 * * * ~/get_garmin.sh >/dev/null 2>&1  
~~~
  
<u>to save</u>: ctrl+X (to escape editing session, then) Y (yes), enter     

4. <b>print active tasks</b> to ensure it was created: ```crontab -l```  

---

## Version Control

#### Git

Store your GitHub login credentials:   
~~~
git	
git config --global user.name {your_github_username}
git config --global user.email {your_github_email}
git config --global --list	
~~~

Initialize a folder as GitHub repository:   
~~~
cd {your_repo}
git init
git add .
git commit -m "message"
git remote add origin https://github.com/{your_username/your_repo}.git

~~~

Push updates to Github repository:    
~~~
git push -u origin main
cd {your_repository_directory}
git add .
## git status
git commit -m "message"
git push
## git push -u origin main	
## git push --set-upstream origin main
~~~

Ignore & delete local changes -- take remote repository:     
~~~
git reset --hard git pull
git pull
~~~


