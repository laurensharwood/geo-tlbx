# geo-tlbx
toolbox of code using geospatial data  

## STANDARDS
improve geographic information's utility & value by increasing its interoperability, reusability, reliability, access, etc.   

* International Organization for Standardization (ISO) standards must be purchased. The American National Standards Institute (ANSI) serves as the US member agency to ISO and provides easier access to the standards and, generally, at a lower cost.   

* Open Geospatial Consortium (OGC) is a diverse array of international groups (govt, academia, private, etc) using geospatial data, settling on standards for sharing & integrating data.

* Federal Geographic Data Committee (FGDC) is a U.S. interagency group with the same mission 


<b> Open Geospatial Consortium (OGC)</b>   
OGC publishes the following documents: 1) implementation standards, 2) abstract specifications, 3) best practices, 4) engineering reports, 5) discussion papers, and 6) change requests.  

Standards:  
* Data Encoding:  
    - [Geography Markup Language (GML)](http://opengeospatial.github.io/e-learning/gml/text/main.html)   
    - [Geopackage (.gpkg)](http://opengeospatial.github.io/e-learning/geopackage/text/introduction.html)   
* Data Access:  
    - [Web Feature Service (WFS)](http://opengeospatial.github.io/e-learning/wfs/text/basic-main.html)   
    - [Web Coverage Service (WCS)](http://opengeospatial.github.io/e-learning/wcs/text/basic-main.html#introduction)   
* Processing:
    - [Web Processing Standards (WPS)](http://opengeospatial.github.io/e-learning/wps/text/basic-main.html)  
* Visualization:  
    - [Web Map Service (WMS)](http://opengeospatial.github.io/e-learning/wms/text/basic-main.html)   
    - [Web Map Tile Service (WMTS)](http://opengeospatial.github.io/e-learning/wmts/text/main.html#introduction)   
* [Metadata and Catalogue Service](http://opengeospatial.github.io/e-learning/metadata/text/specifications.html)

</br>  

<b>Federal Geographic Data Committee (FGDC)</b>  
* FGDC [standards list](https://www.fgdc.gov/standards/list) includes standards from FGDC, along wtih OGC and ISO 

## [Metadata Creation](https://www.usgs.gov/data-management/metadata-creation)

### GIS metadata standards:  
- ISO 19115: Geographic information — Metadata  
- ISO 19139: Geographic information — Metadata — XML schema  
- [FGDC Content Standard for Digital Geospatial Metadata (CSDGM)](https://www.fgdc.gov/metadata/csdgm-standard) to Create System-level Metadata Records  

### Metadata validation:  
* Compares the metadata standard to the XML metadata record to ensure it conforms to the structure of the standard, such that all of the required elements are filled in.
* USGS best practices for [Checking Metadata with Data](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/CheckingMetadataWithData_508-compliant.pdf) with FGDC-CSDGM metadata
* [USGS Metadata Parser (MP)](https://geology.usgs.gov/tools/metadata/tools/doc/mp.html) 
* [Metadata Wizard tool](https://code.usgs.gov/usgs/fort-pymdwizard)


## DATABASES

### postgreSQL  

<b>create database</b>  
launch terminal:  
> createdb -h localhost -p 5432 -U postgres -T template_postgis db_name;  

<b>create table</b>  
launch psql db(db_name=#):  
> \l ## print databases in postgres db server  
> \t ## print tables in connected db  
> CREATE TABLE tablname (id_key INTEGER PRIMARY KEY, fullname varchar(100) NOT NULL, geom geometry(LINESTRING,4269));
    
<b>delete database</b>  
DROP DATABASE db_name;

<b>delete column</b>  
ALTER TABLE table_name DROP COLUMN column_name;


### join

<b>inner join</b>  
links two tables, 'tablname' and lookuptable, 'lut', using a common 'key' column, and returns rows where 'key' value exists in both tables  
> SELECT * FROM tablname INNER JOIN lut USING (key)

<b>left outer join</b>  
returns all rows from first table, tablname, with blank value where 'key' is missing in the second table, lut (will not have rows from second table where key is missing in first table)  
> SELECT * FROM tablname LEFT JOIN lut ON tablname.key = lut.key

<b>right outer join</b>  
returns all rows from lut with blank value where 'key' is missing in tablname (will not have rows from first table whose key)   
> SELECT * FROM tablname RIGHT JOIN lut ON tablname.key = lut.key 

<b>full outer join</b>  
returns all rows from both tables   
> SELECT * FROM tablname FULL OUTER JOIN lut ON tablname.key = lut.key 


### aggregate 

<b>sum minutes column from 'runs_tcx' table</b>   
> SELECT SUM(minutes) FROM runs_tcx  

<b>count number of columns in a table</b>  
> SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = 'public' -- the database AND table_name = 'tablname'

<b>reclassify</b>  

> SELECT * FROM nomnom WHERE cuisine = 'Italian' AND price = '$$$';
SELECT name, CASE
WHEN review > 4.5 THEN 'Extraordinary'
WHEN review > 4 THEN 'Excellent'
WHEN review > 3 THEN 'Good'
WHEN review > 2
THEN 'Fair' ELSE 'Poor'
END AS 'Review' FROM nomnom;

### geopandas (python)  
~~~
import geopandas as gpd  
gdf = gpd.read_file("in_file")  
gdf.to_file("out_file.gpkg", driver="GPKG")   
~~~

* some drivers support mode="a" to append 

| Driver | Extension | 
| ----- | ----- |
| GPKG  | .gpkg    |
| ESRI Shapefile | .shp |
| OpenFileGDB    |.gdb  |
| GeoJSON    | .json   |
| SQLite | .db, .sqlite  |


### osgeo library  

From OSGeo4W Shell:  
> ogr2ogr -f "ESRI Shapefile" "C:\Users\Lauren\EO_ML\USCensus\SB_census_soil.shp" "C:\Users\Lauren
\EO_ML\USCensus\SB_census.gpkg" "soil"

> for /R %f in (*.gpx) do ogr2ogr -f "GPKG" my_geopackage.gpkg "%f"


## LINUX 
- pressing tab following a command (cd , ls , vim ) will list all possible files/folders and autofill if there's only one match 
- run 'pwd' to print working directory and easily copy+paste path later

'|' : pipe that takes first command's output and feeds it into the command after the pipe  
'*' : wildcard matching any character  
'-r' : recursively search subdirectories

<b>list</b>(l) files, sorted by time(t), reverse(r) - last modified file at bottom of list:  
> ls -ltr  

<b>list</b> two last modified files:  
> ls -ltr | tail -2

<b>size</b> of current directory:  
> du -sh .

<b>files sorted based on size</b>:  
> du -sh -- * | sort -rh  

<b>count</b> number of files in current directory:  
> ls | wc -l  

<b>count</b> number of files in directory (that start with 004):   
> ls -dq 004* | wc -l 

<b>move </b> files that contain filterstring (in current dir) and move them into out_dir:  
> mv * filterstring * out_dir  

<b>zip</b> files in folder  
> zip -r out_file.zip in_dir*

<b>delete</b> files (f) in current firectory (.) that match string (start with S1_):  
> find . -type f -name 'S1_*' -delete  
> find . -name gee -exec ls {} \;
> find . -name gee -exec rm -rf {} \;  

<b>delete</b> files (f) <i>recursively</i> in current directiry (.) that match string (start with L3A_LC):   
> find . -name "L3A_LC*" -type f -exec rm -r {} +

<b>delete</b> folders (d) in current firectory (.) that match string (contain the string landsat):  
> find . -name "*landsat*" -type d -exec rm -r {} +


### SLURM 
submit bash script through SLURM (must execute from ~/code/bash directory):  
~~~
cd ~/code/bash 
sbatch {script_name.sh}
~~~

check on bash script progress:
~~~
squeue ## to make sure it's running   
ls -ltr ## from ~/code/bash, prints the last file to be created printed at the bottom. copy filename, bottom_file.err     
cat {bottom_file.err} ## cat command prints the contents of the file, bottom_file.err, in the terminal      
~~~

## VIRTUAL ENVIRONMENTS   

### Mac virtual environment (venv) instructions 
- venv installed in user's base directory 
~~~
python3 -m venv .working
source .working/bin/activate
pip install pandas numpy geopandas earthengine-api geemap
pip install jupyterlab localtileserver jupyter_contrib_nbextensions ipyleaflet ipywidgets
pip install rasterio
pip install gdal

~~~

### Anaconda 
~~~
conda create -n .working python=3.9
conda activate .working
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -c conda-forge pandas numpy geopandas earthengine-api geemap
conda install -c conda-forge jupyterlab localtileserver jupyter_contrib_nbextensions ipyleaflet ipywidgets
conda install -c anaconda ipykernel ## add environment to jupyter notebook ("jupyter lab" to launch)
python -m ipykernel install --user --name=.working ## add environment to jupyter notebook ("jupyter lab" to launch)
conda install -c conda-forge rasterio
conda install -c conda-forge gdal
~~~

### Micromamba: lightweight version of Anaconda
https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html  
~~~
mamba install {package}  
~~~
