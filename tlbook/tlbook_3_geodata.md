# Geographic Data




---

## Coordinate Reference System (CRS)
1. <b>Ellipsoid:</b> Shape of Earth's surface is not perfect sphere, but squashed
   * <b>Geoid:</b> Equal gravitational potential model that estimates mean sea level to define Earth's Ellipsoid 
2. <b>Datum:</b> Defines the two coordinate axis (Longitude: X, Latitude: Y) reference points for measuring any point on Earth's curved surface
   * ex) NAD 83
4. <b>Geographic CRS (3D):</b> Latitude, Longitude for referencing location on Earth's curved/ellipsoid surface. For global-scale projects. Distance is distorted.
   * ex) Web Mercator
4. <b>Projected CRS (2D):</b> Earth's curved surface *projected* onto a 2D surface. Distortion occurs during projection, but 
the goal is to choose a projection method that limits your project area's Easting & Northing distances from the Datum in order to minimize the distortion type critical for your project.    
   - Types: Conformal projections preserve angular distortion, equidistant projections preserve distance, and equal area projections preserve area. 
   * ex) Due to its long shape (elongated North to South), California is divided into 6 State Plane (CASP) Zones. For each CASP Zone, a *Lambert conformal conic projection* is optimal do to the East-West elongation and mid-latitude location. 

<b>[Projection Wizard](https://projectionwizard.org):</b>  Tool to find appropriate projection based on your project area

<b>[PROJ.4 String](https://pygis.io/docs/d_understand_crs_codes.html#proj-4-string):</b>  Multiple parameters needed to describe a CRS. 

<b>EPSG:</b> public registry of CRS info  
* EPSG 4326 - *Unprojected/Geographic* - Used for GPS field survey devices & raw survey data   
* EPSG 3857 - *Projected* - Used for features published in web maps   

<b>CRS Transformation:</b> Transforms geographic data as images(rasters) or coordinates(vectors) between two different CRS using [six parameters](https://rasterio.readthedocs.io/en/stable/topics/transforms.html).  

---

## Common Data Types & Formats 

### Vectors
Vectors represent objects as features (point, line, or polygon shape) with tabular data containing attributes/fields values for each geometry.   
<b>Feature class</b>: homogenous collection of features (points, lines, or polygons)   
  * ESRI Shapefile
  * Database (SQLite, MS Access, PostgreSQL, ESRI file geodatabase)
  * [GeoJSON](https://courses.spatialthoughts.com/python-foundation.html#understanding-json-and-geojson) ([GeoJSON creator](https://geojson.io))
  * KML/KMZ (for Google Earth Pro)
  * [Geoparquet](https://geoparquet.org/)


<b>Networks:</b>   
* Set of connected objects in geographic space to answer questions about connections and flow.   
* Contain objects (as nodes/points and connections/lines). May involve adjacency matrix calculation.  

### Rasters
Rasters are grids with pixels with values that represent continuous fields (elevation, temperature, an aerial image) 
  * GeoTiff: .tiff, .tif
  * netCDF (time-series)
  * HDF5 (time-series)
  * BIL, BIP
  * COG

<b>Spatial Resolution:</b> Pixel size 
- Amount of detail, precision, granularity
- Fine resolution = small pixel size
- Coarse resolution = large pixel size

<b>Light Detection and Ranging (lidar):</b> used to create elevation model rasters      
Lidar processing tool: [rapidlasso](https://rapidlasso.de/)  

Lidar file formats:  
  * ASCII: raw lidar data   
  * [LAS](https://lastools.github.io/): 3D point clouds with XYZ (lon-lat-ele) values    
  * LAZ (zipped LAS)   

<b>[Computer Automated Design](https://pro.arcgis.com/en/pro-app/latest/help/data/cad/what-is-cad-data.htm) (CAD):</b> 
  * From CAD software (AutoCAD, Microstation)
  * Files: .dwg, .dxf
    * [Convert to shp](https://gisgeography.com/dwg-to-shp/)


---

## GIS Standards: 
Improve geographic information's utility & value by increasing its interoperability, reusability, reliability, and access.   
* Example of [City of Fremont CAD standards](https://storymaps.arcgis.com/stories/9767345c01fc4fd5a6b90e970b249dbd)  

International Organization for Standardization (ISO) standards must be purchased. The American National Standards Institute (ANSI) serves as the US member agency to ISO and provides easier access to the standards and, generally, at a lower cost.   

<u> Open Geospatial Consortium (OGC)</u> is a diverse array of international groups (govt, academia, private, etc.) using geospatial data, settling on standards for sharing & integrating data.   
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

<u>Federal Geographic Data Committee (FGDC)</u> is a U.S. interagency group with the same mission 

* FGDC [standards list](https://www.fgdc.gov/standards/list) includes standards from FGDC, along with OGC and ISO

#### GIS metadata standards:  
- ISO 19115: Geographic information — Metadata  
- ISO 19139: Geographic information — Metadata — XML schema  
- [FGDC Content Standard for Digital Geospatial Metadata (CSDGM)](https://www.fgdc.gov/metadata/csdgm-standard) to Create System-level Metadata Records  

#### [Metadata creation best practices](https://www.usgs.gov/data-management/metadata-creation):   
* Gather all information together & reuse information that is already developed, e.g. abstract, purpose, date from grant or funding proposals
* Choose a descriptive title for your data that incorporates who, what, where, when, and scale.
* Choose keywords wisely -- consider all possible interpretations of your word choices.
* Include as many details as you can in the metadata record for future users of the data.
* Update the metadata date (date stamp) so that metadata repositories will know which version of the record is most recent.
* DOI should go in the primary <onlink> in the Citation Information section and should be a URL. 

#### Metadata validation:  
* Compares the metadata standard to the XML metadata record to ensure it conforms to the structure of the standard, such that all of the required elements are filled in.
* USGS best practices for [Checking Metadata with Data](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/CheckingMetadataWithData_508-compliant.pdf) with FGDC-CSDGM metadata
* [USGS Metadata Parser (MP)](https://geology.usgs.gov/tools/metadata/tools/doc/mp.html) 
* [Metadata Wizard tool](https://code.usgs.gov/usgs/fort-pymdwizard)


---

## Publicly available data:  

* [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog) + https://gee-community-catalog.org
* [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/catalog)
* [Planet Labs Basemaps](https://developers.planet.com/docs/basemaps/) 
* [USAID Spatial Data Repository](https://spatialdata.dhsprogram.com/home/)
* [US Govt](https://catalog.data.gov/)
* [USGS](https://data.usgs.gov/datacatalog/search)
* [USFS](https://data-usfs.hub.arcgis.com)
* [California](https://gis.data.ca.gov/) & [CA vegetation](https://wildlife.ca.gov/Data/GIS/Vegetation-Data)

### Google Earth Engine: Data Catalog & Cloud Computing   

#### Data:  
##### ee.FeatureCollection   

| category         |ee.FeatureCollection|
|------------------|---|
| Google buildings |"GOOGLE/Research/open-buildings/v3/polygons"| 

##### ee.Image

| category  |ee.Image| 
|-----------|---| 
| elevation |ee.Image("NASA/NASADEM_HGT/001").select(['elevation'])| 
| elevation |ee.Image("USGS/SRTMGL1_003").select(['elevation'])| 
| landcover |"projects/mapbiomas-public/assets/paraguay/collection1/mapbiomas_paraguay_collection1_integration_v1" | 
| soil      | "ISDASOIL/Africa/v1/fcc" | 
| soil      | "ISDASOIL/Africa/v1/cation_exchange_capacity" | 
| soil      | "ISDASOIL/Africa/v1/cation_exchange_capacity" | 
| soil      | "ISDASOIL/Africa/v1/clay_content" | 
| soil      | "ISDASOIL/Africa/v1/carbon_organic" | 
| soil      | "ISDASOIL/Africa/v1/bedrock_depth" |  

##### ee.ImageCollection

| name                                      |ee.ImageCollection|
|-------------------------------------------|---|
| US NLCD |  ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").select("landcover") | 
| precip                                    |"UCSB-CHG/CHIRPS/DAILY"| 
| Sentinel-1                                |"COPERNICUS/S1_GRD"| 
| Sentinel-2 harmonized surface reflectance |"COPERNICUS/S2_SR_HARMONIZED"| 
| Sentinel-2 cloud probability              |"COPERNICUS/S2_CLOUD_PROBABILITY"| 
| MODIS LST                                 |ee.ImageCollection('MODIS/006/MOD11A1').select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night']) | 
| Planet monthly basemaps                   |"projects/planet-nicfi/assets/basemaps/americas"| 
| Planet monthly basemaps                   |"projects/planet-nicfi/assets/basemaps/africa"| 
| Planet monthly basemaps                   |"projects/planet-nicfi/assets/basemaps/asia"| 
