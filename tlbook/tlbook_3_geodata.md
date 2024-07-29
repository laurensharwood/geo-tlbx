# Geographic Data

---

## Coordinate Reference System (CRS)
1. <b>Ellipsoid:</b> Shape of Earth's surface is not perfect sphere, but squashed
   * <b>Geoid:</b> Equal gravitational potential model that estimates mean sea level to define Earth's Ellipsoid 
2. <b>Datum:</b> Defines the two coordinate axis (Longitude: X, Latitude: Y) reference points for measuring any point on Earth's curved surface
3. <b>Geographic CRS (3D):</b> Latitude, Longitude for referencing location on Earth's curved/ellipsoid surface
4. <b>Projected CRS (2D):</b> Earth's curved surface *projected* onto a 2D surface. Distortion occurs during projection, but 
the <b>goal is to choose a *projection method* that limits your project area's Easting & Northing distances from the Datum in order to minimize this distortion</b>.   
   - Due to its long shape (elongated North to South), California is divided into 5 State Plane (CASP) Zones. For each CASP Zone, a *conformal conic projection* is optimal do to the East-West elongation and mid-latitude location. 

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
Lidar file formats:  
  * ASCII: raw lidar data   
  * [LAS](https://lastools.github.io/): point clouds with XYZ (East-North-Elevation) values    
  * LAZ (zipped LAS)   

<b>[Computer Automated Design](https://pro.arcgis.com/en/pro-app/latest/help/data/cad/what-is-cad-data.htm) (CAD):</b> 
  * From CAD software (AutoCAD, Microstation)
  * Files: .dwg, .dxf
    * [Convert to shp](https://gisgeography.com/dwg-to-shp/)


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



