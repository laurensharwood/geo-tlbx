[Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets)

ee.ImageCollection():  
|ancillary |gee ImageCollection feature|
|---|---|
|precip|ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")| 
|Planet monthly basemaps|ee.ImageCollection("projects/planet-nicfi/assets/basemaps/americas")| 
|Planet monthly basemaps|ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa")| 
|Planet monthly basemaps|ee.ImageCollection("projects/planet-nicfi/assets/basemaps/asia")| 
|Sentinel-1|ee.ImageCollection("COPERNICUS/S1_GRD")| 
|Sentinel-2 harmonized surface reflectance |ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")| 
|Sentinel-2 cloud probability |ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")| 
|MODIS LST |ee.ImageCollection('MODIS/006/MOD11A1').select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night']) | 

ee.Image():   
|ancillary |gee Image feature|
|---|---|
|elevation|ee.Image("NASA/NASADEM_HGT/001").select(['elevation'])| 
|elevation|ee.Image("USGS/SRTMGL1_003").select(['elevation'])| 
|landcover| "USGS/NLCD_RELEASES/2021_REL/NLCD"  | 
|landcover| dynamic world   | 
|landcover|"projects/mapbiomas-public/assets/paraguay/collection1/mapbiomas_paraguay_collection1_integration_v1"   | 
|soil| "ISDASOIL/Africa/v1/fcc"   | 
|soil| "ISDASOIL/Africa/v1/cation_exchange_capacity"   | 
|soil| "ISDASOIL/Africa/v1/cation_exchange_capacity"   | 
|soil| "ISDASOIL/Africa/v1/clay_content"   | 
|soil| "ISDASOIL/Africa/v1/carbon_organic"   | 
|soil| "ISDASOIL/Africa/v1/bedrock_depth" | 



ee.FeatureCollection():
"GOOGLE/Research/open-buildings/v3/polygons"



### Visualize Planet Basemaps monthly time series and download composite 
1) Create a Planet account under [Sign Up For Level 1 User Access](https://www.planet.com/nicfi/#sign-up)     
  *Note: Current Planet data users will need to sign up with a different email not associated with their Planet account.*    
2) Follow [setup instructions](https://developers.planet.com/docs/integrations/gee/nicfi/) to access [NICFI Planet Basemaps - Tropical Americas](https://developers.google.com/earth-engine/datasets/catalog/projects_planet-nicfi_assets_basemaps_americas) in Google Earth Engine (GEE).    
  *Note: The email associated with your GEE account might differ from the email used for your Planet account.*  


