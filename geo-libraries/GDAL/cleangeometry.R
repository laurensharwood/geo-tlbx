# if(!require(pacman)) {
install.packages('pacman')
install.packages('p_load')

# runme 
library(pacman)

p_load(raster,
       devtools,
       dplyr,
       rdgal,
       cleangeo,
       rgeos,
       BiocManager,
       sf,
       rgdal)
library(rgdal) ## 3.4 not 3.6 

## readOGR()
## threat_polys_orig <- readOGR(dsn = "F:/Zone_805/01_REFERENCE_VECTORS", layer = "Zone_805_TVPolys_0405", verbose = TRUE)


## import 
orig_threat_polys <- "L:/DEL_3B/D3/BLOCK_D3_detection_polys.shp"
threat_polys_orig <-  shapefile(orig_threat_polys)

## reproject
threat_polys_proj <- spTransform(threat_polys_orig, '+proj=utm +zone=10 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')

## -1m buffer & repair geometry
threat_polys_proj_nbuff_rep <- clgeo_Clean(gBuffer(threat_polys_proj, width = -1, byid = T))

## export 
writeOGR(threat_polys_proj_nbuff_rep, layer = 'BLOCK_D3_detection_polys_repair', driver = "ESRI Shapefile",  dsn = "L:/", overwrite = T)

