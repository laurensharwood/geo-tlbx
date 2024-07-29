# Geospatial ToolBook

to-do: https://jupyterbook.org/en/stable/basics/create.html 

The purpose of this WORKING book is to support the accessibility of this geographic information & 
* provide anyone with little-to-no (or forgotten) GIS/technical/software knowledge the ability to utilize the vast & increasing amount of geospatial data that exist today.
support collaboration and the use of geospatial data across disciplines:
* This book highlights publicly available & free resources,
  * so this is a non-exhaustive list of geospatial tools available to you today.
* Additionally, [geo-tlbx](https://github.com/laurensharwood/geo-tlbx/) is my public GitHub repository that contains tools of mostly Python code I've written over the last few years to download, manage, analyze, and visualize geospatial data. 

## Resources: 

* Geospatial Blogs: 
  * [Element84](https://element84.com/blog/)
  * [ESRI](https://www.esri.com/arcgis-blog/overview/)   
  * [GIS Geography](https://gisgeography.com/)
* Podcasts:
  * [Mapscaping](https://mapscaping.com/podcasts/)  
  * [Geography is Everything](https://geographyiseverything.substack.com/podcast)
  * [Think Fast, Talk Smart](https://www.gsb.stanford.edu/business-podcasts/think-fast-talk-smart-podcast)   
           *<sup>*Science communication is poor - starting with general communication should help*</sup>
* Data Science Courses: offer free trials   
  * [DataCamp](https://www.datacamp.com/users/sign_up)
  * [CodeAcademy](https://www.codecademy.com/) 
* Data Science Books:
  * [Python for Data Analysis](https://wesmckinney.com/book/) by Wes McKinney
  * [Spatial Statistics for Data Science... with R](https://www.paulamoraga.com/book-spatial/index.html) by Paula Moraga
  * [Earth Engine and Geemap... with Python](https://book.geemap.org/) by Qiusheng Wu
* Geospatial Tutorials:
  * [Geography Realm](https://www.geographyrealm.com/gis/)
  * [Spatial Thoughts](https://courses.spatialthoughts.com) (pyQGIS, GDAL, GEE-JavaScript tutorials)

## Geographic Information System (GIS) standards: 
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

## GIS metadata standards:  
- ISO 19115: Geographic information — Metadata  
- ISO 19139: Geographic information — Metadata — XML schema  
- [FGDC Content Standard for Digital Geospatial Metadata (CSDGM)](https://www.fgdc.gov/metadata/csdgm-standard) to Create System-level Metadata Records  

### [Metadata creation best practices](https://www.usgs.gov/data-management/metadata-creation):   
* Gather all information together & reuse information that is already developed, e.g. abstract, purpose, date from grant or funding proposals
* Choose a descriptive title for your data that incorporates who, what, where, when, and scale.
* Choose keywords wisely -- consider all possible interpretations of your word choices.
* Include as many details as you can in the metadata record for future users of the data.
* Update the metadata date (date stamp) so that metadata repositories will know which version of the record is most recent.
* DOI should go in the primary <onlink> in the Citation Information section and should be a URL. 

### Metadata validation:  
* Compares the metadata standard to the XML metadata record to ensure it conforms to the structure of the standard, such that all of the required elements are filled in.
* USGS best practices for [Checking Metadata with Data](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/CheckingMetadataWithData_508-compliant.pdf) with FGDC-CSDGM metadata
* [USGS Metadata Parser (MP)](https://geology.usgs.gov/tools/metadata/tools/doc/mp.html) 
* [Metadata Wizard tool](https://code.usgs.gov/usgs/fort-pymdwizard)
