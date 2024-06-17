# geo-tlbx
toolbox of code for geospatial processing and analysis 

## Mac virtual environment (venv) instructions 
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
### Micromamba
https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html 


 
### Bash commands     
- pressing tab following a command (cd , ls , vim ) will list all possible files/folders and autofill if there's only one match  
- run 'pwd' to print working directory and easily copy+paste path later   

b>list</b>(l) files, sorted by time(t), reverse(r) - last modified file at bottom of list:  
> ls -ltr  

<b>list</b> two last modified files:  
| : 'pipe' that takes first command's output and feeds it into the command after the pipe
> ls -ltr | tail -2

<b>size</b> of current directory:  
> du -sh .
sort files based on size:  
> du -sh -- * | sort -rh  

<b>count</b> number of files in current directory:  
> ls | wc -l  

<b>count</b> number of files in directory that match * search_string *:   
- adding * to the end of a 'find' command's search string will match any text following the *  
- adding * to the beginning of the search string will match any text preceeding *  
> cd /home/downspout-cel/paraguay_lc/stac/grids  
> ls -dq 004* | wc -l ## count of files that start with 0004      

<b>delete</b> files (f) in current firectory (.) that match string (start with S1_):  
> find . -type f -name 'S1_*' -delete  
> find . -name gee -exec ls {} \;
> find . -name gee -exec rm -rf {} \;  

<b>delete</b> files (f) <i>recursively</i> in current directiry (.) that match string (start with L3A_LC):   
> find . -name "L3A_LC*" -type f -exec rm -r {} +

<b>delete</b> folders (d) in current firectory (.) that match string (contain the string landsat):  
> find . -name "*landsat*" -type d -exec rm -r {} +

<b>move </b> files that contain filterstring (in current dir) and move them into output_folder_path:  
> mv * filterstring * output_folder_path  

<b>zip</b> files: (to bulk download files)    
- adding -r after the zip command will search for files recursively, within subfolders  
- command -r out_file.zip input_folder* (from current directory) (*=everything)
> zip -r WSA_Plot_Locations.zip WSA_Plot_Locations*

## SLURM 
check on bash script progress:  
> cd ~/code/bash   
> sbatch {script_name}.sh

check on bash script progress:  
> squeue  
> cd ~/code/bash   
> ls -ltr  
> cat *bottom_file.err*  
