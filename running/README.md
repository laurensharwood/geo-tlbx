## Step 1) Download activity files (.gpx, .tcx)

### Option 1: Download activity files from Strava:  
i) Login > Dropdown of your profile picture > Settings > My Account   
- under Download or Delete Your Account: select Get Started  
- under 2. Download Request (Optional): select Request Your Archive  
ii) wait for email from Strava containing folder of all activity files     

### Option 2: get activity files files from Garmin:
i) save your Garmin login credentials by opening terminal/command prompt:  
```
export EMAIL={enter your garmin username/email}  
export PASSWORD={enter your garmin password}
```  
ii) set number of days from today to download in the following cell, then run the cell below to execute garmin_api.py




## Step 2) Parse new activity files, add to archive:     
* add parsed activity data into respective archive .csv's and a files to a single archive folder

### Optionally, set up postgreSQL archive database with a table for parsed .tcx & .gpx activity files, respectively  
#### Add postgreSQL install location to PATH(document that contains filepaths where computer looks to execute scripts from):  
i) open the terminal and enter the following command on a Mac: ```sudo nano /etc/paths``` (enter user password when prompted)       
ii) in the last row of the file enter your postgreSQL install location, ex: /Library/PostgreSQL/16/bin   
#### From Terminal/Command Prompt, create postgreSQL database
i) if postgreSQL was added to PATH in the previous step, open a terminal. otherwise, go to the directory where PostgreSQL is installed then into the bin directory, then open a terminal.  
ii) execute the following command to create a database: ```createdb -h localhost -p 5432 -U postgres run_db``` then enter user password when prompted   
#### Launch SQL Shell (psql)
i) Find in install location once then pin to start/taskbar: Entering nothing submits text within brackets-- Server [localhost], Database [run_db], Port[5432], Username[postgres], Password for user
```
\l ## lists databases
\d ## lists tables
```  
#### From psql, create table where each running activity is saved as a row
```
CREATE TABLE gpx_runs (gpx_file TEXT NOT NULL, time VARCHAR(100) NOT NULL, lat FLOAT NOT NULL, lon FLOAT NOT NULL, ele FLOAT NOT NULL, speed FLOAT NOT NULL);
CREATE TABLE tcx_runs (tcx_file TEXT NOT NULL, date VARCHAR(100) NOT NULL, minutes FLOAT NOT NULL, miles FLOAT NOT NULL, vert_m FLOAT NOT NULL, hr_max INTEGER, cad_avg FLOAT);
```
#### v. Populate table with new runs (from TCX activity files) using psycopg2 python package in get_activites.py
- save postgreSQL password  ```export POSTPWD={your postgreSQL password}```




## Step 3) Create plots & webmaps