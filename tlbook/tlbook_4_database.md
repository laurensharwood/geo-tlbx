# Databases

---

<b>Database Design Considerations</b>:    
- *Specific use-case: what is the need?*  
- *What data needs to be stored, & what type of data is it?*
- *How often is will the data be updated and viewed; and by who?*
- *What is the level of security the data requires?* 
- *How should the data be logically organized and what integrity constraints should be applied?* 

<b>NoSQL (nonrelational DBMS)</b>:   
Document-centered rather than table-centred. large data, structure varies. Stored in a [data lake](https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-a-data-lake#data-lake-definition). 
- Unstructured data: No schema. Most data. Ex) Photos, chat lots, MP3     
- Semi-structured data: Self-describing structure but no larger schema. Ex) XML, JSON  


## RDBMS (Relational Database Management System)  

<b>SQL (RDBMS)</b>:   
- Structured data: Follows a schema with defined constraints and relationships. 
- Stored in a [data warehouse](https://www.oracle.com/database/what-is-a-data-warehouse/) (ex: [BigQuery](https://cloud.google.com/bigquery/docs/geospatial-data)) or data lake (ex: database or Excel table). 


<b>Cardinality</b>:   
Relationship between tables, rows, and elements in a database. Ratio denotes the number of entities that another entity can be linked to.   
- 1:1 - one row in a table relates to exactly one row in a second table
- 1:N - one row in a database table relates to many rows in a second table (Primary Key-Foreign Key relationship) 
- N:N - many rows in one table are related to many rows in a second table  

<b>Entity-Relationship Diagrams (ERDs)</b>:  
Charts that depict the entities (objects), relationships, and attributes in a database. 
- https://draw.io 
- https://databasediagram.com/

<u>Crows Notation Chart</u>:
![notation](img/Crow-Notation-Chart.png)

<b>Normalization</b>:   
The process of breaking down tables in a database & managing the relationships between them in order to minimize redundancy.    

*Consider: Should my data have minimal redundancies and dependencies?*    

Non-normalized table issues:   
- missing ID record for other attribute-> deletion anomoly
- update anomoly -> logical inconsistency   
- can't populate new ID because missing one attribute -> insertion anomoly    

Normal form levels assess danger of redundancy:    
- First Normal Form (<b>1NF</b>)  - simplest - need primary key (single column or combination of columns) has one value per key.   
- Second Normal Form (<b>2NF</b>)  - each non-key attribute is dependent on the entire primary key (all primary key columns).   
- Third Normal Form (<b>3NF</b>)  - <b>each non-key attribute should depend on the key, the whole key, and nothing but the key</b>

### Indexes
Similar to indexes in books, indexes in tables improve lookup performance, but by building a tree diagram using a unique ID key 

<b>Clustered Index</b>: index sorted by a column 

*Note different syntax to create clustered index in postgreSQL vs mySQL*: 

* postgreSQL:
> CLUSTER table_name USING column_name;

* mySQL:
> CREATE CLUSTERED INDEX index_name ON db.table_name (column_name);

### Integrity

Refers to the total accuracy, consistency, and completeness of data.

<b>Physical</b>: integrity of the body. Ex) issues from degraded storage, blackouts, hacker attacks   
<b>Logical</b>: integrity across uses. Ex) creating unique primary keys without null values  

<b>ACID</b>: ensures reliability of database transactions   
<b>A</b>tomicity: Transactions are all-or-nothing   
<b>C</b>onsistency: Updates must adhere to data constraints    
<b>I</b>solation: Multiple users can access data at the same time        
<b>D</b>urability: Committed transactions are saved to permanent storage     

### Constraints 

```CONSTRAINT``` types:   
1. Domain: acceptable data type per column 
2. Entity integrity: unique primary key values within a table cannot be null
3. Referential (foreign key) integrity: uses foreign key to ensure consistent relationship between tables

* ```CHECK``` - ensures that values in a column or a group of columns meet a specific condition
> CREATE TABLE table_name (item VARCHAR PRIMARY KEY price NUMERIC CHECK (price > 0));

add a constraint to an existing table:     
> ALTER TABLE table_name ADD CONSTRAINT price CHECK (price > 0);

* ```NOT NULL``` - ensures no NULL values for a given column
* ```UNIQUE``` - ensures all values for a given column are different
* ```PRIMARY KEY``` - combines ```NOT NULL``` and ```UNIQUE```
> CREATE TABLE (activity_names CHAR(14) PRIMARY KEY, activity_gpx CHAR(18) UNIQUE NOT NULL, activity_tcx CHAR(18) UNIQUE NOT NULL);

* ```FOREIGN KEY``` - prevents actions that would destroy links/relationships between tables
> CREATE TABLE gpx_pts (filename CHAR(18) REFERENCES activity_names (activity_gpx), date TIMESTAMP PRIMARY KEY, lat FLOAT NOT NULL, lon FLOAT NOT NULL, ele FLOAT NOT NULL, speed FLOAT NOT NULL);   

* ```CASCADE``` removes link between primary key (activity_names) & foreign key table:   
> DROP TABLE activity_names CASCADE;


* ```CREATE TRIGGER```  - executed when a database event occurs and are used to track edit history & data changes ( [Postgres](https://www.postgresql.org/docs/current/plpgsql-trigger.html) / [PostGIS](https://postgis.net/workshops/postgis-intro/history_tracking.html)  ) 

### Common clauses 
- ```SELECT``` is the clause we use every time we want to query information from a database.
- ```AS``` renames a column or table.
- ```DISTINCT``` return unique values.
- ```WHERE``` is a popular command that lets you filter the results of the query based on conditions that you specify.
- ```LIKE``` and ```BETWEEN``` are special operators.
- ```AND``` and ```OR``` combines multiple conditions.
- ```ORDER BY``` sorts the result.
- ```LIMIT``` specifies the maximum number of rows that the query will return.
- ```COUNT()```: count the number of rows
- 
> SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'table_name';

- ```SUM()```: the sum of the values in a column

> SELECT SUM(minutes) FROM runs_tcx;

- ```MAX()/MIN()```: the largest/smallest value
- ```AVG()```: the average of the values in a column
- ```ROUND()```: round the values in the column  

### Aggregate 
Aggregate functions combine multiple rows together to form a single value of more meaningful information.
- ```GROUP BY``` is a clause used with aggregate functions to combine data from one or more columns.
- ```HAVING``` limit the results of a query based on an aggregate property.

### Data manipulation

```CASE``` = similar to Python's ```if```, ```else``` statement   
```WHEN```, ```THEN```, ```ELSE```, ```END AS``` sets the new column name   

> SELECT season,  
> AVG(CASE WHEN hometeam_id = 8445 AND home_goal > away_goal THEN 1 WHEN hometeam_id = 8445 AND home_goal < away_goal THEN 0 END) AS pct_home_wins,   
> AVG(CASE WHEN awayteam_id = 8445 AND away_goal > home_goal THEN 1 WHEN awayteam_id = 8445 AND away_goal < home_goal THEN 0 END) AS pct_away_wins,   
> FROM match GROUP BY season;  

### Joins
<b>inner join</b>  
links two tables, 'table_name' and lookuptable, 'lut', using a common 'key' column, and returns rows where 'key' value exists in both tables  
> SELECT * FROM table_name INNER JOIN lut USING (key);   

<b>left join</b>  
returns all rows from first table, table_name, with blank value where 'key' is missing in the second table, lut (will not have rows from second table where key is missing in first table)  
> SELECT * FROM table_name LEFT JOIN lut ON table_name.key = lut.key;   

<b>right join</b>  
returns all rows from lut with blank value where 'key' is missing in table_name (will not have rows from first table whose key)   
> SELECT * FROM table_name RIGHT JOIN lut ON table_name.key = lut.key;    

<b>full outer join</b>  
returns all rows from both tables   
> SELECT * FROM table_name FULL OUTER JOIN lut ON table_name.key = lut.key;    


### Views  
A query stored in a data dictionary, which acts as a proxy (virtual table) and does not hold the actual data.  

```CREATE VIEW``` /  ```DROP VIEW```  used to (1) break down more complex operations, and (2) restrict users accessing certain sensitive data.  

*Consider: What queries will be performed most often?*   

View Types:   
- Simple - based on a single table w/ no ```GROUP BY``` clause and functions     
- Complex - based on multiple tables which contain ```GROUP BY``` clause and functions   
- Inline - based on a subquery in ```FROM``` clause, that subquery creates a temp table  
- Materialized - creates replicas of the data to store the definition and data physically   


### Access control   
*Consider: which users should have access to which levels of access(read/update/insert) & tables?*   

[PostGIS](https://postgis.net/workshops/postgis-intro/security.html)

> GRANT INSERT,UPDATE,DELETE ON table_name TO user_name;


SQL Server:  
```GRANT``` { permissions} ON SCHEMA :: {schema} TO {user};    
```DENY``` { permissions} ON SCHEMA :: {schema} TO {user};    

---

## Transformation

Refers to transfering / converting geospatial data between Python objects, ESRI [geodatabases](https://pro.arcgis.com/en/pro-app/latest/help/data/geodatabases/overview/the-architecture-of-a-geodatabase.htm#GUID-739D940C-FD50-4F6F-8600-EBE39B00189A
), and other RDBMS such as [SQL Server](https://pro.arcgis.com/en/pro-app/latest/help/data/geodatabases/manage-sql-server/overview-geodatabases-sqlserver.htm). 


#### Store postgreSQL credentials in Linux:    
> export POSTUSR={your-postgres-username}   
> export POSTPWD={your-postgres-pwd}  

If postgreSQL was added to PATH, open a terminal.   
If not, go to the directory where PostgreSQL is installed then into the bin directory, then open a terminal.  

#### Trust (ignore password step) in Windows:   
Right-click "C:\Program Files\PostgreSQL\16\data\pg_hba.conf" > edit with Notepad: change method from sca-... to trust     

#### Connect to database from psql
i) Find in install location once then pin to start/taskbar: Entering nothing submits text within brackets--   
Server [localhost], Database [db_name], Port[5432], Username[postgres], Password for user

#### From psql, create postgreSQL database
Execute the following command to create a database, then enter user password when prompted:   
> createdb -h localhost -p 5432 -U postgres {db_name}  

print databases in postgres db server:  
> \l   

print tables in connected db:  
> \d 

---

### postgres  -> python (geopandas):  
<b>cursors</b> are database objects that work with tables (for instance: reading or writing) one row at a time
~~~
import pandas as pd
import psycopg2

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
~~~



### OSGEO ```ogr2ogr```:

From OSGeo4W Shell:  

.shp -> .gpkg: 
> ogr2ogr -f "ESRI Shapefile" "input.shp" "output.gpkg" "layer"

PostgreSQL database -> .gpkg:
> ogr2ogr -f PostgreSQL "PG:user=your_username password=your_pwd dbname=your_dbname" out_filename.gpkg

.gpx -> .gpkg:
> for /R %f in (*.gpx) do ogr2ogr -f "GPKG" out_filename.gpkg "%f"


---
