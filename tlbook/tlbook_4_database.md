# Databases




## DBMS (Database Management System)  

<b>NoSQL (nonrelational DBMS)</b>:   
Document-centered rather than table-centred. large data, structure varies.  
- Unstructured data: No schema. Most data. Ex) Photos, chat lots, MP3     
- Semi-structured data: Self-describing sctructure but no larger schema. Ex) NoSQL, XML, JSON  

<b>SQL (relational DBMS)</b>:   
Structured, consistent data that follows a schema with defined data types and relationships / constraints.  




<b>Database Design Considerations</b>:    
- *Specific use-case*  
- *What is being analyzed?*   
- *How often is will data be updated and change?*   

<b>Entity-Relationship Diagrams (ERDs) </b>:  
Charts that depict the entities (objects), relationships, and attributes in a database. 
- https://draw.io 
- https://databasediagram.com/

### Schemas  
*Consider: How should the data be logically organized and what integrity constraints should be applied?* 

Cardinality: Relationship between tables, rows, and elements in a database. Ratio denotes the number of entities that another entity can be linked to   
- 1:1
- 1 to many  
- many to 1
- many to many


### Normalization   
Normalization refers to the process of breaking down tables in a database & managing the relationships between them in order to minimize redundancy.    

*Consider: Should my data have minimal redundancies and dependencies?*    

Non-normalized table issues:   
- missing ID record for other attribute-> deletion anomoly
- update anomoly -> logical inconsistency   
- can't populate new ID because missing one attribute -> insertion anomoly    

Normal form levels assess danger of redundancy:    
- First Normal Form (<b>1NF</b>)  - simplest - need primary key (single column or combination of columns) has one value per key.   
- Second Normal Form (<b>2NF</b>)  - each non-key attribute is dependent on the entire primary key (all primary key columns).   
- Third Normal Form (<b>3NF</b>)  - <b>each non-key attribute should depend on the key, the whole key, and nothing but the key</b>


### Data Integrity
Data integrity refers to the total accuracy, consistency, and completeness of data.

<b>Physical</b>: integrity of the body. Issues may arise from degraded storage, blackouts, hacker attacks 

<b>Logical</b>: integrity across uses.   
Ex) creating unique primary keys without null values. 


#### Ensuring Data Integrity 
<b>Constraints</b>: ```CONSTRAINT```
- Domain: acceptable data type per column 
- Entity: unique primary key values within a table 
- Referential / foreign key: consistent relationship between tables
     Ex) Only add foreign key if that is a unique ID and it also exists in another database 

```CHECK``` constraint ensures that values in a column or a group of columns meet a specific condition

~~~
ALTER TABLE table_name
ADD CONSTRAINT constraint_name CHECK (condition);
~~~


<b>Triggers</b> are executed when a database event occurs, such as inserting or editing an entry. 

```CREATE TRIGGER``` to tracking edit history / data changes in [Postgres](https://www.postgresql.org/docs/current/plpgsql-trigger.html) / [PostGIS](https://postgis.net/workshops/postgis-intro/history_tracking.html)   



### Indexes
Similar to indexes in books, indexes in tables improve lookup performance, but by building a tree diagram using a unique ID key 
Clusterd Index: index sorted by a column 

mySQL:
~~~
CREATE CLUSTERED INDEX index_name ON db.table_name (column_name);
~~~
postgreSQL:

~~~
CLUSTER table_name USING column_name;
~~~

*Note different syntax to create clustered index in mySQL vs postgreSQL*   


### Views  
A view refers to a query stored in a data dictionary. Acts as a proxy, or virtual table, so does not hold the actual data.  
Used to (1) break down more complex operations, and (2) restrict users accessing certain sensitive data.  
*Consider: What queries will be performed most often?*   
```CREATE VIEW```  ```DROP VIEW```  
View Types:   
- Simple - based on a single table w/ no ```GROUP BY``` clause and functions     
- Complex - based on multiple tables which contain ```GROUP BY``` clause and functions   
- Inline - based on a subquery in ```FROM``` clause, that subquery creates a temp table  
- Materialized - creates replicas of the data to store the definition and data physically   


### Access control   
*Consider: which users should have access to which levels of access(read/update/insert) & tables?*   
[PostGIS](https://postgis.net/workshops/postgis-intro/security.html)

```  
GRANT INSERT,UPDATE,DELETE ON table_name TO user_name;
```  


SQL Server:  
```GRANT``` { permissions} ON SCHEMA :: {schema} TO {user};    
```DENY``` { permissions} ON SCHEMA :: {schema} TO {user};    

---

## postgreSQL (psql)   


#### Store postgreSQL credentials in Linux:    
```
export POSTUSR={your-postgres-username}
export POSTPWD={your-postgres-pwd}
```  
If postgreSQL was added to PATH, open a terminal.   
If not, go to the directory where PostgreSQL is installed then into the bin directory, then open a terminal.  

#### Trust (ignore password step) in Windows:   
Right-click "C:\Program Files\PostgreSQL\16\data\pg_hba.conf" > edit with Notepad: change method from sca-... to trust     

#### Connect to database from psql
i) Find in install location once then pin to start/taskbar: Entering nothing submits text within brackets--   
Server [localhost], Database [db_name], Port[5432], Username[postgres], Password for user

#### From psql, create postgreSQL database
Execute the following command to create a database: ```createdb -h localhost -p 5432 -U postgres {db_name}``` then enter user password when prompted

<b>create table</b>  
launch psql db(db_name=#):  
> CREATE TABLE gpx_runs (gpx_file TEXT NOT NULL, time VARCHAR(100) NOT NULL, lat FLOAT NOT NULL, lon FLOAT NOT NULL, ele FLOAT NOT NULL, speed FLOAT NOT NULL);

print databases in postgres db server:  
> \l   

print tables in connected db:  
> \d 


---


## Transformation
Refers to transfering / converting geospatial data between Python objects, ESRI [geodatabases](https://pro.arcgis.com/en/pro-app/latest/help/data/geodatabases/overview/the-architecture-of-a-geodatabase.htm#GUID-739D940C-FD50-4F6F-8600-EBE39B00189A
), and other RDBMS such as [SQL Server](https://pro.arcgis.com/en/pro-app/latest/help/data/geodatabases/manage-sql-server/overview-geodatabases-sqlserver.htm). 



### postgres  -> python (geopandas):  
<b>cursors</b> are database objects that work with tables (for instance: reading or writing) one row at a time
~~~
import pandas as pd
import psycopg2

conn = psycopg2.connect(database = {your_db_name}, user = os.getenv("POSTUSR"), password = os.getenv("POSTPWD"), host = localhost, port = 5432)
cur = conn.cursor()
cur.execute({SQL_query})
items = cur.fetchall()
hits=[]
for i in items:
    hits.append(i)
df=pd.DataFrame(hits)
~~~



### OSGEO ```ogr2ogr```:

From OSGeo4W Shell:  

.shp -> .gpkg: 
> ogr2ogr -f "ESRI Shapefile" "input.shp" "output.gpkg" "layer"

PostgreSQL database -> .gpkg:
> ogr2ogr -f PostgreSQL "PG:user={your_username} password={your_pwd} dbname=your_dbname" {out_filename}.gpkg

.gpx -> .gpkg:
> for /R %f in (*.gpx) do ogr2ogr -f "GPKG" {out_filename}.gpkg "%f"


---

## postgreSQL actions  
<b>create table</b> (in a connected database)
~~~
CREATE TABLE table_name (id_key INTEGER PRIMARY KEY, fullname varchar(100) NOT NULL);
~~~
    
<b>delete database</b> (must be disconnected from the database you're trying to delete)  
~~~
DROP DATABASE {db};
~~~

<b>delete column</b>  
~~~
ALTER TABLE table_name DROP COLUMN column_name;
~~~

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
~~~
SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'table_name'
~~~
- ```SUM()```: the sum of the values in a column
~~~
SELECT SUM(minutes) FROM runs_tcx  
~~~
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

~~~
SELECT 
	CASE WHEN hometeam_id = 10189 THEN 'FC Schalke 04'
        WHEN hometeam_id = 9823 THEN 'FC Bayern Munich'
        ELSE 'Other' 
        END AS home_team,
	COUNT(id) AS total_matches
FROM matches_germany
GROUP BY home_team;
~~~

~~~
SELECT 
	season,
	COUNT(CASE WHEN hometeam_id = 8650 AND home_goal > away_goal THEN 'home win count' END) AS home_wins, 
	COUNT(CASE WHEN awayteam_id = 8650 AND away_goal > home_goal THEN 'away win count' END) AS away_wins, 
FROM match
GROUP BY season;
~~~

~~~
SELECT 
	season,
	AVG(CASE 
	    WHEN hometeam_id = 8445 AND home_goal > away_goal THEN 1 
	    WHEN hometeam_id = 8445 AND home_goal < away_goal THEN 0 
	    END) AS pct_home_wins, 
	AVG(CASE WHEN awayteam_id = 8445 AND away_goal > home_goal THEN 1
	 WHEN awayteam_id = 8445 AND away_goal < home_goal THEN 0 
	 END) AS pct_away_wins, 
FROM match
GROUP BY season;
~~~


### Joins
<b>inner join</b>  
links two tables, 'table_name' and lookuptable, 'lut', using a common 'key' column, and returns rows where 'key' value exists in both tables  
~~~
SELECT * FROM table_name INNER JOIN lut USING (key)
~~~

<b>left join</b>  
returns all rows from first table, table_name, with blank value where 'key' is missing in the second table, lut (will not have rows from second table where key is missing in first table)  
~~~
SELECT * FROM table_name LEFT JOIN lut ON table_name.key = lut.key
~~~

<b>right join</b>  
returns all rows from lut with blank value where 'key' is missing in table_name (will not have rows from first table whose key)   
~~~
SELECT * FROM table_name RIGHT JOIN lut ON table_name.key = lut.key 
~~~

<b>full outer join</b>  
returns all rows from both tables   
~~~
SELECT * FROM table_name FULL OUTER JOIN lut ON table_name.key = lut.key 
~~~
