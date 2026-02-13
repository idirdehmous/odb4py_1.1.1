Usage
-----

**odb4py** provides two main modules:

- **odb4py.core**: A set of low-level functions interfacing Python with the ODB runtime libraries.
- **odb4py.utils**: A collection of pure Python helper utilities, for example to initialize the ODB environment, parse or sanitize SQL queries, and manage auxiliary tasks.

.. note::
   In the following examples, a CCMA ODB database is used.
   However, the same code is fully applicable to an ECMA database.


Opening a CCMA Database
------------------------

To open a given database, the ``odbConnect`` function from the
``odb4py.core`` module must be used. This function returns a
positive value if the connection is successful and a negative
value if it fails.

.. code-block:: python

   #-*- coding: utf-8 -*-
   import sys
   from  odb4py.utils import OdbEnv  

   # Initialize the odb4py  environnment
   env= odbEnv()
   env.InitEnv()

   # Import method
   from odb4py.core import odbConnect

   # Path to a CCMA database
   dbpath = "/imaginary/path/CCMA"

   # Attempt to open the database
   iret = odbConnect(odbdir=dbpath, verbose=True)

   # Check the return code
   if iret < 0:
       print("Failed to open the ODB database:", dbpath)
       sys.exit(1)


Creating DCA (Direct Column Access) Files
------------------------------------------

In order to retrieve rows and columns efficiently, ODB relies on
**DCA (Direct Column Access)** files.

DCA files contain structural metadata describing how the data are
physically stored on disk. This includes, for example:

- The starting and ending memory segments of a given table,
- Column sizes,
- Offsets between tables and columns,
- Internal layout information required for fast access.

DCA files can be created automatically when using ODB binaries such as
``odbdump.x`` or ``odbsql.x``. They can also be generated explicitly
using the ``dcagen`` script or the ``dcagen.x`` binary.

Within the **odb4py** package, a dedicated function ``odbDca`` is provided to
generate DCA files programmatically before executing a query.

Creating DCA files is straightforward: the user must provide the path
to the ODB database. The number of CPUs used for the task and the
verbosity level are optional parameters.

The generated DCA files are written into a subdirectory named
``dca`` inside the corresponding ``ECMA`` or ``CCMA`` directory.



.. code-block:: python 

   #-*- coding: utf-8 -*-
   import os , sys  
   from   odb4py.utils import OdbEnv 

   # Initialize the odb4py  environnment 
   env= odbEnv()
   env.InitEnv()
   
   # Import the  method
   from   odb4py.core  import odbDca 

   db_type  = "CCMA"
   db_path  = "/imaginary/path/"
   dca_path = "/".join( ( dbpath, db_type, "dca" ) )
   NCPU     = 8

   # Check if the files existe 
   if not os.path.isdir (dca_path ):
      ic =odbDca ( db_path=db_path, db= db_type, ncpu= NCPU  ) 
      if ic < 0 :
         print("Failed to create DCA files")
         sys.exist()



SQL Query and Data Retrieval
----------------------------

The SQL syntax used in **odb4py** follows the same conventions as the
official ODB API.

To perform a data query, the function ``odbDict`` must be used.
This function returns both the column names and the corresponding row
values organized as a Python dictionary. In this dictionary, column
names are used as keys, and the associated values correspond to the
retrieved ODB data rows.

Before executing the query, a preliminary syntax validation step is
performed using the ``ParseTokens`` and ``CleanString`` functions inside the ``utils`` module. This includes:
- Removing non-printable characters.
- Filtering out tokens that are not part of the ODB SQL lexicon.
- Parsing SQL tokens to determine the number of ODB SQL functions used within the request.

At the low level, the total number of columns to be retrieved is equal
to the number of explicit column names plus the number of computed
columns resulting from SQL functions.

Consequently, the C backend must internally distinguish between
pure column references and function-based expressions in order to
correctly process and return the requested data.

This preprocessing stage ensures robustness and prevents malformed SQL
queries from reaching the ODB runtime layer.

.. code-block:: python

   #-*- coding: utf-8 -*-   
   from odb4py.utils import  OdbEnv ,OdbObject , StringParser 

   # Initialize the ODB environnment first
   env =OdbEnv()    
   env.InitEnv()  

   # Import the C/Python module
   from odb4py import odbDict  
     

   # Path to ODB
   db_path= "/imaginary/path/../CCMA"

   # Let's get the AMDAR data (obstype =2 )
   # The SQL query  
   sql_query="SELECT statid , obstype, varno, degrees(lat) ,  degrees(lon) , obsvalue   FROM  hdr, body WHERE obstype==2"

   # Check & clean the query 
   p =StringParser()

   # The number of functions in the SQL statement
   nfunc  =p.ParseTokens ( sql_query )    

   # Check and clean before sending !
   sql    =p.CleanString ( sql_query  )  

   # Arguments 
   nfunctions = nfunc    # (type -> integer ) Number of columns considring the functions in the sql statement  (degrees, rad, avg etc ...)
   query_file = None     # (type -> str     ) The sql file if used rather than sql request string 
   poolmask   = None     # (type -> str     ) The ODB pools to consider (  must be a string  "1" , "2", "33" ...  , etc   )
   progress   = True     # (type -> bool    ) Progress bar (very useful in the case of huge ODBs )
   float_fmt  = 5        # (type -> int     ) Number of decimal digits for floats 
   verbose    = False    # (type -> bool    ) Verbosity  on/off   

   # Send the query and get the data 
   data =odbDict  (dbpath ,sql, nfunctions ,float_fmt, query_file , poolmask , progress, verbose  )
   print( data ) 

 
Output :

.. code-block:: python 

   ******** New ODB I/O opened with the following environment
   *******	ODB_WRITE_EMPTY_FILES=0
	  ODB_CONSIDER_TABLES=*
	   ODB_IO_KEEP_INCORE=1
	      ODB_IO_FILESIZE=32 MB
	       ODB_IO_BUFSIZE=4194304 bytes
	       ODB_IO_GRPSIZE=1 (or max no. of pools)
	       ODB_IO_PROFILE=0
	       ODB_IO_VERBOSE=0
	        ODB_IO_METHOD=5
   ODB_CONSIDER_TABLES=*
   ODB_WRITE_TABLES=*
   ***INFO: Poolmasking ignored altogether for database 'CCMA'
   --odb4py : Creating DCA files ...done !

   [##################################################] Complete 100%  (Total: 51 rows)
   {'statid@hdr':['2YIQTRJA', '2YIQTRJA', '2YIQTRJA'....., ],
    'obstype@hdr':[2, 2, 2,....] ,
    'varno@body':[2, 3, 4 ,....],
    'degrees(lat)':[62.38014, 62.38078, 62.38124,...],
    'degrees(lon)':[1.15, 1.12, 1.14,....] , 
    'obsvalue@body':[216.00144, 28.72984, -2.00899 ]}
   



Data Structure and Integration with the Python Ecosystem
---------------------------------------------------------

The choice to return query results as a Python dictionary is deliberate and central to the design of **odb4py**.

Using a dictionary provides a flexible and interoperable data structure, where column names are used as keys and the associated values correspond to the retrieved ODB rows. This approach ensures seamless compatibility with modern Python scientific libraries such as **pandas** and **xarray**.

As a result, the retrieved data can be easily converted into a ``pandas.DataFrame`` for further analysis and processing.

This enables users to perform:
- Statistical analysis
- Data filtering and aggregation
- Visualization
- Export to common formats (ODB2, NetCDF...)
- Integration into operational or research workflows

Example:
Let's consider the code above by considering all the observation types.

.. code-block:: python
   
   #-*- coding: utf-8 -*-
   import pands as pd  
   ...

   # Execute SQL query
   data = odbDict(dbpath, sql_query)

   # Convert to pandas DataFrame
   df = pd.DataFrame(data)
   print(df)


Output :

.. code-block:: python
   
   [##################################################] Complete 100%  (Total: 57484 rows)
   idx     statid@hdr  obstype@hdr  varno@body  degrees(lat)  degrees(lon)  obsvalue@body
   0          26268            1           1     57.816700     29.950000       0.000000
   1          26268            1          39     57.816700     29.950000     276.200000
   2          26268            1          58     57.816700     29.950000       0.940000
   3          26268            1           7     57.816700     29.950000       0.004482
   4          26067            1           1     59.433300     29.500000       0.000000
   ...          ...          ...         ...           ...           ...            ...
   57479      sekrn           13          29     68.494515     21.575417       1.056382
   57480      sekrn           13         192     68.494515     21.575417    -327.669993
   57481      sekrn           13          29     68.494515     21.575417       1.034856
   57482      sekrn           13         192     68.494515     21.575417    -327.669993
   57483      sekrn           13          29     68.494515     21.575417       0.922010

   [57484 rows x 6 columns]



 
