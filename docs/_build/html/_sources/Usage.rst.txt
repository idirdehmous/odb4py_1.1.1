Usage
=====

Open a CCMA database 
--------------------
The package pyodb is written in C/Python API. It contains a set of functions as an interface between python and ODB softeware. 
In order to open a given database one has to import the module "odbConnect". Ths function returns positive value for a seccessful connection and negative if it fails.
We consider the use of an CCMA odb (The same code  works also for ECMA )

.. code-block:: python

   from pyodb import odbConnect   
   iret  = odbConnect(odbdir ="imaginary/../CCMA" , verbose=True, npools="1"  )
   if iret < 0 :
      print("Failed  to open the odb"  , dbpath )
      sys.exit(0)



Create the DCA (Direct Column Access) files 
---------------------------------------------
In order to fetch the columns and rows, the ODB creates the DCA files first. The latters contain the necessary information about how the data are written on the disk.Such as the starting memory segment and the end of a given table, the size and the offsets between the different tables and columns.

The DCA files can be created on the fly when any of the binaries such as odbdump.x or odbsql.x are used. In the pyodb package a C/python function is used to create the files before the data fetch. Creating DCA files is relatiely simple,the user has to provide the path the ODB and the number of CPUs used by the task. The ODB path is requiered however, the number of CPUs and the verbosity arguments are optional. The DCA files are written uin a subdirectory  called "dca" inside the ECMA or CCMA directory. 

.. code-block:: python 

   # Check if the dca directory is already there
   dbpath = "/path to/imaginary"
   db_type= "CCMA" 
   dcapath= "/".join( ( dbpath,db_type, dcapth) )

   if not os.path.isdir (dcapath ):
      os.environ["IOASSIGN "] =dbpath+"/"+db_type+"/IOASSIGN"
      ic=odbDca ( dbpath=dbpath, db= db_type, ncpu=8  )  # in this example 8 CPUs are used 
      # 'ic' is used to check whether the process has successfully ended 
      # Returns a negative value if it fails and postive in success. 



Send and fetch data from a given ECMA or CCMA 
----------------------------------------------
The function used to send an SQL query ,  get the data and store them in a python variable is 'odbDict'. Note that the ODB rows are returned as a python disctionary with the name of the quered columns as keys and the lists of data as values. 
The function needs some requiered and optional arguments and are described hereafter:


.. code-block:: python

   # Pure python modules 
   from pyodb_extra.environment  import  OdbEnv
   from pyodb_extra.odb_ob       import  OdbObject
   from pyodb_extra.parser       import  StringParser


   # C/Python module 
   from pyodb import odbDict  
  
   

   # The SQL query  
   # Let's get the AMDAR/AIREP data only 
   sql_query="select   statid ,obstype, varno, degrees(lat) ,  degrees(lon) ,  obsvalue   FROM  hdr, body WHERE obstype == 2"

   # Check the query 
   p =StringParser()

   # The number of functions in the SQL statement
   # In C ODB code :  N pure columns =   ( N All columns - N functions in sql query)
   nfunc  =p.ParseTokens ( sql_query )    

   # Check and clean before sending ! (It remove eventual non-printable characters )
   sql    =p.CleanString ( sql_query  )  

   nfunctions = nfunc    # (type -> integer ) Number of columns considring the functions in the sql statement  (degrees, rad, avg etc ...)
   query_file = None     # (type -> str     ) The sql file if used rather than sql request string 
   poolmask   = None     # (type -> str     ) The ODB pools to consider (  must be a string  "1" , "2", "33" ...  , etc   )
   progress   = True     # (type -> bool    ) Progress bar (very useful in the case of huge ODBs )
   float_fmt  = 5        # (type -> int     ) Number of digits for floats (same syntax )
   verbose    = False    # (type -> bool    ) Verbosity  on/off   


   # SQL query  
   sql_query="select   statid ,degrees(lat) ,  degrees(lon) , varno, obsvalue   FROM  hdr, body"

   # Path to ODB  
   odbpath= "/".join( (dbpath, db_type) )

   # Send and get the data 
   data =odbDict  (dbpath ,sql, nfunctions ,float_fmt, query_file , poolmask , progress, verbose  )
 
   print( data ) 
   

   [##################################################] Complete 100%  (Total: 51 rows)

   {statid@hdr ['2YIQTRJA', '2YIQTRJA', '2YIQTRJA', 'ZUMAVXZA', 'ZUMAVXZA',....,    ]
   obstype@hdr [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, ....., ]
   varno@body [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4,.....,  ]
   degrees(lat) [62.38, 62.38, 62.38, 61.97, 61.97, 61.97, 61.26,..., ]
   degrees(lon) [1.1, 1.1, 1.1, 0.5, 0.5, 0.5, -0.35, -0.35, -0.35, -1.54,.... ,  ]
   obsvalue@body [216.0, 28.72984, -2.00899, 214.2, 32.57982, -4.5788,...,]  }

