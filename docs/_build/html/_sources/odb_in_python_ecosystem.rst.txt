Integration with python ecosystem
====================================================

The choice to return query results as a Python dictionary is deliberate and central to the design of **odb4py**.

Using a dictionary provides a flexible and interoperable data structure, where column names are used as keys and the associated values correspond to the retrieved ODB rows. This approach ensures seamless compatibility with modern Python scientific libraries such as **pandas** and **xarray**.

As a result, the retrieved data can be easily converted into a ``pandas.DataFrame`` for further analysis and processing.

This enables users to perform:
 - Statistical analysis
 - Data filtering and aggregation
 - Visualization
 - Export to common formats (ODB2, NetCDF...)

Example:
Let's consider the previous code by selecting all the observation types.

.. code-block:: python
   
   #-*- coding: utf-8 -*-
   import pandas as pd  
   ...

   # SQL request 
   sql_query ="SELECT statid , obstype, varno, degrees(lat) ,  degrees(lon) , obsvalue   FROM  hdr, body"

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

   Runtime duration : 0:00:01.96

Visualizing Retrieved Data
---------------------------

Once the data have been retrieved using **odb4py**, they can be easily
visualized using standard Python scientific libraries such as
**matplotlib** and **cartopy**.

Because query results are returned as a Python dictionary, converting
the data into a ``pandas.DataFrame`` makes plotting straightforward.

The folowing example illustrates the retrieve of 2 meter temperature over the `MetCoOp <https://www.met.no/en/projects/metcoop>`_ model domain.
by considering the example above, we add the part which plots the retrieved geopoints and values.

.. code-block:: python 

   #-*- coding: utf-8 -*-
   import cartopy.crs as ccrs
   import cartopy.feature as cfeature
   import matplotlib.pyplot as plt
   from   mpl_toolkits.axes_grid1 import make_axes_locatable
   from datetime import datetime 

   from  odb4py.utils  import  OdbEnv ,  OdbObject  ,  StringParser  
   env= OdbEnv ()
   env.InitEnv ()
   from odb4py.core  import  odbDict , odbConnect, odbClose , odbDca

   # Start 
   start = datetime.now()

   # Path  
   dbpath ="/imaginary/path/CCMA"
   
   # Connect 
   iret  = odbConnect ( odbdir = dbpath )
   if iret < 0 :
      print("Failed  to open the odb"  , dbpath )
      sys.exit(0)

   # DCA files generation 
   if not os.path.isdir ( "/".join( ( dbpath, "dca")))  :
      NCPU=4
      os.environ["IOASSIGN "]  =dbpath+"/IOASSIGN"
      ic=odbDca ( dbpath=dbpath, db= "CCMA" , ncpu=NCPU  )

   # Select SYNOP t2m  
   sql_query="select  statid ,\
              degrees(lat)   ,\
              degrees(lon)   ,\
              varno          ,\
              obsvalue       ,\
              FROM hdr,body WHERE \
              obstype==1 and varno ==39" 

   # Parse the query  
   p      =StringParser()
   nfunc  =p.ParseTokens ( sql_query )   
   sql    =p.CleanString ( sql_query  )  

   # Args 
   nfunctions= nfunc    
   query_file= None      
   poolmask  = None    
   progress  = True     
   float_fmt = 10      
   verbose   = False    

   # Execute the query 
   df = pd.DataFrame(  odbDict  (dbpath ,sql, nfunctions ,float_fmt,query_file , poolmask , progress    ) )

   lats=df["degrees(lat)" ]
   lons=df["degrees(lon)" ]
   obs =df["obsvalue@body"] 

   # Domain boundaries
   if len(lats) != 0:
      ulat=max(lats)+1.
      llat=min(lats)-1.
   if len(lons) != 0:
      ulon=max(lons)+1.
      llon=min(lons)-1.

   # Plot
   fig = plt.figure(figsize=(10, 15))
   ax  = fig.add_subplot(111,projection=ccrs.Mercator())
   ax.autoscale(True)
   ax.coastlines()
   ax.set_extent([llon, ulon  ,llat ,ulat], crs=ccrs.PlateCarree())
   ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='blue')
   ax.gridlines(draw_labels=True)
   sc=plt.scatter ( lons, lats ,c=obs , cmap=plt.cm.jet ,marker='o',s=20, zorder =111,transform=ccrs.PlateCarree() )
   plt.title( "T2m from synop stations. Nobs = 1238 \n Datetime :20240110 00h00 UTC" )
   divider = make_axes_locatable(ax)
   ax_cb = divider.new_horizontal(size="5%", pad=0.9, axes_class=plt.Axes)
   fig.add_axes(ax_cb)
   plt.colorbar(sc, cax=ax_cb)
   plt.show()

   end  = datetime.now()
   duration = end -  start  
   print("Runtime duration:" , duration  )

Runtime duration: 0:00:04.77


.. figure:: source/_static/figures/ccma_metcoop.png
   :width: 80%
   :align: center
   :alt: ODB observations plotted with Cartopy

   Example: visualization of ODB observations (MetCOop domain).

This workflow enables rapid visual diagnostics of observation coverage
and spatial distribution, which are essential in data assimilation and
forecast verification studies.





