import sys  
sys.path.insert( 0, "/hpcperm/cvah/rmib_dev/odb4py_1.1.2/build/lib.linux-x86_64-cpython-312" ) 

from  convert    import  odb2nc  

from  utils  import  OdbObject , SqlParser    
dbpath="/hpcperm/cvah/rmib_dev/odb4py_1.1.2/examples/odb_samples/rmi/ECMA.synop"

sql ="select degrees(lat) ,degrees(lon) , obsvalue from hdr , body"


db      = OdbObject ( dbpath )
db_attr = db.get_attrib()
db_type = db_attr["type"]
db_name = db_attr["name"]
tabs    = db_attr["tables"]

p  = SqlParser()
nf = p.get_nfunc ( sql  )

odb2nc( database = dbpath ,  sql_query=sql , nfunc=nf , ncfile ="output.nc"  )


quit()

conn =  odb_open (dbpath )     #OK
#dc   =  conn.odb_dca  ( dbpath , dbtype="ECMA" , ncpu=8 , verbose =False   )  #OK
data =  conn.odb_dict ( database=dbpath , sql_query= sql , nfunc=nf ,fmt_float=5,  pbar=True , verbose =True   )   #OK
arr  =  conn.odb_array( dbpath , sql  , fmt_float=5  , pbar =True , header=True  )    #OK 

print( arr) 


conn.odb_close()
