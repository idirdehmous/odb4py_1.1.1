#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#include "magicwords.h"
#include "info.h"
#include "odb.h"
#include "privpub.h"
#include "macros.h"
#include <numpy/ndarrayobject.h>


/* ENDIANESS */
extern int ec_is_little_endian();

static PyObject *odbDca_method( PyObject* Py_UNUSED(self) , PyObject *args , PyObject*  kwargs ) {
    char *path     = NULL ; 
    char *database = NULL ;
    int   ncpu   = 1  ; 
    char *dbname =NULL; 
    

    static char *kwlist[] = {"dbpath","db","ncpu"};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|si", kwlist, &path , &database , &ncpu))
    {
        return NULL;
    }

   database=path  ; 
 
   char * dcagen_path ; 
   char * febin  ;  
   char * sysbin ;
   
   febin      =getenv( "ODB_FEBINPATH" )  ;
   sysbin     =getenv( "ODB_SYSPATH"  )   ;
   dcagen_path=getenv( "ODB_BEBINPATH" )  ; 


    putenv(febin);
    putenv(sysbin);
    putenv(dcagen_path);
    
    char scpu[5];
    strtonum(scpu, ncpu);
    char* dbname_arg ;
    char* ncpu_arg ; 
    if (dbname) { dbname_arg = concat ( " -l " , dbname   ); 
        } 
    else 
    { dbname_arg=" " ;  
        };
    if (ncpu  ) { ncpu_arg   = concat ( " -N " , scpu     ); 
    } else 
    { 
     ncpu_arg  =" " ;  
    };

    //if (verb  ) {  quiet = " -q "; } else { quiet =" " ;  };

    char* dca_path   = concat(dcagen_path , "/dcagen ");
    char* cma_path   = concat( " -i "     ,  database );
    char* dca_cmd    = concat(dca_path    ,  cma_path) ;
    char* dca_args   = concat(dca_cmd     ,  " -F -n -q -z -b -P " ) ; 

    char* dca_db     = concat( dca_args   ,  dbname_arg );
    char* dca_cpu    = concat( dca_db     ,  ncpu_arg  );
    //char* dca_verb   = concat(dca_cpu     ,  quiet    ) ; 

    printf ( "%s\n"  , "--odb4py : Creating DCA files ..." ) ; 

    int status= system ( dca_cpu ) ; 
    if ( status != 0){  
    printf("%s  %s\n" , dca_cpu, " Failed to run dcagen : command returned a non Zero value,  !" ) ;
    return PyLong_FromLong( 1 ) ;
    }else {
    free(dca_path) ; 
    free(cma_path) ;
    free(dca_cmd ) ;
    free(dca_args) ;      // deallocate the string  cmd 
    }
    return PyLong_FromLong( 0 ) ; 
}
