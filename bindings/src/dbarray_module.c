#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>

#include "odbdump.h"
#include "rows.h"
#include "progress.h"



// odbArray  : numeric-only fetch  ---->  returns numpy.ndarray
// The variables type 'string' are ignored  
static PyObject * odbArray_method(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    import_array();

    char *database  = NULL;
    char *sql_query = NULL;
//    int   fcols     = 0;
    char *queryfile = NULL;
    char *poolmask  = NULL;
    int   fmt_float = 15  ;   // N decimal in float 
    


    // Boolean args 
    PyObject *phead = Py_None;
    PyObject *pbar  = Py_None;
    PyObject *pverb = Py_None;
    Py_INCREF(pbar  );
    Py_INCREF(pverb );
    Py_INCREF(phead );


    Bool lheader = false ; 
    Bool lpbar   = false;
    Bool verbose = false;


    // keyword list
    static char *kwlist[] = {"database","sql_query", "fmt_float", "queryfile", "poolmask","pbar" , "verbose","header",  NULL};

    // Parse keyword args 
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss|izzOOO", kwlist,   // 2 requiered , 6 optional 
                                     &database,
                                     &sql_query,
//                                     &fcols    ,
                                     &fmt_float,
                                     &queryfile,
                                     &poolmask,
                                     &pbar,
                                     &pverb,
				     &phead)) {
         return NULL  ;
    }

    // Conversion to boolean C variable
    lpbar    = PyObj_ToBool ( pbar  , lpbar     );
    verbose  = PyObj_ToBool ( pverb , verbose   );
    lheader  = PyObj_ToBool ( phead , lheader   );


    // Estimate number of rows                      
    int total_rows = getMaxrows(database, sql_query);
    
    // Progress bar 
    size_t prog_max = 0 ;
    prog_max = (size_t)total_rows;

    if (total_rows <= 0) {
        PyErr_SetString(PyExc_RuntimeError,"--odb4py : SQL query returned zero rows");
        return NULL;
    }


    // Sql string or file 
    if (verbose) {
        if (sql_query)
            printf("--odb4py : Executing query from string statement : %s\n", sql_query);
        else if (queryfile)
            printf("--odb4py : Executing query from file : %s\n", queryfile);
    }


    // Open ODB                                         
    int maxcols = 0;
    void *h = odbdump_open(database,
                           sql_query,
                           NULL,
                           poolmask,
                           NULL,
                           &maxcols);


    if (!h || maxcols <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "--odb4py : Failed to open ODB");
        return NULL;
    }

    // Number of columns taking into account the number of functions in the query  (col pure - n columns function)
    //maxcols  = maxcols - fcols;

    // Prepare reading  the rows                                
    double *d = malloc(sizeof(double) * maxcols);
    if (!d) {
        odbdump_close(h);
        return PyErr_NoMemory();
    }

    
    int new_dataset = 0;
    int nd;

    colinfo_t *ci = NULL;
    int nci = 0;

    //  Mapping numeric columns 
    int *num_index = NULL;
    int  ncols_num = 0;

    double *buffer = NULL;
    int    row_idx = 0;
    size_t ip       = 0 ;

    const char *colname     = NULL;
    PyObject   *header_list = NULL;

    // Loop and read rows        
    while ((nd = odbdump_nextrow(h, d, maxcols, &new_dataset)) > 0) {

        // progress bar if  On 
        if (lpbar) {  ++ip;            print_progress(ip, prog_max); }   // useful for huge ODBs 

        // dataset metadata available 
        if (new_dataset) {
            ci = odbdump_destroy_colinfo(ci, nci);
            ci = odbdump_create_colinfo(h, &nci);

            // Count numeric columns (If STRING , the row is skipped )
            num_index = malloc(sizeof(int)*nci);
            if (!num_index) goto mem_error;

            ncols_num = 0;
            for (int i=0;i<nci;i++) {
                if (ci[i].dtnum != DATATYPE_STRING)
                    num_index[ncols_num++] = i;
            }
            // allocate main numeric buffer 
            buffer = malloc(sizeof(double) * (size_t)total_rows * (size_t)ncols_num);
            if (!buffer) goto mem_error;

             // header 
             if (lheader) {
                header_list = PyList_New(0);
                for (int i = 0; i < nci; ++i) {

                   if (ci[i].dtnum == DATATYPE_STRING)   continue;
		   
                   colname = ci[i].nickname ? ci[i].nickname : ci[i].name;
                   PyObject *py_name = PyUnicode_FromString(colname);

                   if (!py_name ) {  
			PyErr_SetString(PyExc_RuntimeError, "--odb4py : Column name not found !");
                        return NULL;
		   }
		      //int err =   PyList_Insert  (header_list , (Py_ssize_t) h++ , py_name  )  ; 
		      int err =   PyList_Append(header_list, py_name);
		      if (err <0  ) { 
                           PyErr_SetString(PyExc_RuntimeError, "--odb4py : Failed to insert colname to header list ");
			   return NULL ; 
                          }
                    Py_DECREF(py_name);                  
                  }             	       
	     }	     
            new_dataset = 0;
        }


        // reallocate if underestimated rows 
        if (row_idx >= total_rows) {
            total_rows *= 2;
            double *tmp =    realloc(buffer, sizeof(double) * (size_t)total_rows * (size_t)ncols_num);
            if (!tmp) goto mem_error;
            buffer = tmp;
        }


        // Get  numeric columns only 
        for (int j=0; j<ncols_num; ++j) {
            int i = num_index[j];
            colinfo_t *pci = &ci[i];
            double val = d[i];
            // missing value handling   ...Set to NAN 
            if (   fabs (val) == mdi) {       // Missing data indicator 
                buffer[(size_t)row_idx*(size_t)ncols_num + j] = NAN ; 
	    } else {
		      switch (pci->dtnum) {
                               case DATATYPE_STRING:
                                   continue;   // ignore
                                   break;

                               case DATATYPE_INT1:
                               case DATATYPE_INT2:
                               case DATATYPE_INT4:
                               case DATATYPE_YYYYMMDD:
                               case DATATYPE_HHMMSS:
                               buffer[(size_t)row_idx*ncols_num + j] = (double)(int)d[i];
                                    break;

                               default:
                                  {
                                  double float_val = (double)d[i];
                                  double fval = format_float(float_val, fmt_float);
                                  buffer[(size_t)row_idx*ncols_num + j] = fval;
                                  }
                                   break;
                      }  // switch  
        }

	} // cols 
	row_idx++;

    } // while 

    // Cleanup ODB                                       
    ci = odbdump_destroy_colinfo(ci, nci);
    odbdump_close(h);
    free(d);
    free(num_index);



 // Build numpy array  
npy_intp dims[2] = {  (npy_intp)row_idx,  (npy_intp)ncols_num };
PyObject *array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, buffer);

if (!array) {
    free(buffer);
    return PyErr_NoMemory();
}
PyObject *capsule = PyCapsule_New(buffer, NULL, free_when_done);
if (!capsule) {
    Py_DECREF(array);
    free(buffer);
    return PyErr_NoMemory();
}


// NumPy owner 
PyArray_SetBaseObject((PyArrayObject*)array, capsule);

// Create list  
 PyObject *result = PyTuple_New(2);
if (!result) return NULL;

if (lheader){
    PyTuple_SetItem(result, 0, header_list);
    PyTuple_SetItem(result, 1, array);
}else {
    Py_INCREF(Py_None);
    PyTuple_SetItem(result, 0, Py_None);
    PyTuple_SetItem(result, 1, array);
}

return result;


// Free and clean if error 
mem_error:
    if (ci) odbdump_destroy_colinfo(ci, nci);
    if (h) odbdump_close(h);
    free(d);
    free(num_index);
    free(buffer);
    return PyErr_NoMemory();
}



