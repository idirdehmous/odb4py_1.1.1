#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

//NUMPY API 
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#define ODB_STRLEN 8  // 8 chars + '\0' 
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <Python.h>

#include "pyspam.h"
#include "rows.h"
#include "netcdf.h"




typedef struct {
    int odb_col;      // ODB column index 
    colinfo_t *meta;  // metadata 
} nc_column_t;


//  Read ODB rows and build numeric buffer           
static int rows4nc(char *database,
                   char *sql_query,
                   int   fcols,
                   char *poolmask,
                   double **buffer,
                   int *nrows,
                   int *ncols,
                   nc_column_t **cols,
                   colinfo_t **ci)
{
    int fmt_float = 15;
    int maxcols   = 0;
    void *h       = NULL;
    int new_dataset = 0;
    int nci = 0;
    int nd;

    int ncols_num = 0;
    int row_idx   = 0;

    int total_rows = getMaxrows(database, sql_query, poolmask);
    if (total_rows <= 0) {
        printf("--odb4py : ODB query returned zero rows\n");
        return -1;
    }

    h = odbdump_open(database, sql_query, NULL, NULL, NULL, &maxcols);
    if (!h || maxcols <= 0) {
        printf("--odb4py : Failed to open ODB\n");
        return -1;
    }

    double *d = malloc(sizeof(double) * maxcols);
    if (!d) {
        odbdump_close(h);
        return -1;
    }

    while ((nd = odbdump_nextrow(h, d, maxcols, &new_dataset)) > 0) {

        if (new_dataset) {

            *ci = odbdump_destroy_colinfo(*ci, nci);
            *ci = odbdump_create_colinfo(h, &nci);

            *cols = malloc(sizeof(nc_column_t) * nci);
            if (!*cols) goto mem_error;

            ncols_num = 0;

            for (int i = 0; i < nci; i++) {
                if ((*ci)[i].dtnum != DATATYPE_STRING) {
                    (*cols)[ncols_num].odb_col = i;
                    (*cols)[ncols_num].meta    = &(*ci)[i];
                    ncols_num++;
                }
            }

            *buffer = malloc(sizeof(double) * (size_t)total_rows *(size_t)ncols_num);

            if (!*buffer) goto mem_error;

            new_dataset = 0;
        }

        for (int j = 0; j < ncols_num; j++) {

            int i = (*cols)[j].odb_col;
            colinfo_t *pci = (*cols)[j].meta;

            double val = d[i];

            if (fabs(val) == mdi) {
                (*buffer)[(size_t)row_idx*ncols_num + j] = NAN;
            }
            else {

                switch (pci->dtnum) {

                    case DATATYPE_INT1:
                    case DATATYPE_INT2:
                    case DATATYPE_INT4:
                    case DATATYPE_YYYYMMDD:
                    case DATATYPE_HHMMSS:
                        (*buffer)[(size_t)row_idx*ncols_num + j] =(double)(int)d[i];
                        break;

                    default:
                    {
                        double fval =
                            format_float((double)d[i], fmt_float);
                           (*buffer)[(size_t)row_idx*ncols_num + j] = fval;
                    }
                }
            }
        }

        row_idx++;
    }

    *nrows = total_rows;
    *ncols = ncols_num;

    free(d);
    odbdump_close(h);

    return 0;

mem_error:

    if (*buffer) free(*buffer);
    if (*cols)   free(*cols);
    if (*ci)     odbdump_destroy_colinfo(*ci, nci);
    if (h)       odbdump_close(h);
    if (d)       free(d);

    return -1;
}


// Write into  NetCDF file                                    
static int writeNetcdf(const char *outfile,
		       char       *sql_query  ,  
                       double  *buffer,
                       int nrows,
                       int ncols,
                       nc_column_t *col)
{
    int ncid, dimid;
    int *varids = NULL;
    int ret;
    double fill = NAN;


    ret = nc_create(outfile, NC_NETCDF4, &ncid);
    if (ret != NC_NOERR) {
        printf("NetCDF error: %s\n", nc_strerror(ret));
        return -1;
    }


// convention  
nc_put_att_text(ncid, NC_GLOBAL,"Conventions", strlen("CF-1.10"), "CF-1.10");


// The title & global attrib 
nc_put_att_text(ncid, NC_GLOBAL,"title"  , strlen("ODB observations"), "ODB in NetCDF format");
nc_put_att_text(ncid, NC_GLOBAL,"source" , strlen("ODB database"), "ODB database");
nc_put_att_text(ncid, NC_GLOBAL,"history", strlen("created by odb4py"), "created by odb4py");
nc_put_att_text(ncid, NC_GLOBAL,"Institution",strlen("Institution"), "(RMI)");
nc_put_att_text(ncid, NC_GLOBAL, "sql_query",strlen(sql_query), sql_query);
nc_put_att_text(ncid, NC_GLOBAL, "featureType", strlen("point"), "point");


// dims 
 ret = nc_def_dim(ncid, "nobs", nrows, &dimid);
 if (ret != NC_NOERR) {
    printf("NetCDF error: %s\n", nc_strerror(ret));
    return -1;
 }

// Vars 
varids = malloc(sizeof(int) * ncols);
if (!varids) return -1;

// Loop over the nc_colname_t  structure info 
for (int i = 0; i < ncols; i++) {
        const char *name = col[i].meta->nickname ? col[i].meta->nickname: col[i].meta->name;


	// define vars & attributes 	
	if (strcmp(name,"lat")==0 || strcmp(name,"latitude")==0) {
         nc_put_att_text(ncid, varids[i], "units", strlen("degrees_north"),"degrees_north");
         nc_put_att_text(ncid, varids[i],"standard_name",strlen("latitude"),"latitude");
           }

        if (strcmp(name,"lon")==0 || strcmp(name,"longitude")==0) {
         nc_put_att_text(ncid, varids[i], "units", strlen("degrees_east"), "degrees_east");
         nc_put_att_text(ncid, varids[i], "standard_name", strlen("longitude"), "longitude");
           }

// Other variables 
nc_def_var(ncid, name,NC_DOUBLE, 1,  &dimid, &varids[i]);	
// long name 
nc_put_att_text(ncid, varids[i],"long_name",strlen(col[i].meta->name), col[i].meta->name);
// Missing values 
nc_put_att_double(ncid, varids[i],"_FillValue", NC_DOUBLE, 1, &fill);
    }


// Write buffer  values 
size_t start[1], count[1];
count[0]        = nrows;
for (int c = 0; c < ncols; c++) {
    double *colbuf = malloc(sizeof(double) * nrows);
    for (int r = 0; r < nrows; r++)  {
        colbuf[r] = buffer[r*ncols + c];
    }
    start[0] = 0;
    nc_put_vara_double(ncid,  varids[c], start, count, colbuf);
    free(colbuf);
}

    nc_enddef(ncid);
    nc_close(ncid);
    free(varids);
    return 0;
}


// Python wrapper                                           
static PyObject *odb2nc_method(PyObject *Py_UNUSED(self),
                               PyObject *args,
                               PyObject *kwargs)
{
    char *database  = NULL;
    char *sql_query = NULL;
    char *ncfile    =NULL ; 
    int   fcols     = 0   ;    

    static char *kwlist[] = {"database",
	                     "sql_query",
			     "nfunc",
                             "ncfile",
			     NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "ssis",
                                     kwlist,
                                     &database,
                                     &sql_query,
                                     &fcols, 
				     &ncfile ))
        return NULL;

    double *buffer = NULL;
    int nrows = 0;
    int ncols = 0;

    nc_column_t *cols = NULL;
    colinfo_t   *ci   = NULL;

    if (rows4nc(database,
                sql_query,
                fcols,
                NULL,
                &buffer,
                &nrows,
                &ncols,
                &cols,
                &ci) != 0)
    {
        PyErr_SetString(PyExc_RuntimeError,"--odb4py : failed to get rows from ODB before encoding");
        return NULL;
    }

    // To nc  
    writeNetcdf( ncfile , sql_query ,buffer, nrows, ncols, cols);


    free(buffer);
    free(cols);
    if (ci) odbdump_destroy_colinfo(ci, 0);

    Py_RETURN_NONE;
}


