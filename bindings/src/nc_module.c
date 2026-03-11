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
#include <time.h>
#include <Python.h>
#include "pyspam.h"
#include "rows.h"
#include "ncdf.h"
#include "odb.h"
#include "idx.h"
#include "info.h"




// Get the returned rows and to encode into NetCDF 
static int rows4nc(char *database,
                   char *sql_query,
                   int   fcols,
                   char *poolmask,
                   double **buffer,
                   char ***strbufs,
                   int *nrows,
                   int *ncols,
                   nc_column_t **cols,
                   colinfo_t   **ci , 
		   Bool lpbar  )
{
    int fmt_float = 15;
    int maxcols   = 0;
    void *h       = NULL;
    int new_dataset = 0;
    int nci = 0;
    int nd;
    int ncols_all = 0;
    int row_idx   = 0;
    size_t ip       = 0 ;
    size_t prog_max = 0 ;


//info = sql_prepare_func(h , sql_query ? sql_query : queryfile, NULL,  NULL );
//if (!info) {
//    printf("sql_prepare_func failed\n");
//    return;
//}
//if (info->dir)
//    printf("dir: %s\n", info->dir);
// the right db handler 
//db_handle = ODBc_open(database, "r", &npools, &ntables, poolmask);


    int total_rows = getMaxrows(database, sql_query, poolmask);
    if (total_rows <= 0) {
        printf("--odb4py : ODB query returned zero rows\n");
        return -1;
    }

    h = odbdump_open(database, sql_query, NULL, NULL, NULL, &maxcols);
    if (!h || maxcols <= 0) {  printf("--odb4py : Failed to open ODB\n");
        return -1;
    }

    double *d = malloc(sizeof(double) * maxcols);
    if (!d) {
        odbdump_close(h);
        return -1;
    }

    while ((nd = odbdump_nextrow(h, d, maxcols, &new_dataset)) > 0) {
        if (lpbar) {  
	    ++ip;            
	    print_progress(ip, prog_max); 
	}   // useful for huge ODBs 
									 
        if (new_dataset) {
            *ci = odbdump_destroy_colinfo(*ci, nci);
            *ci = odbdump_create_colinfo(h, &nci);

            *cols    = malloc(sizeof(nc_column_t) * nci);
            *strbufs = calloc(nci, sizeof(char*));
            if (!*cols || !*strbufs)
                goto mem_error;
// Realloc if necessary   ( row_idx greater than tot_rows. Worth doing it  ?? )
if (row_idx >= total_rows) {
         total_rows *= 2;	
	 size_t size = (size_t)total_rows * (*ncols);
         *buffer     = realloc(*buffer, sizeof(double)*  size  );
        for (int j=0;j<*ncols;j++) {
           if ((*cols)[j].meta->dtnum == DATATYPE_STRING) {
              (*strbufs)[j] = realloc((*strbufs)[j],total_rows*(ODB_STRLEN+1));
        }
    }
}

            ncols_all = 0;
            for (int i = 0; i < nci; i++) {
                (*cols)[ncols_all].odb_col = i;
                (*cols)[ncols_all].meta    = &(*ci)[i];

                if ((*ci)[i].dtnum == DATATYPE_STRING) {
                    (*strbufs)[ ncols_all ] = calloc(total_rows, ODB_STRLEN + 1);
                    if (!(*strbufs)[ncols_all])
                        goto mem_error;
                      }
                ncols_all++;
            }
            *buffer = malloc(sizeof(double) * total_rows * ncols_all);
            if (!*buffer)
                goto mem_error;
            new_dataset = 0;
        }
        for (int j = 0; j < ncols_all; j++) {
            int i = (*cols)[j].odb_col;
            colinfo_t *pci = (*cols)[j].meta;
            double val = d[i];
            if (pci->dtnum == DATATYPE_STRING) {
                char *dst =
                    &(*strbufs)[j][row_idx * (ODB_STRLEN + 1)];
                if (fabs(val) == mdi) {
                    memset(dst, 0, ODB_STRLEN + 1);
                    strncpy(dst, "NULL", ODB_STRLEN);
                }
                else {
                    union {
                        char s[sizeof(double)];
                        double d;
                    } u;
                    u.d = d[i];
                    memcpy(dst, u.s, ODB_STRLEN);
                    dst[ODB_STRLEN] = '\0';
                    make_printable(dst, ODB_STRLEN);
                }
            }
            else {
                if (fabs(val) == mdi) {
                    (*buffer)[row_idx*ncols_all + j] = NAN;
                }
                else {
                    switch (pci->dtnum) {
                        case DATATYPE_INT1:
                        case DATATYPE_INT2:
                        case DATATYPE_INT4:
                        case DATATYPE_YYYYMMDD:
                        case DATATYPE_HHMMSS:
                            (*buffer)[row_idx*ncols_all + j] =(double)(int)d[i];
                            break;
                        default:
                            (*buffer)[row_idx*ncols_all + j] =
                                format_float(d[i], fmt_float);
                    }
                }
            }
        }

        row_idx++;
    }// while 
    *nrows = total_rows;
    *ncols = ncols_all;
    free(d);
    odbdump_close(h);
    return 0;

// errors Label 
mem_error:
    if (*buffer) free(*buffer);
    if (*strbufs) {
        for (int i = 0; i < ncols_all; i++)
            free((*strbufs)[i]);
           free(*strbufs);
    }
    if (*cols) free(*cols);
    if (*ci) odbdump_destroy_colinfo(*ci, nci);
    if (h) odbdump_close(h);
    if (d) free(d);
    return -1;
}





// Write into  NetCDF file                                    
static int writeNetcdf(const char *database   ,
		       const char *outfile    ,
		       char       *sql_query  ,  
                       double     *buffer     ,
		       char       *strbufs , 
                       int nrows,
                       int ncols,
                       nc_column_t *col)
{


// Debug  
/*printf("---- STRING BUFFER ----\n");
for (int r = 0; r < nrows; r++)
{
    char (*s)[ODB_STRLEN] = (char (*)[ODB_STRLEN])strbufs;
    printf("%.*s\n", ODB_STRLEN, s[r]);
//    printf("%.*s\n", ODB_STRLEN, strbufs + r*ODB_STRLEN);
}*/

// date & Time 
char datetime[64];
time_t now = time(NULL);
struct tm *tm_info = gmtime(&now);   // localtime()
strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S UTC", tm_info);

// Analysis datetime 
// We use a function from odb.h   header  
int date_odb;
int time_odb;
int yyyymmdd_loc, hhmmss_loc ;

codb_analysis_datetime_(&date_odb, &time_odb);
codb_datetime_(&yyyymmdd_loc, &hhmmss_loc);

int size = 512  ; 
char ddfile [ size ]  ;
build_dd_path (database ,  ddfile  , size  ) ; 

int ana_date =0; 
int ana_time =0; 

scan_dd_file ( database  , &ana_date , &ana_time  )  ;
printf( "Path to dd file %s , anadate %d   anatime  =%d  \n" , ddfile , ana_date  , ana_time  )  ; 



//printf( "%d  %d  %d  %d  \n" , date_odb, time_odb,  yyyymmdd_loc, hhmmss_loc ) ; 
//printf ( "%s\n" , database  )  ; 

// dims and vars 
    int ncid, dimid;
    int *varids = NULL;
    int ret;
    double fill = NAN;
    // create the file 
    ret = nc_create(outfile, NC_NETCDF4, &ncid);
    if (ret != NC_NOERR) {
        printf("NetCDF error: %s\n", nc_strerror(ret));
        return -1;
    }
// NC conventions     
nc_put_att_text(ncid, NC_GLOBAL,"Conventions", strlen("CF-1.10"), "CF-1.10");

// The title & global attrib 
const char *title       = "ODB data in NetCDF format";
const char *history     = "created by odb4py";
const char *institution = "(RMI) Royal Meteorological Institute of Belgium";
const char *feature     = "point" ; 
const char *data_source = "ECMWF ODB";
const char *encoding    = "ODB (row-major)" ; 
nc_put_att_text(ncid, NC_GLOBAL, "Title"      ,strlen(title)       , title);
nc_put_att_text(ncid, NC_GLOBAL, "History"    ,strlen(history)     , history);
nc_put_att_text(ncid, NC_GLOBAL, "Institution",strlen(institution) , institution);
nc_put_att_text(ncid, NC_GLOBAL, "Creation_datetime",strlen(datetime) ,  datetime );
nc_put_att_text(ncid, NC_GLOBAL, "Native_fomrat" ,strlen(data_source) , data_source );
nc_put_att_text(ncid, NC_GLOBAL, "Encoding"   ,strlen(encoding)    , encoding);
nc_put_att_text(ncid, NC_GLOBAL, "sql_query"  ,strlen(sql_query)   , sql_query);
nc_put_att_text(ncid, NC_GLOBAL, "featureType",strlen(feature )    , feature );




// dims 
 ret = nc_def_dim(ncid, "nobs", nrows, &dimid);
 if (ret != NC_NOERR) {
    printf("NetCDF error: %s\n", nc_strerror(ret));
    return -1;
 }
int strlen_dimid;
size_t str_len = ODB_STRLEN;
ret = nc_def_dim(ncid, "strlen", str_len, &strlen_dimid);
if (ret != NC_NOERR) {
    printf("NetCDF error: %s\n", nc_strerror(ret));
    return -1;
}
// Vars allocation 
varids = malloc(sizeof(int) * ncols);
if (!varids) return -1;

// Whether to write strings or numeric values 
// the nc_colname_t  structure is here  !
for (int i = 0; i < ncols; i++) {
        const char *name = col[i].meta->nickname ? col[i].meta->nickname: col[i].meta->name;
        char varname[128];
        strncpy(varname, name , sizeof(varname)-1);
        varname[sizeof(varname)-1] = '\0';
        sanitize_name(varname);
	// define vars & attributes 	
	if (strcmp(varname,"lat")==0 || strcmp(varname,"latitude")==0) {
         nc_put_att_text(ncid, varids[i], "units", strlen("degrees_north"),"degrees_north");
         nc_put_att_text(ncid, varids[i],"standard_name",strlen("latitude"),"latitude");           }
        if (strcmp(varname,"lon")==0 || strcmp(varname,"longitude")==0) {
         nc_put_att_text(ncid, varids[i], "units", strlen("degrees_east"), "degrees_east");
         nc_put_att_text(ncid, varids[i], "standard_name", strlen("longitude"), "longitude");           }

if (col[i].meta->dtnum == DATATYPE_STRING) {
    int dimids[2] = {dimid, strlen_dimid};
    ret = nc_def_var(ncid, varname, NC_CHAR, 2, dimids, &varids[i]);
    if (ret != NC_NOERR) {
        printf("NetCDF error: %s\n", nc_strerror(ret));
        return -1;
          }
} else {
    // Other variables
    int dimids[1] = {dimid};
    ret = nc_def_var(ncid, varname, NC_DOUBLE, 1, dimids, &varids[i]);
    // long variable names ( @table   is not ommitted ) 
    nc_put_att_text(ncid, varids[i],"long_name",strlen(col[i].meta->name), col[i].meta->name);

// Missing values 
nc_put_att_double(ncid, varids[i],"_FillValue", NC_DOUBLE, 1, &fill);
if (ret != NC_NOERR) {
    printf("NetCDF error: %s\n", nc_strerror(ret));
    return -1;
     }

} // column type 
}  //  for i 
 
// End definition 
nc_enddef(ncid);


//write strings 
for (int c = 0; c < ncols; c++) {
    if (col[c].meta->dtnum == DATATYPE_STRING) {
        char *src = strbufs;
        size_t start[2] = {0,0};
        size_t count[2] = {nrows, ODB_STRLEN};
        nc_put_vara_text(ncid, varids[c], start, count, src);
    }
}
// Buffer numeric  
size_t start[1] = {0};
size_t count[1] = {nrows};
for (int c = 0; c < ncols; c++) {
    if (col[c].meta->dtnum == DATATYPE_STRING)
        continue;
    double *colbuf = malloc(sizeof(double) * nrows);
    for (int r = 0; r < nrows; r++)
        colbuf[r] = buffer[r*ncols + c];
    //  to netcdf  
    nc_put_vara_double(ncid, varids[c], start, count, colbuf);
    free(colbuf);
}
    nc_close(ncid);
    free(varids);
    return 0;
}





// Main function wrapper to perform conversion ODB--> NetCDF
static PyObject *odb2nc_method(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs){
    char *database  = NULL;
    char *sql_query = NULL;
    char *ncfile    = NULL;
    int fcols = 0;
    Bool ldegree = true;
    Bool lpbar   = false;
    Bool verbose = false;
    static char *kwlist[] = {
        "database",
        "sql_query",
        "nfunc",
        "ncfile",
        "lalon_deg",
        "pbar",
        "verbose",
        NULL
    };
    PyObject *degree_obj = Py_True;
    PyObject *pbar  = Py_None;
    PyObject *pverb = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "ssis|OOO",
                                     kwlist,
                                     &database,
                                     &sql_query,
                                     &fcols,
                                     &ncfile,
                                     &degree_obj,
                                     &pbar,
                                     &pverb))
        return NULL;
    ldegree = PyObj_ToBool(degree_obj, ldegree);
    lpbar   = PyObj_ToBool(pbar, lpbar);
    verbose = PyObj_ToBool(pverb, verbose);
    double *buffer  = NULL;
    char   **strbufs = NULL;
    int nrows = 0;
    int ncols = 0;
    nc_column_t *cols = NULL;
    colinfo_t   *ci   = NULL;

    // Get rows 
    if (rows4nc(database, sql_query, fcols, NULL,
                &buffer, &strbufs,
                &nrows, &ncols,
                &cols, &ci,
                lpbar) != 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "--odb4py : failed to get rows from ODB for NetCDF encoding");
        return NULL;
    }

    //   Convert coordinates        
    //   If it's not in degrees , force conversion  
    //   ldegree is infered from  the sql query  
    if (!ldegree) {
        if (verbose)
            printf("Coordinates converted radians → degrees\n");

        convert_rad_to_deg(buffer, nrows, ncols, cols);
    }

    // Allocate buffer for string 
    size_t str_len = ODB_STRLEN;
    char *buffer_str  = malloc(nrows * str_len);
    if (!buffer_str) {
        PyErr_SetString(PyExc_RuntimeError,
            "--odb4py : Failed to allocate string buffer string for NetCDF encoding");
        return NULL;
    }

    //  Get string values 
    for (int c = 0; c < ncols; c++) {
        if (cols[c].meta->dtnum == DATATYPE_STRING) {
            char *src = strbufs[c];
            char *dst = buffer_str;

            for (int r = 0; r < nrows; r++) {
                memcpy(dst, src, str_len);
                dst += str_len;
                src += str_len + 1;
            }

	    // Better to send the both type at once 
            /*writeNetcdf_string_column(
                ncfile,
                sql_query,
                c,
                col_buf,
                nrows,
                str_len,
                cols
            );*/
        }
    }


    writeNetcdf( database ,  ncfile,sql_query,buffer, buffer_str , nrows, ncols, cols );

    free(buffer_str);
    free(buffer);
    free(cols);

    if (ci)
        odbdump_destroy_colinfo(ci, 0);

    Py_RETURN_NONE;
}
