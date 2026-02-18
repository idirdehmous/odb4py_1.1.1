#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <time.h>




// Check file exists
/*static int  file_exists(  const char *path ) {
struct stat st  ;   
return ( stat( path , &st ) ==  0 && S_ISREG(st.st_mode))  ;
}*/


// Check dir exists 
static int dir_exists(const char *path)
{
    struct stat st;
    return (stat(path, &st) == 0 && S_ISDIR(st.st_mode));
}



// Path  validity 
static int valid_odb_path(const char *path)
{
    struct stat st;
    char tmp[4096];

    /* direct directory */
    if (stat(path,&st)==0 && S_ISDIR(st.st_mode))
        return 1;

    /* try ECMA */
    snprintf(tmp,sizeof(tmp),"%s/ECMA",path);
    if (stat(tmp,&st)==0 && S_ISDIR(st.st_mode))
        return 1;

    /* try CCMA */
    snprintf(tmp,sizeof(tmp),"%s/CCMA",path);
    if (stat(tmp,&st)==0 && S_ISDIR(st.st_mode))
        return 1;

    return 0;
}


/*static int dca_files_exist(const char *dbpath       ,
                           PyObject   *tables       ,
                           char       *missing_table,
                           size_t      missing_size )
{
    if (!tables || !PySequence_Check(tables))
        return  -1 ; 
    Py_ssize_t n = PySequence_Size(tables);
    char filepath[4096]; 
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PySequence_GetItem(tables, i); // New py ref 
        if (!item)
            return -1;
        const char *tname = PyUnicode_AsUTF8(item);
        if (!tname) {
            Py_DECREF(item);
            return -1;
        }
	// Path to   dca/<table_name>.dca 
        snprintf(filepath, sizeof(filepath),"%s/dca/%s.dca", dbpath, tname);
        if (!file_exists(filepath)) {
             copy missing table name 
            snprintf(missing_table, missing_size, "%s", tname);
            Py_DECREF(item);
            return -1;   
        }
        Py_DECREF(item);
    }
    return 0; 
}*/




// odbDca Python method      
static PyObject * odbDca_method(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    const char *dbpath = NULL;
    const char *dbtype = NULL;
    const char *extra  = NULL;
    PyObject *tables = NULL;
    int ncpu = 1;

    static char *kwlist[] = {
        "odbdir",
        "dbtype",
        "tables",
        "ncpu"  , 
        "extra_args",
         NULL
    };

    // Check args  
    if (!PyArg_ParseTupleAndKeywords( args, kwargs, "ss|Ois", kwlist ,
                                                             &dbpath ,
                                                             &dbtype ,
							     &tables ,
                                                             &ncpu,
                                                             &extra))
    {
        return PyLong_FromLong(-1);
    }



    // Check odb path 
    if (!valid_odb_path(dbpath)) {
    fprintf(stderr,
        "--odb4py : invalid ODB directory: %s\n", dbpath);
       return PyLong_FromLong(-1);
     }

  
    // Set number of  CPUs   ( if negative value or not using all system  ressource  , take  _SC_NPROCESSORS_ONLN) 
    if (ncpu < 0   ||  ncpu < sysconf(_SC_NPROCESSORS_ONLN)) { 
      ncpu  =  _SC_NPROCESSORS_ONLN ; 
    }   

    // dca path  
    char dca_dir[4096];
    snprintf(dca_dir, sizeof(dca_dir), "%s/dca", dbpath);

    // Locate dcagen from env 
    const char *bebin = getenv("ODB_BEBINPATH");

    // Check path  
    if (!bebin) {
            fprintf(stderr,
                    "--odb4py : ODB_BEBINPATH not defined\n");
            return PyLong_FromLong(-1);
      }


    // Build dcagen command                               
    char cmd  [4096]  ;
    char cpu_str[32]  ;

    // Concat  
    snprintf(cpu_str, sizeof(cpu_str), "%d", ncpu);
    snprintf(cmd, sizeof(cmd), "%s/dcagen -i '%s' -N %s -q -z -P", bebin, dbpath, cpu_str);


    if (extra && strlen(extra) > 0) {
        strncat(cmd, " ", sizeof(cmd)-strlen(cmd)-1);
        strncat(cmd, extra, sizeof(cmd)-strlen(cmd)-1);
    }


if (tables && !PySequence_Check(tables)) {
    PyErr_SetString(PyExc_TypeError,
        "--odb4py : tables must be a list or tuple of table names seperated by -t .         \n`see the dcagen script documentation`");
     return PyLong_FromLong(-1);
}

// Create dca files only for the table found in the pool (ONLY !) 
// It processes some tables instead of  386 ones  !!
if ( tables &&  PySequence_Check(tables)) {
    Py_ssize_t n = PyList_Size(tables);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PyList_GetItem(tables, i); // borrowed ref
        if (!PyUnicode_Check(item))
            continue;

        const char *tname = PyUnicode_AsUTF8(item);
        strncat(cmd, " -t ", sizeof(cmd)-strlen(cmd)-1);
        strncat(cmd, tname , sizeof(cmd)-strlen(cmd)-1);
    }
}

   
    printf("--odb4py : Creating DCA files...\n");
    // Execute dcagen                                    
    int status= -1;

    // Unlock python GIL for another process 
    Py_BEGIN_ALLOW_THREADS
    status = system(cmd);
    // relock GIL 
    Py_END_ALLOW_THREADS

    // Check failure 
    if (status == -1 ||  !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        fprintf(stderr,
            "--odb4py : dcagen failed\nCommand: %s\n",  cmd);
        return PyLong_FromLong(-1);
    }


    // Check creation                                 
    if (!dir_exists(dca_dir)) {
        fprintf(stderr,
            "--odb4py : dcagen finished but no DCA directory found\n");
        return PyLong_FromLong(-1);
    }  else  {

    printf("--odb4py : DCA files creation done.\n");
    return PyLong_FromLong(0);

    }
}

