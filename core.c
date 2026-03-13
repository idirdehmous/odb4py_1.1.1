#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <Python.h>
#include "pyspam.h"
#include "odb_types.h"
#include "info_module.c"
#include "meta_module.c"
#include "io_module.c"
#include "dca_module.c"
//#include "dict_module.c"
#include "geo_module.c"
#include "dbarray_module.c"
//#include "nc_module.h" 


PyDoc_STRVAR(info_tab_doc  , "Returns all the existing ODB tables.  '386' tables.");
PyDoc_STRVAR(info_var_doc  , "Returns all ODB 'varno' parameters and their descriptions");
PyDoc_STRVAR(info_fun_doc  , "Returns all the possible functions that could be used in ODB sql statement.");
PyDoc_STRVAR(npar_doc      , "Fetch ODB rows as a numpy array with optional header containing column names." );
//PyDoc_STRVAR(dict_doc      , "Fetch ODB rows as a python  dictionnary where the column names are the keys and the values are lists of values.");
PyDoc_STRVAR(dcaf_doc      , "Create DCA files (Direct Column  Access ).");
PyDoc_STRVAR(dist_doc      , "Compute great circle distance between numpy lat/lon pairs. Optimized with Numpy C/API.");
PyDoc_STRVAR(geop_doc      , "Fetch lat,lon,alt,date,time and obsvalue according to some additional custom 'WHERE' SQL conditions.");
//PyDoc_STRVAR(ncdf_doc      , "Convert the columns in the SQL query into NetCDF format (NetCDF Classic  or 4 ).");
PyDoc_STRVAR(vers_doc      , "Returns the odb4py version." );
PyDoc_STRVAR(meta_doc      , "Returns build and Python info.");


PyDoc_STRVAR(module_doc,"C/Python interface to access the ODB1 IFS/ARPEGE databases\nThe original source code has been developed by S.Saarinen et al\n***Copyright (c) 1997-98, 2000 ECMWF. All Rights Reserved !***"); 





/*static PyMethodDef ODBConnection_methods[] = {
    {"odb_close", (PyCFunction)(void(*)(void)) odb_close_method, METH_NOARGS, ""},
    {NULL, NULL, 0, NULL}
};*/

/*static  PyTypeObject ODBConnectionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "core.ODBConnection",
    .tp_basicsize = sizeof(ODBConnection),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = ODBConnection_methods,
//    .tp_dealloc = (destructor)ODBConnection_dealloc
};*/



// Define the functions in the modules 
static PyMethodDef module_methods[] = {

    {"odb_info_tables",(PyCFunction)(void(*)(void))    odb_info_tables_method, METH_VARARGS | METH_KEYWORDS, info_tab_doc },
    {"odb_info_varno" ,(PyCFunction)(void(*)(void))    odb_info_varno_method , METH_VARARGS | METH_KEYWORDS, info_var_doc },
    {"odb_info_funcs" ,(PyCFunction)(void(*)(void))    odb_info_funcs_method , METH_VARARGS | METH_KEYWORDS, info_fun_doc },
    {"odb_array"      ,(PyCFunction)(void(*)(void))    odb_array_method      , METH_VARARGS | METH_KEYWORDS, npar_doc     },
 //   {"odb_dict"       ,(PyCFunction)(void(*)(void))    odb_dict_method       , METH_VARARGS | METH_KEYWORDS, dict_doc     },
    {"odb_dca"        ,(PyCFunction)(void(*)(void))    odb_dca_method        , METH_VARARGS | METH_KEYWORDS, dcaf_doc     },
    {"odb_gcdist"     ,(PyCFunction)(void(*)(void))    odb_gcdist_method     , METH_VARARGS | METH_KEYWORDS, dist_doc     },
//    {"odb_geopoints"  ,(PyCFunction)(void(*)(void))    odb_geopoints_method  , METH_VARARGS | METH_KEYWORDS, geop_doc     },
//    {"odb2nc"         ,(PyCFunction)    odb2nc_method         , METH_VARARGS | METH_KEYWORDS, ncdf_doc     },
    {"version"        ,odb_meta_version, METH_NOARGS          , vers_doc },
    {"info"           ,odb_meta_info,    METH_NOARGS          , meta_doc },
//    {"odb_connect"    ,(PyCFunction)(void(*)(void))    odb_connect_method    , METH_VARARGS | METH_KEYWORDS, connect_doc},
 //   {"odb_close"      ,(PyCFunction)(void(*)(void)) odb_close_method , METH_VARARGS | METH_KEYWORDS,   close_doc  },

};


// Define the module itself 
static struct PyModuleDef   odbmodule = {
    PyModuleDef_HEAD_INIT,
    "core"         ,
    module_doc     , 
    -1             ,
    module_methods ,
    .m_slots =NULL
};



// Called first during python call
PyMODINIT_FUNC PyInit_core (void) {
    if (_import_array() < 0) {
        return NULL;
    }

if (PyType_Ready(&ODBConnectionType) < 0) {
    return NULL;
}

    PyObject*  m  ;
    PyObject* ModuleError ;
    m=PyModule_Create(&odbmodule);
    if ( m == NULL) {
        ModuleError = PyErr_NewException("Failed to create the module : odb4py core", NULL, NULL);
        Py_XINCREF(ModuleError) ;
        return NULL;
}

Py_INCREF(&ODBConnectionType);
PyModule_AddObject(m, "ODBConnection", (PyObject *)&ODBConnectionType);


// Many other Custm Exceptions have to be added here !
//PyOdbEmptyResultError = PyErr_NewException("odb4py.EmptyResultError", NULL, NULL);
//Py_INCREF(PyOdbEmptyResultError);
//PyModule_AddObject(m , "EmptyResultError", PyOdbEmptyResultError);
//PyObject_Print  ( m , stdout , 0 )  ; 
return m  ;
}

