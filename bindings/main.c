#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include "pyspam.h"
#include "info_module.h"
#include "meta_module.h"
#include "io_module.h"
#include "dca_module.h"
#include "dict_module.h"
#include "geo_module.h"


// Define the functions in the modules 
static PyMethodDef module_methods[] = {
    {"odbConnect",  (PyCFunction)(void(*)(void))    odbConnect_method    , METH_VARARGS | METH_KEYWORDS,  "Create ODB connection object"},
    {"odbClose"  ,  (PyCFunction)(void(*)(void))    odbClose_method      , METH_VARARGS | METH_KEYWORDS,  "Close an opened ODB "},
    {"odbTables" ,  (PyCFunction)(void(*)(void))    odbTables_method     , METH_VARARGS | METH_KEYWORDS,  "Show all existing ODB tables"},
    {"odbVarno"  ,  (PyCFunction)(void(*)(void))    odbVarno_method      , METH_VARARGS | METH_KEYWORDS,  "Print all ODB 'varno' parameters and their description"},
    {"odbFunctions",  (PyCFunction)(void(*)(void))  odbFunctions_method  , METH_VARARGS | METH_KEYWORDS,  "Print all the possible functions that could be used in ODB sql statement"},
    {"odbDict"   ,    (PyCFunction)(void(*)(void))  odbDict_method       , METH_VARARGS | METH_KEYWORDS,  "Fetch ODB rows as a python  dictionnary"},
    {"odbDca"    ,    (PyCFunction)(void(*)(void))  odbDca_method        , METH_VARARGS | METH_KEYWORDS,  "Create DCA  (Direct Column  Access ) files "},
    {"odbGcdist" , (PyCFunction)(void(*)(void))  odbGcdistance_method , METH_VARARGS | METH_KEYWORDS,  "Compute great circle distance between numpy lat/lon pairs"},
    {"version"   , odbMeta_version, METH_NOARGS, "Return the odb4py version."  },
    {"info"      , odbMeta_info,    METH_NOARGS, "Return build and Python info."},
};


// Define the module itself 
static struct PyModuleDef   odbmodule = {
    PyModuleDef_HEAD_INIT,
    "core",
    "C/Python interface to access the ODB1 IFS/ARPEGE databases\nThe original source code has been developed by Copyright (c) 1997-98, 2000 ECMWF. All Rights Reserved !",
    -1,
    module_methods ,
     .m_slots =NULL
};


// Called first during python call
PyMODINIT_FUNC PyInit_core (void) {

    if (_import_array() < 0) {
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

PyOdbEmptyResultError = PyErr_NewException("odb4py.EmptyResultError", NULL, NULL);
Py_INCREF(PyOdbEmptyResultError);
PyModule_AddObject(m , "EmptyResultError", PyOdbEmptyResultError);

  return m  ;
}

