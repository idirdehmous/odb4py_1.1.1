#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <Python.h>

#include "nc_module.c"


/*PyDoc_STRVAR(ncdf_doc,"Extract the data from the ODB and writes into NetCDF file" )
PyDoc_STRVAR(conv_doc,"Blala" )
		"Set of C/Python functions to perform ODB to other formats conversion. The original source code has been developed by S.Saarinen et al\n***Copyright (c) 1997-98, 2000 ECMWF. All Rights Reserved !***");*/



static PyMethodDef  convert_methods[] = {
    {"odb2nc"  , (PyCFunction) (void(*)(void)) odb2nc_method  , METH_VARARGS | METH_KEYWORDS,  "" },
    {NULL, NULL, 0, NULL}
};



// Define the module itself 
static struct PyModuleDef   odb_convert = {
    PyModuleDef_HEAD_INIT,
    "convert"         ,
     ""          , 
    -1                ,
    convert_methods   ,
    .m_slots =NULL
};


// Create the IO module 
PyMODINIT_FUNC PyInit_convert   (void) {

    PyObject*  m  ;
    PyObject* ModuleError ;
    m=PyModule_Create(&odb_convert);
    if ( m == NULL) {
        ModuleError = PyErr_NewException("Failed to create the module : odb4py.convert", NULL, NULL);
        Py_XINCREF(ModuleError) ;
        return NULL;
}
return m  ;
}

