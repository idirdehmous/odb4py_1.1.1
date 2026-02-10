# Changelog

All notable changes to this packages 

## [Unreleased]
odb4py_0.1.0


### Version 0.1.0
- v0.1.0 Simple read of the ODB1 rows (equivalent to a dump  )
- v0.1.0 The data types are checked and stored in C arrays but not returned to python 
- v0.1.0 Use of the C API macros such as PyList_New, PyDict_new , PyList_Append etc to get the data into python objects
- v0.1.0 The data values are returned to python layer in a list of columns and lists of rows  (Pylist [nrows x ncols])

### Version 0.1.1
- v0.1.1 The returned values can be processed after the fetch (in the python environment  ) 
- v0.1.1 Add some basic checks concerning the pointers and invalid data handling 


### Version 1.1.0
- v1.1.0 Introduce positional arguments in the C side with PyKeywordsParseArgs  function (more flexibily to add options )
- v1.1.0 Add basic options like: verbosity, get the header , column names etc 
- v1.1.0 Add a rows counter and compute the purcenbage of fetched rows (  pbar  = True ) ->  progress bar
- v1.1.0 Replace the PyList_New structures by numpy arrays from C/numpy  API. The fetch rows process became ~8 times faster 
- v1.1.0 More stable I/O by enhacing the io_module.c  (odbConnect and odbClose methods )
- v1.1.0 The odbFetch function was moved to odbDict to return a python dictionnary with colnames as keys. 

## [Release]
- VERSION 1.1.0
- v1.1.0 Date of release : 15/12/2025
- v1.1.0 Tested with ODBs from different ACCORD countries (RMI , CHMI , CHMZ and the global ARPEGE ODBs )
- v1.1.0 A module called geo_module.c was added. Contains functions to compute the great circle distances from latlon pairs (25-11-2025 )
- v1.1.0 Compilation using many linux and cuibuildwheel + docker environmment  ( 10/02/2026 )
- v1.1.0 First packaging on TestPyPi  
