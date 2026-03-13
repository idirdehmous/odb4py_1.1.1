# odb4py
# Copyright (C) 2026 Royal Meteorological Institute of Belgium (RMI)
#
# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-

__all__ =[  "odb4pyErrMessage"        ,
            "odb4pyBinError"          ,
            "odb4pyLibError"          ,
            "odb4pyWarning"           ,
            "odb4pyInterfaceError"    ,
            "odb4pyDatabaseError"     ,
            "odb4pyProgrammingError"  ,
            "odb4pyIntegrityError"    ,
            "odb4pyDataError"         ,
            "odb4pyNotSupportedError" ,
            "odb4pyPathError"         ,
            "odb4pyEnvError"          ,
            "odb4pyInternalError"     ,
            "odb4pyInstallError" ]




class odb4pyErrMessage:
      def __init__(self):
          return None 
      def _ErrMsg(self):
         msg_list=[ 
       "\n--Path to ODB_INSTALL_DIR is {} is 'None' or not found!\n--Please"+ \
               " export ODB_INSTALL_DIR=/path/../../ where libodb.so is installed",                         # 0

       "\n--Can't find the path to the ODB binaries\n--Set the ODB_INSTALL_DIR nvironement variable",          # 1

       "\n--Something went wrong while loading libodb.so\n--Try to check  the ODB_INSTALL_DIR environment variable",# 2 

       "\n--cdll module can't find shared library libodb.so\n--Please check"+ \
            "that it has been installed  and ODB_INSTALL_LIB var is set",                                   # 3

       "\n--The path {} not found !",                                                                       # 4

       "\n--The ODB env variable{}name not recongnized !",                                                  # 5

       "\n--The ODB doesn't contain .dd   file. Must be : {}.dd",                                           # 6

       "\n--The ODB doesn't contain .sch  file. Must be : {}.sch",                                          # 7

       "\n--{} path not found !"                                                                            # 8
            ]
         return msg_list

class odb4pyBinError(Exception):
    """
    ODB needed bin executables 
    """
    pass 
class  odb4pyLibError(Exception):
    """
    ODB needed libraries 
    """
    pass 

class  odb4pyWarning(Exception):
    """
    Some warnings messages if they exist
    """
    pass 

class  odb4pyInterfaceError(Exception):
    """
    C/Python API communication errors 
    """
    pass


class  odb4pyDatabaseError(Exception):
    """
    ODB internal structure error 
        corrupted odb 
    """
    pass 

class  odb4pyProgrammingError(Exception):
    """
    Programming Error sepcially from the backend side (C language)
    """
    pass 
class  odb4pyIntegrityError(Exception):
    """
    ODB data integrity error 
    """
    pass 

class  odb4pyDataError(Exception):
    """
    Arrays length , data types  etc  
    """
    pass 
class  odb4pyNotSupportedError(Exception):
    """
    Versions and modeules 
    """
    pass 

class odb4pyPathError (Exception):
    """
    Checks path and raises Errors !
    """
    pass 
class odb4pyEnvError(Exception):
    """
    Raises error if a non environmental odb variables is set !
    """ 
    pass 
class odb4pyInternalError(Exception):
    """
    Raises error if a problem is found inside the C code or Python/C communication !
    """
    pass
class odb4pyInstallError(Exception):
    """
    Raises an error if something wrong occured during building the modules
    """
    pass 
