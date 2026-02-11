# -*- coding: utf-8 -*-
import os , sys 
from   ctypes import cdll  , CDLL 

# Check the module import first 
try:
    import odb4py
except ImportError as e:
    print("Failed to import the module odb4py, Please check its installation!", e)
    sys.exit(0)


# The module path check  
mod_path  =  odb4py.__path__[0]
if not os.path.isdir (mod_path):
   print( "Path to the odb4py module not found, Please check that it's correctly installed" )
   sys.exit(0)


# Necessary bins 
binlist=["dcafix"    ,
         "dcagen"    ,
         "dcagen.x"  ,
         "dcaquick"  ,
         "ioassign"  ,
         "odb98be.x" ,
         "odb98.x"   ,
         "odb_prune" ,
         "odbprune"  ,
         "create_ioassign"]

# Necessary libs  (Not all odb libs)
liblist=["libodbdummy.so"   ,
         "libodbec.so"      ,
         "libodbifsaux.so"  ,
         "libodb.so"        ,
         "libodbsqlcompiler.so" ]


# Path to bin , lib  
bindir = mod_path +"/bin"
libdir = mod_path +"/lib"
libname="libodb.so"        


class OdbEnv:
      OdbVars= { "ODB_IO_METHOD"      :"" ,
                 "ODB_CMA"            :"" ,
                 "ODB_STATIC_LINKING" :"",
                 "ODB_SRCPATH_ECMA"   :"" ,
                 "ODB_DATAPATH_ECMA"  :"" ,
                 "ODB_SRCPATH_CCMA"   :"" ,
                 "ODB_DATAPATH_CCMA"  :"" ,
                 "ODB_IDXPATH_ECMA"   :"" ,
                 "IOASSIGN"           :"",
                 "NPES_CCMA"          :"",
                 "SWAPP_ODB_IOASSIGN" :"",
                 "ODB_CCMA_CREATE_POOLMASK":"",
                 "ODB_CCMA_POOLMASK_FILE":"",
                 "TO_ODB_FULL"        :"",
                 "ODB_REPRODUCIBLE_SEQNO":"",
                 "ODB_CTX_DEBUG"      :"",
                 "ODB_REPRODUCIBLE_SEQNO":"",
                 "ODB_CATCH_SIGNALS"  :"",
                 "ODB_TRACE_PROC"     :"",
                 "ODB_TRACE_FILE"     :"" ,
                 "ODB_CONSIDER_TABLES":""}

      def __init__(self):
          pass 

      def InitEnv(self ):
          if os.path.isdir (libdir):
             lib=CDLL ("/".join(    (libdir, libname)    ))


          if os.path.isdir (bindir):
             new_path = os.environ.copy()["PATH"]+":"+bindir
             os.environ["PATH"] = new_path

          odb_bin={
                    "ODB_FEBINPATH"  :bindir,
                    "ODB_SYSPATH"    :bindir,
                    "ODB_BEBINPATH"  :bindir,
                    "ODB_BINPATH"    :bindir
                    }

          # EXPORT 
          if bindir is not None:
             for k , v in odb_bin.items():
                 if os.path.isdir( v ):
                    os.environ[k] = v
          return None 


      def CheckBin(self):
          not_found = []
          for bin_ in binlist:
              if bin_  not in binlist:
                 not_found.append( bin_ )
          if len(not_found )  ==0:
             return (True, "Found ALL needed binaries" )
          else:
             return (False , "Binaries ", not_found )


      def CheckLib(self):
          not_found = []
          for lib_ in liblist:
              if lib_  not in liblist:
                 not_found.append( lib_ )
          if len(not_found )  ==0:
             return (True, "Found ALL needed runtime libraries" )
          else:
             return (False, "Library not found ", not_found )


      def UpdateVar(self, env_vars):
          """
          Update ODB environment variables.
          Parameters :
          env_vars   : Dictionary of ODB_* variables to set.
          """
          if not isinstance(env_vars, dict):
             raise TypeError("env_vars must be a dictionary")

          for key, value in env_vars.items():
              if not isinstance(key, str):
                 raise ValueError("Environment variable names must be strings")

              # Allow only ODB_* variables by default
              if not key.startswith("ODB_") and key not in list(self.ODB_VARS.keys()):
                 raise ValueError(f"Invalid ODB environment variable: {key}")
              os.environ[key] = str(value)


      def GetVar (self, key, default=None):
          """
          Get an ODB environment variable.
          """
          return os.environ.get(key, default)


      def DumpVar(self):
          """
          Return all currently defined ODB_* environment variables.
          """
          return { k: v for k, v in os.environ.items()            if k.startswith("ODB_")  }
