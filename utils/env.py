# -*- coding: utf-8 -*-
import os, sys  
from  ctypes import cdll  

from .exceptions   import  *  


m=odb4pyErrMessage()
msg_err=m._ErrMsg()


__all__ = ["OdbEnv"]


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



      def __init__ (self , odb_basedir , odb_libname):
         self.basedir=odb_basedir
         self.libname=odb_libname 
         self.libdir =self.basedir+"/lib"  #/"+self.libname 
         if self.basedir != None:
            if os.path.isdir ( self.basedir ):
               self.bindir =self.basedir+"/bin"
               self.libdir =self.basedir+"/lib"
         else:
            print(msg_err[0].format(self.basedir)  )
            sys.exit(0)




      def InitEnv(self ):
         try:
             cdll.LoadLibrary(self.libdir+"/"+self.libname  )  
         except:
             OSError
             raise odb4pyLibError(msg_err[2].format(self.libdir ) ) 
         
         syspath=os.environ.copy()
         path=syspath["PATH"]+":"+self.bindir
         odb_bin={
                    "PATH"           :path , 
                    "ODB_FEBINPATH"  :self.bindir,
                    "ODB_SYSPATH"    :self.bindir,
                    "ODB_BEBINPATH"  :self.bindir,
                    "ODB_BINPATH"    :self.bindir
                    }
        
         # EXPORT 
         for k , v in odb_bin.items(): 
             try:
                if os.path.isdir( v ):
                   os.environ[k] = v
             except:
                raise odb4pyPathError(msg_err[3].format( v ))                
         return None 
      
      def OdbUpdateEnv(self, EnvVars =None ):
          if EnvVars != None :
             for k , v in EnvVars.items(): 
                 try:
                    os.environ[k] = v
                 except:
                    raise odb4pyEnvError (err_msg[5].format( k  )  )

