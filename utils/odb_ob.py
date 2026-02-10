#-*- coding; utf-8 -*-  
import sys , os 
import re 
from   glob         import glob  


__all__=["OdbObject"]


from   .parser       import StringParser
from   .exceptions   import *
from   .odb_glossary import OdbLexic 


# INSTANTIATE 
m=odb4pyErrMessage()
msg_err=m._ErrMsg()

lex=OdbLexic ()



class OdbObject:
      def __init__(self, path ):
          if os.path.isdir( path ):
             self.path =  path 
          else:
             print( msg_err[8].format(path) )
             sys.exit(0)
          return None

      @classmethod 
      def CheckExist (self,obj):
          # FOR PATH 
          if isinstance ( obj, str ):
             if os.path.isdir ( obj  ):
                return True 
             else:
                return False 

      @classmethod        
      def GetBasename(self, path):
         self.path = path 
         # MUST HANDLE ANY ODB NAME , TODo ! (WE stay with ECMA CCMA for the moment ! )
         ecma_found = re.search  ( "ECMA" , self.path   )
         ccma_found = re.search  ( "CCMA" , self.path   )
         if ecma_found :
            dbtype = os.path.basename(os.path.normpath(path) )[0:4]
            dbname = os.path.basename(os.path.normpath(path) )
            return dbtype , dbname
#         else:
#            print("ODB path endpoint should be ECMA.<obstype>, got argument :" , self.path ) 
#            sys.exit (0)

         elif ccma_found  :
            dbtype = os.path.basename(os.path.normpath(path) )[0:4]
            dbname = os.path.basename(os.path.normpath(path) )
            return dbtype , dbname 

         else:
            print("ODB path endpoint should be ECAM.<obstype> OR CCMA, but got argument :" , self.path )
            sys.exit (0)


      @classmethod
      def HasFlags(self, path, type_):
          if self.CheckExist(path ):
            flag_file=path+"/"+type_+".flags"
            if os.path.isfile (flag_file):
               return flag_file
            else:
               return None 



      @classmethod
      def HasIoassign(self, path, type_):
          if self.CheckExist(path ):
             io_file  =path+"/IOASSIGN"             # IOASSIGN FILE 
             if os.path.isfile( io_file ):
                return io_file 
             sym_file =path+"/"+type_+".IOASSIGN"   # SYM LINK  EXISTS AFTER RUNNING   dcagen or odbtools.x 
             if os.path.islink ( sym_file ):
                return sym_file 

      @classmethod
      def HasIomap (self , path , type_):
          if self.CheckExist(path ):
             iomap_file  =path+"/"+type_+".iomap"
             if os.path.isfile(iomap_file ):
                return iomap_file
             else:
                return None 


      @classmethod
      def GetSize(self , path):
          total=0
          with os.scandir(path) as it:
               for entry in it:
                   if entry.is_file():
                      total += entry.stat().st_size
                   elif entry.is_dir():
                     total += self.GetSize(entry.path)
          # TOTAL IN Bytes
          return total 


      @classmethod       
      def GetPools(self, path ):
          pools=[]
          if self.CheckExist ( path ):
             content=os.listdir(path) 
             regex=r"^(?:0|[1-9]\d{0,2})$"
             for item in content:
                 pattern =re.findall( regex, item  )  
                 if len(pattern ) != 0:
                    for i in range(len(pattern)):
                        pools.append(pattern[i])

          return pools 


      @classmethod 
      def GetTables(self, path):
          # GET ONLY THE COMMON TABLES in POOLS 
          # THE TABLES ARE MOSTLY IN POOL 1 
          # GOING FURTHER , SHOW THE TABLES OF ALL POOLS IS SHOWING "almost" THE ODB COMPLETE SCHEMA (later )!
          poolmask = self.GetPools(path)
          dbtabs   = lex.odbTables()
          current_tables =[]
          for p in poolmask:
              if p == "1" and os.path.isdir(path+"/"+p):
                 tb=os.listdir(path+"/"+p  )
                 for t in tb:
                     current_tables.append(t)
                 
          return sorted(current_tables )



      def GetAttrib (self):
          self.attrs_={}
          if not self.CheckExist(self.path):
              print(msg_err[8].format(  self.path ))
              sys.exit (0)
          else:
             type_ , name = self.GetBasename ( self.path  ) 
             obstype= name.replace( type_+".", "" )
             self.attrs_["type"]   =type_
             self.attrs_["name"]   =name
             self.attrs_["obstype"]=obstype
          
          # LOOK FOR Data Description File ( OFTEN ECMA.dd or CCMA.dd )  
          if self.CheckExist( self.path ):                                                                       
             ddfile  =glob(self.path+"/*.dd"  )
             schfile =glob(self.path+"/*.sch" )
    
             if   len(ddfile) == 0:
                print( msg_err[6].format( type_ ) )

             elif len(schfile) ==0:
                print( msg_err[7].format( type_ ) )
                sys.exit(0)
             else:
                type_ , name = self.GetBasename ( self.path )
                file_=open( self.path+"/"+type_+".dd" , "r" )    # READ .dd FILE 
                lines=file_.readlines()[:6]
                odb_vers   =lines[0].rstrip()
                creat_date =lines[1].rstrip()
                modif_date =lines[2].rstrip()
                obs_dttm   =lines[3].rstrip()
                npool      =lines[4].rstrip()
                ntables    =lines[5].rstrip()

                db_size =self.GetSize     (self.path )

                poolmask=self.GetPools    (self.path)   
                flags   =self.HasFlags    (self.path, type_  )
                ioassign=self.HasIoassign (self.path, type_  )
                iomap   =self.HasIomap    (self.path, type_  )

                tabs=self.GetTables ( self.path   )

                self.attrs_.update({
                                    "date_creation"           :creat_date,
                                    "last_modification"       :modif_date,
                                    "Observation_date"        :obs_dttm,
                                    "number_pools"            :npool,
                                    "number_of_considered_tables" :ntables,
                                    "odb_total_size"          : db_size ,
                                    "poolmask"                : poolmask,
                                    "tables"                  : tabs ,
                                    "ioassign_file"           : ioassign,
                                    "flags_file"              : flags,
                                    "iomap_file"              : iomap,
                                    "odb_software_release"    : odb_vers} )

          
          return self.attrs_ 



