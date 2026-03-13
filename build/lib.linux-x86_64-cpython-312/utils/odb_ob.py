# odb4py
# Copyright (C) 2026 Royal Meteorological Institute of Belgium (RMI)
#
# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
import sys , os
import re
from   glob         import glob


from   .parser       import SqlParser
from   .exceptions   import *
from   .odb_glossary import OdbLexic



class OdbObject:

    def __init__(self, path):
        if not os.path.isdir(path):
           raise FileNotFoundError ("Path to ODB not found ")
           sys.exit(0)
        self.path   = path
        self.attrs = None 



    def get_basename(self):
        base = os.path.basename(os.path.normpath(self.path))
        if base.startswith("CCMA"):
           cma_type = base 
           db_name  = base 
           return (db_name , cma_type , None    )
        elif base.startswith("ECMA"):
           db_name  = base 
           cma_type = base[:4]
           obs_type = base[5:]
           return (db_name , cma_type , obs_type )
        else:            
           print("ODB endpoint unknown. It should be ECMA.<obstype> OR CCMA :", self.path)
           sys.exit(0)


    # ---------------------
    def has_flag(self, type_):
        flag = os.path.join(self.path, f"{type_}.flags")
        if os.path.isfile(flag):
           return flag  
        else:
           return None


    def has_ioassign(self, type_):
        io_file = os.path.join(self.path, "IOASSIGN")
        if os.path.isfile(io_file):
           return io_file
       
        sym = os.path.join(self.path,   f"{type_}.IOASSIGN")
        if not os.path.islink(sym):
           os.symlink( sym   )
           if not os.path.islink( "/".join(   (self.path, f"IOASSIGN.{type_}")  )):
              os.symlink( self.path    ,   f"IOASSIGN.{type_}" )
              return sym
           return sym
        else:
           return sym


    def has_iomap(self, type_):
        iomap = os.path.join(self.path, f"{type_}.iomap")
        if os.path.isfile(iomap):
           return iomap 
        else: 
           return None

    def get_size(self, path=None):
        """
        Equivalent to bash command : du --apparent-size --total -B1   ../../CCMA  or ECMA.obstype
        Count the file size using st_size and not with file blocks on the disk .
        """
        if path is None:
            path = self.path
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                   total += entry.stat().st_size
                elif entry.is_dir():
                   total += self.get_size(entry.path)
        return total


    # ---------------------
    def get_pools(self):
        pools = []
        regex = r"^(?:0|[1-9]\d{0,2})$"
        for item in os.listdir(self.path):
            if re.match(regex, item):
                pools.append(item)
        return sorted(pools)


    # ---------------------
    def get_tables(self):
        tables = []
        for p in self.get_pools():
            if p == "1":
                pooldir = os.path.join(self.path, p)
                if os.path.isdir(pooldir):
                    tables.extend(os.listdir(pooldir))
        return sorted(tables)


    # Get attributes 
    def _load_attributes(self):
        attrs = {}
        dbname , type_, obstype = self.get_basename()
        attrs.update({ "name": dbname , 
                       "type": type_,    
                       "obstype": obstype,   })

        ddfile  = glob(os.path.join(self.path, "*.dd"))
        if not ddfile:
           raise FileNotFoundError ( type_+".dd  file not foud." )
     
   
        dd_path = os.path.join(self.path, f"{type_}.dd")
        with open(dd_path) as f:
             lines = f.readlines()[:6]

        odb_vers, creat_date, modif_date, obs_dttm, npool, ntables =  [l.rstrip() for l in lines]
        size_unit =" (Bytes)"
        attrs.update({
        "date_creation"     : creat_date,
        "last_modification" : modif_date,
        "observation_date"  : obs_dttm,
        "number_pools"      : npool,
        "odb_total_size"    : str(self.get_size()) + size_unit ,
        "Poolmask"          : self.get_pools(),
        "tables"            : self.get_tables(),
        "ioassign_file"     : self.has_ioassign(type_),
        "flags_file"        : self.has_flag(type_),
        "iomap_file"        : self.has_iomap(type_),
        "number_of_considered_tables"  : ntables,
        "odb_software_release"  : odb_vers,
            })
        return attrs


    def get_attrib(self):
        self.attrs = self._load_attributes()
        return self.attrs
