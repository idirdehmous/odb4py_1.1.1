# odb4py
# Copyright (C) 2026 Royal Meteorological Institute of Belgium (RMI)
#
# Licensed under the Apache License, Version 2.0

# -*- coding: utf-8 -*-

from  .parser        import SqlParser
from  .odb_ob        import OdbObject
from  .odb_glossary  import OdbLexic
from  .setting_env   import OdbEnv


__all__=[ "SqlParser", "OdbObject" , "OdbLexic", "OdbEnv" ]


# Init once
if not globals().get("_ODB_ENV_INITIALIZED", False):
    _env = OdbEnv()
    _env.init()
    _ODB_ENV_INITIALIZED = True

