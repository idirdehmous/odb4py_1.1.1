Introduction 
============

**odb4py** is a C/Python interface designed to access ECMWF ODB1 databases.
It provides direct access to ODB tables, columns, data, and metadata
through SQL queries embedded in Python scripts.

The core of the project is implemented in pure C for performance and
reliability. The routines handling the ODB1 format have been derived and
pruned from the ECMWF ODB_API bundle `version 0.18.1 <https://www.ecmwf.int/sites/default/files/elibrary/2013/13861-using-odb-ecmwf.pdf>`_.
necessary components required to build the runtime libraries used by
odb4py.

The package is distributed as manylinux wheels and embeds the required
ODB runtime libraries, allowing installation via pip without requiring
an external ECMWF ODB installation.
