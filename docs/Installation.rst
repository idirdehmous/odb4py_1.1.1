Installation 
============

.. code-block:: bash

   Clone the source code from github repository:
       git clone https://github.com/idirdehmous/pyodb_1.1.0


   The second step consists in the build and installation of the ODB library. 

       mkdir -p build_odb
       cd build_odb


       ODB_HOME=/path/to/../../odb/install/dir

       The main CMakefiles.txt is inide the python package. It compiles the odb libraries,binaries and copies the necessary include files. 

       cmake -DCMAKE_INSTALL_PREFIX=${ODB_HOME} /path/to/../../pyodb_1.1.0
       make -j ncpu    # Set npcu according to your machine ,  2,4, etc  
       make install


   Once the ODB is built and installed, the following envirnement variables have to be added into the .bashrc file
       
       
       export ODB_INSTALL_DIR=/path/to/../odb/install/dir         # To automatically load the libodb.so when any C/python function is invoked 
       export PATH=${PATH}:/path/to/../../odb/install/dir/bin     # Contains the odb98.x, dcagen that run in background to parse the SQL statement and fetch the data.
       export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/../../odb/install/dir/lib  


.. note::
   The odb directory contained inside the package is not the same as the one provided by the odb_bundle of ECMWF. The user has to careful and make sure that the python scripts are using the libodb*.so and the binaries compiled inside the pyodb package directory. 
