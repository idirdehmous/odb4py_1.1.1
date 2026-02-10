Installation 
============

.. code-block:: bash

   STEP 0 - download the source code:

       git clone https://github.com/idirdehmous/pyodb_1.1.0


   STEP 1 - FIRST, CREATE A BUILD DIRECTORY TO INSTALL THE ODB LIBRARIES.

       mkdir -p build_odb
       cd build_odb

       ODB_HOME=/path/to/../../odb/install/dir

       cmake -DCMAKE_INSTALL_PREFIX=${ODB_HOME} /path/to/../../pyodb_1.1.0
       make -j ncpu      # tested with max 4 CPUs
       make install


   Once the ODB is built and installed, add these environment variables 
   to your ~/.bashrc file:

       export ODB_INSTALL_DIR=/path/to/../../odb/install/dir
       export PATH=${PATH}:/path/to/../../odb/install/dir/bin
       export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/../../odb/install/dir/lib
