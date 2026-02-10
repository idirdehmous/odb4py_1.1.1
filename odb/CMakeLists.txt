cmake_minimum_required(VERSION 3.15..3.27 )

project(odb C Fortran)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake "${CMAKE_CURRENT_SOURCE_DIR}/../ecbuild/cmake")

if(NOT PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake PARENT_SCOPE)
endif()

include(ecbuild_system)
include(odb_add_schema)
include(odb_link_schemas)
include(odb_set_environment)

ecbuild_requires_macro_version(1.9)

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
  set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/module)
else()
  set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/odb/module)
endif()

ecbuild_declare_project()

ecbuild_add_option(FEATURE ODB_SHARED_LIBS
  DESCRIPTION "Build ODB project with shared libraries (overrides BUILD_SHARED_LIBS)"
  DEFAULT ${BUILD_SHARED_LIBS}
  ADVANCED)

set(BUILD_SHARED_LIBS ${ENABLE_ODB_SHARED_LIBS})

test_big_endian(BIG_ENDIAN)

if(NOT BIG_ENDIAN)
  add_definitions(-DLITTLE)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  add_definitions(-DLINUX)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Intel")
    set(CMAKE_C_FLAGS "-fPIC -DINTEL -traceback")
endif()


message ("*************************************************")
message (${CMAKE_C_COMPILER}         )
message (${CMAKE_CXX_COMPILER}       )
message (${CMAKE_CXX_COMPILER_AR}    )
message (${CMAKE_CXX_COMPILER_RANLIB})
message (${CMAKE_C_COMPILER_VERSION} )
message (${CMAKE_Fortran_COMPILER}   )
message (${CMAKE_Fortran_COMPILER_VERSION}  )
message ("*************************************************")


if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU"  )

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -w")
  

  # FOR gcc/gfortran  < 10.x.x 
  if    ( CMAKE_Fortran_COMPILER_VERSION  VERSION_GREATER  "10.0"  )
     message ("gfortran version is GREATER than 10.0")
     set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}  -fPIC -ffree-line-length-none  -fno-automatic   -fallow-argument-mismatch ")
 elseif ( CMAKE_Fortran_COMPILER_VERSION  VERSION_GREATER   "10.0" )
    message ("gfortran version is  LESS  than 10.0")
     set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}  -fPIC -ffree-line-length-none   -Wno-automatic   -Wallow-argument-mismatch ")

 endif ()

  set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -fcheck=bounds")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O2")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")

  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fPIC -traceback -fpe0 -names lowercase")
  set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -check bounds -check pointers")
  set(CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI")

  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fPIC -Ktrap=fp -noswitcherror")
  set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -Mbounds")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O2")

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Cray")

  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -hPIC -emf -M 226,867,7212")
  set(CMAKE_Fortran_FLAGS_DEBUG "-G0 -O0 -hdevelop")
  set(CMAKE_Fortran_FLAGS_RELEASE "-G2 -O2")
  set(CMAKE_C_FLAGS "-fPIC -hnomessage=186:192:223:236:7212:10082")
  set(CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")

endif()

find_package(Dl REQUIRED)
#find_package(PythonInterp REQUIRED)
find_package (Python3 COMPONENTS Interpreter Development)
find_program(ODB_AR NAMES sxar ar)
find_program(ODB_GZIP NAMES gzip)
find_program(ODB_GUNZIP NAMES gunzip)

file(GLOB schema_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src "src/ddl.[A-Z0-9]*")
string(REGEX REPLACE "ddl.([A-Z0-9]*)" "\\1" schemas "${schema_dirs}")
set(ODB_SCHEMAS ${schemas} CACHE STRING "List of pre-compiled ODB schemas")

get_directory_property(ODB_DEFINITIONS COMPILE_DEFINITIONS)

list(APPEND ODB_TPLS Dl)

set(ODB_INCLUDE_DIRS      ${CMAKE_CURRENT_SOURCE_DIR}/src/include ${CMAKE_Fortran_MODULE_DIRECTORY})
set(ODB_LIBRARIES   odb_lib)
set(ODB_CMAKE_DIR         ${CMAKE_CURRENT_SOURCE_DIR}/cmake CACHE INTERNAL "")

odb_set_environment(ODB_ENVIRONMENT ROOT         ${CMAKE_BINARY_DIR})
odb_set_environment(ODB_INSTALL_ENVIRONMENT ROOT ${CMAKE_INSTALL_PREFIX})

ecbuild_find_project_files()

add_subdirectory(cmake)
add_subdirectory(src)

ecbuild_install_project(NAME odb)
ecbuild_print_summary()
