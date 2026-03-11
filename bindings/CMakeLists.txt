# CMakeLists.txt : Compile the C/Python  extension 

execute_process( COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NumPy_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE )

message(STATUS "NumPy include dir: ${NumPy_INCLUDE_DIR}")



# Create the  odb4py  C extension   
add_library(   core    MODULE  main.c  )

# Inclure les headers Python + headers ODB
target_include_directories(core  PRIVATE  ${Python3_INCLUDE_DIRS}     
	                                  ${NumPy_INCLUDE_DIR}     
					  ${CMAKE_SOURCE_DIR}/include 
  					 )

# Use only the python modules 
target_link_libraries(core PRIVATE Python3::Module 
	                           odb_lib  
				   )

# Module output name  (.so) to be imported in python 
set_target_properties(core  PROPERTIES
	OUTPUT_NAME  "core"
        PREFIX ""                
        BUILD_RPATH "\$ORIGIN/lib"
        INSTALL_RPATH "\$ORIGIN/lib"
)

# Install  the module and the other pure python modules 
install( TARGETS core  LIBRARY DESTINATION  . )
install( DIRECTORY ${CMAKE_SOURCE_DIR}/utils/    DESTINATION utils   FILES_MATCHING PATTERN "*.py" )

