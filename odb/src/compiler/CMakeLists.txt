list(APPEND source_files

    cmd.c
    copyfile.c
    defines.c
    genc.c
    list.c
    regex.c
    tree.c
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${ODB_INCLUDE_DIRS})


ecbuild_generate_yy(YYPREFIX yy YACC yacc LEX lex DEPENDANT cmd.c odb98.c YACC_TARGET y.tab)
ecbuild_add_library(TARGET odbsqlcompiler   SOURCES ${source_files} lex.c y.tab.c  LIBS odb_lib)
ecbuild_add_executable(TARGET odb98.x SOURCES odb98.c LIBS odbsqlcompiler  )
#set_target_properties(odb98.x PROPERTIES LINKER_LANGUAGE Fortran) 
#target_link_libraries(odb98.x  gfortran quadmath)

set_target_properties(odb98.x PROPERTIES  LINKER_LANGUAGE Fortran  LINK_FLAGS "-Wl,--no-as-needed" )
target_link_libraries(odb98.x gfortran quadmath)

#if(CMAKE_Fortran_COMPILER_ID MATCHES PGI)
#  set_target_properties(odb98.x PROPERTIES LINKER_LANGUAGE Fortran LINK_FLAGS "-Mnomain")
#endif()

install(CODE "EXECUTE_PROCESS (COMMAND ${CMAKE_COMMAND} -E create_symlink
    odb98.x \${CMAKE_INSTALL_PREFIX}/bin/odb98be.x)")
