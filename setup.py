#-*-coding utf-8 -*- 
import os  , io, sys , re 
from   distutils.core import setup 



from   distutils.core import setup ,Extension
import sysconfig
from   distutils.sysconfig import customize_compiler, get_config_vars
from   Cython.Build import cythonize
from   distutils import log
import numpy as np  
from pathlib import Path

try:
    from Cython.Distutils import build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext



# VERSION 
__version__="1.1.0"



# PATH TO libodb.so
odb_install_dir = Path(__file__).parent.resolve()


# Compilation verbosity  !
log.set_verbosity(log.DEBUG)



class BuildModule:
    def __init__(self, name):
        self.name = name

    def Module(self, src, include):
        lib_dir = _auto_load_lib()
        nclib   ="/usr/local/apps/netcdf4/4.9.3/GNU/14.2/lib"
        print(f"Using ODB lib in: {lib_dir}")

        extra_compile_args = ["-O2" ,"-fPIC", "-Wall", "-Wextra", 
                      "-Wsign-compare", "-Waddress"     , 
                      "-Wunused-variable", "-v"  ]
        

        extra_link_args = [ f"-Wl,-rpath,{lib_dir}"      ]

        m = Extension(
            self.name,
            [src],
            include_dirs=[include, np.get_include(),  "/usr/local/apps/netcdf4/4.9.3/GNU/14.2/include"],
            library_dirs=[lib_dir, nclib ],
            libraries=["odb", "netcdf"],  # libodb.so   lodb  . Without lib  and .so  
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c",
        )
        return m


class NoSuffixBuilder(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        ext = os.path.splitext(filename)[1]
        return filename.replace(suffix, "") + ext




def _auto_load_lib():
    odblib_path = os.getenv("ODB_INSTALL_DIR")
    if odblib_path is not None:
       odblib_path = odblib_path +"/lib"
    else:
        #_thisdir_ = Path(__file__).parent.resolve()
        _thisdir  = Path( "/hpcperm/cvah/rmib_dev/odb4py_1.1.2")
        install_file = _thisdir  / "odb_install_dir"
        if install_file.exists():
            odb_install_path = install_file.read_text().strip()
            libpath = Path(odb_install_path) / "lib"
            libfile = libpath / "libodb.so"
            if libfile.exists():
                return str(libpath)
            else:
                raise FileNotFoundError(f"libodb.so not found in {libpath}")
        else:
            raise FileNotFoundError(f"{install_file} does not exist")
    return odblib_path



def read(path):
    file_path = os.path.join(os.path.dirname(__file__), *path.split("/"))
    return io.open(file_path, encoding="utf-8").read()



def parse_version_from(path):
    version_file = read(path)
    version_match = re.search(r'^__version__ = "(.*)"', version_file, re.M)
    if version_match is None or len(version_match.groups()) > 1:
        raise ValueError("couldn't parse version")
    return version_match.group(1)



# EXTENSION SUFFIX  (only .so)
sfx  =  sysconfig.get_config_var('EXT_SUFFIX')

# SOURCE AND INCLUDE
pyc_src="bindings"
include="include"


# INSTANTIATE  MODULES BY NAME !
cr   =BuildModule("core")
cf   =BuildModule("info")
cv   =BuildModule("convert")

core =cr.Module(   "./bindings/core.c"     , include   )
info =cf.Module(   "./bindings/info.c"     , include   )
conv =cv.Module(   "./bindings/convert.c"  , include  )


module_list=[  conv   ]

setup(  ext_modules = cythonize( module_list )     ,     
        cmdclass={"build_ext": NoSuffixBuilder}    
             )

quit()

