#!/usr/bin/env ksh


# Set the path to find  ODB scripts and binaries 
THIS_FILE=$(realpath "${.sh.file}")
BIN_DIR=${THIS_FILE%/*}

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) PATH="${PATH}:${BIN_DIR}" ;;
esac

# PATH 
export PATH
# The Other vars 
export  ODB_BEBINPATH=${BIN_DIR}
export  ODB_FEBINPATH=${BIN_DIR}
export  ODB_SYSPATH=${BIN_DIR}

