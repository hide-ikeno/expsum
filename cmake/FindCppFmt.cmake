#.rst:
# FindCppFmt
# -------------
#
# Find the fmt, a open-source formatting library for C++
# (https://github.com/fmtlib/fmt).
#
# This module sets the following variables:
#
# ::
#
# CPPFMT_FOUND - set to true if cppformat library is found
# CPPFMT_INCLUDE_DIRS - list of required include directories
# CPPFMT_LIBRARIES - list of libraries to be linked

if (CPPFMT_INCLUDE_DIR AND CPPFMT_LIBRARIES)
  set(CPPFMT_FIND_QUIETLY TRUE)
endif ()

find_path(CPPFMT_INCLUDE_DIR
  NAMES fmt/fmt.h
  PATHS /opt/local/include /sw/include $ENV{CPPFMT_ROOT} ${CPPFMT_ROOT}
  DOC "the directory where fmt/format.h resides"
  )

find_library(CPPFMT_LIBRARY
  NAMES fmt
  PATHS /opt/local/lib /sw/lib $ENV{CPPFMT_ROOT}/lib ${CPPFMT_ROOT}/lib
  )

# Handle the QUIETLY and REQUIRED arguments and set NFFT_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CPPFMT
  DEFAULT_MSG CPPFMT_INCLUDE_DIR CPPFMT_LIBRARY)

if (CPPFMT_FOUND)
  set(CPPFMT_INCLUDE_DIRS ${CPPFMT_INCLUDE_DIR})
  set(CPPFMT_LIBRARIES ${CPPFMT_LIBRARY})
endif()

# Hide internal variables
mark_as_advanced(CPPFMT_INCLUDE_DIR CPPFMT_LIBRARY)
