#.rst:
#
# FindFFTW3
# ---------
#
# Find the FFTW3 library
#
# This module finds the FFTW3, a C subroutine library for computing the discrete
# Fourier transform (DFT) Fourier transforms (see http://www.fftw.org/).
# This module sets the following variables:
#
# ::
#
# FFTW_FOUND - set to true if FFTW library is found
# FFTW_INCLUDE_DIRS - list of required include directories
# FFTW_LIBRARIES - list of libraries to be linked

# if (FFTW_INCLUDES AND FFTW_LIBRARIES)
#   set(FFTW_FIND_QUIETLY TRUE)
# endif (FFTW_INCLUDES AND FFTW_LIBRARIES)

find_path(FFTW_INCLUDE_DIR
  NAMES fftw3.h
  PATHS $ENV{FFTW_ROOT}/include ${FFTW_ROOT}/include
  )

find_library(FFTWF_LIB
  NAMES fftw3f
  PATHS $ENV{FFTWDIR} ${LIB_INSTALL_DIR}
  )
find_library(FFTW_LIB
  NAMES fftw3
  PATHS $ENV{FFTWDIR} ${LIB_INSTALL_DIR}
  )
find_library(FFTWL_LIB
  NAMES fftw3l
  PATHS $ENV{FFTWDIR} ${LIB_INSTALL_DIR}
  )

set(FFTW_LIBRARY "${FFTWF_LIB} ${FFTW_LIB}" )

if(FFTWL_LIB)
  set(FFTW_LIBRARY "${FFTW_LIBRARY} ${FFTWL_LIB}")
endif()

# Handle the QUIETLY and REQUIRED arguments and set NFFT_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW
  DEFAULT_MSG FFTW_INCLUDE_DIR FFTW_LIBRARY
  )

if(FFTW_FOUND)
  set(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR})
  set(FFTW_LIBRARIES ${FFTW_LIBRARY})
endif()

# Hide internal variables
mark_as_advanced(FFTW_INCLUDE_DIR FFTW_LIBRARY)
