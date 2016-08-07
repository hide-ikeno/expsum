# -*- mode: cmake; indent-tabs-mode-nil; -*-
project(expsum C CXX Fortran)

cmake_minimum_required (VERSION 2.8.8)

option(expsum_build_test "Build tests for expsum" ON)


# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. (you may need to remove CMakeCache.txt ")
endif()

#===============================================================================
# Project version info
#===============================================================================

# Project include directories, exectable output path, etc.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# Additional module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#===============================================================================
# Compiler information
#===============================================================================
include(CXXCompilerSetup)
# # Require threads
# set(THREADS_PREFER_PTHREAD_FLAG ON)
# find_package(Threads REQUIRED)

#===============================================================================
# Python Interpreter
#===============================================================================
find_package(PythonInterp)

if (NOT PYTHONINTERP_FOUND)
  message(FATAL_ERROR "Cannot find python interpreter.")
endif()

#===============================================================================
# REQUIRED external packages
#===============================================================================

# External package directory
# set(EXTERNAL_PROJECT_DIR ${PROJECT_SOURCE_DIR}/external)
# add_subdirectory(external)
# include_directories(${EXTERNAL_PROJECT_DIR}/include)
# link_directories(${EXTERNAL_PROJECT_DIR}/lib64)
# link_directories(${EXTERNAL_PROJECT_DIR}/lib)
# 
# include( ExternalProject )

# # Boost
# set(Boost_USE_STATIC_LIBS       OFF)
# set(Boost_USE_MULTITHREADED      ON)
# set(Boost_USE_STATIC_RUNTIME    OFF)
# 
# find_package(Boost COMPONENTS python)
# if(Boost_FOUND)
#   include_directories(${Boost_INCLUDE_DIRS})
# else()
#   message(FATAL_ERROR "Cannot find boost libraries. Project will not be built.")
# endif()


# # Boost NumPy --- very fast C++ logging library
# ExternalProject_Add(
#   Boost.NumPy
#   PREFIX ${EXTERNAL_PROJECT_DIR}
#   GIT_REPOSITORY https://github.com/ndarray/Boost.NumPy.git
#   GIT_TAG "master"
#   CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECT_DIR}
#   TEST_COMMAND ""
#   )


# BLAS/LAPACK
# BLAS / LAPACK
set(BLAS_LAPACK_FOUND FALSE)
set(EXPSUM_USE_MKL FALSE)
if (NOT BLAS_LAPACK_FOUND)
  set(MKL_SDL ON)
  find_package(MKL QUIET)
  if (MKL_FOUND)
    set(BLAS_LAPACK_FOUND TRUE)
    set(BLAS_LAPACK_LIBRARIES ${MKL_MINIMAL_LIBRARIES})
    set(BLAS_LAPACK_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    set(CBLAS_INCLUDE_FILE   "mkl_cblas.h")
    set(LAPACKE_INCLUDE_FILE "mkl_lapacke.h")

    set(EXPSUM_USE_MKL TRUE)
  endif()
endif()

if (NOT BLAS_LAPACK_FOUND)
  find_package(OpenBLAS QUIET)
  if (OpenBLAS_FOUND)
    set(BLAS_LAPACK_FOUND TRUE)
    set(BLAS_LAPACK_LIBRARIES ${OpenBLAS_LIBRARIES})
    set(BLAS_LAPACK_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIRS})
    set(CBLAS_INCLUDE_FILE "cblas.h")
    set(LAPACKE_INCLUDE_FILE "lapacke.h")
  endif()
endif()

if (BLAS_LAPACK_FOUND)
  include_directories(${BLAS_LAPACK_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Cannot find BLAS/LAPACK/CBLAS/LAPACKE. project will not be built.")
endif()

# find_package(LAPACK)
# if (LAPACK_FOUND)
#   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LIKER_FLAGS}")
# else ()
#   message(FATAL_ERROR "Cannot find BLAS/LAPACK libraries. Project will not be built.")
# endif()

# Armadillo -- a C++ linear algeblra library
find_package(Armadillo)
if (ARMADILLO_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DARMA_USE_CXX11")
  include_directories(${ARMADILLO_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Cannot find Armadillo C++ libraries. Project will not be built.")
endif()

# FFTW3
find_package(FFTW)
if (FFTW_FOUND)
  include_directories(${FFTW_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Cannot find FFTW3 libraries. Project will not be built.")
endif()

#===============================================================================
# Sub-directories
#===============================================================================
add_subdirectory(example)
# add_subdirectory(python)

# Testing
# include(CTest)
if (expsum_build_test)
  enable_testing() # must be called from the root CMakeLists
  add_subdirectory(test)
endif()
