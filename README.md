# expsum

Approximate a function by exponential sum

## Description



## Requirement

You need a newer C++ compiler that supports the C++11 standard, such as
GCC (>= 4.8.0) and Clang (>= 3.2).

This library also depends on the following external libraries:

 - [Armadillo: C++ linear algebra library](http://arma.sourceforge.net/)
 - [BLAS](http://www.netlib.org/blas/),
   [LAPACK](http://www.netlib.org/lapack/), or optimized BLAS/LAPACK libraries
   such as [Intel(R) MKL](https://software.intel.com/en-us/intel-mkl/),
   [OpenBLAS](http://www.openblas.net/),
   [Apple Accelerate Framework](https://developer.apple.com/library/mac/documentation/Accelerate/Reference/AccelerateFWRef/)
   etc.
- [FFTW3](http://www.fftw.org/)

## Install

`expsum` is a header only library. You can use it by including header files
under `include` directory.

To build the program using `expsum`, you must link armadillo, blas lapack and
fftw libraries.

## Usage
See, example program in `example` directory.

## Licence

Copyright (c) 2016 Hidekazu Ikeno

Released under the [MIT license](http://opensource.org/licenses/mit-license.php)


## References

