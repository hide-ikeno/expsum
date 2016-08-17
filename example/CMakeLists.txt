macro(add_expsum_example target)
  add_executable(${target} ${target}.cpp)
  # target_link_libraries(${target} ${ARMADILLO_LIBRARIES} ${LAPACK_LIBRARIES} ${FFTW_LIB})
  target_link_libraries(${target} ${ARMADILLO_LIBRARIES} ${BLAS_LAPACK_LIBRARIES} ${FFTW_LIB})
endmacro()

add_expsum_example(approx_pow)
#add_expsum_example(complex_fit)
add_expsum_example(esprit)
add_expsum_example(fast_esprit)

