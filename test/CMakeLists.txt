macro(add_test_numeric target)
  add_executable(test_${target} ${target}.cpp)
  target_link_libraries(test_${target} ${ARMADILLO_LIBRARIES} ${LAPACK_LIBRARIES} ${FFTW_LIB})
endmacro()

add_test_numeric(hankel_matrix)
add_test_numeric(vandermonde_matrix)
add_test_numeric(cholesky_cauchy)

