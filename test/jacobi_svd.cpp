#include <cassert>
#include <cmath>

#include <iomanip>
#include <iostream>

#include "expsum/jacobi_svd.hpp"

using size_type = arma::uword;

template <typename T>
void test_jacobi_svd(size_type nrows, size_type ncols)
{
    using real_type        = typename arma::get_pod_type<T>::result;
    using matrix_type      = arma::Mat<T>;
    using real_vector_type = arma::Col<real_type>;

    const auto tol = arma::Datum<real_type>::eps;
    matrix_type A(nrows, ncols);
    A.randu();

    matrix_type U(A);
    matrix_type V(std::min(nrows, ncols), ncols);
    real_vector_type sigma(std::min(nrows, ncols));

    expsum::jacobi_svd(U, sigma, V, tol);

    auto err = arma::norm(A - U * arma::diagmat(sigma) * V.t(), 2);
    std::cout << "Error = " << err << std::endl;

    // reference
    matrix_type refU, refV;
    real_vector_type ref_sigma;
    arma::svd_econ(refU, ref_sigma, refV, A, "both");

    std::cout << "singular values (computed, reference):\n";
    for (size_type i = 0; i < sigma.size(); ++i)
    {
        std::cout << std::setw(25) << sigma(i) << '\t' << std::setw(25)
                  << ref_sigma(i) << '\n';
    }
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    std::cout << "*** 10x10 real matrix" << std::endl;
    test_jacobi_svd<double>(20, 20);
    std::cout << "*** 20x10 real matrix" << std::endl;
    test_jacobi_svd<double>(50, 30);
    std::cout << "*** 20x10 complex matrix" << std::endl;
    test_jacobi_svd<std::complex<double>>(50, 20);

    return 0;
}
