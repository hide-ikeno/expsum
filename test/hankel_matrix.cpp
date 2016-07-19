// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "expsum/hankel_matrix.hpp"

using size_type     = arma::uword;
using real_matrix    = arma::mat;
using complex_matrix = arma::cx_mat;

template <typename T>
void test_hankel_matvec(size_type nrows, size_type ncols)
{
    using MatVec           = expsum::hankel_gemv<T>;
    using vector_type      = arma::Col<T>;
    using real_vector_type = arma::Col<typename vector_type::pod_type>;

    const size_type niter = 10;

    vector_type h(nrows + ncols - 1, arma::fill::randn);
    MatVec matvec(nrows, ncols);

    matvec.set_coeffs(h);
    auto A = expsum::make_dense_hankel(nrows, ncols, h);

    vector_type x1(ncols);
    vector_type x2(nrows);
    vector_type y1(nrows);
    vector_type y2(nrows);
    vector_type y3(ncols);
    vector_type y4(ncols);
    real_vector_type r(nrows);

    for (size_type i = 0; i < niter; ++i)
    {
        x1.randu();
        matvec.apply(x1, T(), y1);
        y2          = A * x1;
        r           = arma::abs(y1 - y2);
        auto maxerr = arma::max(r);
        auto rnorm  = arma::norm(r, 2);
        std::cout << "A     * x( " << std::setw(2) << i
                  << "): max|error| = " << maxerr << ", ||error|| = " << rnorm
                  << '\n';

        x2.randu();
        matvec.apply_trans(x2, T(), y3);
        y4     = A.t() * x2;
        r      = arma::abs(y3 - y4);
        maxerr = arma::max(arma::abs(r));
        rnorm  = arma::norm(r, 2);
        std::cout << "A.t() * x( " << std::setw(2) << i
                  << "): max|error| = " << maxerr << ", ||error|| = " << rnorm
                  << '\n';
    }

    return;
}

template <typename T>
void test_hankel_fnorm(size_type nrows, size_type ncols)
{
    using vector_type     = arma::Col<T>;
    const size_type niter = 10;
    vector_type h(nrows + ncols - 1);

    for (size_type i = 0; i < niter; ++i)
    {
        h.randn();
        auto fnrm = expsum::fnorm_hankel(nrows, ncols, h);
        auto A    = expsum::make_dense_hankel(nrows, ncols, h);
        std::cout << "Test " << std::setw(2) << i << ": ||A||_F = " << fnrm
                  << ", error = " << std::abs(fnrm - arma::norm(A, "fro"))
                  << '\n';
    }
}

int main()
{
    std::cout << "# Test fast Hankel matrix-vector product:\n";

    std::cout << "  (50 x 50) real matrix\n";
    test_hankel_matvec<double>(50, 50);
    std::cout << "  (50 x 71) real matrix\n";
    test_hankel_matvec<double>(50, 71);
    std::cout << "  (71 x 50) real matrix\n";
    test_hankel_matvec<double>(71, 50);

    std::cout << "  (50 x 50) complex matrix\n";
    test_hankel_matvec<std::complex<double>>(50, 50);
    std::cout << "  (50 x 71) complex matrix\n";
    test_hankel_matvec<std::complex<double>>(50, 71);
    std::cout << "  (71 x 50) complex matrix\n";
    test_hankel_matvec<std::complex<double>>(71, 50);

    std::cout << "\n# Test frobenius norm general Hankel matrix:\n";

    std::cout << "  (50 x 50) real matrix\n";
    test_hankel_fnorm<double>(50, 50);
    std::cout << "  (50 x 71) real matrix\n";
    test_hankel_fnorm<double>(50, 71);
    std::cout << "  (71 x 50) real matrix\n";
    test_hankel_fnorm<double>(71, 50);

    std::cout << "  (50 x 50) complex matrix\n";
    test_hankel_fnorm<std::complex<double>>(50, 50);
    std::cout << "  (50 x 71) complex matrix\n";
    test_hankel_fnorm<std::complex<double>>(50, 71);
    std::cout << "  (71 x 50) complex matrix\n";
    test_hankel_fnorm<std::complex<double>>(71, 50);

    return 0;
}
