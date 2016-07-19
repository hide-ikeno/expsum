// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

#include "expsum/vandermonde_matrix.hpp"

using size_type      = arma::uword;
using real_type      = double;
using complex_type   = std::complex<real_type>;
using real_matrix    = arma::Mat<real_type>;
using complex_matrix = arma::Mat<complex_type>;

template <typename T>
void test_vandermonde_matvec(size_type nrows, size_type ncols)
{
    using vandermonde_matrix = expsum::vandermonde_matrix<T>;
    using vector_type        = typename vandermonde_matrix::vector_type;
    using real_vector_type   = arma::Col<real_type>;

    const size_type niter = 100;

    arma::arma_rng::set_seed_random();

    vector_type h(ncols, arma::fill::randu);
    vandermonde_matrix matV(nrows, h);

    auto V = matV.as_dense_matrix();

    vector_type x1(ncols);
    vector_type x2(nrows);
    vector_type y1(nrows);
    vector_type y2(nrows);
    vector_type y3(ncols);
    vector_type y4(ncols);
    real_vector_type r(nrows);

    for (size_type i = 0; i < niter; ++i)
    {
        std::cout << "--- Trial " << i << '\n';
        x1.randn();
        y1.zeros();
        matV.apply(x1, T(), y1);
        y2          = V * x1;
        r           = arma::abs(y1 - y2);
        auto maxerr = arma::max(r);
        auto rnorm  = arma::norm(r);
        std::cout << "A     * x: max|error| = " << maxerr
                  << ", |error| = " << rnorm << '\n';

        x2.randn();
        matV.apply_trans(x2, T(), y3);
        y4     = V.t() * x2;
        r      = arma::abs(y3 - y4);
        maxerr = arma::max(r);
        rnorm  = arma::norm(r);
        std::cout << "A.t() * x: max|error| = " << maxerr
                  << ", |error| = " << rnorm << '\n';
    }

    return;
}

template <typename T>
void test_ldlt_vandermonde_gramian(size_type nrows, size_type ncols)
{
    using vandermonde_matrix = expsum::vandermonde_matrix<T>;
    using vector_type        = typename vandermonde_matrix::vector_type;
    using matrix_type        = arma::Mat<T>;

    const size_type niter = 10;

    matrix_type mat(ncols, ncols);
    matrix_type work(ncols, 4);

    vector_type h(ncols);
    vandermonde_matrix matV(nrows, ncols);
    matrix_type gramian(ncols, ncols);
    matrix_type reconstructed(ncols, ncols);

    for (size_type i = 0; i < niter; ++i)
    {
        std::cout << "--- Trial " << i << '\n';
        h.randu();
        matV.set_coeffs(h);
        auto V  = matV.as_dense_matrix();
        gramian = V.t() * V;

        expsum::ldlt_vandermonde_gramian(matV, mat, work);

        vector_type d = mat.diag();
        mat.diag().ones();
        auto matL     = arma::trimatl(mat);
        reconstructed = matL * arma::diagmat(d) * matL.t();
        std::cout << "|A - L L^T| / |A| = "
                  << arma::norm(gramian - reconstructed) / arma::norm(gramian)
                  << std::endl;
    }
}

int main()
{
    std::cout << "# Test Vandermonde matrix-vector product:\n";

    std::cout << "# (50 x 50) real matrix\n";
    test_vandermonde_matvec<double>(50, 50);
    std::cout << "# (50 x 71) real matrix\n";
    test_vandermonde_matvec<double>(50, 71);
    std::cout << "# (71 x 50) real matrix\n";
    test_vandermonde_matvec<double>(71, 50);

    std::cout << "# (50 x 50) complex matrix\n";
    test_vandermonde_matvec<std::complex<double>>(50, 50);
    std::cout << "# (50 x 71) complex matrix\n";
    test_vandermonde_matvec<std::complex<double>>(50, 71);
    std::cout << "# (71 x 50) complex matrix\n";
    test_vandermonde_matvec<std::complex<double>>(71, 50);

    std::cout
        << "# Test Cholesky decomposition of the gramian of Vandermonde\n";
    std::cout << "# (100 x 20) double matrix\n";
    test_ldlt_vandermonde_gramian<double>(100, 20);
    std::cout << "# (100 x 20) complex matrix\n";
    test_ldlt_vandermonde_gramian<std::complex<double>>(100, 20);

    return 0;
}
