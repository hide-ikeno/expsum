#include <iostream>
#include <iomanip>

#include <random>

#include "expsum/cholesky_cauchy.hpp"

using size_type      = arma::uword;
using real_type      = double;
using complex_type   = std::complex<double>;
using real_vector    = arma::Col<real_type>;
using complex_vector = arma::Col<complex_type>;
using real_matrix    = arma::Mat<real_type>;
using complex_matrix = arma::Mat<complex_type>;

void test_cholesky_quasi_cauchy(size_type n)
{
    // static const auto pi2 = 2 * arma::Datum<real_type>::pi;
    const real_type delta = 1.0e-14;
    std::random_device rnd;
    std::mt19937 gen(rnd());
    std::uniform_real_distribution<real_type> dist;

    complex_vector a(n, arma::fill::randu);
    complex_vector b(n, arma::fill::randu);
    complex_matrix P(n, n);
    for (size_type j = 0; j < n; ++j)
    {
        for (size_type i = 0; i < n; ++i)
        {
            P(i, j) = b(i) * std::conj(b(j)) / (a(i) + std::conj(a(j)));
        }
    }

    expsum::cholesky_cauchy_rrd<complex_type> chol;

    complex_matrix matL(chol.run(a, b, delta));

    complex_matrix P_rec(chol.reconstruct(matL));

    const auto norm_P = arma::norm(P);
    const auto resid  = arma::norm(P - P_rec);

    std::cout << "Test Cholesky factorization of quasi-Cauchy matrix:\n"
              << "  rank = " << matL.n_cols
              << ", |A - L D**2 L**H| / |A| = " << resid / norm_P << '\n';
}

int main()
{
    std::cout << "# Test Cholesky factorization of quasi-Cauchy matrix:\n";
    for (size_type i = 0; i < 20; ++i)
    {
        std::cout << "--- trial " << i + 1 << '\n';
        test_cholesky_quasi_cauchy(100);
    }
    return 0;
}

