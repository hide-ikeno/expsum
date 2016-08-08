#include <iomanip>
#include <iostream>

#include <random>

#include "expsum/cholesky_quasi_cauchy.hpp"

using size_type = arma::uword;

template <typename T>
void test_cholesky_quasi_cauchy(size_type n)
{
    using cholesky_t       = expsum::cholesky_quasi_cauchy<T>;
    using real_type        = typename cholesky_t::real_type;
    using matrix_type      = typename cholesky_t::matrix_type;
    using vector_type      = typename cholesky_t::vector_type;
    using real_vector_type = typename cholesky_t::real_vector_type;

    const real_type delta = arma::Datum<real_type>::eps * std::sqrt(n);

    // vector_type alpha(n, arma::fill::randu);
    // vector_type gamma(n, arma::fill::randu);

    // vector_type a(arma::sqrt(alpha) / gamma);
    // vector_type b(arma::sqrt(arma::conj(alpha)));
    // vector_type x(T(1) / gamma);
    // vector_type y(arma::conj(-gamma));

    vector_type a(n, arma::fill::randu);
    vector_type b(arma::conj(a));
    vector_type x(n, arma::fill::randu);
    vector_type y(arma::conj(x));

    matrix_type P(n, n);
    for (size_type j = 0; j < n; ++j)
    {
        for (size_type i = 0; i < n; ++i)
        {
            P(i, j) = a(i) * b(j) / (x(i) + y(j));
        }
    }

    arma::uvec ipiv(n);
    vector_type work1(n);
    vector_type work2(n);
    auto rank = cholesky_t::pivot_order(a, b, x, y, delta, ipiv, work1);

    matrix_type X(n, rank);
    real_vector_type d(rank);

    cholesky_t::factorize(a, b, x, y, X, d, work1, work2);
    matrix_type P_rec(cholesky_t::reconstruct(X, ipiv, d));

    const auto norm_P = arma::norm(P);
    const auto resid  = arma::norm(P - P_rec);

    std::cout << "  rank = " << rank
              << ", |A - L * D^2 * L^H| / |A| = " << resid / norm_P
              << " (|A| = " << norm_P << ")" << '\n';
}

int main()
{
    arma::arma_rng::set_seed_random();

    const size_type n = 200;
    std::cout << "# Test Cholesky factorization of real quasi-Cauchy matrix:\n";
    for (size_type i = 0; i < 20; ++i)
    {
        std::cout << "--- trial " << i + 1 << ": ";
        test_cholesky_quasi_cauchy<double>(n);
    }

    std::cout
        << "\n# Test Cholesky factorization of complex quasi-Cauchy matrix:\n";
    for (size_type i = 0; i < 20; ++i)
    {
        std::cout << "--- trial " << i + 1 << ": ";
        test_cholesky_quasi_cauchy<std::complex<double>>(n);
    }

    return 0;
}
