#include <iostream>
#include <iomanip>

#include <random>

#include "expsum/cholesky_quasi_cauchy.hpp"

using size_type      = arma::uword;

template <typename T>
void test_cholesky_quasi_cauchy(size_type n)
{
    using cholesky_t       = expsum::cholesky_quasi_cauchy<T>;
    using real_type        = typename cholesky_t::real_type;
    using matrix_type      = typename cholesky_t::matrix_type;
    using vector_type      = typename cholesky_t::vector_type;
    using real_vector_type = typename cholesky_t::real_vector_type;

    const real_type delta = arma::Datum<real_type>::eps * std::sqrt(n);

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

    cholesky_t chol;

    matrix_type X;
    real_vector_type d;
    chol.run(a, b, x, y, delta, X, d);
    matrix_type P_rec(chol.reconstruct(X, d));

    const auto norm_P = arma::norm(P);
    const auto resid  = arma::norm(P - P_rec);

    std::cout << "Test Cholesky factorization of quasi-Cauchy matrix:\n"
              << "  rank = " << X.n_cols
              << ", |A - L D**2 L**H| / |A| = " << resid / norm_P << '\n';
}

int main()
{
    arma::arma_rng::set_seed_random();
    std::cout << "# Test Cholesky factorization of real quasi-Cauchy matrix:\n";
    for (size_type i = 0; i < 20; ++i)
    {
        std::cout << "--- trial " << i + 1 << '\n';
        test_cholesky_quasi_cauchy<double>(100);
    }

    std::cout
        << "\n# Test Cholesky factorization of complex quasi-Cauchy matrix:\n";
    for (size_type i = 0; i < 20; ++i)
    {
        std::cout << "--- trial " << i + 1 << '\n';
        test_cholesky_quasi_cauchy<std::complex<double> >(100);
    }
    return 0;
}


