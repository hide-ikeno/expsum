#ifndef EXPSUM_BALANCED_TRUNCATION_HPP
#define EXPSUM_BALANCED_TRUNCATION_HPP

#include <armadillo>
#include "expsum/cholesky_quasi_cauchy.hpp"
// #include "expsum/qr_col_pivot.hpp"
#include "expsum/jacobi_svd.hpp"

namespace expsum
{

template <typename T>
struct balanced_truncation
{
    using value_type = T;
    using real_type  = typename arma::get_pod_type<T>::result;
    using size_type  = arma::uword;

    using vector_type      = arma::Col<value_type>;
    using matrix_type      = arma::Mat<value_type>;

    using real_vector_type = arma::Col<real_type>;

    void run(const vector_type& p, const vector_type& w, real_type tol);
};

template <typename T>
void balanced_truncation<T>::run(const vector_type& p, const vector_type& w,
                                real_type tol)
{
    const size_type n = p.size();
    assert(w.size() == n);

    vector_type a(p);
    vector_type b(arma::conj(a));
    vector_type x(arma::sqrt(w));
    vector_type y(arma::conj(x));
    //
    // Compute Cholesky factors of Gramian matrix
    //
    cholesky_quasi_cauchy<value_type> chol;
    matrix_type X;
    real_vector_type d;

    chol.run(a, b, x, y, tol, X, d); // P = X * D^2 * X.t()
    const size_type m = d.size();
    // G = (D2 * X2.t() * X1 * D1) = U * S * V.t()
    matrix_type G(X.t() * arma::conj(X));
    for (size_type j = 0; j < m; ++j)
    {
        for (size_type i = 0; i < m; ++i)
        {
            G(i, j) *= d(i) * d(j);
        }
    }

    // QR factorization  G = Q * R
    matrix_type Q(arma::size(G));
    matrix_type R(arma::size(G));
    arma::qr_econ(Q, R, G);

    // Form matrix R1 = D^(-1) * R * R^(-1)
    matrix_type R1(R);
    for (size_type j = 0; j < m; ++j)
    {
        for (size_type i = 0; i <= j; ++i)
        {
            R1(i, j) /= d(i) * d(j);
        }
    }
    // R = W * S * V.t()
    real_vector_type sigma(m);
    matrix_type V(m, m);
    const auto ctol = arma::Datum<real_type>::eps * std::sqrt(m);
    jacobi_svd(R, sigma, V, ctol); // R is overwritten by W

    std::cout << sigma << std::endl;
    //
    // Find truncation size.
    //
    size_type k = m;
    {
        auto sum = real_type();

        while (k > 0)
        {
            sum += sigma(k - 1);
            if (real_type(2) * sum > tol)
            {
                break;
            }
            --k;
        }
    }
    std::cout << "*** Truncation size: " << k << std::endl;

    // Y = D * conj(V) * S^(-1/2)
    //   = R1.inv() * (D^(-1) * conj(W) * sqrt(S))
    for (size_type j = 0; j < m; ++j)
    {
        auto sj = std::sqrt(sigma(j));
        for (size_type i = 0; i < m; ++i)
        {
            R(i, j) *= sj / d(i);
        }
    }

    matrix_type Y(m, k);
    arma::solve(Y, arma::trimatu(R1), R.head_cols(k));
    // Z = D * U * S^(-1/2)
    //   = D * Q * D^(-1) * R1.inv() * (D^(-1) * V * sqrt(S))
    for (size_type j = 0; j < m; ++j)
    {
        auto sj = std::sqrt(sigma(j));
        for (size_type i = 0; i < m; ++i)
        {
            V(i, j) *= sj / d(i);
        }
    }

    for (size_type j = 0; j < m; ++j)
    {
        auto dj = real_type(1) / d(j);
        for (size_type i = 0; i < m; ++i)
        {
            Q(i, j) *= d(i) * dj;
        }
    }
    matrix_type Z(m, k);
    arma::solve(Z, arma::trimatu(R1), V.head_cols(k));

    Z = Q * Z;

    std::cout << "*** Transformation matrices status\n"
              << "    |I - Y^H * Z| = "
              << arma::norm(arma::eye<matrix_type>(k, k) - Y.t() * Z, 2) << std::endl; 

    // G = Y.t() * X.st() * A * X * Z
    G = X.st() * arma::diagmat(p) * X;
    matrix_type A2(Y.t() * G * Z);

    vector_type b2(Y.t() * (X.st() * x));
    vector_type c2(Z.st() * (X.st() * x));

}

}  // namespace: expsum

#endif /* EXPSUM_BALANCED_TRUNCATION_HPP */
