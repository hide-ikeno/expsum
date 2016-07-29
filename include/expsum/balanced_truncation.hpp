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
    matrix_type X1, X2;
    real_vector_type d1, d2;

    chol.run(a, b, x, y, tol, X1, d1); // P1 = X1 * D1^2 * X1.t()
    chol.run(b, a, x, y, tol, X2, d2); // P2 = X2 * D2^2 * X2.t()
    const size_type m = d1.size();
    assert(d2.size() == m);
    for (size_type  i = 0; i < m; ++i)
    {
        std::cout << d1(i) << '\t' << d2(i) << '\n';
    }
    // G = (D2 * X2.t() * X1 * D1) = U * S * V.t()
    matrix_type G(X2.t() * X1);
    for (size_type j = 0; j < m; ++j)
    {
        for (size_type i = 0; i < m; ++i)
        {
            G(i, j) *= d2(i) * d1(j);
        }
    }

    // QR factorization with column pivoting, W = Q * R * P.t()
    matrix_type Q(arma::size(G));
    matrix_type R(arma::size(G));
    arma::qr_econ(Q, R, G);

    real_vector_type sigma(m);
    matrix_type V(m, m);
    const auto ctol = arma::Datum<real_type>::eps * std::sqrt(m);
    jacobi_svd(R, sigma, V, ctol);

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

}

}  // namespace: expsum

#endif /* EXPSUM_BALANCED_TRUNCATION_HPP */
