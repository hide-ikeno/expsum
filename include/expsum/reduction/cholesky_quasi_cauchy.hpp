#ifndef EXPSUM_REDUCTION_CHOLESKY_CAUCHY_HPP
#define EXPSUM_REDUCTION_CHOLESKY_CAUCHY_HPP

#include <armadillo>
#include <cassert>

namespace expsum
{
//
// Rank-revealing Cholesky decomposition for a positive-definite quasi-Cauchy
// matrix.
//
// This class computes the Cholesky decomposition of ``$n \times n$``
// quasi-Cauchy matrix defined as
//
// ``` math
//  C_{ij} = \frac{a_{i}^{} b_{j}^{}}{x_{i}^{} + y_{i}^{}}.
// ```
//
template <typename T>
struct cholesky_quasi_cauchy
{
public:
    using value_type = T;
    using real_type  = typename arma::get_pod_type<T>::result;
    using size_type  = arma::uword;

    using vector_type      = arma::Col<value_type>;
    using matrix_type      = arma::Mat<value_type>;
    using real_vector_type = arma::Col<real_type>;
    //
    // Pre-compute pivot order for the Cholesky factorization of Cauchy matrix.
    //
    static size_type pivot_order(vector_type& a, vector_type& b, vector_type& x,
                                 vector_type& y, real_type delta,
                                 arma::uvec& ipiv, vector_type& g);
    //
    // Compute Cholesky factors (`X` and diagonal part of `D`).
    //
    // The arrays `a,b,x,y` must be properly reordered by calling `pivot_order`
    // beforehand, so that the diagonal part of Cholesky factors appear in
    // decesnding order.
    //
    // @a vector of length ``$n$`` defining quasi-Cauchy matrix (reordered)
    // @b vector of length ``$n$`` defining quasi-Cauchy matrix (reordered)
    // @x vector of length ``$n$`` defining quasi-Cauchy matrix (reordered)
    // @y vector of length ``$n$`` defining quasi-Cauchy matrix (reordered)
    // @L Cholesky factor (lower triangular matrix)
    // @d diagonal elements of Cholesky factor ``$D$``
    // @alpha vector of length ``$n$`` as working space
    // @beta  vector of length ``$n$`` as working space
    //
    static void factorize(const vector_type& a, const vector_type& b,
                          const vector_type& x, const vector_type& y,
                          matrix_type& L, real_vector_type& d,
                          vector_type& alpha, vector_type& beta);
    //
    // Apply permutation matrix generated by previous decomposition *in-place.*
    //
    // @X $n \times k$ matrix that the row-permutation matrix is applied.
    // @ipiv vector of index with size $n$. Permutation index obtained by
    //       `pivot_order` or `pivot_order_sym`.
    // @work working space of size $n$.
    //
    template <typename MatX>
    static void apply_row_permutation(MatX& X, const arma::uvec& ipiv,
                                      vector_type& work)
    {
        const size_type n = ipiv.size();
        assert(X.n_rows == n && work.n_elem == n);

        for (size_type j = 0; j < X.n_cols; ++j)
        {
            for (size_type i = 0; i < n; ++i)
            {
                work(ipiv(i)) = X(i, j);
            }
            X.col(j) = work;
        }
    }
    //
    // Reconstruct matrix from Cholesky factor
    //
    // @X Cholesky factor computed by `cholesky_quasi_cauchy::run`.
    // @d Cholesky factor computed by `cholesky_quasi_cauchy::run`.
    //
    static matrix_type reconstruct(const matrix_type& X, const arma::uvec& ipiv,
                                   const real_vector_type& d)
    {
        matrix_type PX(X);
        vector_type work(X.n_rows);
        apply_row_permutation(PX, ipiv, work);

        return matrix_type(PX * arma::diagmat(arma::square(d)) * PX.t());
    }
};

//------------------------------------------------------------------------------
// Implementation of member functions
//------------------------------------------------------------------------------
template <typename T>
typename cholesky_quasi_cauchy<T>::size_type
cholesky_quasi_cauchy<T>::pivot_order(vector_type& a, vector_type& b,
                                      vector_type& x, vector_type& y,
                                      real_type delta, arma::uvec& ipiv,
                                      vector_type& g)
{
    const size_type n = a.size();
    assert(b.size() == n);
    assert(x.size() == n);
    assert(y.size() == n);
    assert(ipiv.size() == n);
    assert(g.size() == n);
    //
    // Set cutoff for GECP termination
    //
    const auto eta = arma::Datum<real_type>::eps * delta * delta;
    //
    // Form vector g(i) = a(i) * b(i) / (x(i) + y(i))
    //
    g = (a % b) / (x + y);
    //
    // Initialize permutation matrix
    //
    ipiv = arma::linspace<arma::uvec>(0, n - 1, n);

    size_type m = 0;
    while (m < n)
    {
        //
        // Find m <= l < n such that |g(l)| = max_{m<=k<n}|g(k)|
        //
        const auto l    = arma::abs(g.tail(n - m)).index_max() + m;
        const auto gmax = std::abs(g(l));

        if (gmax < eta)
        {
            break;
        }

        if (l != m)
        {
            // Swap elements
            std::swap(g(l), g(m));
            std::swap(a(l), a(m));
            std::swap(b(l), b(m));
            std::swap(x(l), x(m));
            std::swap(y(l), y(m));
            // Swap _rows_ of permutation matrix
            std::swap(ipiv(l), ipiv(m));
        }

        // Update diagonal of Schur complement
        const auto xm = x(m);
        const auto ym = y(m);
        for (size_type k = m + 1; k < n; ++k)
        {
            g(k) *= (x(k) - xm) * (y(k) - ym) / ((x(k) + ym) * (y(k) + xm));
        }
        ++m;
    }
    //
    // Returns the truncation size
    //
    return m;
}

template <typename T>
void cholesky_quasi_cauchy<T>::factorize(const vector_type& a,
                                         const vector_type& b,
                                         const vector_type& x,
                                         const vector_type& y, matrix_type& L,
                                         real_vector_type& d,
                                         vector_type& alpha, vector_type& beta)
{
    const auto n = L.n_rows;
    const auto m = L.n_cols;
    assert(a.size() == n);
    assert(b.size() == n);
    assert(x.size() == n);
    assert(y.size() == n);
    assert(d.size() == m);
    assert(alpha.size() == n);
    assert(beta.size() == n);

    alpha = a;
    beta  = b;

    L.zeros();
    for (size_type l = 0; l < n; ++l)
    {
        L(l, 0) = alpha(l) * beta(0) / (x(l) + y(0));
    }

    for (size_type k = 1; k < m; ++k)
    {
        // Upgrade generators
        const auto xkm1 = x(k - 1);
        const auto ykm1 = y(k - 1);
        for (size_type l = k; l < n; ++l)
        {
            alpha(l) *= (x(l) - xkm1) / (x(l) + ykm1);
            beta(l) *= (y(l) - ykm1) / (y(l) + xkm1);
        }
        // Extract k-th column for Cholesky factors
        for (size_type l = k; l < n; ++l)
        {
            L(l, k) = alpha(l) * beta(k) / (x(l) + y(k));
        }
    }
    //
    // Scale strictly lower triangular part of G
    //   - diagonal part of G contains D**2
    //   - L = tril(G) * D^{-2} + I
    //
    for (size_type j = 0; j < m; ++j)
    {
        const auto djj   = std::real(L(j, j));
        const auto scale = real_type(1) / djj;
        d(j)             = std::sqrt(djj);
        L(j, j) = real_type(1);
        for (size_type i = j + 1; i < n; ++i)
        {
            L(i, j) *= scale;
        }
    }

    return;
}

} // namespace: expsum

#endif /* EXPSUM_REDUCTION_CHOLESKY_CAUCHY_HPP */