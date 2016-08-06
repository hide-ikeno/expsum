#ifndef EXPSUM_REDUCTION_HPP
#define EXPSUM_REDUCTION_HPP

#include <algorithm>
#include <cassert>

#include "expsum/numeric.hpp"
#include <armadillo>

namespace expsum
{
namespace detail
{
//
// Rank-revealing Cholesky decomposition for a positive-definite quasi-Cauchy
// matrix.
//
// This class computes the Cholesky decomposition of ``$n \times n$``
// quasi-Cauchy matrix defined as
//
// ``` math
//  C_{ij}=\frac{a_{i}^{} a_{j}^{\ast}}{x_{i}^{}+x_{i}^{\ast}}.
// ```
//
template <typename T>
class cholesky_quasi_cauchy
{
public:
    using value_type = T;
    using real_type  = typename arma::get_pod_type<T>::result;
    using size_type  = arma::uword;

    using vector_type      = arma::Col<value_type>;
    using matrix_type      = arma::Mat<value_type>;
    using real_vector_type = arma::Col<real_type>;
    //
    // Preconpute pivot order for the Cholesky factorization of $n \times n$
    // positive-definite Caunchy matrix $C_{ij}=a_{i}b_{j}/(x_{i}+y_{j}).$
    //
    static size_type pivot_order(vector_type& a, vector_type& x,
                                 arma::uvec& ipiv, real_type delta,
                                 real_vector_type& g)
    {
        const size_type n = a.size();
        assert(x.size() == n);
        assert(g.size() == n);
        assert(ipiv.size() == n);
        //
        // Set cutoff for GECP termination
        //
        const auto eta = arma::Datum<real_type>::eps * delta * delta;
        //
        // Form vector g(i) = a(i) * b(i) / (x(i) + y(i))
        //
        for (size_type i = 0; i < n; ++i)
        {
            g(i) = numeric::abs2(a(i)) / (real_type(2) * std::real(x(i)));
        }
        //
        // Initialize permutation matrix
        //
        std::iota(std::begin(ipiv), std::end(ipiv), size_type());

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
                std::swap(x(l), x(m));
                // Swap _rows_ of permutation matrix
                std::swap(ipiv(l), ipiv(m));
            }

            // Update diagonal of Schur complement
            const auto xm = x(m);
            for (size_type k = m + 1; k < n; ++k)
            {
                g(k) *= numeric::abs2(x(k) - xm) / numeric::abs2(x(k) + xm);
            }
            ++m;
        }
        //
        // Returns the truncation size
        //
        return m;
    }
    //
    // Compute Cholesky factors (`L` and diagonal part of `D`).
    //
    // The arrays `a,x` must be properly reordered by calling `pivot_order`
    // beforehand, so that the diagonal part of Cholesky factors appear in
    // decesnding order.
    //
    // @a vector of length ``$n$`` defining quasi-Cauchy matrix (reordered)
    // @x vector of length ``$n$`` defining quasi-Cauchy matrix (reordered)
    // @X Cholesky factor (lower triangular matrix)
    // @d diagonal elements of Cholesky factor ``$D$``
    // @alpha working space
    //
    static void factorize(const vector_type& a, const vector_type& x,
                          matrix_type& L, real_vector_type& d,
                          vector_type& alpha)
    {
        const auto n = L.n_rows;
        const auto m = L.n_cols;
        assert(a.size() == n);
        assert(x.size() == n);
        assert(d.size() == m);
        assert(alpha.size() == n);

        alpha = a;

        L.zeros();
        for (size_type l = 0; l < n; ++l)
        {
            L(l, 0) = alpha(l) * numeric::conj(alpha(0)) /
                      (x(l) + numeric::conj(x(0)));
        }

        for (size_type k = 1; k < m; ++k)
        {
            // Upgrade generators
            const auto xkm1 = x(k - 1);
            const auto ykm1 = numeric::conj(xkm1);
            for (size_type l = k; l < n; ++l)
            {
                alpha(l) *= (x(l) - xkm1) / (x(l) + ykm1);
            }
            // Extract k-th column for Cholesky factors
            const auto beta_k = numeric::conj(alpha(k));
            const auto y_k    = numeric::conj(x(k));
            for (size_type l = k; l < n; ++l)
            {
                L(l, k) = alpha(l) * beta_k / (x(l) + y_k);
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

    //
    // Apply permutation matrix generated by previous decomposition
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
        matrix_type XD(X * arma::diagmat(d));
        matrix_type PXD(arma::size(XD));
        apply_row_permutation(XD, ipiv, PXD);

        return matrix_type(PXD * PXD.t());
    }
};

} // namespace: detail

template <typename T>
class reduction_body
{
public:
    using size_type    = arma::uword;
    using value_type   = T;
    using real_type    = typename arma::get_pod_type<T>::result;
    using complex_type = std::complex<real_type>;

    using vector_type         = arma::Col<value_type>;
    using index_vector_type   = arma::uvec;
    using real_vector_type    = arma::Col<real_type>;
    using complex_vector_type = arma::Col<complex_type>;

    using matrix_type         = arma::Mat<value_type>;
    using real_matrix_type    = arma::Mat<real_type>;
    using complex_matrix_type = arma::Mat<complex_type>;

private:
    arma::uvec ipiv_;
    real_vector_type d_;
    matrix_type X_;

public:
    template <typename VecP, typename VecW>
    typename std::enable_if<(arma::is_basevec<VecP>::value &&
                             arma::is_basevec<VecW>::value),
                            void>::type
    run(const VecP& p, const VecW& w, real_type tol);

    void resize(size_type n)
    {
        ipiv_.set_size(n);
        d_.set_size(n);
        X_.set_size(n, n);
    }

private:
    //
    // Compute Cholesky decomposition of quasi-Cauchy matrix by Gaussian
    // elimination with complete pivoting.
    //
    // Factorize the matrix P = diagmat(a)  / (x + conj(x)) * diagmat(conj(a))
    // as
    //
    //   P = L * D^2 * L.t()
    //
    // where L is unit triangular with size (n x m), and D is a (m x m) diagonal
    // matrix, where m is the extracted rank of the matrix P.
    //
    size_type cholesky_rrd(vector_type& a, vector_type& x, real_type tol);
};

template <typename T>
template <typename VecP, typename VecW>
typename std::enable_if<(arma::is_basevec<VecP>::value &&
                         arma::is_basevec<VecW>::value),
                        void>::type
reduction_body<T>::run(const VecP& p, const VecW& w, real_type tol)
{
    const auto n = p.size();
    assert(w.size() == n);
    resize(n);

    vector_type a(arma::sqrt(w));
    vector_type x(p);

    cholesky_rrd(a, x, tol);
}

template <typename T>
typename reduction_body<T>::size_type
reduction_body<T>::cholesky_rrd(vector_type& a, vector_type& x, real_type delta)
{
    const size_type n = a.size();

#ifdef DEBUG
    matrix_type P(n, n);
    for (size_type j = 0; j < n; ++j)
    {
        for (size_type i = 0; i < n; ++i)
        {
            P(i, j) = a(i) * numeric::conj(a(j)) / (x(i) + numeric::conj(x(j)));
        }
    }
#endif

    vector_type work_(n);
    real_vector_type g(reinterpret_cast<real_type*>(work_.memptr()), n,
                       /*copy_aux_mem*/ false, /*strict*/ true);
    // Pre-compute correct pivot order of Cholesky decomposition
    const size_type m =
        detail::cholesky_quasi_cauchy<T>::pivot_order(a, x, ipiv_, delta, g);
    matrix_type L(X_.memptr(), n, m, false, true);
    real_vector_type d(d_.memptr(), m, false, true);
    // Compute Cholesky factors
    detail::cholesky_quasi_cauchy<T>::factorize(a, x, L, d, work_);
    // Apply permutation matrix
    detail::cholesky_quasi_cauchy<T>::apply_row_permutation(L, ipiv_, work_);

#ifdef DEBUG
    std::cout << "*** Cholesky-Cauchy:\n"
              << "    |P - L * D^2 * L.t()| = "
              << arma::norm(P - L * arma::diagmat(arma::square(d)) * L.t())
              << std::endl;
#endif

    return m;
}

} // namespace: expsum

#endif /* EXPSUM_REDUCTION_HPP */
