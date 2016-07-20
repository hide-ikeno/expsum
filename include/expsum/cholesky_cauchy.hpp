#ifndef EXPSUM_CHOLESKY_CAUCHY_HPP
#define EXPSUM_CHOLESKY_CAUCHY_HPP

#include <cassert>
#include <armadillo>

#include "expsum/numeric.hpp"

namespace expsum
{
//
// Rank-revealing Cholesky decomposition for a quasi-Cauchy matrix.
//
// This class computes the Cholesky decomposition of ``$n \times n$``
// quasi-Cauchy matrix defined as
//
// ``` math
//  C_{ij}=\frac{b_{i}^{} b_{j}^{\ast}}{a_{i}^{}+a_{i}^{\ast}}.
// ```
//
template <typename T>
class cholesky_cauchy_rrd
{
public:
    using value_type = T;
    using real_type  = typename arma::get_pod_type<T>::result;
    using size_type  = arma::uword;

    using vector_type      = arma::Col<value_type>;
    using matrix_type      = arma::Mat<value_type>;
    using real_vector_type = arma::Col<real_type>;

private:
    size_type n_;
    arma::uvec ipiv_;  // pivot order of rows
    vector_type work_; // pivot order of rows

public:
    // Default constructor
    cholesky_cauchy_rrd() = default;
    // Constructor with allocating workspace
    explicit cholesky_cauchy_rrd(size_type n) : n_(n), ipiv_(n), work_(n)
    {
    }
    // Default copy constructor
    cholesky_cauchy_rrd(const cholesky_cauchy_rrd&) = default;
    // Default move constructor
    cholesky_cauchy_rrd(cholesky_cauchy_rrd&&) = default;
    // Default destructor
    ~cholesky_cauchy_rrd() = default;
    // Copy assignment operator
    cholesky_cauchy_rrd& operator=(const cholesky_cauchy_rrd&) = default;
    // Move assignment operator
    cholesky_cauchy_rrd& operator=(cholesky_cauchy_rrd&&) = default;
    //
    // Compute Cholesky decomposition.
    //
    // @a vector of length ``$n$`` defining quasi-Cauchy matrix
    // @b vector of length ``$n$`` defining quasi-Cauchy matrix
    // @delta target size
    //
    matrix_type run(vector_type& a, vector_type& b, real_type delta)
    {
        const size_type n = a.size();
        resize(n);
        const size_type m = pivod_order(a, b, delta); // m <= n
        matrix_type mat_L(n, m, arma::fill::zeros);
        cholesky_impl(a, b, mat_L);

        return mat_L;
    }

    // Reserve memory for working space
    void resize(size_type n)
    {
        n_ = n;
        if (ipiv_.size() < n)
        {
            ipiv_.set_size(n);
            work_.set_size(n);
        }
    }
    //
    // Apply permutation matrix generated by previous decomposition
    //
    template <typename Mat1, typename Mat2>
    void apply_row_permutation(const Mat1& src, Mat2& dest) const
    {
        using size_type = arma::uword;

        assert(src.n_rows == n_ && dest.n_rows == n_ &&
               src.n_cols == dest.n_cols);

        for (size_type j = 0; j < src.n_cols; ++j)
        {
            for (size_type i = 0; i < n_; ++i)
            {
                dest(ipiv_(i), j) = src(i, j);
            }
        }
    }
    //
    // Reconstruct matrix from Cholesky factor
    //
    // @matL Cholesky factor computed by `cholesky_cauchy_rrd::run`.
    //
    matrix_type reconstruct(const matrix_type& matL)
    {
        assert(matL.n_rows == n_);
        matrix_type ret(n_, n_);
        auto sigma = work_.head(matL.n_cols);
        sigma = matL.diag();
        // *** Ugly const-cast hack ***
        auto& lower = const_cast<matrix_type&>(matL);
        lower.diag().ones();    // make unit lower
        matrix_type X(arma::size(lower));
        apply_row_permutation(lower, X);
        // *** Restore diagonal part of matL
        lower.diag() = sigma;

        return matrix_type(X * arma::diagmat(arma::square(sigma)) * X.t());
    }
private:
    size_type pivod_order(vector_type& a, vector_type& b, real_type delta);
    void cholesky_impl(const vector_type& a, const vector_type& b,
                       matrix_type& L);
};

template <typename T>
typename cholesky_cauchy_rrd<T>::size_type
cholesky_cauchy_rrd<T>::pivod_order(vector_type& a, vector_type& b,
                                    real_type delta)
{
    constexpr const real_type eps = std::numeric_limits<real_type>::epsilon();

    const auto n = a.size();

    assert(b.n_elem == n);
    assert(ipiv_.n_elem >= n);
    assert(work_.n_elem >= n);

    arma::uvec ipiv(ipiv_.memptr(), n, /*copy_aux_mem*/ false, /*strict*/ true);
    real_type* ptr = reinterpret_cast<real_type*>(work_.memptr());
    arma::Col<real_type> g(ptr, n, /*copy_aux_mem*/ false, /*strict*/ true);

    // Init permutation matrix
    for (size_type i = 0; i < n; ++i)
    {
        ipiv(i) = i;
    }

    // g[i] = b[i] * conj(b[i]) / (a[i] + conj(a[i]));
    for (size_type i = 0; i < n; ++i)
    {
        g(i) = numeric::abs2(b(i)) / (2 * std::real(a(i)));
    }

    //
    // Gaussian elimination with complete pivoting
    //
    const real_type cutoff = eps * delta * delta;

    size_type m = 0;
    for (; m < n - 1; ++m)
    {
        const auto l = arma::abs(g.subvec(m, n - 1)).index_max() + m;
        if (l != m)
        {
            // Swap element
            std::swap(g(m), g(l));
            std::swap(a(m), a(l));
            std::swap(b(m), b(l));
            // Swap rows of permutation matrix
            std::swap(ipiv(m), ipiv(l));
        }
        // Update diagonal of Schur complement
        const auto am = a(m);

        for (size_type k = m + 1; k < n; ++k)
        {
            const auto numer = numeric::abs2(a(k) - am);
            const auto denom = numeric::abs2(a(k) + am);
            g(k) *= numer / denom;
        }

        if (std::abs(g(m)) < cutoff)
        {
            ++m;
            break;
        }
    }
    //
    // Return truncation size
    //
    return m;
}

template <typename T>
void cholesky_cauchy_rrd<T>::cholesky_impl(const vector_type& a,
                                           const vector_type& b, matrix_type& L)
{
    const auto n = a.size();
    assert(b.size() == n);
    assert(L.n_rows == n);
    const auto rank = L.n_cols;
    assert(rank <= n);

    auto alpha = work_.head(n);
    alpha       = b;

    const auto beta0 = arma::access::alt_conj(b(0));
    const auto c_a0  = arma::access::alt_conj(a(0));
    for (size_type l = 0; l < n; ++l)
    {
        L(l, 0) = alpha(l) * beta0 / (a(l) + c_a0);
    }

    for (size_type k = 1; k < rank; ++k)
    {
        // Upgrade generators
        const auto akm1   = a(k - 1);
        const auto c_akm1 = arma::access::alt_conj(akm1);
        for (size_type l = k; l < n; ++l)
        {
            alpha(l) *= (a(l) - akm1) / (a(l) + c_akm1);
        }
        // Extract k-th column for Cholesky factors
        const auto beta_k = arma::access::alt_conj(alpha(k));
        const auto c_ak   = arma::access::alt_conj(a(k));
        for (size_type l = k; l < n; ++l)
        {
            L(l, k) = alpha(l) * beta_k / (a(l) + c_ak);
        }
    }
    //
    // Scale strictly lower triangular part of G
    //   - diagonal part of G contains D**2
    //   - L = tril(G) * D^{-2} + I
    //
    for (size_type j = 0; j < rank; ++j)
    {
        const auto djj = std::real(L(j, j));
        L(j, j) = std::sqrt(djj);

        for (size_type i = j + 1; i < n; ++i)
        {
            L(i, j) /= djj;
        }
    }

    return;
}

} // namespace: expsum

#endif /* EXPSUM_CHOLESKY_CAUCHY_HPP */
