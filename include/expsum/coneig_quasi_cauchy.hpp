// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-

#ifndef EXPSUM_CONEIG_QUASI_CAUCHY_HPP
#define EXPSUM_CONEIG_QUASI_CAUCHY_HPP

#include <cassert>
#include <type_traits>

#include <armadillo>

#include "arma/lapack_extra.hpp"
#include "expsum/jacobi_svd.hpp"
#include "expsum/numeric.hpp"
#include "expsum/qr_col_pivot.hpp"

namespace expsum
{

//
// Computes accurate con-eigenvalue decomposition of matrix of the form
// ``$X D^{2} X^{\ast}$``
//
// This function solves the con-eigenvalue problem
//
// ``` math
//   A\boldsymbol{u}=\lambda\overline{\boldsymbol{u}},
// ```
//
// where ``$A$`` is a symmetic matrix of dimension ``$n \times n$`` and is
// having a rank-revealing decomposition
//
// ``` math
//  A = X D^{2} X^{\ast},
// ```
//
// where rank-revealing factor ``$X$`` is a ``$n \times m$`` matrix and ``$D$``
// is a ``$m \times m$`` diagonal matrix.
//
// @X rank-revealing factor ``$X$`` of size ``$n \times m$``
// @d diagonal part of matrix ``$D$`` given as a vector of size ``$m$``
//
//
//
// X * D * (G.t() * G) * v = d^2 * X * D * v
//
// R = U * S * V.t()  or  R * V = U * S
//   ==> V = R^(-1) * U * S
//   ==> D * V * S^{-1/2} = D * R^(-1) * U * S^{1/2}
//
template <typename T>
void coneig_rrd(arma::Mat<T>& X,
                 arma::Col<typename arma::get_pod_type<T>::result>& d)
{
    using size_type = arma::uword;
    using real_type = typename arma::get_pod_type<T>::result;

    using matrix_type      = arma::Mat<T>;
    using real_vector_type = arma::Col<real_type>;

    assert(X.n_rows >= X.n_cols);
    assert(d.n_elem == X.n_cols);

    const auto n = X.n_rows;
    const auto m = X.n_cols;
    //
    // Form G = D * (X.st() * X) * D
    //
    matrix_type G(m, m);
    G = X.st() * X;
    for (size_type j = 0; j < m; ++j)
    {
        for (size_type i = 0; i < m; ++i)
        {
            G(i, j) *= d(i) * d(j);
        }
    }
    //
    // Compute G = Q * R * P.t() by Householder QR factorization with column
    // pivoting
    //
    std::cout << "***** G = Q * R" << std::endl;
    matrix_type Q(m, m), R(m, m);
    arma::qr_econ(Q, R, G); // G = Q * R

    std::cout << "|I-Q**H Q| = "
              << arma::norm(arma::eye<matrix_type>(m, m) - Q.t() * Q, 2) << '\n'
              << "|G - QR| = " << arma::norm(G - Q * R, 2) << '\n';
    std::cout << "***** diag(R)\n" << R.diag() << std::endl;

    //
    // Compute R1 = D^(-1) * (R * P.t()) * D^(-1)
    //
    std::cout << "***** R1 = D^(-1) * R * D^(-1)" << std::endl;
    matrix_type R1(R);
    for (size_type j = 0; j < m; ++j)
    {
        for (size_type i = 0; i <= j; ++i)
        {
            R1(i, j) /= d(i) * d(j);
        }
    }
    //
    // Compute SVD of R * P.t() = U * S * V.t(). We need singular values and
    // left singular vectors.
    //
    std::cout << "***** R = U * S * V.t()" << std::endl;
    real_vector_type sigma(m);
    // const auto ctol = arma::Datum<real_type>::eps * std::sqrt(real_type(m));
    const auto ctol = arma::Datum<real_type>::eps;
    matrix_type V(m, m);
    matrix_type tmp(R);
    jacobi_svd(R, sigma, V, ctol);
    std::cout << "     |R - U * S * V.t()| = "
              << arma::norm(tmp - R * arma::diagmat(sigma) * V.t(), 2)
              << std::endl;
    //
    // Compute X1 = D^(-1) * U * S^{1/2}
    //
    matrix_type X1(R);
    for (size_type j = 0; j < m; ++j)
    {
        const auto sj = std::sqrt(sigma(j));
        for (size_type i = 0; i < m; ++i)
        {
            X1(i, j) *= sj / d(i);
        }
    }

    std::cout << "***** solve R1 * Y1 = X1" << std::endl;
    matrix_type Y1(m, m);
    arma::solve(Y1, arma::trimatu(R1), X1);

    matrix_type coneigvec(n, m);
    coneigvec = arma::conj(X) * arma::conj(Y1);
    // coneigvec = X * Y1;
    std::cout << "***** exit" << std::endl;

    X = coneigvec;
    d = sigma;
}

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

private:
    size_type n_;
    arma::uvec ipiv_; // pivot order of rows
    vector_type work1_;
    vector_type work2_;

public:
    // Default constructor
    cholesky_quasi_cauchy() = default;
    // Constructor with allocating workspace
    explicit cholesky_quasi_cauchy(size_type n)
        : n_(n), ipiv_(n), work1_(n), work2_(n)
    {
    }
    // Default copy constructor
    cholesky_quasi_cauchy(const cholesky_quasi_cauchy&) = default;
    // Default move constructor
    cholesky_quasi_cauchy(cholesky_quasi_cauchy&&) = default;
    // Default destructor
    ~cholesky_quasi_cauchy() = default;
    // Copy assignment operator
    cholesky_quasi_cauchy& operator=(const cholesky_quasi_cauchy&) = default;
    // Move assignment operator
    cholesky_quasi_cauchy& operator=(cholesky_quasi_cauchy&&) = default;
    //
    // Compute Cholesky decomposition.
    //
    // @a vector of length ``$n$`` defining quasi-Cauchy matrix
    // @b vector of length ``$n$`` defining quasi-Cauchy matrix
    // @x vector of length ``$n$`` defining quasi-Cauchy matrix
    // @y vector of length ``$n$`` defining quasi-Cauchy matrix
    // @delta target size
    // @X matrix of dimention ``$n \times m$``, where ``$m$`` is the rank of
    //    Cauchy matrix ``$C$`` and is internaly determined.
    //    On exit, `X` holds Cholesky factor ``$X$`` which is a unit trianular
    //    matrix.
    // @d vector of length ``$m$``. On exit, `d` holds diagonal elements of
    //    Cholesky factor ``$D$``.
    //
    void run(vector_type& a, vector_type& b, vector_type& x, vector_type& y,
             real_type delta, matrix_type& X, real_vector_type& d)
    {
        const size_type n = a.size();
        resize(n);
        // Reorder vectors defining quasi-Cauchy matrix
        const size_type m = pivot_order(a, b, x, y, delta, ipiv_, work1_);

        // Compute Cholesky factor
        X.set_size(n, m);
        X.zeros();
        d.set_size(m);
        factorize(a, b, x, y, X, d, work1_, work2_);

        return;
    }
    //
    // Reserve memory for working space
    //
    void resize(size_type n)
    {
        n_ = n;
        ipiv_.set_size(n);
        work1_.set_size(n);
        work2_.set_size(n);
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
    // @X Cholesky factor computed by `cholesky_quasi_cauchy::run`.
    // @d Cholesky factor computed by `cholesky_quasi_cauchy::run`.
    //
    matrix_type reconstruct(const matrix_type& X, const real_vector_type& d)
    {
        matrix_type XD(X * arma::diagmat(d));
        matrix_type PXD(arma::size(XD));
        apply_row_permutation(XD, PXD);

        return matrix_type(PXD * PXD.t());
    }

private:
    //
    // Preconpute pivot order for the Cholesky factorization of $n \times n$
    // positive-definite Caunchy matrix $C_{ij}=a_{i}b_{j}/(x_{i}+y_{j}).$
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
    // @X Cholesky factor (lower triangular matrix)
    // @d diagonal elements of Cholesky factor ``$D$``
    // @alpha working space
    // @beta  working space
    //
    static void factorize(const vector_type& a, const vector_type& b,
                          const vector_type& x, const vector_type& y,
                          matrix_type& X, real_vector_type& d,
                          vector_type& alpha, vector_type& beta);

private:
};

//------------------------------------------------------------------------------
// Private member functions
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
                                         const vector_type& y, matrix_type& X,
                                         real_vector_type& d,
                                         vector_type& alpha, vector_type& beta)
{
    const auto n = X.n_rows;
    const auto m = X.n_cols;
    assert(a.size() == n);
    assert(b.size() == n);
    assert(x.size() == n);
    assert(y.size() == n);
    assert(d.size() == m);
    assert(alpha.size() == n);
    assert(beta.size() == n);

    alpha = a;
    beta  = b;

    X.zeros();
    for (size_type l = 0; l < n; ++l)
    {
        X(l, 0) = alpha(l) * beta(0) / (x(l) + y(0));
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
            X(l, k) = alpha(l) * beta(k) / (x(l) + y(k));
        }
    }
    //
    // Scale strictly lower triangular part of G
    //   - diagonal part of G contains D**2
    //   - L = tril(G) * D^{-2} + I
    //
    for (size_type j = 0; j < m; ++j)
    {
        auto djj = X(j, j);
        d(j)     = std::sqrt(std::real(djj));
        X(j, j) = real_type(1);
        for (size_type i = j + 1; i < n; ++i)
        {
            X(i, j) /= djj;
        }
    }

    return;
}

/*!
 * Rank revealing con-eigenvalue decomposition of positive-definite Cauchy-like
 * matrices.
 *
 * This class compute the con-eigenvalue decomposition of positive-definite
 * Cauchy-like matrices
 *
 * \f[
 * C\bm{u}&=\lambda_{m}\overline{\bm{u}}, \\
 * C_{ij}&= \frac{a_{i}b_{j}}{x_{i}+y_{j}}, \quad (i,j=1,2,\dots,n).
 * \f]
 *
 * This con-eigenvalue problem can be reduced to an eigenvalue problem as
 *
 * \f[
 * \overline{C} C \bm{u}&=\lambda\overline{C}\overline{\bm{u}}
 *                       =\lambda^{2}\bm{u}.
 * \f]
 *
 * First, the rank-revealing decomposition of Cauchy matrix is made such as \f$
 * C=XDX^{\ast} \f$ where X is a (well-conditioned) \f$ n \times m \f$ matrix
 * \f$ n \geq m \f$, and D is an \f$ m \times m \f$ diagonal matrix with
 * positive (real) non-increasing entries. The eigenvalues of the matrix \f$
 * \overline{C}C \f$ can be obtained as squares of the the sigular values of the
 * matrix \f$ G=D(X^{T}X)D \f$, while the corresponding eigenvectors are
 * obtained as \f$ \overline{X}D\bm{v}_{i} \f$ where \f$ \bm{v}_{i} \f$ is i-th
 * right singualr vector of \f$ G \f$.
 *
 */

template <typename T>
class coneig_quasi_cauchy
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

public:
    template <typename VecA, typename VecB, typename VecX, typename VecY>
    typename std::enable_if<
        (arma::is_basevec<VecA>::value && arma::is_basevec<VecA>::value &&
         arma::is_basevec<VecA>::value && arma::is_basevec<VecA>::value),
        void>::type
    compute(const VecA& a, const VecB& b, const VecX& x, const VecY& y,
            real_type delta, real_vector_type& coneigvals,
            matrix_type& coneigvecs);
};

template <typename T>
template <typename VecA, typename VecB, typename VecX, typename VecY>
typename std::enable_if<
    (arma::is_basevec<VecA>::value && arma::is_basevec<VecA>::value &&
     arma::is_basevec<VecA>::value && arma::is_basevec<VecA>::value),
    void>::type
coneig_quasi_cauchy<T>::compute(const VecA& a, const VecB& b, const VecX& x,
                                const VecY& y, real_type delta,
                                real_vector_type& coneigvals,
                                matrix_type& coneigvecs)
{
    assert(a.n_elem == b.n_elem);
    assert(a.n_elem == x.n_elem);
    assert(a.n_elem == y.n_elem);

    const auto n = a.size();

#ifdef DEBUG
    matrix_type C(n, n);
    for (size_type j = 0; j < n; ++j)
    {
        for (size_type i = 0; i < n; ++i)
        {
            C(i, j) = a(i) * b(j) / (x(i) + y(j));
        }
    }
#endif

    vector_type a_(n);
    vector_type b_(n);
    vector_type x_(n);
    vector_type y_(n);

    a_ = a;
    b_ = b;
    x_ = x;
    y_ = y;
    //
    // Compute partial Cholesky factorization of quasi-Cauchy matrix with
    // pivotting.
    //
    matrix_type X;
    real_vector_type d;
    cholesky_quasi_cauchy<value_type> chol;
    chol.run(a_, b_, x_, y_, delta, X, d);
    matrix_type PX(arma::size(X));
    chol.apply_row_permutation(X, PX);

#ifdef DEBUG
    {
        matrix_type W = PX * arma::diagmat(arma::square(d)) * PX.t();
        std::cout << "*** Cholesky factors of quasi-Cauchy matrix:\n"
                     "  rank = "
                  << PX.n_cols                             // rank
                  << "  ||C - (PL) * D**2 * (PL).t()|| = " // residual
                  << arma::norm(C - W, 2) << std::endl;
        // for (size_type j = 0; j < rank_; ++j)
        // {
        //     std::cout << d(j) << '\t' << arma::norm(viewX.col(j), 2) << '\n';
        // }
    }
#endif
    //
    // Compute con-eigenvalues and coresponding con-eigenvectors
    //
    std::cout << "*** coneig_rrd" << std::endl;
    coneig_rrd(PX, d);
    //
    // find largest index m such that coneigvals_(m) >= delta
    //
    coneigvals.set_size(arma::size(d));
    coneigvecs.set_size(arma::size(PX));
    coneigvals = d;
    coneigvecs = PX;

}

} // namespace: expsum

#endif /* EXPSUM_CONEIG_QUASI_CAUCHY_HPP */
