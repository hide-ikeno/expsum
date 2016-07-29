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

#include "expsum/cholesky_quasi_cauchy.hpp"

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
    // std::cout << "***** diag(R)\n" << R.diag() << std::endl;

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
