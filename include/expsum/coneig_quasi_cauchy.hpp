// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-

#ifndef EXPSUM_CONEIG_QUASI_CAUCHY_HPP
#define EXPSUM_CONEIG_QUASI_CAUCHY_HPP

#include <cassert>
#include <type_traits>

#include <armadillo>

#include "arma/lapack_extra.hpp"
#include "expsum/numeric.hpp"
#include "expsum/jacobi_svd.hpp"
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

    matrix_type G(arma::diagmat(d) * (X.st() * X) * arma::diagmat(d));
    //
    // Compute G = Q * R * P.t() by Householder QR factorization with column
    // pivoting
    //
    matrix_type G_orig(G);          // tentative copy

    qr_col_pivot<T> qr;
    qr.run(G);
    matrix_type RPT(qr.get_matrix_RPT(G));

    qr.make_matrix_Q(G);
    std::cout << "|I-Q**H Q| = "
              << arma::norm(arma::eye<matrix_type>(n, n) - G.t() * G, 2)
              << '\n'
              << "|G - QR| = " << arma::norm(G_orig - G * RPT, 2) << '\n';

    //
    // Compute R1 = D^(-1) * (R * P.t()) * D^(-1)
    //
    matrix_type R1(RPT);
    for (size_type j = 0; j < m; ++j)
    {
        for (size_type i = 0; i < m; ++i)
        {
            R1(i, j) /= d(i) * d(j);
        }
    }
    //
    // Compute SVD of R * P.t() = U * S * V.t(). We need singular values and
    // left singular vectors.
    //
    jacobi_svd<T> svd;
    real_vector_type sigma(m);
    svd.run(RPT, sigma, /*compute_U*/true);

    //
    // Compute X1 = D^(-1) * U * S^{1/2}
    //
    matrix_type X1(RPT);
    for (size_type j = 0; j < m; ++j)
    {
        const auto sj = std::sqrt(sigma(j));
        for (size_type i = 0; i < m; ++i)
        {
            X1(i, j) *= sj / d(i);
        }
        std::cout << arma::norm(X1.col(j), 2) << '\n';
    }

    matrix_type Y1 = arma::solve(R1, X1);

    matrix_type coneigvec(arma::conj(X) * arma::conj(Y1));

    X = coneigvec;
    d = sigma;
}

//
// A = X D**2 X**H
//
// G = D X**T  X D = W S V**H
//
// Con-eigenvalues of A:  lambda(i) = S(i)
// Con-eigenvectors of A: U = conj(X) * D * conj(V)
//
// Algorithm:
// ----------
//     G = U S V**H
//     G**H G = V S**2 V**H
//
// let Z = X D V S**(-1/2), then
//   conj(A) A Z = X D (D X**H conj(X) D) (D X**T X D) V S**(-1/2)
//               = X D V S**2 V**H  V S**(-1/2)
//               = X D V S**2 S**(-1/2)
//               = Z S**2
//
//   G = Q R P**T = U S V**H
//
// - Compute matrix G = D X**T X D
// - Compute QR factorization of G by column pivot Householder QR method:
//     GP = QR
// - compute SVD of RP**T = W S V**H
//   or D.inv() R P**T D.inv() D V = D.inv() W S**(1/2)**2
//
// let
//   R1 = D.inv() * R * D.inv()
//   D2 = D * P**T * D.inv();
//   X1 = D.inv() * W * S**(1/2)
// then
//   R1 * D2 * (D V S**(-1/2)) = X1
//   D2 * (D V S**(-1/2)) = R1.inv() X1
//   (D V S**(-1/2)) = D * P * D.inv() R1.inv() X1
//
// let
//   R1 = D.inv() * R * P**T * D.inv()
//      = (D.inv() * R * D.inv()) * (D * P**T * D.inv())
//   X1 = D.inv() * W * S**(1/2)
// then
//   R1 * D V S**(-1/2) = X1
//
// i.e., D V S**(1/2) = R1.inv() X1
//
template <typename T>
void coneigenvalueRRD(arma::Mat<T>&                                      X,
                      arma::Col<typename arma::get_pod_type<T>::result>& d)
{
    using UIndex     = arma::uword;
    using Real       = typename arma::get_pod_type<T>::result;
    using Matrix     = arma::Mat<T>;
    using RealVector = arma::Col<Real>;

    static const auto jacobi_tolerance = arma::Datum<Real>::eps;
    const auto n = X.n_cols;

    Matrix G(n, n);
    Matrix QR(n, n);
    Matrix R(n, n);
    Matrix RP(n, n);
    Matrix U(n, n);
    Matrix X1(n, n);
    Matrix Y1(n, n);
    Matrix Y2(n, n);

    G = arma::diagmat(d) * (X.st() * X) * arma::diagmat(d);
    //
    // Compute SVD of matrix G in two-step:
    //
    // 1. Compute Housefolder QR with column pivotting:
    //    GP = QR (P is a permutation matrix)
    //
    QR = G;
    auto N     = static_cast<arma::blas_int>(n);
    auto lwork = static_cast<arma::blas_int>((n + 1) * n);
    arma::Col<T> work(static_cast<UIndex>(lwork));
    arma::blas_int info = 0;
    arma::Col<arma::blas_int> jpiv(n, arma::fill::zeros);
    arma::Col<T> tau(n);
    arma::Col<Real> rwork(2 * n);
    RealVector sigma(n);
    arma::lapack::geqp3(&N, &N, QR.memptr(), &N, jpiv.memptr(), tau.memptr(),
                        work.memptr(), &lwork, rwork.memptr(), &info);
    // std::cout << QR.diag() << std::endl;
    // check
    RP.zeros();
    for (UIndex j = 0; j < n; ++j)
    {
        arma::blas_int idx = j + 1;
        UIndex icol = 0;
        while (icol < n)
        {
            if (jpiv(icol) == idx)
            {
                break;
            }
            ++icol;
        }
        RP.col(j).subvec(0, icol) = QR.col(icol).subvec(0, icol);
    }
    // 2. Compute SVD of R P**T = W * S * V**H
    //    R     <-- W
    //    sigma <-- S.diag()
    //    V is not necessary
    // Upper triangular part of QR holds matrix R.
    R = arma::trimatu(QR);
    arma::lapack::ungqr(&N, &N, &N, QR.memptr(), &N, tau.memptr(),
                        work.memptr(), &lwork, &info);
    std::cout << "|I-Q**H Q| = "
              << arma::norm(arma::eye<Matrix>(n, n) - QR.t() * QR, 2) << '\n'
              << "|G - QR| = "
              << arma::norm(G - QR * RP, 2) << '\n';

    Matrix R1(RP);
    for (UIndex j = 0; j < n; ++j)
    {
        for (UIndex i = 0; i < n; ++i)
        {
            R1(i, j) /= d(i) * d(j);
        }
    }
    // std::cout << R1.diag() << std::endl;
    // U = R;
    U = RP;
    oneSidedJacobiSVD(U, sigma, jacobi_tolerance);
    // std::cout << sigma << std::endl;
    // Matrix& U = R;              // Right singular vector of G
    // R1 = D.inv() * R * D.inv()
    // X1 = D.inv() * U * sqrt(S)
    for (UIndex j = 0; j < n; ++j)
    {
        const auto sj = std::sqrt(sigma(j));
        for (UIndex i = 0; i < n; ++i)
        {
            U(i, j) *= sj / d(i);
        }
        std::cout << arma::norm(U.col(j), 2) << '\n';
    }
    // Y1 = R1.inv() * X1
    // RealVector dinv(Real(1) / d);
    // arma::solve(Y1, arma::trimatu(R1), U);
    arma::solve(Y1, R1, U);
    // Matrix Y1(arma::solve(R1, U));
    // Matrix Y2(n, n);
    // Y2 = D * P * D.inv() * Y1
    // for (UIndex j = 0; j < n; ++j)
    // {
    //     // for (UIndex i = 0; i < n; ++i)
    //     // {
    //     //     UIndex irow = static_cast<UIndex>(jpiv(i) - 1);
    //     //     Y2(irow, j) = d(j) * Y1(i, j) / d(i);
    //     //     // Y2(irow, j) = d(irow) * Y1(i, j) / d(i);
    //     // }
    //     for (UIndex i = 0; i < n; ++i)
    //     {
    //         UIndex irow = static_cast<UIndex>(jpiv(i) - 1);
    //         Y2(irow, j) = Y1(i, j);
    //     }
    //     std::cout << arma::norm(Y1.col(j), 2) << '\t'
    //               << arma::norm(Y2.col(j), 2) << '\n';
    // }
    // Matrix Xtmp(X.n_rows, X.n_cols);
    // Xtmp = X * Y2;
    // X    = Xtmp;
    // X *= Y2;
    X *= Y1;
    d.subvec(0, n - 1) = sigma;
}

// template <typename T>
// void coneigenvalueRRD(arma::Mat<T>&                                      X,
//                       arma::Col<typename arma::get_pod_type<T>::result>& d,
//                       arma::Col<T>&                                      work,
//                       arma::Col<typename arma::get_pod_type<T>::result>& rwork)
// {
//     using UIndex     = arma::uword;
//     using Real       = typename arma::get_pod_type<T>::result;
//     using Matrix     = arma::Mat<T>;
//     using RealVector = arma::Col<Real>;

//     static const auto jacobi_tolerance = arma::Datum<Real>::eps;

//     // const UIndex m = X.n_rows;
//     const UIndex n = X.n_cols;

//     Matrix G(work.memptr(), n, n, /*copy_aux_mem*/ false, /*strict*/ true);
//     G = arma::diagmat(d) * (X.st() * X) * arma::diagmat(d);

//     // Matrix Q(work.memptr() +     n * n, n, n, false, true);
//     // Matrix R(work.memptr() + 2 * n * n, n, n, false, true);
//     //
//     // Compute QR decomposition of G with pivotting
//     //
//     // G * P = Q * R
//     //
//     auto N     = static_cast<arma::blas_int>(n);
//     auto lwork = static_cast<arma::blas_int>((n + 1) * n);
//     arma::blas_int info = 0;
//     arma::Col<arma::blas_int> jpiv(n, arma::fill::zeros);
//     arma::Col<T> tau(n);
//     arma::lapack::geqp3(&N, &N, G.memptr(), &N, jpiv.memptr(), tau.memptr(),
//                         work.memptr() + n * n, &lwork, rwork.memptr(), &info);
//     std::cout << "(gepq3 info): "  << info << std::endl;
//     std::cout << "(gepq3 pivot): " << jpiv << std::endl;
//     //
//     // Uppre triangular part of G contains R
//     //
//     // Set U = R
//     Matrix U(work.memptr() + n * n, n, n, false, true);
//     // U.zeros();                 // initialized by 0
//     // for (UIndex j = 0; j < n; ++j)
//     // {
//     //     auto jcol = static_cast<UIndex>(jpiv(j) - 1);
//     //     for (UIndex i = 0; i <= jcol; ++i)
//     //     {
//     //         U(i, j) = G(i, jcol);
//     //     }
//     // }
//     // G = U;
//     U = arma::trimatu(G);
//     //
//     // Comput SVD of R (= U * S * V.t())
//     //  - G is overwritten by matrix U.
//     //  - V is not computed
//     //
//     // Matrix V(n, n);
//     RealVector sigma(rwork.memptr(), n, false, true);
//     // oneSidedJacobiSVD(U, sigma, V, jacobi_tolerance);
//     oneSidedJacobiSVD(U, sigma, jacobi_tolerance);
//     //
//     // R1 = D.inv() * R * D.inv() -- overwrite G
//     //
//     Matrix RP(work.memptr() + 2 * n * n, n, n, false, true);
//     RP.zeros();
//     for (UIndex j = 0; j < n; ++j)
//     {
//         UIndex jcol = static_cast<UIndex>(jpiv(j) - 1);
//         for (UIndex i = 0; i <= jcol; ++i)
//         {
//             RP(i, j) =  G(i, jcol) / (d(i) * d(j));
//             // RP(i, j) =  G(i, jcol) / (d(i) * d(j));
//         }
//     }
//     // auto R1 = arma::trimatu(G);
//     //
//     // Form X1 = D.inv() * U * sqrt(S).
//     //  - Overwrite matrix R.
//     //
//     Matrix& X1 = U;
//     for (UIndex j = 0; j < n; ++j)
//     {
//         const auto sj = std::sqrt(sigma(j));
//         for (UIndex i = 0; i < n; ++i)
//         {
//             X1(i, j) *= sj / d(i);
//         }
//     }
//     Matrix Y1(work.memptr() + 2 * n * n, n, n, false, true);
//     //
//     // Solve R1 * Y1 = X1
//     //
//     arma::solve(Y1, RP, X1);
//     //
//     // Set con-eigenvalues and con-eigenvectors
//     //
//     // d  = sigma;
//     // X *= Y1;
//     // X  = arma::conj(X);
//     // for (UIndex i = 0; i < n; ++i)
//     // {
//     //     std::cout << arma::norm(X.col(i), 2) << '\n';
//     // }
//     X *= Y1;
//     X  = arma::conj(X);
//     // for (UIndex i = 0; i < n; ++i)
//     // {
//     //     auto nrm = arma::norm(X.col(i), 2);
//     //     d(i) = sigma(i) * nrm;
//     //     X.col(i) /= nrm;
//     // }
// }

///
/// Preconpute pivot order for the Cholesky factorization of \f$ n\times n \f$
/// positive-definite Caunchy matrix \f$ C_{ij}=a_{i}b_{j}/(x_{i}+y_{j}). \f$
///
template <typename T>
arma::uword
cauchyPivotOrder(arma::Col<T>&                          a,
                 arma::Col<T>&                          b,
                 arma::Col<T>&                          x,
                 arma::Col<T>&                          y,
                 typename arma::get_pod_type<T>::result delta,
                 arma::Col<arma::uword>&                ipiv,
                 arma::Col<T>&                          work)
{
    using UIndex = arma::uword;
    using Real   = typename arma::get_pod_type<T>::result;

    const UIndex n = a.size();

    assert(b.size() == n && x.size() == n && y.size() == n);
    assert(ipiv.size() == n && work.size() >= n);

    // Set cutoff for GECP termination
    const auto eta = arma::Datum<Real>::eps * delta * delta;
    // Form vector g(i) = a(i) * b(i) / (x(i) + y(i))
    arma::Col<T> g(work.memptr(), n, /*copy_aux_mem*/ false, /*strict*/ true);
    g    = (a % b) / (x + y);
    // Initialize permutation matrix
    ipiv = arma::linspace<arma::Col<UIndex> >(0, n - 1, n);

    UIndex m = 0;
    while (m < n)
    {
        UIndex l  = m;
        auto gmax = std::abs(g(m));
        //
        // Find m <= l < n such that |g(l)| = max_{m<=k<n}|g(k)|
        //
        for (UIndex k = m + 1; k < n; ++k)
        {
            auto gk = std::abs(g(k));
            if (gk > gmax)
            {
                l    = k;
                gmax = gk;
            }
        }

        if (gmax < eta)
        {
            break;
        }

        if (l != m)
        {
            // Swap elements
            std::swap(g(l),    g(m));
            std::swap(a(l),    a(m));
            std::swap(b(l),    b(m));
            std::swap(x(l),    x(m));
            std::swap(y(l),    y(m));
            // Swap _rows_ of permutation matrix
            std::swap(ipiv(l), ipiv(m));
        }

        // Update diagonal of Schur complement
        const auto xm = x(m);
        const auto ym = y(m);
        for (UIndex k = m + 1; k < n; ++k)
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

///
/// Partial Cholesky decomposition of positive-definite Cauchy matrix.
///
template <typename T>
arma::uword choleskyCauchy(arma::Col<T>&                          a,
                           arma::Col<T>&                          b,
                           arma::Col<T>&                          x,
                           arma::Col<T>&                          y,
                           typename arma::get_pod_type<T>::result delta,
                           arma::Mat<T>&                          G,
                           arma::Col<arma::uword>&                ipiv,
                           arma::Col<T>&                          alpha,
                           arma::Col<T>&                          beta)
{
    using UIndex = arma::uword;
    using Real   = typename arma::get_pod_type<T>::result;

    const auto n = a.size();
    assert(b.size() == n && x.size() == n && y.size() == n && ipiv.size() == n);
    assert(G.n_rows == n);

    const auto rank = cauchyPivotOrder(a, b, x, y, delta, ipiv, alpha);

    alpha = a;
    beta  = b;

    G.zeros();
    // G.col(0) = (alpha % beta) / (x + y);
    for (UIndex l = 0; l < n; ++l)
    {
        G(l, 0) = alpha(l) * beta(0) / (x(l) + y(0));
    }

    for (UIndex k = 1; k < rank; ++k)
    {
        // Upgrade generators
        const auto xkm1 = x(k - 1);
        const auto ykm1 = y(k - 1);
        for (UIndex l = k; l < n; ++l)
        {
            alpha(l) *= (x(l) - xkm1) / (x(l) + ykm1);
            beta(l)  *= (y(l) - ykm1) / (y(l) + xkm1);
        }
        // Extract k-th column for Cholesky factors
        for (UIndex l = k; l < n; ++l)
        {
            G(l, k) = alpha(l) * beta(k) / (x(l) + y(k));
        }
    }
    //
    // Scale strictly lower triangular part of G
    //   - diagonal part of G contains D**2
    //   - L = tril(G) * D^{-2} + I
    //
    for (UIndex j = 0; j < rank; ++j)
    {
        auto djj = G(j, j);
        G(j, j) = std::sqrt(djj);
        for (UIndex i = j + 1; i < n; ++i)
        {
            G(i, j) /= djj;
        }
    }

    return rank;
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

private:
    size_type rank_;
    real_vector_type coneigvals_;
    matrix_type coneigvecs_;
    index_vector_type ipiv_;
    vector_type work_;
    real_vector_type rwork_;

public:
    template <typename VecA, typename VecB, typename VecX, typename VecY>
    typename std::enable_if<(arma::is_basevec<VecA>::value&&
                             arma::is_basevec<VecA>::value&&
                             arma::is_basevec<VecA>::value&&
                             arma::is_basevec<VecA>::value), void>::type
    compute(const VecA& a, const VecB& b,
            const VecX& x, const VecY& y, real_type delta);

    size_type rank() const
    {
        return rank_;
    }

    void resize(size_type n)
    {
        coneigvals_.set_size(n);
        coneigvecs_.set_size(n, n);
        ipiv_.set_size(n);
        work_.set_size(std::max<size_type>(3 * n, 6) * n);
        rwork_.set_size(2 * n);
    }

    const real_vector& coneigenvalues() const
    {
        return coneigvals_;
    }

    const Matrix& coneigenvectors() const
    {
        return coneigvecs_;
    }
};

template <typename T>
template <typename VecA, typename VecB, typename VecX, typename VecY>
typename std::enable_if<(arma::is_basevec<VecA>::value&&
                         arma::is_basevec<VecA>::value&&
                         arma::is_basevec<VecA>::value&&
                         arma::is_basevec<VecA>::value), void>::type
coneig_quasi_cauchy<T>::compute(const VecA& a, const VecB& b,
                                 const VecX& x, const VecY& y, real_type delta)
{
    assert(a.n_elem == b.n_elem);
    assert(a.n_elem == x.n_elem);
    assert(a.n_elem == y.n_elem);

    const auto n = a.size();
    resize(n);

#ifdef DEBUG
    Matrix C(n, n);
    for (size_type j = 0; j < n; ++j)
    {
        for (size_type i = 0; i < n; ++i)
        {
            C(i, j) = a(i) * b(j) / (x(i) + y(j));
        }
    }
#endif

    value_type* p = work_.memptr();
    {
        Vector a_(p + 0 * n, n, /*copy_aux_mem*/ false, /*strict*/ true);
        Vector b_(p + 1 * n, n, /*copy_aux_mem*/ false, /*strict*/ true);
        Vector x_(p + 2 * n, n, /*copy_aux_mem*/ false, /*strict*/ true);
        Vector y_(p + 3 * n, n, /*copy_aux_mem*/ false, /*strict*/ true);
        Vector alpha(p + 5 * n, n, /*copy_aux_mem*/ false, /*strict*/ true);
        Vector beta(p + 6 * n, n, /*copy_aux_mem*/ false, /*strict*/ true);
        a_ = a;
        b_ = b;
        x_ = x;
        y_ = y;
        //
        // Compute partial Cholesky factorization of quasi-Cauchy matrix with
        // pivotting.
        //
        Matrix& X  = coneigvecs_;
        rank_ = choleskyCauchy(a_, b_, x_, y_, delta, X, ipiv_, alpha, beta);
        auto viewX = X.cols(0, rank_ - 1);
        auto d     = coneigvals_.subvec(0, rank_ - 1);
        //
        // Copy diagonal elements of D.
        //
        // Note that, as the input matrix is positive-definite and Hermitian,
        // D.diag() must have real, positive values.
        //
        d = arma::real(viewX.diag());
        //
        // L has unit diagonal.
        //
        viewX.diag().ones();
        //
        // Apply permutation matrix to form X = P * L
        //
        for (size_type j = 0; j < rank_; ++j)
        {
            // Vector a_ is used as an temporal space
            for (size_type i = 0; i < n; ++i)
            {
                a_(ipiv_(i)) = X(i, j);
            }
            X.col(j) = a_;
        }

#ifdef DEBUG
        Matrix W = viewX * arma::diagmat(arma::square(d)) * viewX.t();
        std::cout << "(choleskyCauchy):\n"
                     "  rank = "      << rank_
                  << "  ||C - (PL) * D**2 * (PL).t()|| = "
                  << arma::norm(C - W, 2) << std::endl;
        for (size_type j = 0; j < rank_; ++j)
        {
            std::cout << d(j) << '\t' << arma::norm(viewX.col(j), 2) << '\n';
        }
#endif
    }

    //
    // Compute con-eigenvalues and coresponding con-eigenvectors
    //
    {
        Matrix X(coneigvecs_.memptr(), n, rank_, false, true);
        real_vector sigma(coneigvals_.memptr(), rank_, false, true);
        std::cout << "coneigenvalueRRD" << std::endl;
        // coneigenvalueRRD(X, sigma, work_, rwork_);
        coneigenvalueRRD(X, sigma);
        std::cout << "done" << std::endl;
        //
        // find largest index m such that coneigvals_(m) >= delta
        //
        size_type m = 0;
        while (m < rank_)
        {
            if (coneigvals_(m) < delta)
            {
                break;
            }
            ++m;
        }
        rank_ = m;

#ifdef DEBUG
        // auto viewX = X.cols(0, rank_ - 1);
        // auto viewsigma = sigma.subvec(0, rank_ - 1);
        // Matrix W = arma::conj(viewX) * arma::diagmat(viewsigma) * viewX.t();
        // std::cout << "(coneigenvalueRRD):\n"
        //         "  rank = "      << rank_
        //           << "\n  ||C - conj(X) * D * (X).t()|| = "
        //           << arma::norm(C - W, 2) << std::endl;
        std::cout << "(coneigenvalueRRD):\n"
                     "  rank = "      << rank_ << '\n';
        std::cout << "# con-eigenvalue, norm(x(i)), err1, err2\n";
        for (size_type k = 0; k < rank_; ++k)
        {
            auto uk   = X.col(k);
            auto err1 = arma::norm(C * uk - sigma(k) * arma::conj(uk), 2);
            auto err2 = arma::norm(C * arma::conj(uk) - sigma(k) * uk, 2);
            std::cout << sigma(k) << '\t' << arma::norm(uk, 2) << '\t'
                      << err1 << '\t' << err2 << '\n';
        }
#endif
    }
}
}   // namespace: expsum

#endif /* EXPSUM_CONEIG_QUASI_CAUCHY_HPP */
