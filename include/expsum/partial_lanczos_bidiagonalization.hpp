#ifndef EXPSUM_PARTIAL_LANCZOS_BIDIAGONALIZATION_HPP
#define EXPSUM_PARTIAL_LANCZOS_BIDIAGONALIZATION_HPP

#include <armadillo>

namespace expsum
{
///
/// Partial Lanczos bidiagonalization with full reorthogonalization
///
/// @param matvec  compute `y = a * A) * x + b * y` with matrix-free form
/// @param matvec_trans compute `y = a * A.t() * x + b * y` with matrix-free
///   form
/// @param[out] alpha  real vector of size `n` that stores diagonal part of B.
/// @param[out] beta   real vector of size `n - 1` that stores superdiagonal
///   part of B.
/// @param[out] P  on exit matrix P
/// @param[out] Q  on exit matrix Q
/// @param[out] rank  on exit rank of matrix A
/// @param[inout] tol_error on entry tolerance of residuals, and on exit an
///   esitmation of residual.
/// @param[inout] work workspace
///
template <typename MatVec, typename MatVecTrans, typename T>
void partial_lanczos_bidiagonalization(
    MatVec matvec, MatVecTrans matvec_trans,
    arma::Col<typename arma::Col<T>::pod_type>& alpha,
    arma::Col<typename arma::Col<T>::pod_type>& beta, arma::Mat<T>& P,
    arma::Mat<T>& Q, arma::uword& rank,
    typename arma::Col<T>::pod_type& tol_error, arma::Col<T>& work)
{
    using value_type = T;
    using real_type  = typename arma::Col<T>::pod_type;

    // constexpr static const value_type zero = value_type();
    // constexpr static const value_type one  = value_type(1);

    arma::uword m        = P.n_rows;
    arma::uword n        = Q.n_rows;
    arma::uword k        = P.n_cols;
    arma::uword max_rank = std::min({m, n, k});

    assert(Q.n_cols == k);
    assert(work.size() >= max_rank);
    assert(real_type() < tol_error && tol_error < real_type(1));

    auto q0 = Q.col(0);
    q0.randn();
    q0 /= arma::norm(q0);
    auto p0 = P.col(0);
    // p0 <-- A * q0
    matvec(q0, value_type(), p0);
    auto a1 = arma::norm(p0);
    if (a1 > real_type())
    {
        p0 *= real_type(1) / a1;
    }
    alpha(0) = a1;

    const real_type tol2 = tol_error * tol_error;
    // Estimation of the Frobenius norm of A
    real_type fnormA = a1 * a1;
    // Estimation of relative error
    real_type error = real_type();

    rank = 0;

    while (++rank < max_rank)
    {
        auto p1 = P.col(rank - 1);
        auto p2 = P.col(rank);
        auto q1 = Q.col(rank - 1);
        auto q2 = Q.col(rank);
        //
        // --- Recursion for right Lanczos vector
        //
        // q2 <-- A.t() * p1 - a1 * q1
        matvec_trans(p1, value_type(), q2);
        q2 -= a1 * q1;
        // Reorthogonalization
        auto tmp   = work.head(rank);
        auto viewQ = Q.head_cols(rank);
        tmp        = viewQ.t() * q2;
        q2 -= viewQ * tmp;

        auto b1 = arma::norm(q2);
        if (b1 > real_type())
        {
            q2 *= real_type(1) / b1;
        }
        beta(rank - 1) = b1;
        //
        // --- Recursion for left Lanczos vector
        //
        // p2 <-- A * p2 - b1 * p1
        matvec(q2, value_type(), p2);
        p2 -= b1 * p1;
        // Reorthogonalization
        auto viewP = P.head_cols(rank);
        tmp        = viewP.t() * p2;
        p2 -= viewP * tmp;

        const auto a2 = arma::norm(p2);
        if (a2 > real_type())
        {
            p2 *= real_type(1) / a2;
        }
        alpha(rank) = a2;

        const auto t = a2 * a2 + b1 * b1;
        //
        // Estimation of the Frobenius norm of A
        //
        // \|A\|_{F}^{2}= \sum_{K=1}^{rank(A)-1}
        //              (\alpha_{K}^{2} + \beta_{K}^{2}) + \alpha_{rank(A)}}
        //
        fnormA += t;
        error = t;
        if (t <= tol2 * fnormA)
        {
            // Converged
            break;
        }

        a1 = a2;
    }

    tol_error = sqrt(error);
    return;
}

} // namespace: expsum

#endif /* NUMERIC_PARTIAL_LANCZOS_BIDIAGONALIZATION_HPP */
