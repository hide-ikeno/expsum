#ifndef EXPSUM_LSQR_HPP
#define EXPSUM_LSQR_HPP

#include <armadillo>

namespace expsum
{
///
/// lsqr
///
/// `lsqr` solves a least squares problem
///
/// ``` math
///   \min \| A\bm{x} - \bm{b} \|_{2},
/// ```
///
/// where ``$A \in \mathcal{C}^{m \times n}, \, \bm{b} \in \mathcal{C}^{n} $``
/// with ``$m \geq n$``.
///
/// @param[in]  matvec A function object that compute matrix-vector product
///   `y += A * x`
/// @param[in]  matvec_trans A function object that compute matrix-vector
///   product `y += A.t() * x`
/// @param[inout] u A vector of length m. On input the right hand side vector
///     ``$b$``, on exit, `u` is overwritten.
/// @param[out] x A vector of length n. On input and initial guess of solution,
///    on exit, the computed solution.
/// @param[in] precond_solve Functor for applying preconditioner to a vector.
/// @param[out] work A matrix of size ``$n \times 3$`` used as workspace.
/// @param[inout] iterations  On input the max number of iteration, on exit the
///    number of performed iterations.
/// @param[inout] tol_error  On input the tolerance error, on exit as estimation
///    of the relative error.
///

template <typename MatVec, typename MatVecTrans, typename VectorU,
          typename VectorX, typename Preconditioner, typename MatrixWork>
void lsqr(MatVec matvec, MatVecTrans matvec_trans, VectorU& u, VectorX& x,
          const Preconditioner& apply_preconditioner, MatrixWork& work,
          arma::uword& iterations, typename VectorX::pod_type& tol_error)
{
    using value_type = typename VectorX::elem_type;
    using real_type  = typename VectorX::pod_type;
    using size_type  = arma::uword;

    using std::sqrt;
    using std::real;
    using std::abs;
    using std::norm;

    const real_type atol = tol_error;
    const real_type btol = tol_error;

    auto p = work.col(0);
    auto v = work.col(1);
    auto w = work.col(2);
    //
    // --- Initialization
    //
    x.zeros();
    real_type alpha = real_type();
    real_type beta  = arma::norm(u);
    if (beta > real_type())
    {
        u *= real_type(1) / beta;         // normalize
        matvec_trans(u, value_type(), p); // p <-- A.t() * u;
        apply_preconditioner(p, v);       // v <-- M.inv() * p
        alpha = sqrt(real(arma::cdot(v, p)));
        if (alpha > real_type())
        {
            v *= real_type(1) / alpha; // normalize
            w = v;
        }
    }

    // Estimation of the norm of `A.t() * r`
    real_type norm_Atr = alpha * beta;
    if (norm_Atr == real_type())
    {
        // x = 0 is the exact solution
        return;
    }

    real_type rhobar = alpha;
    real_type phibar = beta;

    // Estimations of the Frobenius norm of matrix A
    real_type norm_A = real_type();
    // Norm of r.h.s. vector b
    real_type norm_b = beta;
    // Estimation of the norm of residual vector `r = b - A * x`.
    real_type norm_r = beta;
    // Estimation of the norm of solution vector x.
    real_type norm_x = real_type();

    real_type cs2     = real_type(-1);
    real_type sn2     = real_type();
    real_type z       = real_type();
    real_type norm_bb = real_type();
    real_type norm_xx = real_type();

    size_type iter = 0;
    while (++iter <= iterations)
    {
        //
        // --- Bidiagonalization
        //
        // Perform the next step of the bidiagonalization with M-weighted inner
        // product.  `beta, u, alpha, v`. These satisfy the relations
        //
        // ``` math
        //  \beta_{i+1} \bm{u}_{i+1}
        //    &= A \bm{v}_{i} - \alpha_{i} \bm{u}_{i},
        //  \alpha_{i+1} \bm{v}_{i+1}
        //    &= M^{-1} A^{H} \bm{u}_{i+1} - \beta_{i+1} \bm{v}_{i},
        //  \alpha_{i+1} &= (\bm{v}_{i+1}^{H} M \bm{v}_{i+1}^{})^{1/2}
        // ```
        //
        matvec(v, -alpha, u); // u <-- A * v - alpha * u
        beta = arma::norm(u);
        norm_bb += norm(alpha) + norm(beta);

        if (beta > real_type())
        {
            u *= real_type(1) / beta;
            matvec_trans(u, -beta, p);  // p <-- A.t() * u - beta * p
            apply_preconditioner(p, v); // v <-- M.inv() * p
            alpha = sqrt(real(arma::cdot(v, p)));
            if (alpha > real_type())
            {
                v *= real_type(1) / alpha;
            }
        }
        //
        // --- Orthogonal transformation
        //
        // Construct and apply next orthogonal transformation (plane rotation)
        // to eliminate the subdiagonal element (beta) of the lower-bidiagonal
        // matrix, giving an upper-bidiagonal matrix. The explicit form of the
        // transformation is given as
        //
        // [cs  sn][rhobar      0  phibar]   [rho   theta     phi ]
        // [sn -cs][  beta  alpha       0] = [  0  rhobar' phibar']
        //
        real_type rho   = sqrt(norm(rhobar) + norm(beta));
        real_type cs    = rhobar / rho;
        real_type sn    = beta / rho;
        real_type theta = sn * alpha;
        rhobar          = -cs * alpha;
        real_type phi   = cs * phibar;
        phibar *= sn;
        //
        // Update vectors
        //
        // norm_dd += (w / rho).squaredNorm();
        x += (phi / rho) * w;
        w = v - (theta / rho) * w;
        //
        // Estimate norm of x using the result of plane rotation
        //
        real_type delta     = sn2 * rho;
        real_type gamma_bar = -cs2 * rho;
        real_type rhs       = phi - delta * z;
        real_type zbar      = rhs / gamma_bar;
        norm_x              = sqrt(norm_xx + norm(zbar));
        real_type gamma     = sqrt(norm(gamma_bar) + norm(theta));
        cs2                 = gamma_bar / gamma;
        sn2                 = theta / gamma;
        z                   = rhs / gamma;
        norm_xx += norm(z);

        //
        // Test for convergence.
        //
        norm_A = sqrt(norm_bb);
        // cond_A   = norm_A * sqrt(norm_dd);
        norm_r   = phibar;
        norm_Atr = phibar * alpha * abs(cs);
        //
        // Stop iteration if
        //
        // |A^{T} \bm{r}|_{F} / |A^{T}|_F |\bm{r}| < tol
        //
        // if (norm_Atr <= norm_A * norm_r * tol_error)
        // {
        //     break;
        // }
        if (norm_r <= btol * norm_b + atol * norm_A * norm_x)
        {
            break;
        }
    }

    tol_error  = norm_r;
    iterations = iter;
}

} // namespace expsum

#endif /* EXPSUM_LSQR_HPP */
