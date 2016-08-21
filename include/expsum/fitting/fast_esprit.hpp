#ifndef EXPSUM_FAST_ESPRIT_HPP
#define EXPSUM_FAST_ESPRIT_HPP

#include <algorithm>
#include <cassert>
#include <tuple>

#include "arma/lapack_extra.hpp"

#include "expsum/fitting/hankel_matrix.hpp"
#include "expsum/fitting/partial_lanczos_bidiagonalization.hpp"
#include "expsum/fitting/vandermonde_matrix.hpp"

namespace expsum
{
namespace detail
{
//
// Wrapper class of LAPACK xGGEV to compute the eigenvalues of the matrix pencil
// matrix pencil `z * F - G`. Left and right eigenvectors are not computed.
//
template <typename T>
struct lapack_ggev
{
    using value_type   = T;
    using real_type    = typename arma::get_pod_type<value_type>::result;
    using complex_type = std::complex<real_type>;

    static arma::blas_int invoke(arma::blas_int n, real_type* A, real_type* B,
                                 complex_type* alpha, real_type* beta,
                                 real_type* work, arma::blas_int lwork,
                                 real_type* rwork)
    {
        const bool query = (lwork == arma::blas_int(-1));

        char jobvl            = 'N';
        char jobvr            = 'N';
        real_type* alphar     = rwork;
        real_type* alphai     = rwork + n;
        real_type dummy       = 0;
        arma::blas_int idummy = 1;
        arma::blas_int info;

        arma::lapack::ggev(&jobvl, &jobvr, &n, A, &n, B, &n, alphar, alphai,
                           beta, &dummy, &idummy, &dummy, &idummy, work, &lwork,
                           &info);

        if (!query && info == arma::blas_int())
        {
            for (arma::blas_int i = 0; i < n; ++i)
            {
                alpha[i] = complex_type(alphar[i], alphai[i]);
            }
        }
        return info;
    }

    // Specialization for complex scalar type
    static arma::blas_int invoke(arma::blas_int n, complex_type* A,
                                 complex_type* B, complex_type* alpha,
                                 complex_type* beta, complex_type* work,
                                 arma::blas_int lwork, real_type* rwork)
    {
        char jobvl            = 'N';
        char jobvr            = 'N';
        complex_type dummy    = complex_type();
        arma::blas_int idummy = 1;
        arma::blas_int info;
        arma::lapack::cx_ggev(&jobvl, &jobvr, &n, A, &n, B, &n, alpha, beta,
                              &dummy, &idummy, &dummy, &idummy, work, &lwork,
                              rwork, &info);
        return info;
    }
};

//
// Compute the roots of Prony polynomials
//
template <typename T>
struct prony_root_solver
{
    using size_type    = arma::uword;
    using value_type   = T;
    using real_type    = typename arma::get_pod_type<value_type>::result;
    using complex_type = std::complex<real_type>;

    using matrix_type      = arma::Mat<value_type>;
    using vector_type      = arma::Col<value_type>;
    using real_vector_type = arma::Col<real_type>;

    using hankel_gemv_type = hankel_gemv<value_type>;

private:
    struct matvec
    {
        matvec(const hankel_gemv_type& h) : h_(h)
        {
        }

        template <typename U1, typename U2>
        void operator()(const U1& x, value_type beta, U2& y) const
        {
            h_.apply(x, beta, y);
        }

        const hankel_gemv_type& h_;
    };

    struct matvec_trans
    {
        matvec_trans(const hankel_gemv_type& h) : h_(h)
        {
        }

        template <typename U1, typename U2>
        void operator()(const U1& x, value_type beta, U2& y) const
        {
            h_.apply_trans(x, beta, y);
        }

        const hankel_gemv_type& h_;
    };

    //
    // Compute partial Lanczos bidiagonalization for the low-rank approximation
    // of trajectory matrix H, such that
    //
    //   |H - P * B * Q'|_F < eps * |H|_F,
    //
    // where,
    //
    //  - H: nr x nc trajectory (Hankel) matrix
    //  - P: nr x K matrix, orthornormal columns
    //  - Q: nc x K matrix, orthornormal columns
    //  - nr >= nc > K >= 1
    //
    // Here K = rank(H). The the value of K might be greater than or equals to
    // max_rank: this means that the matrix H has full rank. This may happen in
    // the cases:
    //
    //  - too few sampling points
    //  - sapling data contains large noise
    //  - eps is too small
    //
    static void run_pldb(const hankel_gemv_type& mat_h, value_type* ptr_p,
                         value_type* ptr_q, value_type* ptr_v,
                         real_type* ptr_alpha, real_type* ptr_beta,
                         real_type& tol_error, size_type& rank)
    {
        const size_type nr = mat_h.nrows();
        const size_type nc = mat_h.ncols();

        matvec mv(mat_h);
        matvec_trans mv_trans(mat_h);

        matrix_type P(ptr_p, nr, rank, /*copy_aux_mem*/ false, /*strict*/ true);
        matrix_type Q(ptr_q, nc, rank, /*copy_aux_mem*/ false, /*strict*/ true);
        vector_type vwork(ptr_v, nc, /*copy_aux_mem*/ false, /*strict*/ true);
        real_vector_type alpha(ptr_alpha, rank, false, true);
        real_vector_type beta(ptr_beta, rank, false, true);

        partial_lanczos_bidiagonalization(mv, mv_trans, alpha, beta, P, Q, rank,
                                          tol_error, vwork);

#ifdef DEBUG
        std::cout << "partial Lanczos bidiagonalizagion:\n"
                  << "  rank: " << rank << '\n'
                  << "  error estimation (|A - P * B * Q.t()|): " << tol_error
                  << '\n';
#endif
    }

public:
    //
    // Compute roots of prony polynomials using partial Lanczos
    // bidiagonalization.
    //
    // Compute partial Lanczos bidiagonalization for the low-rank approximation
    // of trajectory matrix H, such that
    //
    //   |H - P * B * Q'|_F < eps * |H|_F,
    //
    // Let us define
    //
    //   Q0: removing the last row of matrix Q
    //   Q1: removing the first row of matrix Q
    //   F = Q0.t() * Q0
    //   G = Q1.t() * Q0.
    //
    // Then the roots of Prony polynomials are obtained as the eigenvalues of
    // matrix pencil `z * F - G`.
    //
    // The above eigenvalues can be efficiently computed by generalized Schur
    // decomposition (QZ-algorithm), which be computed using LAPACK `xGGEV` or
    // `xGGEV3` routine.
    //
    // Work and rwork must be properly aligned
    static size_type solve(hankel_gemv<value_type>& mat_h, size_type max_rank,
                           complex_type* dst, real_type& tol_error,
                           value_type* work, real_type* rwork)
    {
        const size_type nr = mat_h.nrows();
        const size_type nc = mat_h.ncols();
        //
        // Partial Lanczos bidiagonalization of Hankel matrix `matH`.
        //
        value_type* ptr_q    = work;
        value_type* ptr_p    = ptr_q + nc * max_rank;
        value_type* ptr_v    = ptr_p + nr * max_rank;
        real_type* ptr_alpha = rwork;
        real_type* ptr_beta  = rwork + max_rank;
        size_type rank       = max_rank;
        run_pldb(mat_h, ptr_p, ptr_q, ptr_v, ptr_alpha, ptr_beta, tol_error,
                 rank);
        //
        // Compute generalized eigenvalues of matrix pencil `z * F - G`.
        //
        matrix_type Q(ptr_q, nc, rank, max_rank);
        value_type* ptr_f     = ptr_p;
        value_type* ptr_g     = ptr_f + max_rank * max_rank;
        value_type* ptr_denom = ptr_g + max_rank * max_rank;

        auto Q0 = Q(arma::span(0, nc - 2), arma::span(0, rank - 1));
        auto Q1 = Q(arma::span(1, nc - 1), arma::span(0, rank - 1));

        matrix_type F(ptr_f, rank, rank);
        matrix_type G(ptr_g, rank, rank);

        F = Q0.t() * Q0;
        G = Q1.t() * Q0;

        auto n     = static_cast<arma::blas_int>(rank);
        auto lwork = static_cast<arma::blas_int>(Q.n_elem);
        // matrix Q is used as workspace
        auto info = lapack_ggev<value_type>::invoke(n, G.memptr(), F.memptr(),
                                                    dst, ptr_denom, Q.memptr(),
                                                    lwork, rwork);
        if (info)
        {
            std::ostringstream msg;
            msg << "Lapack xGGEV failed with info = " << info;
            throw std::runtime_error(msg.str());
        }

        for (size_type i = 0; i < rank; ++i)
        {
            dst[i] /= ptr_denom[i];
        }

        return rank;
    }
};

} // namespace detail

///
/// ESPRIT algorihtm based on the partial Lanczos bidiagonalizagion.
///
template <typename T>
class fast_esprit
{
public:
    using size_type   = arma::uword;
    using value_type  = T;
    using vector_type = arma::Col<value_type>;
    using matrix_type = arma::Mat<value_type>;

    using real_type        = typename matrix_type::pod_type;
    using real_vector_type = arma::Col<real_type>;
    using real_matrix      = arma::Mat<real_type>;

    using complex_type        = std::complex<real_type>;
    using complex_vector_type = arma::Col<complex_type>;
    using complex_matrix_type = arma::Mat<complex_type>;

private:
    using hankel_gemv_type = hankel_gemv<value_type>;
    using vandermonde_type = vandermonde_matrix<complex_type>;
    constexpr static const bool is_complex =
        arma::is_complex<value_type>::value;

    size_type nrows_;
    size_type ncols_;
    size_type nterms_;
    complex_vector_type exponent_;
    complex_vector_type weight_;

    real_type error_; // estimation of error
    hankel_gemv_type mat_h_;
    vandermonde_type mat_v_;
    vector_type mem_work_;
    real_vector_type mem_rwork_;

public:
    fast_esprit() = default;

    /*!
     * \param[in] N number of sample data
     * \param[in] L window size. This is equals to the number of rows of
     * generalized Hankel matrix.
     * \param[in] M maxumum number of terms used for the exponential sum.
     * \pre <tt> N >= L >= N / 2 >= 1 </tt> and <tt> N - L + 1 >= M >= 1.
     */
    fast_esprit(size_type N, size_type L, size_type M)
        : nrows_(L),
          ncols_(N - L + 1),
          nterms_(),
          exponent_(M),
          weight_(M),
          mat_h_(nrows_, ncols_),
          mat_v_(N, M),
          mem_work_(workspace_size(nrows_, ncols_, M)),
          mem_rwork_((arma::is_complex<value_type>::value ? 8 : 2) * M)
    {
        assert(nrows_ >= M && ncols_ >= M && M >= 1);
    }

    fast_esprit(const fast_esprit&) = default;
    fast_esprit(fast_esprit&&)      = default;
    ~fast_esprit()                  = default;

    fast_esprit& operator=(const fast_esprit& rhs) = default;
    fast_esprit& operator=(fast_esprit&& rhs) = default;

    /*!
     * \return number of sampling data
     * */
    size_type size() const
    {
        return nrows_ + ncols_ - 1;
    }

    /*!
     * \return number of rows of trajectory matrix
     */
    size_type nrows() const
    {
        return nrows_;
    }

    /*!
     * \return number of columns of trajectory matrix. This should be a upper
     * bound of the number of exponential functions.
     */
    size_type ncols() const
    {
        return ncols_;
    }

    /*!
     * Reset data sizes and reallocate memories for working space, if necessary.
     *
     * \param[in] N number of sample data
     * \param[in] L window size. This is equals to the number of rows of
     * generalized Hankel matrix.
     * \param[in] M maxumum number of terms used for the exponential sum.
     * \pre <tt> N >= L >= N / 2 >= 1 </tt> and <tt> N - L + 1 >= M >= 1.
     */
    void resize(size_type N, size_type L, size_type M)
    {
        nrows_ = L;
        ncols_ = N - L + 1;

        assert(nrows_ >= M && ncols_ >= M && M >= 1);

        nterms_ = 0;
        exponent_.resize(M);
        weight_.resize(M);

        mat_h_.resize(nrows_, ncols_);
        mat_v_.resize(N, M);
        mem_work_.resize(workspace_size(nrows_, ncols_, M));
        mem_rwork_.resize((is_complex ? 8 : 2) * M);
    }

    /*!
     * Compute non-linear approximation of by as the exponential sums.
     *
     * \param[in] f values of the target function sampled on the equispaced
     * grid. The first \c numSamples() elemnets of \c f are used as a sampled
     * data. In case <tt> f.size() < numSamples() </tt> then, last <tt>
     * numSamples() - f.size() </tt> elements are padded by zeros.
     * \param[in] eps small positive number <tt>(0 < eps < 1)</tt> that
     * controlls the accuracy of the fit.
     * \param[in] x0 argument of first sampling data
     * \param[in] delta spacing between neighbouring sample points
     */
    template <typename U>
    typename std::enable_if<arma::is_arma_type<U>::value>::type
    fit(const U& f, real_type x0, real_type delta, real_type eps);

    ///
    /// @return Vector view to the exponents.
    ///
    auto exponents() const -> decltype(exponent_.head(nterms_))
    {
        return exponent_.head(nterms_);
    }

    ///
    /// @return Vector view to the weights.
    ///
    auto weights() const -> decltype(weight_.head(nterms_))
    {
        return weight_.head(nterms_);
    }

    ///
    /// Evaluate exponential sum at a point
    ///
    complex_type eval_at(real_type x) const
    {
        return arma::sum(arma::exp(-x * exponents()) % weights());
    }

private:
    static size_type workspace_size(size_type nrows, size_type ncols,
                                    size_type max_terms)
    {
        const size_type mm = max_terms * max_terms;
        const size_type s1 = nrows * max_terms;
        const size_type s2 = ncols * max_terms;
        const size_type s3 = 2 * mm;
        const size_type s4 = max_terms;
        const size_type s5 = (is_complex ? 1 : 2) * (mm + nrows + ncols);

        return std::max(s1 + std::max(s2, s3) + s4, s5);
    }
};

//------------------------------------------------------------------------------
// Implementation of public member functions
//------------------------------------------------------------------------------

template <typename T>
template <typename U>
typename std::enable_if<arma::is_arma_type<U>::value>::type
fast_esprit<T>::fit(const U& f, real_type x0, real_type delta, real_type eps)
{
    assert(f.is_vec());
    assert(real_type() < eps && eps < real_type(1));
    //
    // Setup general fast Hankel matrix-vector product.
    //
    mat_h_.set_coeffs(f);
    //
    // Compute roots of prony polynomials
    //
    const size_type max_rank = exponent_.size();
    real_type tol_error      = eps;
    //
    // NOTE: on exit, the `nterms_` holds number of roots, and first `nterms_`
    // elements of `exponent_` stores values of Prony roots, z_{i}.
    // z_{i} is related to the exponent \zeta_{i} as z(i) = \exp(\zeta_{i}), so
    // we need to calculate log(z(i)) later.
    //
    nterms_ = detail::prony_root_solver<value_type>::solve(
        mat_h_, max_rank, exponent_.memptr(), tol_error, mem_work_.memptr(),
        mem_rwork_.memptr());

    error_ = tol_error;

#ifdef DEBUG
    std::cout << "(fast_esprit::fit)\n"
                 "  number of prony roots: "
              << nterms_
              << "\n  estimated error for prony system: " << tol_error
              << std::endl;
#endif /* DEBUG */
    //
    // Calculate weights of exponentials
    //
    if (nterms_ > 0)
    {
        //
        // Solve overdetermined Vandermonde system
        //
        vandermonde_least_squares<complex_type> ls_solver;
        ls_solver.set_tolerance(eps);
        mat_v_.set_coeffs(exponents());
        auto dst = weight_.head(nterms_);
        complex_type* cwork =
            reinterpret_cast<complex_type*>(mem_work_.memptr());
        ls_solver.solve(mat_v_, f, dst, cwork);
        //
        // Adjust computed paremeters
        //
        auto xi_ = exponent_.head(nterms_);
        auto w_  = weight_.head(nterms_);
        xi_      = -arma::log(xi_) / delta;
        if (x0 != real_type())
        {
            w_ %= arma::exp(-xi_ * x0);
        }

#ifdef DEBUG
        std::cout << "\n  estimated residual of least squares solution: "
                  << tol_error << std::endl;
#endif /* DEBUG */
        error_ = std::max(tol_error, error_);
    }

    return;
}

} // namespace: expsum

#endif /* EXPSUM_FAST_ESPRIT_HPP */
