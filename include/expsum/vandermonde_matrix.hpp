#ifndef EXPSUM_VANDERMONDE_HPP
#define EXPSUM_VANDERMONDE_HPP

#include <cassert>

#include <armadillo>

#include "expsum/lsqr.hpp"

namespace expsum
{
namespace detail
{
//
// Fused multiply-add
//
// Computes `(x * y) + z` as if to infinite precision and rounded only once to
// fit the result type.
//
template <typename T>
inline T fma(T x, T y, T z)
{
    return std::fma(x, y, z);
}

template <typename T>
inline std::complex<T> fma(const std::complex<T>& x, const std::complex<T>& y,
                           const std::complex<T>& z)
{
    const auto re =
        std::fma(-x.imag(), y.imag(), std::fma(x.real(), y.real(), z.real()));
    const auto im =
        std::fma(x.real(), y.imag(), std::fma(x.imag(), y.real(), z.imag()));
    return {re, im};
}

} // namespace detail

//
// Generalized Vandermonde matrix
//
// This class represents a ``$m \times n$`` generalized Vandermonde matrix of
// the form
//
// ``` math
//   \bm{V}(\bm{t}) = \left[ \begin{array}{}
//     1         & 1         & \dots  & 1         \\
//     t_{1}^{}  & t_{2}^{}  & \dots  & t_{n}^{}  \\
//     \vdots    & \vdots    & \ddots & \vdots    \\
//     t_{1}^{m} & t_{2}^{m} & \dots  & t_{n}^{m} \\
//   \end{array} \right]
// ```
template <typename T>
class vandermonde_matrix
{
public:
    using value_type  = T;
    using real_type   = typename arma::get_pod_type<value_type>::result;
    using size_type   = arma::uword;
    using vector_type = arma::Col<value_type>;
    using matrix_type = arma::Mat<value_type>;

private:
    size_type nrows_;
    size_type ncols_;

    vector_type coeffs_;
    mutable vector_type work_;

public:
    vandermonde_matrix()                          = default;
    vandermonde_matrix(const vandermonde_matrix&) = default;
    vandermonde_matrix(vandermonde_matrix&&)      = default;
    ~vandermonde_matrix()                         = default;
    vandermonde_matrix& operator=(const vandermonde_matrix&) = default;
    vandermonde_matrix& operator=(vandermonde_matrix&&) = default;

    vandermonde_matrix(size_type nrows, size_type ncols)
        : nrows_(nrows), ncols_(ncols), coeffs_(ncols), work_(ncols)
    {
    }

    template <typename T1>
    vandermonde_matrix(
        size_type nrows, const T1& coeffs,
        typename std::enable_if<arma::is_arma_type<T1>::value>::type* = 0)
        : nrows_(nrows), ncols_(coeffs.n_elem), coeffs_(coeffs), work_(ncols_)
    {
    }

    size_type nrows() const
    {
        return nrows_;
    }

    size_type ncols() const
    {
        return ncols_;
    }

    // Returns size of vector that defines the Vandermonde matrix
    size_type size() const
    {
        return ncols();
    }

    // Resize matrix
    void resize(size_type nrows, size_type ncols)
    {
        nrows_ = nrows;
        ncols_ = ncols;
        coeffs_.resize(ncols);
        work_.resize(ncols);
    }

    // Set elements of Vandermonde matrix
    template <typename T1>
    typename std::enable_if<arma::is_arma_type<T1>::value>::type
    set_coeffs(const T1& coeffs)
    {
        ncols_ = coeffs.n_elem;
        if (coeffs_.size() < ncols_)
        {
            coeffs_.resize(ncols_);
            work_.resize(ncols_);
        }
        coeffs_.head(ncols_) = coeffs;
    }

    auto coeffs() const -> decltype(coeffs_.head(ncols_))
    {
        return coeffs_.head(ncols_);
    }

    matrix_type as_dense_matrix() const
    {
        matrix_type ret(nrows(), ncols());
        for (size_type j = 0; j < ncols(); ++j)
        {
            auto x = coeffs_(j);
            auto v = value_type(1);
            ret(0, j) = v;
            for (size_type i = 1; i < nrows(); ++i)
            {
                v *= x;
                ret(i, j) = v;
            }
        }
        return ret;
    }
    ///
    /// Compute `y = A * x + beta * y`
    ///
    template <typename U1, typename U2>
    typename std::enable_if<(arma::is_arma_type<U1>::value &&
                             arma::is_arma_type<U2>::value),
                            void>::type
    apply(const U1& x, value_type beta, U2& y) const
    {
        assert(x.n_elem == ncols());
        assert(y.n_elem == nrows());

        auto w = work_.head(ncols_);
        w      = x;

        y(0) = arma::sum(w) + beta * y(0);
        for (size_type i = 1; i < nrows(); ++i)
        {
            w %= coeffs();
            y(i) = arma::sum(w) + beta * y(i);
        }
    }
    ///
    /// Compute `y = A.t() * x + beta * y`
    ///
    template <typename U1, typename U2>
    typename std::enable_if<(arma::is_arma_type<U1>::value &&
                             arma::is_arma_type<U2>::value),
                            void>::type
    apply_trans(const U1& x, value_type beta, U2& y) const
    {
        assert(x.n_elem == nrows());
        assert(y.n_elem == ncols());

        for (size_type i = 0; i < ncols(); ++i)
        {
            const auto zi = arma::access::alt_conj(coeffs_(i));
            // Evaluate polynomial using Honer's method
            auto s = value_type();
            for (size_type j = 0; j < nrows(); ++j)
            {
                // s = detail::fma(s, zi, x(nrows() - j - 1));
                s = s * zi + x(nrows() - j - 1);
            }

            y(i) = s + beta * y(i);
        }
    }
};

//
// Compute LDL^H factorization of the gramian matrix of column Vandermonde
// matrix.
//
template <typename T, typename MatrixT, typename MatrixWork>
void ldlt_vandermonde_gramian(const vandermonde_matrix<T>& V, MatrixT& mat,
                              MatrixWork& work)
{
    using value_type  = typename vandermonde_matrix<T>::value_type;
    using vector_type = typename vandermonde_matrix<T>::vector_type;
    using real_type   = typename arma::get_pod_type<value_type>::result;
    using size_type   = arma::uword;

    using std::abs;
    using std::real;
    using std::sqrt;

    constexpr const auto one = value_type(1);
    static const auto tiny   = sqrt(std::numeric_limits<real_type>::min());

    const auto z      = V.coeffs();
    const size_type m = V.nrows();
    const size_type n = V.ncols();

    assert(mat.n_rows == n && mat.n_cols == n);
    assert(work.n_rows == n && work.n_cols >= 4);

    // ----- Initialization
    auto y1 = work.col(0);
    auto y2 = work.col(1);
    auto x1 = work.col(2);
    auto x2 = work.col(3);

    auto gramian = [&](size_type i, size_type j) {
        const auto arg = arma::access::alt_conj(z(i)) * z(j);
        return arg == one ? value_type(m)
                          : (one - std::pow(arg, m)) / (one - arg);
    };

    auto sigma2 = gramian(0, 0);
    auto b0     = mat.col(0);
    b0(0)       = one;
    for (size_type j = 1; j < n; ++j)
    {
        b0(j) = gramian(j, 0) / sigma2;
    }

    y1 = arma::ones<vector_type>(n) - b0;
    y2 = arma::pow(arma::conj(z), m);
    y2 -= y2(0) * b0;
    x1 = value_type(1) / arma::conj(z);
    x1 -= x1(0) * b0;
    x2 = arma::pow(-arma::conj(z), m - 1);
    x2 -= x2(0) * b0;

    b0(0) = sigma2;

    for (size_type k = 1; k < n; ++k)
    {
        auto bk = mat.col(k);

        auto mu1 = x1(k);
        auto mu2 = x2(k);
        auto nu1 = arma::access::alt_conj(y1(k));
        auto nu2 = arma::access::alt_conj(y2(k));

        auto zk     = z(k);
        auto zk_inv = one / zk;
        auto denom  = arma::access::alt_conj(zk_inv) - zk;

        if (abs(denom) < tiny)
        {
            sigma2         = real_type(m);
            const auto bkk = real(mat(k, k));
            for (size_type j = 0; j < k; ++j)
            {
                const auto bkj = mat(k, j);
                sigma2 -= std::norm(bkj) * bkk;
            }
        }
        else
        {
            sigma2 = (mu1 * nu1 + mu2 * nu2) / denom;
        }

        bk(k) = sigma2;

        for (size_type i = k + 1; i < n; ++i)
        {
            const auto d1 = (arma::access::alt_conj(mu1) * y1(i) +
                             arma::access::alt_conj(mu2) * y2(i));
            const auto d2 = sigma2 * (zk_inv - arma::access::alt_conj(z(i)));
            bk(i)         = d1 / d2;
        }

        const size_type nt = n - k - 1;

        x1.tail(nt) -= mu1 * bk.tail(nt);
        x2.tail(nt) -= mu2 * bk.tail(nt);
        y1.tail(nt) -= arma::access::alt_conj(nu1) * bk.tail(nt);
        y2.tail(nt) -= arma::access::alt_conj(nu2) * bk.tail(nt);
    }

    // arma::Mat<T> tmp_V(V.as_dense_matrix());
    // arma::Mat<T> tmp_H(V.t() * V);

    // for (size_type k = 0; k < n; ++k)
    // {
    //     auto bk          = mat.col(k);
    //     const auto sigma = sqrt(real(mat(k, k)));

    //     bk(k) = sigma;
    //     bk.tail(n - k - 1) *= sigma;
    // }

    return;
}

///
/// Solve overdetermined Vandermonde system
///
template <typename T>
struct vandermonde_least_squares
{
    using size_type  = arma::uword;
    using value_type = T;
    using real_type  = typename arma::get_pod_type<value_type>::result;

    using matrix_type             = arma::Mat<value_type>;
    using vector_type             = arma::Col<value_type>;
    using vandermonde_matrix_type = vandermonde_matrix<value_type>;

private:
    struct matvec
    {
        matvec(const vandermonde_matrix_type& mat_V) : mat_V_(mat_V)
        {
        }

        template <typename U1, typename U2>
        void operator()(const U1& x, value_type beta, U2& y) const
        {
            mat_V_.apply(x, beta, y);
        }

        const vandermonde_matrix_type& mat_V_;
    };

    struct matvec_trans
    {
        matvec_trans(const vandermonde_matrix_type& mat_V) : mat_V_(mat_V)
        {
        }

        template <typename U1, typename U2>
        void operator()(const U1& x, value_type beta, U2& y) const
        {
            mat_V_.apply_trans(x, beta, y);
        }

        const vandermonde_matrix_type& mat_V_;
    };

    struct preconditioner
    {
        // ldlt -- Result of Cholesky (LDL^T) decomposition of V.t() * V
        preconditioner(const matrix_type& ldlt) : ldlt_(ldlt){};

        template <typename Rhs, typename Dest>
        void operator()(const Rhs& rhs, Dest& dst) const
        {
            constexpr const real_type tol =
                real_type(1) / std::numeric_limits<real_type>::max();
            // auto mat_L = arma::trimatl(chol_VtV_);
            // dst = arma::solve(mat_L, rhs);
            // dst = arma::solve(mat_L.t(), dst);
            char uplo           = 'L';
            char trans1         = 'N';
            char trans2         = 'C';
            char diag           = 'U';
            arma::blas_int n    = arma::blas_int(ldlt_.n_rows);
            arma::blas_int nrhs = dst.n_cols;
            arma::blas_int info = 0;
            dst                 = rhs;

            arma::lapack::trtrs(&uplo, &trans1, &diag, &n, &nrhs,
                                ldlt_.memptr(), &n, &dst[0], &n, &info);

            for (size_type i = 0; i < ldlt_.n_rows; ++i)
            {
                if (std::abs(ldlt_(i, i)) > tol)
                {
                    dst(i) /= ldlt_(i, i);
                }
                else
                {
                    dst(i) = value_type();
                }
            }

            arma::lapack::trtrs(&uplo, &trans2, &diag, &n, &nrhs,
                                ldlt_.memptr(), &n, &dst[0], &n, &info);

            return;
        }

        const matrix_type& ldlt_;
    };

public:
    vandermonde_least_squares()
        : max_iterations_(),
          iterations_(),
          tolerance_(arma::Datum<real_type>::eps),
          error_()
    {
    }

    static size_type
    inquery_workspace_size(const vandermonde_matrix_type& mat_V)
    {
        return inquery_workspace_size(mat_V.nrows(), mat_V.ncols());
    }

    static size_type inquery_workspace_size(size_type nrows, size_type ncols)
    {
        return (ncols + 3) * ncols + std::max(nrows, ncols);
    }

    // Solve overdetermined Vandermonde problem.
    template <typename Rhs, typename Dest>
    void solve(const vandermonde_matrix_type& mat_V, const Rhs& b, Dest& x,
               value_type* work)
    {
        const size_type m = mat_V.nrows();
        const size_type n = mat_V.ncols();

        assert(b.n_rows == m && x.n_rows == n);

        //
        // Compute Cholesky decomposition of `V.t() * V = L.t() * L`, where `V`
        // is input Vandermonde matrix while `L` is a lower triangular matrix.
        //
        value_type* ptr_chol = work;
        value_type* ptr_vec1 = ptr_chol + n * n;
        value_type* ptr_vec2 = ptr_vec1 + m;
        matrix_type chol_VtV(ptr_chol, n, n, /* copy_aux_mem */ false,
                             /* strict */ true);
        matrix_type vecs(ptr_vec1, n, 4, /* copy_aux_mem */ false,
                         /* strict */ true);
        ldlt_vandermonde_gramian(mat_V, chol_VtV, vecs);
        //
        // Solve `V * x = b` using preconditioned LSQR
        //
        matvec mv(mat_V);
        matvec_trans mv_trans(mat_V);
        preconditioner precond(chol_VtV);

        vector_type u(ptr_vec1, m, /*copy_aux_mem*/ false, /*strict*/ true);
        matrix_type tmp(ptr_vec2, n, 3, /*copy_aux_mem*/ false,
                        /*strict*/ true);
        u           = arma::conv_to<vector_type>::from(b);
        iterations_ = max_iterations_ ? max_iterations_ : 2 * n;
        error_      = tolerance_;
        lsqr(mv, mv_trans, u, x, precond, tmp, iterations_, error_);

        return;
    }

    /// Check the convergence
    bool converged() const
    {
        return iterations_ <= max_iterations_;
    }

    real_type tolerance() const
    {
        return tolerance_;
    }

    void set_tolerance(real_type tol)
    {
        tolerance_ = tol;
    }

    size_type iterations() const
    {
        return iterations_;
    }

    void set_max_iterations(size_type max_iter)
    {
        max_iterations_ = max_iter;
    }

    /// @return An estimation of residual error
    real_type error() const
    {
        return error_;
    }

private:
    size_type max_iterations_;
    size_type iterations_;
    real_type tolerance_;
    real_type error_;
};

} // namespace expsum

#endif /* EXPSUM_VANDERMONDE_HPP */
