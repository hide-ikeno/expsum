#ifndef EXPSUM_BALANCED_TRUNCATION_HPP
#define EXPSUM_BALANCED_TRUNCATION_HPP

#include <cassert>

#include <armadillo>

#include "expsum/cholesky_quasi_cauchy.hpp"
#include "expsum/coneig_sym_rrd.hpp"

namespace expsum
{

template <typename T>
class balanced_truncation
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
    size_type size_;
    vector_type exponent_;
    vector_type weight_;

    index_vector_type ipiv_;
    vector_type work_;
    real_vector_type rwork_;

public:
    template <typename VecP, typename VecW>
    typename std::enable_if<(arma::is_basevec<VecP>::value &&
                             arma::is_basevec<VecW>::value),
                            void>::type
    run(const VecP& p, const VecW& w, real_type tol);

    void resize(size_type n)
    {
        if (exponent_.size() < n)
        {
            exponent_.set_size(n);
            weight_.set_size(n);
            ipiv_.set_size(n);
            work_.set_size(std::max(n * (n + 6), n * (3 * n + 1)));
            rwork_.set_size(arma::is_complex<T>::value ? 3 * n : n);
        }
    }

    size_type size() const
    {
        return size_;
    }

    //
    // @return Vector view to the exponents.
    //
    auto exponents() const -> decltype(exponent_.head(size_))
    {
        return exponent_.head(size_);
    }
    //
    // @return Vector view to the weights.
    //
    auto weights() const -> decltype(weight_.head(size_))
    {
        return weight_.head(size_);
    }

private:
    //
    // Compute eigenvalue decomposition of (n x n) symmetric matrix A.
    //
    // @A On input, matrix A, on exit, A is overwritten.
    // @d On exit, eigenvalues of matrix A
    //
    // --- for real matrix
    //     workspace size (lwork): n * n
    static void diagonalize(real_matrix_type& A, real_matrix_type& X,
                            real_vector_type& d, real_type* work,
                            size_type lwork, real_type* /*dummy*/)
    {
        char jobz = 'V';
        char uplo = 'U';

        auto n_     = arma::blas_int(A.n_rows);
        auto lwork_ = static_cast<arma::blas_int>(lwork);
        auto info_  = arma::blas_int();

        arma::lapack::syev(&jobz, &uplo, &n_, A.memptr(), &n_, d.memptr(), work,
                           &lwork_, &info_);
        if (info_)
        {
            std::ostringstream msg;
            msg << "[s/d]SYEV error: failed with info " << info_;
            throw std::logic_error(msg.str());
        }

        X = A;
    }

    // --- for complex matrix
    //     workspace size (lwork): n * n
    //     real workspace size: 2 * n
    static void diagonalize(complex_matrix_type& A, complex_matrix_type& X,
                            complex_vector_type& d, complex_type* work,
                            size_type lwork, real_type* rwork)
    {
        char jobvl = 'N'; // Do not compute left eigenvectors of A
        char jobvr = 'V'; // Compute right eigenvectors of A

        complex_type vl[2]; // dummy

        auto n_     = arma::blas_int(A.n_rows);
        auto ldvl_  = arma::blas_int(1);
        auto lwork_ = static_cast<arma::blas_int>(lwork);
        auto info_  = arma::blas_int();

        complex_type* work_ = work + A.n_elem;

        arma::lapack::cx_geev(&jobvl, &jobvr, &n_, A.memptr(), &n_, d.memptr(),
                              &vl[0], &ldvl_, X.memptr(), &n_, work_, &lwork_,
                              rwork, &info_);
        if (info_)
        {
            std::ostringstream msg;
            msg << "[c/z]GEEV error: failed with info " << info_;
            throw std::logic_error(msg.str());
        }

        for (size_type j = 0; j < X.n_cols; ++j)
        {
            auto xj          = X.col(j);
            const auto t     = arma::dot(xj, xj);
            const auto scale = value_type(1) / std::sqrt(t);
            xj *= scale;
        }
    }
};

template <typename T>
template <typename VecP, typename VecW>
typename std::enable_if<(arma::is_basevec<VecP>::value &&
                         arma::is_basevec<VecW>::value),
                        void>::type
balanced_truncation<T>::run(const VecP& p, const VecW& w, real_type tol)
{
    using cholesky_rrd = cholesky_quasi_cauchy<value_type>;

    assert(p.n_elem == w.n_elem);

    const auto n = p.n_elem;
    resize(n);
    if (n <= size_type(1))
    {
        return;
    }

    value_type* ptr_X = work_.memptr();
    value_type* ptr_a = ptr_X + n * n;
    real_type* ptr_d  = reinterpret_cast<real_type*>(exponent_.memptr());

#ifdef DEBUG
    matrix_type P(n, n);
#endif /* DEBUG */

    //
    // Set the factors that defines the quasi-Cauchy matrix
    //
    // P(i, j) = a(i) * b(j) /(x(i) + y(j))
    //
    // This is the controllability Gramian matrix of the system.
    //
    vector_type a(ptr_a + 0 * n, n, false, true);
    vector_type b(ptr_a + 1 * n, n, false, true);
    vector_type x(ptr_a + 2 * n, n, false, true);
    vector_type y(ptr_a + 3 * n, n, false, true);
    vector_type work1(ptr_a + 4 * n, n, false, true);
    vector_type work2(ptr_a + 5 * n, n, false, true);

    a = arma::sqrt(w);
    b = arma::conj(a);
    x = p;
    y = arma::conj(x);

#ifdef DEBUG
    for (size_type j = 0; j < n; ++j)
    {
        for (size_type i = 0; i < n; ++i)
        {
            P(i, j) = a(i) * b(j) / (x(i) + y(j));
        }
    }

    std::cout << "*** cholesky_quasi_cauchy:" << std::endl;
#endif /* DEBUG */

    //
    // Rank-revealing Cholesky factorization of Gramian matrix given as
    //
    // P = X * D^2 * X.t()
    //
    // Cholesky factorization can be computed accurately using the Gaussian
    // elimination with complete pivoting (GECP).
    //
    index_vector_type ipiv(ipiv_.memptr(), n, false, true);
    size_ = cholesky_rrd::pivot_order(a, b, x, y, tol, ipiv, work1);

    matrix_type X1(ptr_X, n, size_, false, true);
    real_vector_type d(ptr_d, size_, false, true);

    cholesky_rrd::factorize(a, b, x, y, X1, d, work1, work2);
    cholesky_rrd::apply_row_permutation(X1, ipiv, work1);

#ifdef DEBUG
    std::cout << "*** coneig_sym_rrd:" << std::endl;
#endif /* DEBUG */

    //
    // Compute the state-space transformation matrix.
    //
    // First compute eigenvalue decomposition of L * Q * L.t(), where Q is
    // observability Gramian matrix of the system,
    // which can be obtained as Q = P.st()
    //
    // Let us define G = D * X.st() * X * D, then eigenvalue decomposition L * Q
    // * L.t() = G.t() * G = X * S^2 * X.t(). As  described by Haut and Beylkin
    // (2011), the eigenvectors of G.t() * G become con-eigenvectors of matrix
    // P, i.e.,
    //
    // P = conj(X) * S * X.t().
    //
    // The matrix X is the desired transformation matrix of the system.
    //
    // NOTE: Required memory for workspace
    //
    // work size : 2 * n * (n + 1)
    // rwork size:
    //    if `T` is real type   : n
    //    if `T` is complex type: 3 * n
    //
    size_ = coneig_sym_rrd<T>::run(X1, d, tol, ptr_a, rwork_.memptr());

#ifdef DEBUG
    std::cout << "*** found " << size_ << " con-eigenpairs" << std::endl;
    auto Xv           = X1.head_cols(size_);
    auto dv           = d.head(size_);
    const auto norm_P = arma::norm(P);
    const auto resid_norm =
        arma::norm(P - arma::conj(Xv) * arma::diagmat(dv) * Xv.st());
    const auto resid_orth =
        arma::norm(arma::eye<matrix_type>(size_, size_) - Xv.st() * Xv);
    std::cout << "    |P - conj(X) * D * X.st()|       = " << resid_norm << '\n'
              << "    |P - conj(X) * D * X.st()| / |P| = "
              << resid_norm / norm_P << '\n'
              << "    |I - X.st() * X|                 = " << resid_orth
              << '\n';
#endif /* DEBUG */

    //
    // Apply transformation matrix
    //
    auto viewX = X1.head_cols(size_);
    matrix_type A1(ptr_a, size_, size_, false, true);
    vector_type p_(exponent_.memptr(), size_, false, true);
    vector_type w_(weight_.memptr(), size_, false, true);
    A1 = viewX.t() * arma::diagmat(p) * arma::conj(viewX);
    w_ = viewX.t() * arma::sqrt(w);

    matrix_type X2(ptr_X, size_, size_, false, true);
#ifdef DEBUG
    std::cout << "*** eigen-decomposition" << std::endl;
    matrix_type A1_orig(A1);
#endif

    diagonalize(A1, X2, p_, ptr_a + n * n, n * n, rwork_.memptr());

#ifdef DEBUG
    // const auto resid1 = arma::norm(A1_orig * X2 - X2 * arma::diagmat(p_));
    // const auto resid2 =
    //     arma::norm(X2.st() * A1_orig - arma::diagmat(p_) * X2.st());
    const auto norm_A1 = arma::norm(A1_orig);
    const auto resid2  = arma::norm(A1_orig - X2 * arma::diagmat(p_) * X2.st());

    const auto resid_orth2 =
        arma::norm(arma::eye<matrix_type>(size_, size_) - X2.st() * X2);
    // std::cout << "    |A * X - X * D|           = " << resid1 << '\n'
    //           << "    |X.st() * A - D * X.st()| = " << resid2 << '\n'
    //           << "    |I - X.st() * X|          = " << resid_orth2 <<
    //           std::endl;
    std::cout << "    |A - X * D * X.st()|       = " << resid2 << '\n'
              << "    |A - X * D * X.st()| / |A| = " << resid2 / norm_A1 << '\n'
              << "    |I - X.st() * X|           = " << resid_orth2
              << std::endl;
#endif /* DEBUG */

    A1.col(0) = w_;
    w_        = X2.st() * A1.col(0);
    w_        = arma::square(w_);

    return;
}

} // namespace: expsum

#endif /* EXPSUM_BALANCED_TRUNCATION_HPP */
