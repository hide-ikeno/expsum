#ifndef EXPSUM_REDUCTION_HPP
#define EXPSUM_REDUCTION_HPP

#include <cassert>

#include <armadillo>

#include "expsum/cholesky_quasi_cauchy.hpp"
#include "expsum/coneig_sym_rrd.hpp"

namespace expsum
{

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
};

template <typename T>
template <typename VecP, typename VecW>
typename std::enable_if<(arma::is_basevec<VecP>::value &&
                         arma::is_basevec<VecW>::value),
                        void>::type
reduction_body<T>::run(const VecP& p, const VecW& w, real_type tol)
{
    using cholesky_rrd = cholesky_quasi_cauchy<value_type>;

    assert(p.n_elem == w.n_elem);

    const auto n = p.size();
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
    // Rank-revealing Cholesky factorization of quasi-Cauchy matrix
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

    size_ = cholesky_rrd::pivot_order(a, b, x, y, tol, ipiv_, work1);

    matrix_type X(ptr_X, n, size_, false, true);
    real_vector_type d(ptr_d, size_, false, true);

    cholesky_rrd::factorize(a, b, x, y, X, d, work1, work2);
    cholesky_rrd::apply_row_permutation(X, ipiv_, work1);

#ifdef DEBUG
    std::cout << "*** coneig_sym_rrd:" << std::endl;
#endif /* DEBUG */

    //
    // Compute con-eigenvalue decomposition of matrix C = X * D^2 * X.t()
    //
    //
    // NOTE: Required memory for workspace
    //
    // work size:
    //    for matrix G : n * n
    //    for matrix Y : n * n
    //    for workspace: 2 * n
    //    ------------------------------
    //    Total        : 2 * n * (n + 1)
    //
    // rwork size:
    //    if `T` is real type   : n
    //    if `T` is complex type: 3 * n
    //
    coneig_sym_rrd<T>::run(X, d, ptr_a, rwork_.memptr());

#ifdef DEBUG
    const auto norm_P = arma::norm(P);
    const auto resid_norm =
        arma::norm(P - arma::conj(X) * arma::diagmat(d) * X.st());
    const auto resid_orth =
        arma::norm(arma::eye<matrix_type>(size_, size_) - X.st() * X);
    std::cout << "    |p - conj(X) * D * X.st()|       = " << resid_norm << '\n'
              << "    |P - conj(X) * D * X.st()| / |P| = "
              << resid_norm / norm_P << '\n'
              << "    |I - X.st() * X|                 = " << resid_orth
              << '\n';
#endif /* DEBUG */

    auto sum_d  = real_type();
    size_type k = size_;
    while (k)
    {
        sum_d += d(k - 1);
        if (2 * sum_d > tol)
        {
            break;
        }
        --k;
    }
    size_ = k;

    std::cout << "*** truncation size = " << k << std::endl;
    if (k == size_type())
    {
        return;
    }

    auto viewX = X.head_cols(k);
    matrix_type A(ptr_a, k, k, false, true);

    a = p;
    b = viewX.st() * arma::sqrt(w);
    A = viewX.st() * arma::diagmat(a) * viewX;

    return;
}

} // namespace: expsum

#endif /* EXPSUM_REDUCTION_HPP */
