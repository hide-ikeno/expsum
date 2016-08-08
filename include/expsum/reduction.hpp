#ifndef EXPSUM_REDUCTION_HPP
#define EXPSUM_REDUCTION_HPP

#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>

#include <armadillo>

#include "expsum/jacobi_svd.hpp"
#include "expsum/numeric.hpp"

namespace expsum
{
namespace detail
{
//
// Rank-revealing Cholesky decomposition for a positive-definite quasi-Cauchy
// matrix.
//
// This class computes the Cholesky decomposition of ``$n \times n$``
// quasi-Cauchy matrix defined as
//
// ``` math
//  C_{ij}=\frac{a_{i}^{} a_{j}^{\ast}}{x_{i}^{}+x_{i}^{\ast}}.
// ```
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
    //
    // Preconpute pivot order for the Cholesky factorization of $n \times n$
    // positive-definite Caunchy matrix $C_{ij}=a_{i}b_{j}/(x_{i}+y_{j}).$
    //
    static size_type pivot_order(vector_type& a, vector_type& x,
                                 arma::uvec& ipiv, real_type delta,
                                 real_vector_type& g)
    {
        const size_type n = a.size();
        assert(x.size() == n);
        assert(g.size() == n);
        assert(ipiv.size() == n);
        //
        // Set cutoff for GECP termination
        //
        const auto eta = arma::Datum<real_type>::eps * delta * delta;
        //
        // Form vector g(i) = a(i) * b(i) / (x(i) + y(i))
        //
        for (size_type i = 0; i < n; ++i)
        {
            g(i) = numeric::abs2(a(i)) / (real_type(2) * std::real(x(i)));
        }
        //
        // Initialize permutation matrix
        //
        std::iota(std::begin(ipiv), std::end(ipiv), size_type());

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
                std::swap(x(l), x(m));
                // Swap _rows_ of permutation matrix
                std::swap(ipiv(l), ipiv(m));
            }

            // Update diagonal of Schur complement
            const auto xm = x(m);
            for (size_type k = m + 1; k < n; ++k)
            {
                g(k) *= numeric::abs2(x(k) - xm) / numeric::abs2(x(k) + xm);
            }
            ++m;
        }
        //
        // Returns the truncation size
        //
        return m;
    }
    //
    // Compute Cholesky factors (`L` and diagonal part of `D`).
    //
    // The arrays `a,x` must be properly reordered by calling `pivot_order`
    // beforehand, so that the diagonal part of Cholesky factors appear in
    // decesnding order.
    //
    // @a vector of length ``$n$`` defining quasi-Cauchy matrix (reordered)
    // @x vector of length ``$n$`` defining quasi-Cauchy matrix (reordered)
    // @X Cholesky factor (lower triangular matrix)
    // @d diagonal elements of Cholesky factor ``$D$``
    // @alpha working space
    //
    static void factorize(const vector_type& a, const vector_type& x,
                          matrix_type& L, real_vector_type& d,
                          vector_type& alpha)
    {
        const auto n = L.n_rows;
        const auto m = L.n_cols;
        assert(a.size() == n);
        assert(x.size() == n);
        assert(d.size() == m);
        assert(alpha.size() == n);

        alpha = a;

        L.zeros();
        for (size_type l = 0; l < n; ++l)
        {
            L(l, 0) = alpha(l) * numeric::conj(alpha(0)) /
                      (x(l) + numeric::conj(x(0)));
        }

        for (size_type k = 1; k < m; ++k)
        {
            // Upgrade generators
            const auto xkm1 = x(k - 1);
            const auto ykm1 = numeric::conj(xkm1);
            for (size_type l = k; l < n; ++l)
            {
                alpha(l) *= (x(l) - xkm1) / (x(l) + ykm1);
            }
            // Extract k-th column for Cholesky factors
            const auto beta_k = numeric::conj(alpha(k));
            const auto y_k    = numeric::conj(x(k));
            for (size_type l = k; l < n; ++l)
            {
                L(l, k) = alpha(l) * beta_k / (x(l) + y_k);
            }
        }
        //
        // Scale strictly lower triangular part of G
        //   - diagonal part of G contains D**2
        //   - L = tril(G) * D^{-2} + I
        //
        for (size_type j = 0; j < m; ++j)
        {
            const auto djj   = std::real(L(j, j));
            const auto scale = real_type(1) / djj;
            d(j)             = std::sqrt(djj);
            L(j, j) = real_type(1);
            for (size_type i = j + 1; i < n; ++i)
            {
                L(i, j) *= scale;
            }
        }

        return;
    }

    //
    // Apply permutation matrix generated by previous decomposition
    //
    template <typename MatX>
    static void apply_row_permutation(MatX& X, const arma::uvec& ipiv,
                                      vector_type& work)
    {
        const size_type n = ipiv.size();
        assert(X.n_rows == n && work.n_elem == n);

        for (size_type j = 0; j < X.n_cols; ++j)
        {
            for (size_type i = 0; i < n; ++i)
            {
                work(ipiv(i)) = X(i, j);
            }
            X.col(j) = work;
        }
    }
    //
    // Reconstruct matrix from Cholesky factor
    //
    // @X Cholesky factor computed by `cholesky_quasi_cauchy::run`.
    // @d Cholesky factor computed by `cholesky_quasi_cauchy::run`.
    //
    static matrix_type reconstruct(const matrix_type& X, const arma::uvec& ipiv,
                                   const real_vector_type& d)
    {
        matrix_type XD(X * arma::diagmat(d));
        matrix_type PXD(arma::size(XD));
        apply_row_permutation(XD, ipiv, PXD);

        return matrix_type(PXD * PXD.t());
    }
};

} // namespace: detail

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

    arma::uvec ipiv_;
    matrix_type work1_;
    matrix_type work2_;
    matrix_type work3_;

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
            work1_.set_size(n, n);
            work2_.set_size(n, n);
            work3_.set_size(n, n);
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

private:
    //
    // Compute Cholesky decomposition of quasi-Cauchy matrix by Gaussian
    // elimination with complete pivoting.
    //
    // Factorize the matrix P = diagmat(a)  / (x + conj(x)) * diagmat(conj(a))
    // as
    //
    //   P = L * D^2 * L.t()
    //
    // where L is unit triangular with size (n x m), and D is a (m x m) diagonal
    // matrix, where m is the extracted rank of the matrix P.
    //
    static size_type cholesky_rrd(vector_type& a, vector_type& x,
                                  real_type delta, arma::uvec& ipiv,
                                  value_type* L, real_type* d,
                                  vector_type& work);
    //
    // Eigenvalue decomposition of the product of Gramian matrices
    //
    //   conj(P) * P = U * diagmat(sigma) * U.t()
    //
    // where P has a RRD of the form, P = X * D^2 * X.t()
    //
    static void coneig_gramian(size_type n, size_type m, value_type* ptr_X,
                               real_type* ptr_d, real_type* ptr_dinv,
                               value_type* ptr_U, value_type* ptr_W);
    // void eig_prod_gramian(matrix_type X, real_vector_type& d);
};

template <typename T>
template <typename VecP, typename VecW>
typename std::enable_if<(arma::is_basevec<VecP>::value &&
                         arma::is_basevec<VecW>::value),
                        void>::type
reduction_body<T>::run(const VecP& p, const VecW& w, real_type tol)
{
    const auto n = p.size();
    assert(w.size() == n);
    resize(n);
    if (n <= size_type(1))
    {
        return;
    }

    real_type* ptr_d    = reinterpret_cast<real_type*>(exponent_.memptr());
    real_type* ptr_dinv = reinterpret_cast<real_type*>(weight_.memptr());
    value_type* ptr_L   = work1_.memptr();
    {
        value_type* ptr_a    = weight_.memptr();
        value_type* ptr_x    = work2_.memptr();
        value_type* ptr_work = work2_.colptr(1);

        vector_type a(ptr_a, n, false, true);
        vector_type x(ptr_x, n, false, true);
        vector_type work(ptr_work, n, false, true);
        a     = arma::sqrt(w);
        x     = p;
        size_ = cholesky_rrd(a, x, tol, ipiv_, ptr_L, ptr_d, work);

        for (size_type i = 0; i < size_; ++i)
        {
            ptr_dinv[i] = real_type(1) / ptr_d[i];
        }
    }

    value_type* ptr_U = work2_.memptr();
    {
        coneig_gramian(n, size_, ptr_L, ptr_d, ptr_dinv, ptr_U,
                       work3_.memptr());
        auto sum_d  = real_type();
        size_type k = size_;
        while (k)
        {
            sum_d += ptr_d[k - 1];
            if (2 * sum_d > tol)
            {
                break;
            }
            --k;
        }
    }
}

template <typename T>
typename reduction_body<T>::size_type
reduction_body<T>::cholesky_rrd(vector_type& a, vector_type& x, real_type delta,
                                arma::uvec& ipiv, value_type* ptr_L,
                                real_type* ptr_d, vector_type& work)
{
    const size_type n = a.size();

#ifdef DEBUG
    matrix_type P(n, n);
    for (size_type j = 0; j < n; ++j)
    {
        for (size_type i = 0; i < n; ++i)
        {
            P(i, j) = a(i) * numeric::conj(a(j)) / (x(i) + numeric::conj(x(j)));
        }
    }
#endif

    real_vector_type g(reinterpret_cast<real_type*>(work.memptr()), n,
                       /*copy_aux_mem*/ false, /*strict*/ true);
    // Pre-compute correct pivot order of Cholesky decomposition
    const size_type m =
        detail::cholesky_quasi_cauchy<T>::pivot_order(a, x, ipiv, delta, g);
    matrix_type L(ptr_L, n, m, false, true);
    real_vector_type d(ptr_d, m, false, true);
    // Compute Cholesky factors
    detail::cholesky_quasi_cauchy<T>::factorize(a, x, L, d, work);
    // Apply permutation matrix
    detail::cholesky_quasi_cauchy<T>::apply_row_permutation(L, ipiv, work);

#ifdef DEBUG
    std::cout << "*** Cholesky-Cauchy:\n"
              << "    |P - L * D^2 * L.t()| = "
              << arma::norm(P - L * arma::diagmat(arma::square(d)) * L.t())
              << std::endl;
#endif

    return m;
}

template <typename T>
void reduction_body<T>::coneig_gramian(size_type n, size_type m,
                                       value_type* ptr_X, real_type* ptr_d,
                                       real_type* ptr_dinv, value_type* ptr_U,
                                       value_type* ptr_Y)
{
    matrix_type X(ptr_X, n, m, false, true);
    real_vector_type d(ptr_d, m, false, true);
    real_vector_type dinv(ptr_dinv, m, false, true);

    matrix_type G(ptr_U, m, m, false, true);

    auto D    = arma::diagmat(d);
    auto Dinv = arma::diagmat(dinv);

    matrix_type P(X * arma::diagmat(arma::square(d)) * X.t());

    //
    // Form G = D * (X.st() * X) * D
    //
    G = D * X.st() * X * D;
    //
    // Compute G = Q * R by Householder QR factorization
    //
    {
        auto m_         = static_cast<arma::blas_int>(m);
        auto lwork_     = static_cast<arma::blas_int>(n * (n - 1));
        value_type* tau = ptr_Y;
        arma::blas_int info;
        arma::lapack::geqrf(&m_, &m_, G.memptr(), &m_, tau, ptr_Y + m, &lwork_,
                            &info);
        if (info)
        {
            std::ostringstream msg;
            msg << "(reduction_body::eig_prod_gramian) xGEQRF failed with "
                   "info "
                << info;
            throw std::runtime_error(msg.str());
        }
    }
    //
    // Compute SVD of R = Y * S * V.t() using one-sided Jacobi method. We need
    // singular values and left singular vectors here.
    //
    matrix_type Y(ptr_Y, m, m, false, true);
    // overwrite d by singular value sigma
    real_vector_type sigma(ptr_d, m, false, true);
    Y.zeros();
    Y = arma::trimatu(G); // U = R
    // const auto ctol = arma::Datum<real_type>::eps * std::sqrt(real_type(m));
    const auto ctol = arma::Datum<real_type>::eps;
    jacobi_svd(Y, sigma, ctol);
    //
    // The eigenvectors of conj(P) * P are obtained as conj(X) * D * conj(V)
    //
    // Compute X1 = D^(-1) * U * S^{1/2}
    //
    for (size_type j = 0; j < m; ++j)
    {
        const auto sj = std::sqrt(sigma(j));
        for (size_type i = 0; i < m; ++i)
        {
            Y(i, j) *= sj * dinv(i);
        }
    }

    //
    // Compute R1 = D^(-1) * (R * P.t()) * D^(-1)
    //
    std::cout << "***** R1 = D^(-1) * R * D^(-1)" << std::endl;
    for (size_type j = 0; j < m; ++j)
    {
        for (size_type i = 0; i <= j; ++i)
        {
            G(i, j) *= dinv(i) * dinv(j);
        }
    }

    //
    // Solve R1 * Y1 = X1 in-place
    //
    std::cout << "***** solve R1 * Y1 = X1" << std::endl;
    {
        char uplo  = 'U';
        char trans = 'N';
        char diag  = 'N';
        auto m_    = static_cast<arma::blas_int>(m);
        auto nrhs_ = m_;
        arma::blas_int info;
        arma::lapack::trtrs(&uplo, &trans, &diag, &m_, &nrhs_, G.memptr(), &m_,
                            Y.memptr(), &m_, &info);
        if (info)
        {
            std::ostringstream msg;
            msg << "(reduction_body::eig_prod_gramian) xTRTRS failed with "
                   "info "
                << info;
            throw std::runtime_error(msg.str());
        }
    }
    matrix_type U(ptr_U, n, m, false, true);
    U = arma::conj(X) * arma::conj(Y);

    if (arma::is_complex<T>::value)
    {
        // conj(A) * A = U * S * U.t()
        for (size_type j = 0; j < m; ++j)
        {
            auto uj = U.col(j);
            // auto phi   = numeric::arg(arma::dot(uj, uj));
            // auto phase = std::polar(real_type(1), -phi / 2);
            // uj *= phase;
            // auto d     = arma::dot(uj, uj);
            // auto phase = d / std::abs(d);
            auto phase = arma::dot(uj, uj);
            auto scale = std::sqrt(numeric::conj(phase));
            uj *= scale;
        }
    }

    for (size_type j = 0; j < m; ++j)
    {
        auto uj = U.col(j);
        std::cout << sigma(j) << '\t' << arma::norm(uj) << '\t'
                  << arma::norm(P * arma::conj(uj) - sigma(j) * uj) << '\t'
                  << arma::norm(P * uj - sigma(j) * arma::conj(uj)) << '\n';
    }
}

} // namespace: expsum

#endif /* EXPSUM_REDUCTION_HPP */
