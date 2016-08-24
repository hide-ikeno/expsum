#ifndef EXPSUM_REDUCTION_CONEIG_SYM_RRD_HPP
#define EXPSUM_REDUCTION_CONEIG_SYM_RRD_HPP

#include <cassert>

#include <armadillo>

#include "arma/lapack_extra.hpp"

namespace expsum
{

//
// Compute accurate con-eigenvalue decomposition of the matrix $A=XD^{2}X^{H}$
//
// This class computes the con-eigenvalue decomposition
//
// ``` math
//   A U = \Lambda \overline{U},
// ```
//
// of a real symmetric (or complex Hermitian) and positive-definite matrix $A$
// having a rank-revealing decomposition of the form, $A = X D^{2} X^{H}$.
//
// @X  A matrix of type `T` of dimension $m \times n$.
//     On entry, the rank-revealing factor $X$.
//     On exit, orthonormal con-eigenvectors of the matrix $A$.
// @d  A vector of real values with size $n$.
//     On entry, the diagonal elements of rank-revealing factor $D$. The
//     diagonal of $D$ must be all positive and decreasing.
//     On exit, con-eigenvalues of the matrix $A$.
// @threshold threshold value for con-eigenvalues.
//

template <typename T>
class coneig_sym_rrd
{
public:
    // Scalar types
    using size_type    = arma::uword;
    using value_type   = T;
    using real_type    = typename arma::get_pod_type<T>::result;
    using complex_type = std::complex<real_type>;

    // Matrix/Vector types
    using vector_type         = arma::Col<value_type>;
    using matrix_type         = arma::Mat<value_type>;
    using index_vector_type   = arma::uvec;
    using real_vector_type    = arma::Col<real_type>;
    using real_matrix_type    = arma::Mat<real_type>;
    using complex_vector_type = arma::Col<complex_type>;
    using complex_matrix_type = arma::Mat<complex_type>;

    static size_type run(matrix_type& X, real_vector_type& d,
                         real_type threshold, value_type* work,
                         real_type* rwork);

private:
    //
    // Compute QR factorization of matrix G = Q * R.
    //
    static void qr_factorization(matrix_type& G, value_type* work)
    {
        auto n          = static_cast<arma::blas_int>(G.n_rows);
        auto lwork      = n * n;
        value_type* tau = work + n * n;
        arma::blas_int info;
        arma::lapack::geqrf(&n, &n, G.memptr(), &n, tau, work, &lwork, &info);
        if (info)
        {
            std::ostringstream msg;
            msg << "(coneig_sym_rrd) xGEQRF failed with info " << info;
            throw std::runtime_error(msg.str());
        }
    }
    //
    // Solve R * Y = X, where R is upper triangular matrix.
    //
    static void tri_solve(matrix_type& R, matrix_type& X)
    {
        char uplo  = 'U';
        char trans = 'N';
        char diag  = 'N';
        auto m     = static_cast<arma::blas_int>(R.n_rows);
        auto nrhs  = static_cast<arma::blas_int>(X.n_cols);
        arma::blas_int info;
        arma::lapack::trtrs(&uplo, &trans, &diag, &m, &nrhs, R.memptr(), &m,
                            X.memptr(), &m, &info);
        if (info)
        {
            std::ostringstream msg;
            msg << "(coneig_sym_rrd) xTRTRS failed with info " << info;
            throw std::runtime_error(msg.str());
        }
    }
    //
    // Compute singular values and corresponding left singular vectors of upper
    // triangular matrix R using one-sided Jacobi method.
    //
    // --- for real matrix
    static void jacobi_svd(matrix_type& R, real_vector_type& sigma,
                           real_type* work, real_type* /*dummy*/)
    {
        assert(R.n_rows == R.n_cols);
        assert(sigma.n_elem == R.n_cols);

        char joba = 'U'; // Input matrix R is upper triangular matrix
        char jobu = 'U'; // Compute left singular vectors
        char jobv = 'N'; // Do not compute right singular vectors
        auto n    = static_cast<arma::blas_int>(R.n_cols);
        auto mv   = arma::blas_int();
        auto ldv  = arma::blas_int(2);

        real_type dummy_v[2];
        auto lwork = 2 * n;

        arma::blas_int info;

        arma::lapack::gesvj(&joba, &jobu, &jobv, &n, &n, R.memptr(), &n,
                            sigma.memptr(), &mv, &dummy_v[0], &ldv, work,
                            &lwork, &info);

        if (info < arma::blas_int())
        {
            std::ostringstream msg;
            msg << "[s/d]GESVJ error: " << -info
                << " th argument had an illegal value";
            throw std::logic_error(msg.str());
        }

        if (info > arma::blas_int())
        {
            std::ostringstream msg;
            msg << "[s/d]GESVJ did not converge in the maximal allowed number "
                << info << " of sweeps";
            throw std::runtime_error(msg.str());
        }
    }
    // --- for complex matrix
    static void jacobi_svd(matrix_type& R, real_vector_type& sigma,
                           complex_type* work, real_type* rwork)
    {
        assert(R.n_rows == R.n_cols);
        assert(sigma.n_elem == R.n_cols);

        char joba = 'U'; // Input matrix R is upper triangular matrix
        char jobu = 'U'; // Compute left singular vectors
        char jobv = 'N'; // Do not compute right singular vectors
        auto n    = static_cast<arma::blas_int>(R.n_cols);
        auto mv   = arma::blas_int();
        auto ldv  = arma::blas_int(2);

        complex_type dummy_v[2];
        auto lwork  = 2 * n;
        auto lrwork = std::max(arma::blas_int(6), 2 * n);

        arma::blas_int info;

        arma::lapack::gesvj(&joba, &jobu, &jobv, &n, &n, R.memptr(), &n,
                            sigma.memptr(), &mv, &dummy_v[0], &ldv, work,
                            &lwork, rwork, &lrwork, &info);

        if (info < arma::blas_int())
        {
            std::ostringstream msg;
            msg << "[c/z]GESVJ error: " << -info
                << " th argument had an illegal value";
            throw std::logic_error(msg.str());
        }

        if (info > arma::blas_int())
        {
            std::ostringstream msg;
            msg << "[c/z]GESVJ did not converge in the maximal allowed number "
                << info << " of sweeps";
            throw std::runtime_error(msg.str());
        }
    }
};

//
// Required memory for workspace
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
template <typename T>
typename coneig_sym_rrd<T>::size_type
coneig_sym_rrd<T>::run(matrix_type& X, real_vector_type& d, real_type threshold,
                       value_type* work, real_type* rwork)
{
    // const size_type m = X.n_rows;
    const size_type n = X.n_cols;

    value_type* ptr1 = work;
    value_type* ptr2 = work + n * n;
    value_type* ptr3 = work + 2 * n * n;

    real_vector_type dinv(rwork, n, false, true);
    dinv = real_type(1) / d;

    //
    // Form G = D * (X.st() * X) * D
    //
    matrix_type G(ptr1, n, n, false, true);
    G = X.st() * X;
    for (size_type j = 0; j < n; ++j)
    {
        for (size_type i = 0; i < n; ++i)
        {
            G(i, j) *= d(i) * d(j);
        }
    }
    //
    // Compute G = Q * R by Householder QR factorization
    //
    qr_factorization(G, ptr2);
    matrix_type& R = G; // R = trimatu(G), on exit
    //
    // Compute SVD of R = U * S * V.t() using one-sided Jacobi method.
    // We need singular values and left singular vectors here.
    //
    real_vector_type& sigma = d;
    matrix_type U(ptr2, n, n, false, true);
    U = arma::trimatu(R); // make a copy of matrix R
    jacobi_svd(U, sigma, ptr3, rwork + n);

    //-------------------------------------------------------------------------
    // Truncation: discard con-eigenvalues if negligibly small
    //-------------------------------------------------------------------------
    auto sum_d     = real_type();
    size_type nvec = n;
    while (nvec)
    {
        sum_d += d(nvec - 1);
        if (2 * sum_d > threshold)
        {
            break;
        }
        --nvec;
    }

    if (nvec == size_type(0))
    {
        return size_type();
    }

    //-------------------------------------------------------------------------
    //
    // The eigenvectors of A * conj(A) are given as
    //
    //   conj(X') = X * D * V * S^{-1/2}.
    //
    // However, direct evaluation of eigenvectors with this formula might be
    // inaccurate since D is ill-conditioned. The following formula are used
    // instead.
    //
    //   conj(X') = X * D * R^(-1) * U * S^{1/2}
    //            = X * (D.inv() * R * D.inv())^(-1) * (D.inv() * U * S^{1/2})
    //            = X * R1.inv() * X1
    //
    //-------------------------------------------------------------------------
    //
    // Compute X1 = D^(-1) * U * S^{1/2} [in-place]
    //
    matrix_type U_trunc(ptr2, n, nvec, false, true);
    for (size_type j = 0; j < nvec; ++j)
    {
        const auto sj = std::sqrt(sigma(j));
        for (size_type i = 0; i < n; ++i)
        {
            U_trunc(i, j) *= sj * dinv(i);
        }
    }
    //
    // Compute R1 = D^(-1) * R * D^(-1) [in-place]
    //
    for (size_type j = 0; j < n; ++j)
    {
        for (size_type i = 0; i <= j; ++i)
        {
            R(i, j) *= dinv(i) * dinv(j);
        }
    }
    //
    // Solve R1 * Y1 = X1 in-place
    //
    tri_solve(R, U_trunc);
    //
    // Compute con-eigenvectors U = conj(X) * conj(Y).
    //
    X.head_cols(nvec) = arma::conj(X) * arma::conj(U_trunc);
    //
    // Adjust phase factor of each con-eigenvectors, so that U^{T} * U = I
    //
    if (arma::is_complex<T>::value)
    {
        for (size_type j = 0; j < nvec; ++j)
        {
            auto xj          = X.col(j);
            const auto t     = arma::dot(xj, xj);
            const auto phase = t / std::abs(t);
            const auto scale = std::sqrt(arma::access::alt_conj(phase));
            xj *= scale;
        }
    }

    return nvec;
}

} // namespace: expsum

#endif /* EXPSUM_REDUCTION_CONEIG_SYM_RRD_HPP */
