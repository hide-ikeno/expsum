// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-
#ifndef EXPSUM_JACOBI_SVD_HPP
#define EXPSUM_JACOBI_SVD_HPP

#include <armadillo>
#include <cassert>
#ifdef DEBUG
#include <iomanip>
#include <iostream>
#endif /* DEBUG */

#include "expsum/numeric.hpp"

namespace expsum
{
namespace detail
{
//
// Make Jacobi rotation matrix \c J of the form
//
//  J = [ cs conj(sn)]
//      [-sn conj(cs)]
//
// that diagonalize the 2x2 matirix
//
// B = [     a  c]
//     [conj(c) b]
//
// as D = J.t() * B * J
//
// J.t() = [ conj(cs)    -conj(sn)] = J.inv()
//         [      sn           cs ]
//
// J * [a11 a12] = [ cs*a11 + conj(sn)*a21   cs*a12 + conj(sn)*a22]
//     [a21 a22]   [-sn*a11 + conj(cs)*a21  -sn*a12 + conj(cs)*a22]
//
// [a11 a12] * J = [     cs *a11 -      sn *a12        cs *a21 -      sn *a22]
// [a21 a22]       [conj(sn)*a11 + conj(cs)*a12  -conj(sn)*a21 + conj(cs)*a22]
//
template <typename RealT, typename T>
bool make_jacobi_rotation(RealT a, RealT b, T c, RealT& cs, T& sn)
{
    if (c == T())
    {
        cs = RealT(1);
        sn = T();
        return false;
    }

    auto zeta = (a - b) / (RealT(2) * std::abs(c));
    auto w    = std::sqrt(numeric::abs2(zeta) + RealT(1));
    auto t = (zeta > RealT() ? RealT(1) / (zeta + w) : RealT(1) / (zeta - w));

    cs = RealT(1) / std::sqrt(numeric::abs2(t) + RealT(1));
    sn = -t * cs * (numeric::conj(c) / std::abs(c));

    return true;
}

template <typename T>
struct jacobi_helper
{
    using size_type  = arma::uword;
    using value_type = T;
    using real_type  = typename arma::get_pod_type<T>::result;
    //
    // Sum of the square of the absolute value of the elments in the i-th column
    // of the matrix G, i.e. \f$ \sum_{k=0}^{n-1} |G_{ki}|^{2}. \f$
    //
    template <typename MatG>
    static real_type submat_diag(const MatG& G, size_type i)
    {
        auto ret = real_type();
        // Since G is an upper triangular matrix, we only need to take the sum
        // over k up to i.
        for (size_type k = 0; k < G.n_rows; ++k)
        {
            ret += numeric::abs2(G(k, i));
        }
        return ret;
    }

    template <typename MatG>
    static value_type submat_offdiag(const MatG& G, size_type i, size_type j)
    {
        return arma::cdot(G.col(i), G.col(j));
    }

    template <typename MatU>
    static void update_vecs(MatU& U, size_type i, size_type j, real_type cs,
                            value_type sn)
    {
        for (size_type k = 0; k < U.n_rows; ++k)
        {
            const auto t1 = U(k, i);
            const auto t2 = U(k, j);
            // Apply Jacobi rotation matrix on the right
            U(k, i) = cs * t1 - sn * t2;
            U(k, j) = numeric::conj(sn) * t1 + cs * t2;
        }
    }
};

} // namespace: detail

/*!
 * Compute the singular value decomposition (SVD) by one-sided Jacobi algorithm.
 *
 * This function compute the singular value decomposition (SVD) using modified
 * one-sided Jacobi algorithm.
 *
 * - James Demmel and Kresimir Veselic, "Jacobi's method is more accurate than
 *   QR", LAPACK working note 15 (lawn15) (1989), Algorithm 4.1.
 *
 * \param[in,out] A On entry, an \f$ m \times n \f$ matrix to be decomposed.
 *  On exit, columns of \c A holds left singular vectors.
 * \param[out] sigma singular values of \c A.
 * \param[out] V     right singular vectors.
 * \param[in]  tol   small positive real number that determines stopping
 * criterion of Jacobi algorithm.
 */
template <typename MatA, typename VecS, typename MatV>
void jacobi_svd(MatA& A, VecS& sigma, MatV& V, typename MatA::pod_type tol)
{
    using size_type  = arma::uword;
    using value_type = typename MatA::elem_type;
    using real_type  = typename MatA::pod_type;
    using jacobi_aux = detail::jacobi_helper<value_type>;

    auto n = A.n_cols;
    assert(V.n_rows == A.n_cols);
    //
    // Initialize right singular vectors
    //
    V.eye();

    value_type sn;
    real_type cs;
    real_type max_resid;
    do
    {
        max_resid = real_type();
        for (size_type j = 1; j < n; ++j)
        {
            auto b = jacobi_aux::submat_diag(A, j);
            for (size_type i = 0; i < j; ++i)
            {
                //
                // For all pairs i < j, compute the 2x2 submatrix of A.t() * A
                // constructed from i-th and j-th columns, such as
                //
                //   M = [     a   c]
                //       [conj(c)  b]
                //
                // where
                //
                //   a = \sum_{k=0}^{n-1} |A(k,i)|^{2} (computed in outer loop)
                //   b = \sum_{k=0}^{n-1} |A(k,j)|^{2}
                //   c = \sum_{k=0}^{n-1} conj(A(k,i)) * A(k,j)
                //
                auto a    = jacobi_aux::submat_diag(A, i);
                auto c    = jacobi_aux::submat_offdiag(A, i, j);
                max_resid = std::max(max_resid, std::abs(c) / std::sqrt(a * b));
                //
                // Compute the Jacobi rotation matrix which diagonalize M.
                //
                if (detail::make_jacobi_rotation(a, b, c, cs, sn))
                {
                    // If the return vlaue of make_jacobi_rotation is false, no
                    // rotation is made.
                    //
                    // Update columns i and j of A
                    //
                    jacobi_aux::update_vecs(A, i, j, cs, sn);
                    //
                    // Update right singular vector V
                    //
                    jacobi_aux::update_vecs(V, i, j, cs, sn);
                }
            }
        }
    } while (max_resid > tol);

    //
    // Set singular values and left singular vectors.
    //
    for (size_type j = 0; j < A.n_cols; ++j)
    {
        // Singular values are the norms of the columns of the final A
        sigma(j) = arma::norm(A.col(j), 2);
        // Left singular vectors are the normalized columns of the final A
        A.col(j) /= sigma(j);
    }
    //
    // Sort singular values in descending order. The corresponding singular
    // vectors are also rearranged.
    //
    for (size_type i = 0; i < n - 1; ++i)
    {
        //
        // Find the index of the maximum value of a sub-array, sigma(i:n-1).
        //
        size_type imax = i;
        for (size_type k = i + 1; k < n; ++k)
        {
            if (sigma(k) > sigma(imax))
            {
                imax = k;
            }
        }

        if (imax != i)
        {
            // Move the largest singular value to the beggining of the
            // sub-array, and corresponsing singular vectors by swapping columns
            // of A and V.
            std::swap(sigma(i), sigma(imax));
            A.swap_cols(i, imax);
            V.swap_cols(i, imax);
        }
    }
}

// Compute singular values and left singular vectors of matrix A by one-sided
// Jacobi algorithm. Right singular vectors are not computed.
template <typename MatA, typename VecS>
void jacobi_svd(MatA& A, VecS& sigma, typename MatA::pod_type tol)
{
    using size_type  = arma::uword;
    using value_type = typename MatA::elem_type;
    using real_type  = typename MatA::pod_type;
    using jacobi_aux = detail::jacobi_helper<value_type>;

    auto n = A.n_cols;
    //
    // Initialize right singular vectors
    //
    value_type sn;
    real_type cs;
    real_type max_resid;
    do
    {
        max_resid = real_type();
        for (size_type j = 1; j < n; ++j)
        {
            auto b = jacobi_aux::submat_diag(A, j);
            for (size_type i = 0; i < j; ++i)
            {
                //
                // For all pairs i < j, compute the 2x2 submatrix of A.t() * A
                // constructed from i-th and j-th columns, such as
                //
                //   M = [     a   c]
                //       [conj(c)  b]
                //
                // where
                //
                //   a = \sum_{k=0}^{n-1} |A(k,i)|^{2} (computed in outer loop)
                //   b = \sum_{k=0}^{n-1} |A(k,j)|^{2}
                //   c = \sum_{k=0}^{n-1} conj(A(k,i)) * A(k,j)
                //
                auto a    = jacobi_aux::submat_diag(A, i);
                auto c    = jacobi_aux::submat_offdiag(A, i, j);
                max_resid = std::max(max_resid, std::abs(c) / std::sqrt(a * b));
                //
                // Compute the Jacobi rotation matrix which diagonalize M.
                //
                if (detail::make_jacobi_rotation(a, b, c, cs, sn))
                {
                    // If the return vlaue of make_jacobi_rotation is false, no
                    // rotation is made.
                    //
                    // Update columns i and j of A
                    //
                    jacobi_aux::update_vecs(A, i, j, cs, sn);
                }
            }
        }
    } while (max_resid > tol);

    //
    // Set singular values and left singular vectors.
    //
    for (size_type j = 0; j < A.n_cols; ++j)
    {
        // Singular values are the norms of the columns of the final A
        sigma(j) = arma::norm(A.col(j), 2);
        // Left singular vectors are the normalized columns of the final A
        A.col(j) /= sigma(j);
    }
    //
    // Sort singular values in descending order. The corresponding singular
    // vectors are also rearranged.
    //
    for (size_type i = 0; i < n - 1; ++i)
    {
        //
        // Find the index of the maximum value of a sub-array, sigma(i:n-1).
        //
        size_type imax = i;
        for (size_type k = i + 1; k < n; ++k)
        {
            if (sigma(k) > sigma(imax))
            {
                imax = k;
            }
        }

        if (imax != i)
        {
            // Move the largest singular value to the beggining of the
            // sub-array, and corresponsing singular vectors by swapping columns
            // of A and V.
            std::swap(sigma(i), sigma(imax));
            A.swap_cols(i, imax);
        }
    }
}

} // namespace: expsum

#endif /* EXPSUM_JACOBI_SVD_HPP */
