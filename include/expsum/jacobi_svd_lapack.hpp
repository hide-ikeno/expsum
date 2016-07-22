// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-
#ifndef EXPSUM_JACOBI_SVD_HPP
#define EXPSUM_JACOBI_SVD_HPP

#include <cassert>

#include <sstream>
#include <stdexcept>

#include <armadillo>
#ifdef DEBUG
#include <iomanip>
#include <iostream>
#endif /* DEBUG */

#include "arma/lapack_extra.hpp"

namespace expsum
{

template <typename T>
struct jacobi_svd
{
    using value_type = T;
    using real_type  = T;
    using size_type  = arma::uword;

    void run(arma::Mat<value_type>& A, arma::Col<real_type>& sva,
             bool compute_U = false)
    {
        resize(A.n_rows, A.n_cols);

        char joba = 'G';
        char jobu = compute_U ? 'U' : 'N';
        char jobv = 'N';
        auto m    = static_cast<arma::blas_int>(A.n_rows);
        auto n    = static_cast<arma::blas_int>(A.n_cols);
        assert(m >= n);
        assert(sva.n_elem == A.n_cols);
        auto mv  = arma::blas_int();
        auto ldv = arma::blas_int(2);
        value_type dummy_v[2];
        auto lwork = static_cast<arma::blas_int>(work.size());

        arma::blas_int info;

        arma::lapack::gesvj(&joba, &jobu, &jobv, &m, &n, A.memptr(), &m,
                            sva.memptr(), &mv, &dummy_v[0], &ldv, work.memptr(),
                            &lwork, &info);

        if (info < arma::blas_int())
        {
            std::ostringstream msg;
            msg << "xGESVJ error: " << -info
                << " th argument had an illegal value";
            throw std::logic_error(msg.str());
        }
        if (info > arma::blas_int())
        {
            std::ostringstream msg;
            msg << "xGESVJ did not converge in the maximal allowed number "
                << info << " of sweeps";
            throw std::runtime_error(msg.str());
        }
    }

    void run(arma::Mat<value_type>& A, arma::Col<real_type>& sva,
             arma::Mat<value_type>& V, bool compute_U = false)
    {
        resize(A.n_rows, A.n_cols);

        char joba = 'G';
        char jobu = compute_U ? 'U' : 'N';
        char jobv = 'V';
        auto m    = static_cast<arma::blas_int>(A.n_rows);
        auto n    = static_cast<arma::blas_int>(A.n_cols);
        assert(m >= n);
        assert(sva.n_elem == A.n_cols);
        assert(V.n_rows == A.n_cols && V.n_cols == A.n_cols);

        auto mv    = arma::blas_int();
        auto lwork = static_cast<arma::blas_int>(work.size());

        arma::blas_int info;

        arma::lapack::gesvj(&joba, &jobu, &jobv, &m, &n, A.memptr(), &m,
                            sva.memptr(), &mv, V.memptr(), &n, work.memptr(),
                            &lwork, &info);

        if (info < arma::blas_int())
        {
            std::ostringstream msg;
            msg << "[S/D]GESVJ error: " << -info
                << " th argument had an illegal value";
            throw std::logic_error(msg.str());
        }

        if (info > arma::blas_int())
        {
            std::ostringstream msg;
            msg << "[S/D]GESVJ did not converge in the maximal allowed number "
                << info << " of sweeps";
            throw std::runtime_error(msg.str());
        }
    }

    void resize(size_type m, size_type n)
    {
        auto length = std::max(size_type(6), m + n);
        if (work.size() < length)
        {
            work.set_size(length);
        }
    }

    arma::Col<value_type> work;
};

//==============================================================================
// Specialization for complex matrix
//==============================================================================

template <typename T>
struct jacobi_svd<std::complex<T>>
{
    using value_type = std::complex<T>;
    using real_type  = T;
    using size_type  = arma::uword;

    void run(arma::Mat<value_type>& A, arma::Col<real_type>& sva,
             bool compute_U = false)
    {
        resize(A.n_rows, A.n_cols);

        char joba = 'G';
        char jobu = compute_U ? 'U' : 'N';
        char jobv = 'N';
        auto m    = static_cast<arma::blas_int>(A.n_rows);
        auto n    = static_cast<arma::blas_int>(A.n_cols);
        assert(m >= n);
        assert(sva.n_elem == A.n_cols);
        // assert(sva.n_elem == A.n_cols);
        auto mv  = arma::blas_int();
        auto ldv = arma::blas_int(1);
        value_type dummy_v[2];
        auto lwork  = static_cast<arma::blas_int>(cwork.size());
        auto lrwork = static_cast<arma::blas_int>(rwork.size());

        arma::blas_int info;

        arma::lapack::gesvj(&joba, &jobu, &jobv, &m, &n, A.memptr(), &m,
                            sva.memptr(), &mv, &dummy_v[0], &ldv,
                            cwork.memptr(), &lwork, rwork.memptr(), &lrwork,
                            &info);

        if (info < arma::blas_int())
        {
            std::ostringstream msg;
            msg << "[C/Z]GESVJ error: " << -info
                << " th argument had an illegal value";
            throw std::logic_error(msg.str());
        }

        if (info > arma::blas_int())
        {
            std::ostringstream msg;
            msg << "[C/Z]GESVJ did not converge in the maximal allowed number "
                << info << " of sweeps";
            throw std::runtime_error(msg.str());
        }
    }

    void run(arma::Mat<value_type>& A, arma::Col<real_type>& sva,
             arma::Mat<value_type>& V, bool compute_U = false)
    {
        resize(A.n_rows, A.n_cols);

        char joba = 'G';
        char jobu = compute_U ? 'U' : 'N';
        char jobv = 'V';
        auto m    = static_cast<arma::blas_int>(A.n_rows);
        auto n    = static_cast<arma::blas_int>(A.n_cols);
        assert(m >= n);
        assert(sva.n_elem == A.n_cols);
        assert(V.n_rows == A.n_cols && V.n_cols == A.n_cols);

        auto mv     = arma::blas_int();
        auto lwork  = static_cast<arma::blas_int>(cwork.size());
        auto lrwork = static_cast<arma::blas_int>(rwork.size());

        arma::blas_int info;

        arma::lapack::gesvj(&joba, &jobu, &jobv, &m, &n, A.memptr(), &m,
                            sva.memptr(), &mv, V.memptr(), &n, cwork.memptr(),
                            &lwork, rwork.memptr(), &lrwork, &info);

        if (info < arma::blas_int())
        {
            std::ostringstream msg;
            msg << "xGESVJ error: " << -info
                << " th argument had an illegal value";
            throw std::logic_error(msg.str());
        }
        if (info > arma::blas_int())
        {
            std::ostringstream msg;
            msg << "xGESVJ did not converge in the maximal allowed number "
                << info << " of sweeps";
            throw std::runtime_error(msg.str());
        }
    }

    void resize(size_type m, size_type n)
    {
        auto length = std::max(size_type(6), m + n);
        if (cwork.size() < length)
        {
            cwork.set_size(length);
            rwork.set_size(length);
        }
    }

    arma::Col<value_type> cwork;
    arma::Col<real_type> rwork;
};

} // namespace: expsum

#endif /* EXPSUM_JACOBI_SVD_HPP */
