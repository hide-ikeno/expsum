#ifndef EXPSUM_QR_COL_PIVOT_HPP
#define EXPSUM_QR_COL_PIVOT_HPP

#include <sstream>
#include <stdexcept>

#include "arma/lapack_extra.hpp"

namespace expsum
{

template <typename T>
struct qr_col_pivot
{
    using value_type = T;
    using real_type  = T;
    using size_type  = arma::uword;

    void run(arma::Mat<value_type>& A)
    {
        resize(A.n_rows, A.n_cols);

        auto m     = static_cast<arma::blas_int>(A.n_rows);
        auto n     = static_cast<arma::blas_int>(A.n_cols);
        auto lwork = static_cast<arma::blas_int>(work.n_elem);
        jpiv.zeros();

        invoke(m, n, A.memptr(), m, jpiv.memptr(), tau.memptr(), work.memptr(),
               lwork);
    }

    void resize(size_type m, size_type n)
    {
        jpiv.set_size(n);
        tau.set_size(std::min(m, n));
        const auto lwork = query(static_cast<arma::blas_int>(m),
                                 static_cast<arma::blas_int>(n));
        if (work.size() < lwork)
        {
            work.set_size(lwork);
        }
    }
    //
    // From output matrix of GEQP3, return matrix R * P.t()
    //
    // @A output matrix of `qr_col_pivot::run`
    //
    arma::Mat<value_type> get_matrix_RPT(const arma::Mat<value_type>& A) const
    {
        assert(A.n_cols == jpiv.size());
        arma::Mat<value_type> RPT(std::min(A.n_rows, A.n_cols), A.n_cols,
                                  arma::fill::zeros);
        arma::uvec ipiv(jpiv.size());
        for (size_type i = 0; i < jpiv.size(); ++i)
        {
            ipiv(static_cast<size_type>(jpiv(i) - 1)) = i;
        }

        for (size_type i = 0; i < ipiv.size(); ++i)
        {
            auto n = std::min(i + 1, A.n_cols);
            RPT.col(ipiv(i)).head(n) = A.col(i).head(n);
        }

        return RPT;
    }

    //
    // Compute matrix Q from result matrix of `qr_col_pivot::run`
    //
    // @A On entry, result matrix of `qr_col_pivot::run` and on exit, matrix Q.
    //
    void make_matrix_Q(arma::Mat<value_type>& A)
    {
        auto m     = static_cast<arma::blas_int>(A.n_rows);
        auto n     = static_cast<arma::blas_int>(A.n_cols);
        auto k     = std::min(m, n);
        auto lwork = static_cast<arma::blas_int>(work.n_elem);
        arma::blas_int info;
        arma::lapack::orgqr(&m, &n, &k, A.memptr(), &m, tau.memptr(),
                            work.memptr(), &lwork, &info);

        if (info < arma::blas_int())
        {
            std::ostringstream msg;
            msg << "[S/D]ORGQR error: " << -info
                << " th argument had an illegal value";
            throw std::logic_error(msg.str());
        }
    }

private:
    arma::Col<arma::blas_int> jpiv;
    arma::Col<real_type> tau;
    arma::Col<real_type> work;

    // Get optimal workspace size
    static size_type query(arma::blas_int m, arma::blas_int n)
    {
        value_type dummy[2];
        arma::blas_int lwork = -1;
        arma::blas_int jpiv[2];
        arma::blas_int info;
        arma::lapack::geqp3(&m, &n, &dummy[0], &m, &jpiv[0], &dummy[0],
                            &dummy[0], &lwork, &info);
        return static_cast<size_type>(dummy[0]);
    }

    static arma::blas_int invoke(arma::blas_int m, arma::blas_int n,
                                 value_type* A, arma::blas_int lda,
                                 arma::blas_int* jpiv, value_type* tau,
                                 value_type* work, arma::blas_int lwork)
    {
        arma::blas_int info;
        arma::lapack::geqp3(&m, &n, A, &lda, jpiv, tau, work, &lwork, &info);

        return info;
    }
};


template <typename T>
struct qr_col_pivot<std::complex<T> >
{
    using value_type = std::complex<T>;
    using real_type  = T;
    using size_type  = arma::uword;

    void run(arma::Mat<value_type>& A)
    {
        resize(A.n_rows, A.n_cols);

        auto m     = static_cast<arma::blas_int>(A.n_rows);
        auto n     = static_cast<arma::blas_int>(A.n_cols);
        auto lwork = static_cast<arma::blas_int>(work_.n_elem);
        jpiv_.zeros();

        invoke(m, n, A.memptr(), m, jpiv_.memptr(), tau_.memptr(),
               work_.memptr(), lwork, rwork_.memptr());
    }

    void resize(size_type m, size_type n)
    {
        jpiv_.set_size(n);
        tau_.set_size(std::min(m, n));
        const auto lwork = query(static_cast<arma::blas_int>(m),
                                 static_cast<arma::blas_int>(n));
        if (work_.size() < lwork)
        {
            work_.set_size(lwork);
        }
        rwork_.set_size(2 * n);
    }
    //
    // From output matrix of GEQP3, return matrix R * P.t()
    //
    // @A output matrix of `qr_col_pivot::run`
    //
    arma::Mat<value_type> get_matrix_RPT(const arma::Mat<value_type>& A) const
    {
        assert(A.n_cols == jpiv_.size());
        arma::Mat<value_type> RPT(std::min(A.n_rows, A.n_cols), A.n_cols,
                                  arma::fill::zeros);
        arma::uvec ipiv(jpiv_.size());
        for (size_type i = 0; i < jpiv_.size(); ++i)
        {
            ipiv(static_cast<size_type>(jpiv_(i) - 1)) = i;
        }

        for (size_type i = 0; i < ipiv.size(); ++i)
        {
            auto n = std::min(i + 1, A.n_cols);
            RPT.col(ipiv(i)).head(n) = A.col(i).head(n);
        }

        return RPT;
    }

    //
    // Compute matrix Q from result matrix of `qr_col_pivot::run`
    //
    // @A On entry, result matrix of `qr_col_pivot::run` and on exit, matrix Q.
    //
    void make_matrix_Q(arma::Mat<value_type>& A)
    {
        auto m     = static_cast<arma::blas_int>(A.n_rows);
        auto n     = static_cast<arma::blas_int>(A.n_cols);
        auto k     = std::min(m, n);
        auto lwork = static_cast<arma::blas_int>(work_.n_elem);
        arma::blas_int info;
        arma::lapack::ungqr(&m, &n, &k, A.memptr(), &m, tau_.memptr(),
                            work_.memptr(), &lwork, &info);

        if (info < arma::blas_int())
        {
            std::ostringstream msg;
            msg << "[C/Z]UNGQR error: " << -info
                << " th argument had an illegal value";
            throw std::logic_error(msg.str());
        }
    }
private:
    arma::Col<arma::blas_int> jpiv_;
    arma::Col<value_type> tau_;
    arma::Col<value_type> work_;
    arma::Col<real_type> rwork_;

    // Get optimal workspace size
    static size_type query(arma::blas_int m, arma::blas_int n)
    {
        value_type dummy[1];
        arma::blas_int lwork = -1;
        arma::blas_int jpiv[1];
        real_type rwork[2];
        arma::blas_int info;
        arma::lapack::geqp3(&m, &n, &dummy[0], &m, &jpiv[0], &dummy[0],
                            &dummy[0], &lwork, &rwork[0], &info);
        return static_cast<size_type>(std::real(dummy[0]));
    }

    static arma::blas_int invoke(arma::blas_int m, arma::blas_int n,
                                 value_type* A, arma::blas_int lda,
                                 arma::blas_int* jpiv, value_type* tau,
                                 value_type* work, arma::blas_int lwork,
                                 real_type* rwork)
    {
        arma::blas_int info;
        arma::lapack::geqp3(&m, &n, A, &lda, jpiv, tau, work, &lwork, rwork,
                            &info);
        return info;
    }
};

} // namespace: expsum

#endif /* EXPSUM_QR_COL_PIVOT_HPP */
