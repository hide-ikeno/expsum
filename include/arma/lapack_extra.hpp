// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-

#ifndef ARMA_LAPACK_EXTRA_HPP
#define ARMA_LAPACK_EXTRA_HPP

#include <armadillo>

#if !defined(ARMA_BLAS_CAPITALS)

#define arma_sbdsqr sbdsqr
#define arma_dbdsqr dbdsqr

/* Least-square problem */
#define arma_sgglse sgglse
#define arma_dgglse dgglse
#define arma_cgglse cgglse
#define arma_zgglse zgglse

/* QR factorization with column pivoting */
#define arma_sgeqp3 sgeqp3
#define arma_dgeqp3 dgeqp3
#define arma_cgeqp3 cgeqp3
#define arma_zgeqp3 zgeqp3

/* Eigenvalue decomposition of upper Hessenberg matrix */
#define arma_shseqr shseqr
#define arma_dhseqr dhseqr
#define arma_chseqr chseqr
#define arma_zhseqr zhseqr

/* Jacobi SVD */
#define arma_sgesvj sgesvj
#define arma_dgesvj dgesvj
#define arma_cgesvj cgesvj
#define arma_zgesvj zgesvj

/* Another Jacobi SVD */
#define arma_sgejsv sgejsv
#define arma_dgejsv dgejsv
#define arma_cgejsv cgejsv
#define arma_zgejsv zgejsv

#else

#define arma_sbdsqr SBDSQR
#define arma_dbdsqr DBDSQR

/* Least-square problem */
#define arma_sgglse SGGLSE
#define arma_dgglse DGGLSE
#define arma_cgglse CGGLSE
#define arma_zgglse ZGGLSE

/* QR factorization with column pivoting */
#define arma_sgeqp3 SGEQP3
#define arma_dgeqp3 DGEQP3
#define arma_cgeqp3 CGEQP3
#define arma_zgeqp3 ZGEQP3

/* Eigenvalue decomposition of upper Hessenberg matrix */
#define arma_shseqr SHSEQR
#define arma_dhseqr DHSEQR
#define arma_chseqr CHSEQR
#define arma_zhseqr ZHSEQR

/* Jacobi SVD */
#define arma_sgesvj SGESVJ
#define arma_dgesvj DGESVJ
#define arma_cgesvj CGESVJ
#define arma_zgesvj ZGESVJ

/* Another Jacobi SVD */
#define arma_sgejsv SGEJSV
#define arma_dgejsv DGEJSV
#define arma_cgejsv CGEJSV
#define arma_zgejsv ZGEJSV

#endif /* ARMA_BLAS_CAPITALS */

namespace arma
{

extern "C" {
// SVD of real bidiagonal matrix
void arma_fortran_noprefix(arma_sbdsqr)(char* uplo, blas_int* n, blas_int* ncvt,
                                        blas_int* nru, blas_int* ncc, float* d,
                                        float* e, float* vt, blas_int* ldvt,
                                        float* u, blas_int* ldu, float* c,
                                        blas_int* ldc, float* work,
                                        blas_int* info);

void arma_fortran_noprefix(arma_dbdsqr)(char* uplo, blas_int* n, blas_int* ncvt,
                                        blas_int* nru, blas_int* ncc, double* d,
                                        double* e, double* vt, blas_int* ldvt,
                                        double* u, blas_int* ldu, double* c,
                                        blas_int* ldc, double* work,
                                        blas_int* info);

// Solve generalized eigenvalue problem
void arma_fortran_noprefix(arma_sggev)(char* jobl, char* jobr, blas_int* n,
                                       float* a, blas_int* lda, float* b,
                                       blas_int* ldb, float* alphar,
                                       float* alphai, float* beta, float* vl,
                                       blas_int* ldvl, float* vr,
                                       blas_int* ldvr, float* work,
                                       blas_int* lwork, blas_int* info);

void arma_fortran_noprefix(arma_dggev)(char* jobl, char* jobr, blas_int* n,
                                       double* a, blas_int* lda, double* b,
                                       blas_int* ldb, double* alphar,
                                       double* alphai, double* beta, double* vl,
                                       blas_int* ldvl, double* vr,
                                       blas_int* ldvr, double* work,
                                       blas_int* lwork, blas_int* info);

void arma_fortran_noprefix(arma_cggev)(
    char* jobl, char* jobr, blas_int* n, void* a, blas_int* lda, void* b,
    blas_int* ldb, void* alpha, void* beta, void* vl, blas_int* ldvl, void* vr,
    blas_int* ldvr, void* work, blas_int* lwork, float* rwork, blas_int* info);

void arma_fortran_noprefix(arma_zggev)(
    char* jobl, char* jobr, blas_int* n, void* a, blas_int* lda, void* b,
    blas_int* ldb, void* alpha, void* beta, void* vl, blas_int* ldvl, void* vr,
    blas_int* ldvr, void* work, blas_int* lwork, double* rwork, blas_int* info);

// linear equality-constrained least squares problem (LSE)
void arma_fortran_noprefix(arma_sgglse)(blas_int* m, blas_int* n, blas_int* p,
                                        float* a, blas_int* lda, float* b,
                                        blas_int* ldb, float* c, float* d,
                                        float* x, float* work, blas_int* lwork,
                                        blas_int* info);

void arma_fortran_noprefix(arma_dgglse)(blas_int* m, blas_int* n, blas_int* p,
                                        double* a, blas_int* lda, double* b,
                                        blas_int* ldb, double* c, double* d,
                                        double* x, double* work,
                                        blas_int* lwork, blas_int* info);

void arma_fortran_noprefix(arma_cgglse)(blas_int* m, blas_int* n, blas_int* p,
                                        void* a, blas_int* lda, void* b,
                                        blas_int* ldb, void* c, void* d,
                                        void* x, void* work, blas_int* lwork,
                                        blas_int* info);

void arma_fortran_noprefix(arma_zgglse)(blas_int* m, blas_int* n, blas_int* p,
                                        void* a, blas_int* lda, void* b,
                                        blas_int* ldb, void* c, void* d,
                                        void* x, void* work, blas_int* lwork,
                                        blas_int* info);
// QR factorization with column pivoting
void arma_fortran_noprefix(arma_sgeqp3)(blas_int* m, blas_int* n, float* a,
                                        blas_int* lda, blas_int* jpiv,
                                        float* tau, float* work,
                                        blas_int* lwork, blas_int* info);

void arma_fortran_noprefix(arma_dgeqp3)(blas_int* m, blas_int* n, double* a,
                                        blas_int* lda, blas_int* jpiv,
                                        double* tau, double* work,
                                        blas_int* lwork, blas_int* info);

void arma_fortran_noprefix(arma_cgeqp3)(blas_int* m, blas_int* n, void* a,
                                        blas_int* lda, blas_int* jpiv,
                                        void* tau, void* work, blas_int* lwork,
                                        float* rwork, blas_int* info);
void arma_fortran_noprefix(arma_zgeqp3)(blas_int* m, blas_int* n, void* a,
                                        blas_int* lda, blas_int* jpiv,
                                        void* tau, void* work, blas_int* lwork,
                                        double* rwork, blas_int* info);

// xHSEQR --- Eigendecomposition of upper Hessenberg matrix by mutishift QR
void arma_fortran_noprefix(arma_shseqr)(char* job, char* compz, blas_int* n,
                                        blas_int* ilo, blas_int* ihi, float* h,
                                        blas_int* ldh, float* wr, float* wi,
                                        float* z, blas_int* ldz, float* work,
                                        blas_int* lwork, blas_int* info);

void arma_fortran_noprefix(arma_dhseqr)(char* job, char* compz, blas_int* n,
                                        blas_int* ilo, blas_int* ihi, double* h,
                                        blas_int* ldh, double* wr, double* wi,
                                        double* z, blas_int* ldz, double* work,
                                        blas_int* lwork, blas_int* info);

void arma_fortran_noprefix(arma_chseqr)(char* job, char* compz, blas_int* n,
                                        blas_int* ilo, blas_int* ihi, void* h,
                                        blas_int* ldh, void* w, void* z,
                                        blas_int* ldz, void* work,
                                        blas_int* lwork, blas_int* info);

void arma_fortran_noprefix(arma_zhseqr)(char* job, char* compz, blas_int* n,
                                        blas_int* ilo, blas_int* ihi, void* h,
                                        blas_int* ldh, void* w, void* z,
                                        blas_int* ldz, void* work,
                                        blas_int* lwork, blas_int* info);

// xGESVJ --- Jacobi SVD
void arma_fortran_noprefix(arma_sgesvj)(char* joba, char* jobu, char* jobv,
                                        blas_int* m, blas_int* n, float* a,
                                        blas_int* lda, float* sva, blas_int* mv,
                                        float* v, blas_int* ldv, float* work,
                                        blas_int* lwork, blas_int* info);
void arma_fortran_noprefix(arma_dgesvj)(char* joba, char* jobu, char* jobv,
                                        blas_int* m, blas_int* n, double* a,
                                        blas_int* lda, double* sva,
                                        blas_int* mv, double* v, blas_int* ldv,
                                        double* work, blas_int* lwork,
                                        blas_int* info);
void arma_fortran_noprefix(arma_cgesvj)(char* joba, char* jobu, char* jobv,
                                        blas_int* m, blas_int* n, void* a,
                                        blas_int* lda, float* sva, blas_int* mv,
                                        void* v, blas_int* ldv, void* cwork,
                                        blas_int* lwork, float* rwork,
                                        blas_int* lrwork, blas_int* info);
void arma_fortran_noprefix(arma_zgesvj)(char* joba, char* jobu, char* jobv,
                                        blas_int* m, blas_int* n, void* a,
                                        blas_int* lda, double* sva,
                                        blas_int* mv, void* v, blas_int* ldv,
                                        void* cwork, blas_int* lwork,
                                        double* rwork, blas_int* lrwork,
                                        blas_int* info);

// xGEJSV --- Jacobi SVD
void arma_fortran_noprefix(arma_sgejsv)(char* joba, char* jobu, char* jobv,
                                        char* jobr, char* jobt, char* jobp,
                                        blas_int* m, blas_int* n, float* a,
                                        blas_int* lda, float* sva, float* u,
                                        blas_int* ldu, float* v, blas_int* ldv,
                                        float* work, blas_int* lwork,
                                        blas_int* iwork, blas_int* info);

void arma_fortran_noprefix(arma_dgejsv)(char* joba, char* jobu, char* jobv,
                                        char* jobr, char* jobt, char* jobp,
                                        blas_int* m, blas_int* n, double* a,
                                        blas_int* lda, double* sva, double* u,
                                        blas_int* ldu, double* v, blas_int* ldv,
                                        double* work, blas_int* lwork,
                                        blas_int* iwork, blas_int* info);

void arma_fortran_noprefix(arma_cgejsv)(
    char* joba, char* jobu, char* jobv, char* jobr, char* jobt, char* jobp,
    blas_int* m, blas_int* n, void* a, blas_int* lda, float* sva, void* u,
    blas_int* ldu, void* v, blas_int* ldv, void* cwork, blas_int* lwork,
    float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* info);

void arma_fortran_noprefix(arma_zgejsv)(
    char* joba, char* jobu, char* jobv, char* jobr, char* jobt, char* jobp,
    blas_int* m, blas_int* n, void* a, blas_int* lda, double* sva, void* u,
    blas_int* ldu, void* v, blas_int* ldv, void* cwork, blas_int* lwork,
    double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* info);
}

namespace lapack
{

// xBDSQR
inline void bdsqr(char* uplo, blas_int* n, blas_int* ncvt, blas_int* nru,
                  blas_int* ncc, float* d, float* e, float* vt, blas_int* ldvt,
                  float* u, blas_int* ldu, float* c, blas_int* ldc, float* work,
                  blas_int* info)
{
    arma_fortran_noprefix(arma_sbdsqr)(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt,
                                       u, ldu, c, ldc, work, info);
}

inline void bdsqr(char* uplo, blas_int* n, blas_int* ncvt, blas_int* nru,
                  blas_int* ncc, double* d, double* e, double* vt,
                  blas_int* ldvt, double* u, blas_int* ldu, double* c,
                  blas_int* ldc, double* work, blas_int* info)
{
    arma_fortran_noprefix(arma_dbdsqr)(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt,
                                       u, ldu, c, ldc, work, info);
}

// xGGLSE
inline void gglse(blas_int* m, blas_int* n, blas_int* p, float* a,
                  blas_int* lda, float* b, blas_int* ldb, float* c, float* d,
                  float* x, float* work, blas_int* lwork, blas_int* info)
{
    arma_fortran_noprefix(arma_sgglse)(m, n, p, a, lda, b, ldb, c, d, x, work,
                                       lwork, info);
}

inline void gglse(blas_int* m, blas_int* n, blas_int* p, double* a,
                  blas_int* lda, double* b, blas_int* ldb, double* c, double* d,
                  double* x, double* work, blas_int* lwork, blas_int* info)
{
    arma_fortran_noprefix(arma_dgglse)(m, n, p, a, lda, b, ldb, c, d, x, work,
                                       lwork, info);
}

inline void gglse(blas_int* m, blas_int* n, blas_int* p, std::complex<float>* a,
                  blas_int* lda, std::complex<float>* b, blas_int* ldb,
                  std::complex<float>* c, std::complex<float>* d,
                  std::complex<float>* x, std::complex<float>* work,
                  blas_int* lwork, blas_int* info)
{
    arma_fortran_noprefix(arma_cgglse)(m, n, p, a, lda, b, ldb, c, d, x, work,
                                       lwork, info);
}

inline void gglse(blas_int* m, blas_int* n, blas_int* p,
                  std::complex<double>* a, blas_int* lda,
                  std::complex<double>* b, blas_int* ldb,
                  std::complex<double>* c, std::complex<double>* d,
                  std::complex<double>* x, std::complex<double>* work,
                  blas_int* lwork, blas_int* info)
{
    arma_fortran_noprefix(arma_zgglse)(m, n, p, a, lda, b, ldb, c, d, x, work,
                                       lwork, info);
}

// xGEQP3
inline void geqp3(blas_int* m, blas_int* n, float* a, blas_int* lda,
                  blas_int* jpiv, float* tau, float* work, blas_int* lwork,
                  blas_int* info)
{
    arma_fortran_noprefix(sgeqp3)(m, n, a, lda, jpiv, tau, work, lwork, info);
}

inline void geqp3(blas_int* m, blas_int* n, double* a, blas_int* lda,
                  blas_int* jpiv, double* tau, double* work, blas_int* lwork,
                  blas_int* info)
{
    arma_fortran_noprefix(dgeqp3)(m, n, a, lda, jpiv, tau, work, lwork, info);
}

inline void geqp3(blas_int* m, blas_int* n, std::complex<float>* a,
                  blas_int* lda, blas_int* jpiv, std::complex<float>* tau,
                  std::complex<float>* work, blas_int* lwork, float* rwork,
                  blas_int* info)
{
    arma_fortran_noprefix(cgeqp3)(m, n, a, lda, jpiv, tau, work, lwork, rwork,
                                  info);
}

inline void geqp3(blas_int* m, blas_int* n, std::complex<double>* a,
                  blas_int* lda, blas_int* jpiv, std::complex<double>* tau,
                  std::complex<double>* work, blas_int* lwork, double* rwork,
                  blas_int* info)
{
    arma_fortran_noprefix(zgeqp3)(m, n, a, lda, jpiv, tau, work, lwork, rwork,
                                  info);
}

// xHSEQR
inline void hseqr(char* job, char* compz, blas_int* n, blas_int* ilo,
                  blas_int* ihi, float* h, blas_int* ldh, float* wr, float* wi,
                  float* z, blas_int* ldz, float* work, blas_int* lwork,
                  blas_int* info)
{
    arma_fortran_noprefix(arma_shseqr)(job, compz, n, ilo, ihi, h, ldh, wr, wi,
                                       z, ldz, work, lwork, info);
}

inline void hseqr(char* job, char* compz, blas_int* n, blas_int* ilo,
                  blas_int* ihi, double* h, blas_int* ldh, double* wr,
                  double* wi, double* z, blas_int* ldz, double* work,
                  blas_int* lwork, blas_int* info)
{
    arma_fortran_noprefix(arma_dhseqr)(job, compz, n, ilo, ihi, h, ldh, wr, wi,
                                       z, ldz, work, lwork, info);
}

inline void hseqr(char* job, char* compz, blas_int* n, blas_int* ilo,
                  blas_int* ihi, std::complex<float>* h, blas_int* ldh,
                  std::complex<float>* w, std::complex<float>* z, blas_int* ldz,
                  std::complex<float>* work, blas_int* lwork, blas_int* info)
{
    arma_fortran_noprefix(arma_chseqr)(job, compz, n, ilo, ihi, h, ldh, w, z,
                                       ldz, work, lwork, info);
}

inline void hseqr(char* job, char* compz, blas_int* n, blas_int* ilo,
                  blas_int* ihi, std::complex<double>* h, blas_int* ldh,
                  std::complex<double>* w, std::complex<double>* z,
                  blas_int* ldz, std::complex<double>* work, blas_int* lwork,
                  blas_int* info)
{
    arma_fortran_noprefix(arma_zhseqr)(job, compz, n, ilo, ihi, h, ldh, w, z,
                                       ldz, work, lwork, info);
}

// Jacobi SVD
inline void gesvj(char* joba, char* jobu, char* jobv, blas_int* m, blas_int* n,
                  float* a, blas_int* lda, float* sva, blas_int* mv, float* v,
                  blas_int* ldv, float* work, blas_int* lwork, blas_int* info)
{
    arma_fortran_noprefix(arma_sgesvj)(joba, jobu, jobv, m, n, a, lda, sva, mv,
                                       v, ldv, work, lwork, info);
}

inline void gesvj(char* joba, char* jobu, char* jobv, blas_int* m, blas_int* n,
                  double* a, blas_int* lda, double* sva, blas_int* mv,
                  double* v, blas_int* ldv, double* work, blas_int* lwork,
                  blas_int* info)
{
    arma_fortran_noprefix(arma_dgesvj)(joba, jobu, jobv, m, n, a, lda, sva, mv,
                                       v, ldv, work, lwork, info);
}

inline void gesvj(char* joba, char* jobu, char* jobv, blas_int* m, blas_int* n,
                  std::complex<float>* a, blas_int* lda, float* sva,
                  blas_int* mv, std::complex<float>* v, blas_int* ldv,
                  std::complex<float>* cwork, blas_int* lwork, float* rwork,
                  blas_int* lrwork, blas_int* info)
{
    using complex_t = std::complex<float>;
    arma_fortran_noprefix(arma_cgesvj)(
        joba, jobu, jobv, m, n, (complex_t*)a, lda, sva, mv, (complex_t*)v, ldv,
        (complex_t*)cwork, lwork, rwork, lrwork, info);
}

inline void gesvj(char* joba, char* jobu, char* jobv, blas_int* m, blas_int* n,
                  std::complex<double>* a, blas_int* lda, double* sva,
                  blas_int* mv, std::complex<double>* v, blas_int* ldv,
                  std::complex<double>* cwork, blas_int* lwork, double* rwork,
                  blas_int* lrwork, blas_int* info)
{
    using complex_t = std::complex<float>;
    arma_fortran_noprefix(arma_zgesvj)(
        joba, jobu, jobv, m, n, (complex_t*)a, lda, sva, mv, (complex_t*)v, ldv,
        (complex_t*)cwork, lwork, rwork, lrwork, info);
}

// Another Jacobi SVD
inline void gejsv(char* joba, char* jobu, char* jobv, char* jobr, char* jobt,
                  char* jobp, blas_int* m, blas_int* n, float* a, blas_int* lda,
                  float* sva, float* u, blas_int* ldu, float* v, blas_int* ldv,
                  float* work, blas_int* lwork, blas_int* iwork, blas_int* info)
{
    arma_fortran_noprefix(arma_sgejsv)(joba, jobu, jobv, jobr, jobt, jobp, m, n,
                                       a, lda, sva, u, ldu, v, ldv, work, lwork,
                                       iwork, info);
}

inline void gejsv(char* joba, char* jobu, char* jobv, char* jobr, char* jobt,
                  char* jobp, blas_int* m, blas_int* n, double* a,
                  blas_int* lda, double* sva, double* u, blas_int* ldu,
                  double* v, blas_int* ldv, double* work, blas_int* lwork,
                  blas_int* iwork, blas_int* info)
{
    arma_fortran_noprefix(arma_dgejsv)(joba, jobu, jobv, jobr, jobt, jobp, m, n,
                                       a, lda, sva, u, ldu, v, ldv, work, lwork,
                                       iwork, info);
}

inline void gejsv(char* joba, char* jobu, char* jobv, char* jobr, char* jobt,
                  char* jobp, blas_int* m, blas_int* n, std::complex<float>* a,
                  blas_int* lda, float* sva, std::complex<float>* u,
                  blas_int* ldu, std::complex<float>* v, blas_int* ldv,
                  std::complex<float>* cwork, blas_int* lwork, float* rwork,
                  blas_int* lrwork, blas_int* iwork, blas_int* info)
{
    arma_fortran_noprefix(arma_cgejsv)(joba, jobu, jobv, jobr, jobt, jobp, m, n,
                                       a, lda, sva, u, ldu, v, ldv, cwork,
                                       lwork, rwork, lrwork, iwork, info);
}

inline void gejsv(char* joba, char* jobu, char* jobv, char* jobr, char* jobt,
                  char* jobp, blas_int* m, blas_int* n, std::complex<double>* a,
                  blas_int* lda, double* sva, std::complex<double>* u,
                  blas_int* ldu, std::complex<double>* v, blas_int* ldv,
                  std::complex<double>* cwork, blas_int* lwork, double* rwork,
                  blas_int* lrwork, blas_int* iwork, blas_int* info)
{
    arma_fortran_noprefix(arma_zgejsv)(joba, jobu, jobv, jobr, jobt, jobp, m, n,
                                       a, lda, sva, u, ldu, v, ldv, cwork,
                                       lwork, rwork, lrwork, iwork, info);
}

} // namespace: lapack
} // namespace: arma

#endif /* ARMA_LAPACK_EXTRA_HPP */
