#ifndef EXPSUM_HANKEL_MATRIX_VECTOR_PRODUCT_HPP
#define EXPSUM_HANKEL_MATRIX_VECTOR_PRODUCT_HPP

#include <armadillo>

#include "fftw3/shared_plan.hpp"

namespace expsum
{
/*!
 * Fast matrix-vector product for generalized Hankel matrix.
 *
 * A general Hankel matrix is a matrix of special form give as
 *
 * \f[
 *   A = \left[ \begin{array}{cccccc}
 *     h_0 & h_1 & h_2 & h_3 & \cdots & h_{n-1} \\
 *     h_1 & h_2 & h_3 & h_4 & \cdots & h_{n}   \\
 *     h_2 & h_3 & h_4 & h_5 & \cdots & h_{n+1} \\
 *     h_3 & h_4 & h_5 & h_6 & \cdots & h_{n+2} \\
 *     \vdots & \vdots & \vdots &\vdots &\ddots & \vdots \\
 *     h_{m-1} & h_{m} & h_{m+1} & h_{m+2} & \cdots & h_{m+n-1} \\
 *   \end{array} \right],
 * \f]
 *
 * where \f$ m \f$ and \f$ n \f$ are the number of rows and colums of matrix.
 * From the definition above, a general Hankel matrix \f$ A \f$ can fully be
 * determined by a vector \f$ h = [h_0,h_1,...,h_{n+m-1}]^{T}\f$ composed of the
 * elements of first column and last row where \f$ A_{ij} = h_{i+j}.\f$
 *
 * This class compute the matrix-vector product,
 *
 * \f[
 *   \bm{y} = A \bm{x},
 * \f]
 *
 * where \f$ \bm{x} = [x_0,x_1,\dots,x_{n-1}]^{T} \f$ and \f$ \bm{y} =
 * [y_0,y_1,\dots,y_{m-1}]^{T}. \f$ This product can be efficiently computed by
 * the fast Fourier transfor (FFT), as follows.
 *
 * Let define a new vector \f$ \hat{\bm{c}} \f$ of size \f$ n + m - 1 \f$ as
 *
 * \f[
 *   \hat{\bm{c}}=[h_{n-1},\dots,h_{n+m-2},h_{0},\dots,h_{n-2}]^{T},
 * \f]
 *
 * and corresponding circulant matrix
 *
 * \f[
 *   C = \left[ \begin{array}{}
 *         c_0     & c_{n+m-2} & \cdots    & c_{2}  & c_{1} \\
 *         c_1     & c_{0}     & c_{n+m-2} &        & c_{2} \\
 *         \vdots  & c_{1}     & c_{0}     & \ddots & \vdots    \\
 *         c_{n+m-3} &         & \ddots    & \ddots & c_{n+m-2} \\
 *         c_{n+m-2} & c_{n+m-3} & \cdots  & c_{1}  & c_{0}
 *       \end{array} \right].
 * \f]
 *
 * For a given vector \f$ \bm{x} \f$ of length \f$ n, \f$ we define a auxilialy
 * vector of length \f$ n + m - 1 \f$ as
 *
 * \f[
 *   \hat{\bm{x}}=[x_{n-1},x_{n-2},\dots,x_{0},0,\dots,0]^{T}.
 * \f]
 *
 * Then, the result vector \f$ \bm{y} \f$ can be obtained as the first
 * \f$ m \f$-elemets of the vector \f$ \hat{\bm{y}}\equiv C\hat{\bm{x}}. \f$
 * The produt \f$ C\hat{\bm{x}} \f$ can be evaluated as,
 *
 * \f[
 *   \hat{\bm{y}} = \text{IFFT}(\text{FFT}(\hat{\bm{c}}) \odot
 *                              \text{FFT}(\hat{\bm{x}})).
 * \f]
 *
 * Here, \$f \text{FFT}(\bm{v}) \f$ and \$f \text{IFFT}(\bm{v}) \f$ denote
 * one-dimensional FFT and inverse FFT of vector \bm{v}, and \f$ \odot \f$
 * denotes a element-wise multiplication of two vectors.
 *
 * The computational complexity of this algorithm is \f$
 * \mathcal{O}((m+n-1)\log(m+n-1)) \f$, rather than \f$ \mathcal{O}(mn) \f$
 * for ordinary dense matrix-vector operation in BLAS2.
 *
 * For the computational efficiency, the FFT of the vector \f$ \hat{\bm{c}} \f$
 * is pre-computed and stored internally.
 *
 */

template <typename T>
class hankel_gemv
{
public:
    using size_type   = arma::uword;
    using value_type  = T;
    using vector_type = arma::Col<value_type>;

    using real_type           = typename vector_type::pod_type;
    using complex_type        = std::complex<real_type>;
    using complex_vector_type = arma::Col<complex_type>;

private:
    using fft  = fftw3::fft<real_type>;
    using ifft = fftw3::ifft<real_type>;

    typename fft::plan_pointer fft_plan_;
    typename ifft::plan_pointer ifft_plan_;

    size_type nrows_;
    size_type ncols_;

    mutable vector_type work_;
    complex_vector_type caux_;
    mutable complex_vector_type xaux_;

public:
    /// Default constructor
    hankel_gemv() = default;

    /// Create an Hankel matrix operator with memory preallocation.
    hankel_gemv(size_type nrows, size_type ncols, size_type fft_size = 0)
        : fft_plan_(),
          ifft_plan_(),
          nrows_(nrows),
          ncols_(ncols),
          work_(std::max(fft_size, nrows + ncols - 1)),
          caux_(arma::is_complex<value_type>::value ? work_.size()
                                                    : work_.size() / 2 + 1),
          xaux_(caux_.size())
    {
        set_fft_plans();
    }

    /// Copy constructor (default)
    hankel_gemv(const hankel_gemv&) = default;

    /// Move constructor (default)
    hankel_gemv(hankel_gemv&&) = default;

    /// Destructor (default)
    ~hankel_gemv() = default;

    /// Copy assignment operator
    hankel_gemv& operator=(const hankel_gemv&) = default;

    /// Move assignment operator
    hankel_gemv& operator=(hankel_gemv&&) = default;

    /// @return the number of rows of the Hankel matrix
    size_type nrows() const
    {
        return nrows_;
    }
    /// @return the number of columns of the Hankel matrix
    size_type ncols() const
    {
        return ncols_;
    }
    /// @return number of coefficients that defines this Hankel matrix
    size_type size() const
    {
        return nrows() + ncols() - 1;
    }

    /// Reallocate internal memory space
    void resize(size_type nrows, size_type ncols, size_type fft_size = 0)
    {
        fft_size = std::max(fft_size, nrows + ncols - 1);

        nrows_ = nrows;
        ncols_ = ncols;
        work_.set_size(fft_size);
        caux_.set_size(arma::is_complex<value_type>::value
                           ? work_.size()
                           : work_.size() / 2 + 1);
        xaux_.set_size(caux_.size());

        set_fft_plans();
    }
    ///
    /// Set coefficients that defines the Hankel matrix.
    ///
    template <typename T1>
    typename std::enable_if<arma::is_arma_type<T1>::value>::type
    set_coeffs(const T1& coeffs)
    {
        assert(coeffs.is_vec() && coeffs.n_elem == size());
        //
        // Set first column of circulant matrix C. Then compute the discrete
        // Fourier transform this vector and store the result into \c caux.
        //
        const auto nhead    = nrows();
        const auto ntail    = ncols() - 1;
        const auto npadding = work_.size() - nhead - ntail;

        work_.head(nhead) = coeffs.tail(nhead);
        if (npadding > size_type())
        {
            work_.subvec(nhead, nhead + npadding - 1).zeros();
        }
        work_.tail(ntail) = coeffs.head(ntail);

        // caux_ <-- FFT[work_]
        fft::run(fft_plan_, work_.memptr(), caux_.memptr());
        caux_ *= real_type(1) / work_.size();
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
        assert(x.is_vec() && x.n_elem == ncols());
        assert(y.is_vec() && y.n_elem == nrows());
        //
        // Form new vector x' = [x(n-1),x(n-2),...,x(0),0....0] of length
        // n + m - 1, and compute FFT.
        //
        work_.head(ncols()) = arma::flipud(x);
        work_.tail(work_.size() - ncols()).zeros();
        // xaux_ <-- FFT[work_]
        fft::run(fft_plan_, work_.memptr(), xaux_.memptr());
        //
        // y[0:nrows] = IFFT(FFT(c') * FFT(x'))[0:nrows]
        //
        xaux_ %= caux_;
        ifft::run(ifft_plan_, xaux_.memptr(), work_.memptr());
        if (beta == value_type())
        {
            y = work_.head(nrows());
        }
        else
        {
            y = work_.head(nrows()) + beta * y;
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
        assert(x.is_vec() && x.n_rows == nrows());
        assert(y.is_vec() && y.n_rows == ncols());
        //
        // Form new vector x' = [0,0,...,0,x(m-1),x(m-2),...,x(0)] of length
        // n + m - 1, and compute FFT.
        //
        work_.head(work_.size() - nrows()).zeros();
        work_.tail(nrows()) = arma::conj(arma::flipud(x));
        // xaux_ <-- FFT[work_]
        fft::run(fft_plan_, work_.memptr(), xaux_.memptr());
        //
        // y[0:nrows-1] = IFFT(FFT(c') * FFT(x'))[0:nrows-1]
        //
        xaux_ %= caux_;
        ifft::run(ifft_plan_, xaux_.memptr(), work_.memptr());
        if (beta == value_type())
        {
            y = arma::conj(work_.tail(ncols()));
        }
        else
        {
            y = arma::conj(work_.tail(ncols())) + beta * y;
        }
    }

private:
    void set_fft_plans()
    {
        const int n       = static_cast<int>(work_.size());
        const int howmany = 1;
        fft_plan_ = fft::make_plan(n, howmany, work_.memptr(), xaux_.memptr());
        ifft_plan_ =
            ifft::make_plan(n, howmany, xaux_.memptr(), work_.memptr());
    }
};

/*!
 * Create a Hankel matrix in dense form from the sequence of elements.
 *
 * This function creates a \f$ m \times n \f$ Hankel matrix \f$ A \f$ in the
 * dense form from a given vector of matrix element \f$ h =
 * [h_0,h_1,...,h_{n+m-1}]^{T}\f$ such that,
 *
 * \f[
 *   A = \left[ \begin{array}{}
 *     h_0 & h_1 & h_2 & h_3 & \cdots & h_{n-1} \\
 *     h_1 & h_2 & h_3 & h_4 & \cdots & h_{n}   \\
 *     h_2 & h_3 & h_4 & h_5 & \cdots & h_{n+1} \\
 *     h_3 & h_4 & h_5 & h_6 & \cdots & h_{n+2} \\
 *     \vdots & \vdots & \vdots &\vdots &\ddots & \vdots \\
 *     h_{m-1} & h_{m} & h_{m+1} & h_{m+2} & \cdots & h_{m+n-1} \\
 *   \end{array} \right]
 * \f]
 *
 * \param[in] nrows number of rows, \f$ m \f$
 * \param[in] ncols number of columns, \f$ n \f$
 * \param[in] h vector of elments of Hankel matrix with length \c nrows+ncols-1
 * \return \c arma::Mat with same scalar type of input vector type \c T1.
 */
template <typename T1>
typename std::enable_if<arma::is_arma_type<T1>::value,
                        arma::Mat<typename T1::elem_type>>::type
make_dense_hankel(arma::uword nrows, arma::uword ncols, const T1& h)
{
    assert(h.n_elem == nrows + ncols - 1);
    arma::Mat<typename T1::elem_type> A(nrows, ncols);

    for (arma::uword col = 0; col < ncols; ++col)
    {
        for (arma::uword row = 0; row < nrows; ++row)
        {
            A(row, col) = h(row + col);
        }
    }

    return A;
}

namespace detail
{
template <typename T>
inline T abs2(T x)
{
    return x * x;
}

template <typename T>
inline T abs2(std::complex<T> x)
{
    return std::real(x) * std::real(x) + std::imag(x) * std::imag(x);
}
} // namespace: detail

/*!
 * Compute the Frobenius norm of general Hankel matrix.
 *
 * This function computes the Frobenius norm of \c nrows-by-ncols general Hankel
 * matrix defined by the given vector of elements.
 *
 * \param[in] nrows number of rows
 * \param[in] ncols number of columns
 * \param[in] h a vector that determines the Hankel matrix \f$ A \f$.  If <tt>
 * h.size() >= N </tt> with <tt> N = nrows + ncols - 1, </tt> first \c N elemnts
 * are refered as the elements of Hankel matrix. If <tt> h.size() < N, </tt>
 * rest of elements are assumed to be zero.
 */

template <typename T1>
typename T1::pod_type fnorm_hankel(arma::uword nrows, arma::uword ncols,
                                   const T1& h)
{
    arma::uword m, n;
    std::tie(m, n) = std::minmax(nrows, ncols);
    arma::uword l = m + n - 1;

    auto sqsum = typename T1::pod_type();

    if (h.n_elem > m)
    {
        for (arma::uword i = 0; i < m; ++i)
        {
            sqsum += (i + 1) * std::norm(h(i));
        }

        if (h.n_elem > n)
        {
            for (arma::uword i = m; i < n; ++i)
            {
                sqsum += m * std::norm(h(i));
            }

            for (arma::uword i = n; i < std::min(l, h.n_elem); ++i)
            {
                sqsum += (l - i) * std::norm(h(i));
            }
        }
        else
        {
            for (arma::uword i = m; i < h.n_elem; ++i)
            {
                sqsum += m * std::norm(h(i));
            }
        }
    }
    else
    {
        for (arma::uword i = 0; i < h.n_elem; ++i)
        {
            sqsum += (i + 1) * std::norm(h(i));
        }
    }

    return std::sqrt(sqsum);
}
} // namespace: expsum

#endif /* EXPSUM_HANKEL_MATRIX_VECTOR_PRODUCT_HPP */
