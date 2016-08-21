#ifndef EXPSUM_ESPRIT_HPP
#define EXPSUM_ESPRIT_HPP

#include <armadillo>

namespace expsum
{

//
// Naive implementation of ESPRIT algorithm
//
template <typename T>
class esprit
{
public:
    using size_type   = arma::uword;
    using value_type  = T;
    using vector_type = arma::Col<value_type>;
    using matrix_type = arma::Mat<value_type>;

    using real_type        = typename matrix_type::pod_type;
    using real_vector_type = arma::Col<real_type>;
    using real_matrix      = arma::Mat<real_type>;

    using complex_type        = std::complex<real_type>;
    using complex_vector_type = arma::Col<complex_type>;
    using complex_matrix_type = arma::Mat<complex_type>;

private:
    constexpr static const bool is_complex =
        arma::is_complex<value_type>::value;

    size_type nrows_;
    size_type ncols_;
    size_type nterms_;
    complex_vector_type exponent_;
    complex_vector_type weight_;

public:
    // Default constructor
    esprit() = default;

    //
    // @N number of sample data
    // @L window size. This is equals to the number of
    // rows of generalized Hankel matrix.
    //
    // @M maxumum number of terms used for the exponential sum.
    // `N >= L >= N / 2 + 1 >= 1` and `N - L + 1 >= M >= 1`.
    //
    esprit(size_type N, size_type L)
        : nrows_(L),
          ncols_(N - L + 1),
          nterms_(),
          exponent_(ncols_),
          weight_(ncols_)
    {
        assert(nrows_ >= ncols_ && ncols_ >= 1);
    }
    // Default copy constructor
    esprit(const esprit&) = default;
    // Defautl move constructor
    esprit(esprit&&) = default;
    // Default destructor
    ~esprit() = default;
    // Default assignment operator
    esprit& operator=(const esprit&) = default;
    // Default move assignment operator
    esprit& operator=(esprit&& rhs) = default;

    // @return number of sampling data
    size_type size() const
    {
        return nrows_ + ncols_ - 1;
    }

    // @return number of rows of trajectory matrix
    size_type nrows() const
    {
        return nrows_;
    }

    // @return number of columns of trajectory matrix. This should be a upper
    //         bound of the number of exponential functions.
    size_type ncols() const
    {
        return ncols_;
    }

    //
    //  Reset data sizes and reallocate memories for working space, if
    // necessary.
    //
    // @N number of sample data
    // @L window size. This is equals to the number of rows of generalized
    //    Hankel matrix.
    // @M maxumum number of terms used for the exponential sum.
    //    `N >= L >= N / 2 >= 1` and `N - L + 1 >= M >= 1`.
    //
    void resize(size_type N, size_type L, size_type M)
    {
        nrows_ = L;
        ncols_ = N - L + 1;

        assert(nrows_ >= M && ncols_ >= M && M >= 1);

        nterms_ = 0;
        exponent_.resize(M);
        weight_.resize(M);
    }

    //
    // Compute non-linear approximation of by as the exponential sums.
    //
    // @f values of the target function sampled on the equispaced grid. The
    //    first `size()` elemnets of `f` are used as a sampled data.
    //
    // @eps small positive number `(0 < eps < 1)` that controlls the accuracy of
    //      the fit.
    //
    // @x0 argument of first sampling data
    //
    // @delta spacing between neighbouring sample points
    //
    template <typename U>
    typename std::enable_if<arma::is_arma_type<U>::value>::type
    fit(const U& f, real_type x0, real_type delta, real_type eps);

    //
    // @return Vector view to the exponents.
    //
    auto exponents() const -> decltype(exponent_.head(nterms_))
    {
        return exponent_.head(nterms_);
    }
    //
    // @return Vector view to the weights.
    //
    auto weights() const -> decltype(weight_.head(nterms_))
    {
        return weight_.head(nterms_);
    }

    //
    // Evaluate exponential sum at a point
    //
    complex_type eval_at(real_type x) const
    {
        return arma::sum(arma::exp(-x * exponents()) % weights());
    }

private:
    template <typename U>
    typename std::enable_if<arma::is_arma_type<U>::value>::type
    compute_nodes(const U& f, real_type eps);
    void compute_weights(complex_vector_type& b);
};

template <typename T>
template <typename U>
typename std::enable_if<arma::is_arma_type<U>::value>::type
esprit<T>::fit(const U& f, real_type x0, real_type delta, real_type eps)
{
    assert(f.is_vec() && f.n_elem >= size());
    assert(real_type() < eps && eps < real_type(1));
    compute_nodes(f, eps);
    //
    // Calculate weights of exponentials
    //
    if (nterms_ > 0)
    {
        complex_vector_type b(size());
        for (size_type k = 0; k < size(); ++k)
        {
            b(k) = f(k);
        }
        compute_weights(b);
        //
        // Adjust computed paremeters
        //
        auto xi_ = exponent_.head(nterms_);
        auto w_  = weight_.head(nterms_);
        xi_      = -arma::log(xi_) / delta;
        if (x0 != real_type())
        {
            w_ %= arma::exp(-xi_ * x0);
        }
    }
}

template <typename T>
template <typename U>
typename std::enable_if<arma::is_arma_type<U>::value>::type
esprit<T>::compute_nodes(const U& f, real_type eps)
{
    //
    // Setup general fast Hankel matrix-vector product.
    //
    matrix_type H(nrows(), ncols());
    for (size_type j = 0; j < ncols(); ++j)
    {
        for (size_type i = 0; i < nrows(); ++i)
        {
            H(i, j) = f(i + j);
        }
    }
    // Compute SVD of Hankel matrix H.f
    real_vector_type sigma(ncols());
    matrix_type W(ncols(), ncols()); // right singular vector
    matrix_type dummy;               // left singular vector (not computed)
    arma::svd_econ(dummy, sigma, W, H, "right");

    size_type rank = 1;
    auto cutoff    = sigma(0) * eps;
    while (rank < sigma.size())
    {
        if (sigma(rank) < cutoff)
        {
            break;
        }
        ++rank;
    }

    auto W0 = W.submat(arma::span(0, ncols() - 2), arma::span(0, rank - 1));
    auto W1 = W.submat(arma::span(1, ncols() - 1), arma::span(0, rank - 1));
    //
    // Compute eigenvalue of matrix pencil zA - B, with matrix A, B defined
    // as follows:
    //
    matrix_type A(W0.t() * W0);
    matrix_type B(W1.t() * W0);

    complex_vector_type eigvals(exponent_.memptr(), rank,
                                /*copy_aux_mem*/ false, /*struct*/ true);
    if (!arma::eig_pair(eigvals, B, A))
    {
        throw std::runtime_error("(esprit::fit) eig_pair failed...");
    }

    nterms_ = rank;
}

template <typename T>
void esprit<T>::compute_weights(complex_vector_type& b)
{
    //
    // Solve (dual) Vanermonde system  V * x = b
    //
    //
    // REMARK: Here exponent(i) contains the value of exp(zeta(i)), not
    // exponet zeta(i) itself.
    //

    complex_matrix_type V(b.size(), nterms_);
    for (size_type i = 0; i < V.n_cols; ++i)
    {
        const auto z = exponent_(i);
        V(0, i) = real_type(1);
        for (size_type j = 1; j < V.n_rows; ++j)
        {
            V(j, i) = V(j - 1, i) * z; // z[i]**j
        }
    }

    complex_vector_type x(weight_.memptr(), nterms_, /*copy_aux_mem*/ false,
                          /*strict*/ true);
    arma::solve(x, V, b);
    return;
}

} // namespace: expsum

#endif /* EXPSUM_ESPRIT_HPP */
