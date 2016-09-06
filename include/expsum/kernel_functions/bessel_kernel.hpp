#ifndef EXPSUM_KERNEL_FUNCTIONS_BESSEL_KERNEL_HPP
#define EXPSUM_KERNEL_FUNCTIONS_BESSEL_KERNEL_HPP

#include <cassert>
#include <cmath>

#include <armadillo>

namespace expsum
{

template <typename T>
struct bessel_j_kernel
{
public:
    using size_type   = arma::uword;
    using real_type   = T;
    using vector_type = arma::Col<T>;
    using matrix_type = arma::Mat<T>;

private:
    vector_type exponent_;
    vector_type weight_;
    real_type v_;   // order of Bessel function Jv(x)
    real_type eps_; // tolerance

public:
    void compute(real_type v, real_type eps);

    size_type size() const
    {
        return exponent_.size();
    }

    const vector_type& exponents() const
    {
        return exponent_;
    }

    const vector_type& weights() const
    {
        return weight_;
    }

private:
};

template <typename T>
void bessel_j_kernel<T>::compute(real_type v, real_type eps)
{
}

} // namespace: expsum

#endif /* EXPSUM_KERNEL_FUNCTIONS_BESSEL_KERNEL_HPP */
