#ifndef EXPFIT_NUMERIC_HPP
#define EXPFIT_NUMERIC_HPP

#include <complex>

namespace expsum
{

namespace numeric
{

//
// Complex conjugate of a given scalar.
//
template <typename T>
T conj(T x)
{
    return x;
}

template <typename T>
std::complex<T> conj(const std::complex<T>& x)
{
    return std::conj(x);
}


//
// Square of the absolute value
//
template <typename T>
T abs2(const T& x)
{
    return x * x;
}

template <typename T>
T abs2(const std::complex<T>& x)
{
    const auto re = std::real(x);
    const auto im = std::imag(x);
    return re * re + im * im;
}

} // namespace numeric

}  // namespace: expsum

#endif /* EXPFIT_NUMERIC_HPP */
