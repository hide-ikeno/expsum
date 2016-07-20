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

//
// Fused multiply-add
//
// Computes `(x * y) + z` as if to infinite precision and rounded only once to
// fit the result type.
//
template <typename T>
inline T fma(T x, T y, T z)
{
    return std::fma(x, y, z);
}

template <typename T>
inline std::complex<T> fma(const std::complex<T>& x, const std::complex<T>& y,
                           const std::complex<T>& z)
{
    const auto re =
        std::fma(-x.imag(), y.imag(), std::fma(x.real(), y.real(), z.real()));
    const auto im =
        std::fma(x.real(), y.imag(), std::fma(x.imag(), y.real(), z.imag()));
    return {re, im};
}

} // namespace numeric
} // namespace expsum

#endif /* EXPFIT_NUMERIC_HPP */
