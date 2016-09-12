#ifndef EXPSUM_KERNEL_FUNCTIONS_SPH_BESSEL_KERNEL_HPP
#define EXPSUM_KERNEL_FUNCTIONS_SPH_BESSEL_KERNEL_HPP

#include <armadillo>

#include "expsum/constants.hpp"
#include "expsum/kernel_functions/gamma.hpp"

namespace expsum
{

template <typename T>
struct sph_bessel_kernel
{
public:
    using size_type    = arma::uword;
    using real_type    = T;
    using complex_type = std::complex<T>;

    using vector_type = arma::Col<complex_type>;
    using matrix_type = arma::Mat<complex_type>;

private:
    vector_type exponent_;
    vector_type weight_;

public:
    void compute(size_type n, real_type band_limit, real_type eps);

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
};

template <typename T>
void sph_bessel_kernel<T>::compute(size_type n, real_type band_limit,
                                   real_type eps)
{
    constexpr const real_type one  = real_type(1);
    constexpr const real_type zero = real_type();
    // i^{n % 4}
    constexpr const complex_type phase_p[4] = {
        complex_type(one, zero), complex_type(zero, one),
        complex_type(-one, zero), complex_type(zero, -one)};
    // (-i)^{n % 4}
    constexpr const complex_type phase_m[4] = {
        complex_type(one, zero), complex_type(zero, -one),
        complex_type(-one, zero), complex_type(zero, one)};

    // k mod 4 = k & mask_mod4
    constexpr const size_type mask_mod4 = 3; // 0b11

    // static const auto huge = std::sqrt(std::numeric_limits<T>::max());
    // static const auto huge = std::numeric_limits<T>::max() / 10;

    const auto delta = 1 / band_limit;

    real_type coeff = real_type(1);
    real_type t_lower, t_upper, h = real_type(1);

    for (size_type k = 0; k <= n; ++k)
    {
        const auto beta    = static_cast<real_type>(k + 1);
        const auto log_eps = std::log(eps / coeff);
        // Lower bound
        t_lower = std::min((log_eps + std::lgamma(beta + 1)) / beta, t_lower);
        t_upper = std::max(t_upper, -std::log(delta) + std::log(-log_eps) +
                                        std::log(beta) + real_type(0.5));
        h = std::min(h, 2 * constant<T>::pi /
                            (std::log(T(3)) - beta * std::log(std::cos(T(1))) -
                             log_eps));
        std::cout << "  (" << n << ',' << k << "):\n"
                  << "    t_lower = " << t_lower << '\n'
                  << "    t_upper = " << t_upper << '\n'
                  << "    a(n, k) = " << coeff << '\n'
                  << "    h       = " << h << std::endl;
        coeff *= static_cast<real_type>((n - k) * (n + k + 1)) / (2 * k + 2);
    }

    const auto N = static_cast<size_type>(std::floor((t_upper - t_lower) / h));

    vector_type a(2 * N);
    vector_type w(2 * N);

    for (size_type i = 0; i < N; ++i)
    {
        const auto ai = std::exp(t_lower + i * h);
        a(2 * i + 0)  = complex_type(ai, real_type(-1));
        a(2 * i + 1)  = complex_type(ai, real_type(1));
    }
    // a(2 * N) = huge;

    for (size_type i = 0; i < N; ++i)
    {
        const auto ti = t_lower + i * h;
        coeff         = real_type(0.5);
        auto w0       = complex_type();
        auto w1       = complex_type();
        for (size_type k = 0; k <= n; ++k)
        {
            auto base_w = coeff * h * std::exp((k + 1) * ti);
            w0 += base_w * phase_m[(n - k + 1) & mask_mod4];
            w1 += base_w * phase_p[(n - k + 1) & mask_mod4];
            coeff *= static_cast<real_type>((n - k) * (n + k + 1)) /
                     (2 * (k + 1) * (k + 1));
        }
        w(2 * i + 0) = w0;
        w(2 * i + 1) = w1;
    }
    // w(2 * N) = -arma::sum(w.head(2 * N)) + (n == 0 ? one : zero);

    exponent_.swap(a);
    weight_.swap(w);
    return;
}

} // namespace: expsum

#endif /* EXPSUM_KERNEL_FUNCTIONS_SPH_BESSEL_KERNEL_HPP */
