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

private:
    template <typename UnaryFunction1, typename UnaryFunction2,
              typename UnaryFunction3>
    static real_type newton(real_type guess, real_type tol,
                            UnaryFunction1 transform, UnaryFunction2 f,
                            UnaryFunction3 df)
    {
        // Max iterations in the Newton method
        static const size_type max_iter = 10000;

        auto counter = max_iter;
        auto t0      = guess;

        while (--counter > 0)
        {
            // Transformation of an argument
            const auto x  = transform(t0);
            const auto fx = f(x);
            // We assume df(x) never to be too small.
            const auto dfx = df(x);
            const auto t1  = t0 - fx / dfx;
            std::cout << t0 << '\t' << x << '\t' << fx << '\t' << dfx
                      << std::endl;
            if (std::abs(t1 - t0) < std::abs(t1) * tol)
            {
                break;
            }
            t0 = t1;
        }

        return t0;
    }

    static real_type lower_bound(real_type beta, real_type /*delta*/,
                                 real_type eps, real_type tolerance)
    {
        const auto log_eps = std::log(eps);
        const auto scale   = 1 / std::tgamma(beta);
        const auto guess   = (log_eps + std::lgamma(beta + 1)) / beta;

        auto xt = [=](real_type t) { return std::exp(t); };
        auto fx = [=](real_type x) { return gamma_p(beta, x) - eps; };
        auto df = [=](real_type x) {
            return scale * std::exp(beta * std::log(x) - x);
        };

        return newton(guess, tolerance, xt, fx, df);
    }

    static real_type upper_bound(real_type beta, real_type delta, real_type eps,
                                 real_type tolerance)
    {
        const auto log_eps   = std::log(eps);
        const auto log_delta = std::log(delta);
        const auto guess =
            -log_delta + std::log(-log_eps) + std::log(beta) + real_type(0.5);
        const auto scale = 1 / std::tgamma(beta);

        auto xt = [=](real_type t) { return delta * std::exp(t); };
        auto fx = [=](real_type x) { return eps - gamma_q(beta, x); };
        auto df = [=](real_type x) {
            return scale * std::exp(beta * std::log(x) - x);
        };

        return newton(guess, tolerance, xt, fx, df);
    }
};

template <typename T>
void sph_bessel_kernel<T>::compute(size_type n, real_type band_limit,
                                   real_type eps)
{
    const auto delta = 1 / band_limit;

    real_type coeff = real_type(1);

    for (size_type k = 0; k <= n; ++k)
    {
        // auto t_lower =
        //     lower_bound(static_cast<real_type>(k + 1), delta, eps / coeff,
        //     eps);
        // std::cout << t_lower << std::endl;
        auto t_upper =
            upper_bound(static_cast<real_type>(k + 1), delta, eps / coeff, eps);
        std::cout << t_upper << std::endl;
        // std::cout << "(" << n << ',' << k << "): " << t_lower << "\t" <<
        // t_upper
        //           << std::endl;
        coeff *= static_cast<real_type>((n - k) * (n + k + 1)) / (2 * k + 2);
    }
}

} // namespace: expsum

#endif /* EXPSUM_KERNEL_FUNCTIONS_SPH_BESSEL_KERNEL_HPP */
