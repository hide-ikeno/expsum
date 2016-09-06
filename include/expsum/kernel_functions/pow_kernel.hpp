#ifndef EXPSUM_KERNEL_FUNCTIONS_POW_KERNEL_HPP
#define EXPSUM_KERNEL_FUNCTIONS_POW_KERNEL_HPP

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <armadillo>

#include "expsum/constants.hpp"
#include "expsum/exponential_sum.hpp"
#include "expsum/kernel_functions/gamma.hpp"
#include "expsum/kernel_functions/gauss_quadrature.hpp"

namespace expsum
{

//
// Approximate power function by an exponential sum.
//
// For given accuracy ``$\epsilon > 0$`` and distance to the singularity
// ``$\delta > 0$``, this find the approximation of power function
// ``$f(r)=r^{-\beta}$`` with a linear combination of exponential functions
// such that
//
// ``` math
// \left| r^{-\beta}-\sum_{m=1}^{M}w_{m}e^{-a_{m}r} \right|
//    \leq r^{-\beta}\epsilon
// ```
//
// for ``$r\in [\delta,1]$.
//
// @beta   power factor ``$beta > 0$``
// @delta  distance to the singularity ``$0 < \delta < 1$``
// @eps    required accuracy ``$0 < \epsilon < e^{-1}$``
// @return pair of vectors holding expnents ``$a_{m}$`` and weights ``$w_{m}$``
//

template <typename T>
struct pow_kernel
{
public:
    using size_type   = arma::uword;
    using real_type   = T;
    using vector_type = arma::Col<T>;
    using matrix_type = arma::Mat<T>;

private:
    vector_type exponent_;
    vector_type weight_;
    real_type beta_;
    real_type delta_;
    real_type eps_;

public:
    void compute(real_type beta__, real_type delta__, real_type eps__);

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

    real_type beta() const
    {
        return beta_;
    }

    real_type delta() const
    {
        return delta_;
    }

    real_type eps() const
    {
        return eps_;
    }

private:
    static void check_params(real_type beta__, real_type delta__,
                             real_type eps__);
    static size_type get_num_intervals(real_type beta__, real_type delta__,
                                       real_type eps__);
    static std::tuple<size_type, real_type>
    get_num_gauss_jacobi(real_type beta__, real_type eps__);
    static std::tuple<size_type, real_type>
    get_num_gauss_legendre(real_type beta__, real_type eps__);
};

template <typename T>
void pow_kernel<T>::compute(real_type beta__, real_type delta__,
                            real_type eps__)
{
    check_params(beta__, delta__, eps__);

    beta_  = beta__;
    delta_ = delta__;
    eps_   = eps__;

    //
    // Find the sub-optimal sum-of-exponential approximation by discretizing the
    // integral representation of power function
    //
    // ``` math
    //  r^{-\beta} = \frac{1}{\Gamma(\beta)}
    //               \int_{0}^{\infty} e^{-rx} x^{\beta-1} dx \quad (t > 0).
    // ```
    //
    // Split the integral into $J+1$ sub-intervals as
    //
    // ``` math
    //  r^{-\beta}
    //     = \frac{1}{\Gamma(\beta)}
    //       \left( \int_{0}^{2} + \int_{4}^{2} + \int_{8}^{4} + \cdots
    //             + \int_{2^{J-1}}^{2^J} + \int_{2^J}^{\infty} \right)
    //       e^{-ry} x^{\beta-1} dy.
    // ```
    //
    // The integral over the first interval $[0,2]$ is approximated by the $N_1$
    // point Gauss-Jacobi rule, while the integral over the inteval
    // $[2^{j-1},2^j]$ is approximated by the $N_2$ point Gauss-Legendre rule.
    //
    // The number of intervals $J$ and quadrature points $N_1,N_2$ are
    // determined so as to the relative error of the truncation becomes smaller
    // than $\epsilon$.
    //
    size_type J, N1, N2;
    T b1, b2;
    J = get_num_intervals(beta__, delta__ / 2, eps__);
    std::tie(N1, b1) = get_num_gauss_jacobi(beta__, eps__ / (J + 1));
    std::tie(N2, b2) = get_num_gauss_legendre(beta__, eps__ / (J + 1));
    //
    // Discritization of the integral representation by applying the quadrature
    // rule to each interval
    //
    const size_type N = N1 + (J - 1) * N2;
    gauss_jacobi_rule<T> gaujac(N1, T(), beta__ - 1);
    gauss_legendre_rule<T> gauleg(N2);

    assert(gaujac.size() == N1);
    assert(gauleg.size() == N2);

    vector_type a(N);
    vector_type w(N);

    auto ait = std::begin(a);
    auto wit = std::begin(w);

    const auto scale = T(1) / std::tgamma(beta__);

    for (size_type i = 0; i < gaujac.size(); ++i)
    {
        *ait = gaujac.x(i) + T(1);
        *wit = scale * gaujac.w(i);
        ++ait;
        ++wit;
    }

    // The upper/lower bound of interval
    auto lower = T(1);
    auto upper = T(2);
    for (size_type k = 1; k < J; ++k)
    {
        lower *= 2; // lower = 2^(k-1)
        upper *= 2; // upper = 2^k
        const auto a1 = (upper - lower) / 2;
        const auto b1 = (upper + lower) / 2;
        for (size_type i = 0; i < gauleg.size(); ++i)
        {
            const auto y = a1 * gauleg.x(i) + b1;
            *ait         = y;
            *wit         = a1 * scale * std::pow(y, beta__ - 1) * gauleg.w(i);
            ++ait;
            ++wit;
        }
    }

    std::swap(exponent_, a);
    std::swap(weight_, w);
    return;
}

//------------------------------------------------------------------------------
// Private member functions
//------------------------------------------------------------------------------
template <typename T>
void pow_kernel<T>::check_params(real_type beta__, real_type delta__,
                                 real_type eps__)
{
    if (!(beta__ > real_type()))
    {
        std::ostringstream msg;
        msg << "Invalid value for the argument `beta': "
               "beta > 0 expected, but beta = "
            << beta__ << " is given";
        throw std::invalid_argument(msg.str());
    }

    if (!(real_type() < delta__ && delta__ < real_type(1)))
    {
        std::ostringstream msg;
        msg << "Invalid value for the argument `delta': "
               "0 < delta < 1 expected, but delta = "
            << delta__ << " is given";
        throw std::invalid_argument(msg.str());
    }

    if (!(real_type() < eps__ &&
          eps__ < real_type(1) / arma::Datum<real_type>::e))
    {
        std::ostringstream msg;
        msg << "Invalid value for the argument `eps': "
               "0 < eps < 1/e expected, but "
            << eps__ << " is given";
        throw std::invalid_argument(msg.str());
    }
}

template <typename T>
typename pow_kernel<T>::size_type
pow_kernel<T>::get_num_intervals(real_type beta__, real_type delta__,
                                 real_type eps__)
{
    //
    // Find minimal integer J, such that
    //
    //  Gamma(beta, delta * 2^J) / Gamma(beta) <= eps / (J + 1)
    //
    size_type J1 = 0;
    size_type J2 = 20;

    while (true)
    {
        auto x  = delta__ * std::pow(T(2), J2);
        auto fj = gamma_q(beta__, x);

        if (fj * (J2 + 1) <= eps__)
        {
            if (J2 - J1 <= 1)
            {
                break;
            }

            J2 = (J1 + J2) / 2;
        }
        else
        {
            J1 = J2;
            J2 = J2 + 5;
        }
    }

    return J2;
}

template <typename T>
std::tuple<typename pow_kernel<T>::size_type, T>
pow_kernel<T>::get_num_gauss_jacobi(real_type beta__, real_type eps__)
{
    //
    // Find N such that R(N, b) <= eps with
    //
    // R(N, x) = 2^(beta + 2) / Gamma(beta + 1) * exp(cosh(x)-1)
    //         * exp(-2*N*x) / (1 - exp(-2*x)).
    //
    // Here, the parameter b is chosen to minimize R(N, x) for each N.
    //

    // Logarithm of pre-factor of R(N, x),
    const auto ln_pre = (beta__ + 2) * std::log(T(2)) - std::lgamma(beta__ + 1);

    const size_type max_iter = 100;
    const auto ln_eps        = std::log(eps__);

    // Initial guesses of parameters
    size_type n1 = 0;
    size_type n2 = 4;
    auto x       = T(4);

    while (true)
    {
        // For given N = n2, minimize R(N, x) w.r.t. x
        for (size_type i = 0; i < max_iter; ++i)
        {
            // R'(N, x) without pre-factor
            auto df = std::sinh(x) + 2 / std::expm1(2 * x) - 2 * n2;
            // R''(N, x) without pre-factor
            auto t   = std::expm1(2 * x);
            auto ddf = std::cosh(x) - 4 * std::exp(2 * x) / (t * t);
            auto dx  = -df / ddf;

            if (x + dx < T()) // ensure x > 0
            {
                x *= T(0.5);
            }
            else
            {
                x += dx;
            }

            if (std::abs(dx) <= x * eps__)
            {
                break;
            }
        }

        // ln(R(N, b))
        auto ln_resid = (std::expm1(x) + std::expm1(-x)) / 2 // cosh(x) - 1
                        - 2 * n2 * x                         // ln(exp(-2*n*x))
                        - std::log(-std::expm1(-2 * x)); // log(1 - exp(-2*x))

        if (ln_pre + ln_resid <= ln_eps) // equiv to R(N, b) <= eps
        {
            if (n2 - n1 <= 1)
            {
                break;
            }

            n2 = (n1 + n2) / 2;
        }
        else
        {
            n1 = n2;
            n2 = n2 + 2;
        }
    }

    return {n2, x};
}

template <typename T>
std::tuple<typename pow_kernel<T>::size_type, T>
pow_kernel<T>::get_num_gauss_legendre(real_type beta__, real_type eps__)
{
    //
    // Find N such that R(N, b) <= eps with
    //
    // R(N, x) = 8 / Gamma(beta) * (beta / (e*x))^(beta)
    //         * rho^(-2*N) / (1 - rho^(-2)) * g(x),
    //
    // for 0 < x < 2, where rho(x) = 3 - x + sqrt((2 - x) * (4 - x)), and
    //
    //  g(x) = x^(beta-1)        if 0 < beta <= 1,
    //  g(x) = (6 - x)^(beta-1)  if beta > 1
    //

    // Logarithm of pre-factor of R(N, x),
    const auto ln_pre = std::log(T(8)) - std::lgamma(beta__) +
                        beta__ * std::log(beta__ / constant<T>::e);

    const size_type max_iter = 100;
    const auto ln_eps        = std::log(eps__);

    // Initial guesses of parameters

    size_type n1 = 0;
    size_type n2 = 4;
    auto x       = T(1);

    // function log(g(x))
    auto fn_g = [=](T z) {
        return beta__ <= T(1)
                   ? -std::log(z)
                   : (beta__ - 1) * std::log(6 - z) - beta__ * std::log(z);
    };

    // first derivative of log(g(x))
    auto fn_dg = [=](T z) {
        return beta__ <= T(1) ? -1 / z : (1 - beta__) / (6 - z) - beta__ / z;
    };

    // second derivative of log(g(x))
    auto fn_d2g = [=](T z) {
        return beta__ <= T(1)
                   ? 1 / (z * z)
                   : (1 - beta__) / ((6 - z) * (6 - z)) + beta__ / (z * z);
    };

    while (true)
    {
        // For given N = n2, minimize R(N, x) w.r.t. x
        for (size_type i = 0; i < max_iter; ++i)
        {
            auto p    = std::sqrt((2 - x) * (4 - x));
            auto pinv = 1 / p;
            // rho(x) = 3 - x + sqrt((2 - x) * (4 - x))
            auto rho = 3 - x + p;
            // d rho(x) / d x = -1 - (3 - x) / sqrt((2 - x) * (4 - x))
            //                = - rho / p
            // auto drho = -rho * pinv;
            // d^2 rho(x) / d x^2
            // auto d2rho = rho * pinv * pinv * (1 - (3 - x) * pinv);

            //
            // Let f(x) = log(rho^{-2N}(x) / (1 -rho^{2}(x))) and compute its
            // first and second derivatives
            //
            auto t   = 1 / (rho + 1) * (rho - 1);
            auto df  = 2 * (n2 - t) * pinv;
            auto d2f = (df * (3 - x) - 2 * rho * rho * t * t) * pinv * pinv;

            //
            // Compute first and second derivatives of log(g(x))
            //
            auto dg  = fn_dg(x);
            auto d2g = fn_d2g(x);

            // d log(R(N, x)) / d x without pre-factor
            auto dr = df + dg;
            // d^2 log(R(N, x)) / d x^2  without pre-factor
            auto d2r = d2f + d2g;

            // Update x
            auto dx = -dr / d2r;

            if (x + dx < T()) // ensure x > 0
            {
                x *= T(0.5);
            }
            else if (x + dx > T(2)) // ensure x < 2
            {
                x += (2 - x) / T(2);
            }
            else
            {
                x += dx;
            }

            if (std::abs(dx) <= x * eps__)
            {
                break;
            }
        }

        // ln(R(N, b))
        const auto rho = 3 - x + std::sqrt((2 - x) * (4 - x));
        const auto fx =
            -T(2 * n2) * std::log(rho) - std::log1p(-T(1) / rho / rho);
        const auto ln_resid = fx + fn_g(x);

        if (ln_pre + ln_resid <= ln_eps) // equiv to R(N, b) <= eps
        {
            if (n2 - n1 <= 1)
            {
                break;
            }

            n2 = (n1 + n2) / 2;
        }
        else
        {
            n1 = n2;
            n2 = n2 + 2;
        }
    }

    return {n2, std::acosh(3 - x)};
}
} // namespace: expsum

#endif /* EXPSUM_KERNEL_FUNCTIONS_POW_KERNEL_HPP */
