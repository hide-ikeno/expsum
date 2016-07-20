#ifndef EXPSUM_APPROX_POW_HPP
#define EXPSUM_APPROX_POW_HPP

#include <cassert>
#include <tuple>

#include <armadillo>

#include "expsum/exponential_sum.hpp"
#include "expsum/gamma.hpp"

namespace expsum
{

namespace detail
{
template <typename T, typename UnaryFunction1, typename UnaryFunction2,
          typename UnaryFunction3>
T newton_solve(T initial_guess, T tol, UnaryFunction1 transform,
               UnaryFunction2 f, UnaryFunction3 df)
{
    // Max iterations in the Newton method
    constexpr const std::size_t max_iter = 10000;

    std::size_t counter = max_iter;
    auto t0             = initial_guess;

    while (--counter > 0)
    {
        // Transformation of an argument
        const auto x  = transform(t0);
        const auto fx = f(x);
        // We assume df(x) never to be too small.
        const auto dfx = df(x);
        const auto t1  = t0 - fx / dfx;
        if (std::abs(t1 - t0) < std::abs(t1) * tol)
        {
            break;
        }
        t0 = t1;
    }

    return t0;
}

} // namespace: detail

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
exponential_sum<T, T> approx_pow(T beta, T delta, T eps)
{
    using size_type   = arma::uword;
    using result_type = exponential_sum<T, T>;

    assert(beta > T());
    assert(T() < delta && delta < T(1));
    assert(T() < eps && eps < std::exp(T(-1)));

    const T log_beta  = std::log(beta);
    const T log_delta = std::log(delta);
    const T log_eps   = std::log(eps);
    //
    // Parameters required for obtaining sub-optimal expansion
    //
    const T newton_tol = eps;
    const T scale_l    = T(1) / std::tgamma(beta);
    // Eq. (31) of [Beylkin2010]
    auto t_lower = detail::newton_solve(
        // Initial guess --- Eq. (33) of [Beylkin2010]
        (log_eps + std::lgamma(T(1) + beta)) / beta,
        // Tolerance
        newton_tol,
        // transformation: x = exp(t)
        [=](T t) { return std::exp(t); },
        // f(x)  = gamma_p(x) - eps (with x = exp(t / 2))
        [=](T x) { return gamma_p(beta, x) - eps; },
        // f'(x) * dx/dt = exp(-x) * pow(x, beta-1)  / tgamma(beta) * x
        [=](T x) { return std::exp(beta * std::log(x) - x) * scale_l; });

    // std::cout << "# t_lower = "   << t_lower << std::endl;

    // Eq. (32) of [Beylkin2010]
    const auto scale_u = scale_l / delta;
    auto t_upper       = detail::newton_solve(
        // Initial guess --- Eq. (34) of [Beylkin2010]
        std::log(-log_eps) - log_delta + log_beta + T(0.5),
        // Tolerance
        newton_tol,
        // x = delta * exp(t/2)
        [=](T t) { return delta * std::exp(t); },
        // f(x)
        [=](T x) { return eps - gamma_q(beta, x); },
        // f'(x) = df /dx * dx / dt
        [=](T x) { return std::exp(beta * std::log(x) - x) * scale_u; });

    // std::cout << "# t_upper = " << t_upper << std::endl;

    // ----- Spacing of discritization (eq. (15) of [Beylkin2010])
    const T q = std::log(T(3)) - beta * std::log(std::cos(T(1)));
    auto h0   = T(2) * arma::Datum<T>::pi / (q - log_eps);
    //
    // Make sub-optimal approximation with exponential sum
    //
    const auto n0 = static_cast<size_type>(std::ceil((t_upper - t_lower) / h0));

    result_type ret(n0);
    const auto pre = h0 * scale_l; // h0 / tgamma(beta);
    for (size_type i = 0; i < n0; ++i)
    {
        ret.exponent(i) = std::exp(t_lower + h0 * i);
        ret.weight(i)   = pre * std::exp(beta * (t_lower + h0 * i));
    }

    ret.truncate(eps);
    // ret.remove_small_terms(eps / ret.size());

    return ret;
}

} // namespace: expsum

#endif /* EXPSUM_APPROX_POW_HPP */
