#ifndef EXPSUM_APPROX_POW_HPP
#define EXPSUM_APPROX_POW_HPP

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <armadillo>

#include "expsum/balanced_truncation.hpp"
#include "expsum/exponential_sum.hpp"
#include "expsum/gamma.hpp"
#include "expsum/modified_prony_truncation.hpp"

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
    static real_type eval_at(real_type x, const vector_type& p,
                             const vector_type& w)
    {
        return arma::sum(w % arma::exp(-x * p));
    }

    void optimal_discritization(real_type t_lower, real_type t_upper,
                                real_type spacing);
};

template <typename T>
void pow_kernel<T>::compute(real_type beta__, real_type delta__,
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

    beta_  = beta__;
    delta_ = delta__;
    eps_   = eps__;

    const auto log_beta  = std::log(beta_);
    const auto log_delta = std::log(delta_);
    const auto log_eps   = std::log(eps_);
    //
    // Parameters required for obtaining sub-optimal expansion
    //
    const auto newton_tol = eps_;
    const auto scale_l    = T(1) / std::tgamma(beta_);
    // Eq. (31) of [Beylkin2010]
    auto t_lower = detail::newton_solve(
        // Initial guess --- Eq. (33) of [Beylkin2010]
        (log_eps + std::lgamma(T(1) + beta_)) / beta_,
        // Tolerance
        newton_tol,
        // transformation: x = exp(t)
        [=](T t) { return std::exp(t); },
        // f(x)  = gamma_p(x) - eps (with x = exp(t / 2))
        [=](T x) { return gamma_p(beta_, x) - eps_; },
        // f'(x) * dx/dt = exp(-x) * pow(x, beta-1)  / tgamma(beta) * x
        [=](T x) { return std::exp(beta_ * std::log(x) - x) * scale_l; });

    // Eq. (32) of [Beylkin2010]
    const auto scale_u = scale_l / delta_;
    auto t_upper       = detail::newton_solve(
        // Initial guess --- Eq. (34) of [Beylkin2010]
        std::log(-log_eps) - log_delta + log_beta + T(0.5),
        // Tolerance
        newton_tol,
        // x = delta * exp(t/2)
        [=](T t) { return delta_ * std::exp(t); },
        // f(x)
        [=](T x) { return eps_ - gamma_q(beta_, x); },
        // f'(x) = df /dx * dx / dt
        [=](T x) { return std::exp(beta_ * std::log(x) - x) * scale_u; });

    // ----- Spacing of discritization (eq. (15) of [Beylkin2010])
    const auto q = std::log(T(3)) - beta_ * std::log(std::cos(T(1)));
    auto h0      = T(2) * arma::Datum<T>::pi / (q - log_eps);

    // Discritization of integral representation of power function
    optimal_discritization(t_lower, t_upper, h0);

    // Truncation of terms with small exponents
    size_type m1 = 0;
    for (; m1 < exponent_.size(); ++m1)
    {
        if (exponent_(m1) >= real_type(1))
        {
            break;
        }
    }
    size_type m2 = exponent_.size() - m1;

    modified_prony_truncation<T> trunc1;
    trunc1.run(exponent_.head(m1), weight_.head(m1), arma::Datum<T>::eps);

    balanced_truncation<T> trunc2;
    trunc2.run(exponent_.tail(m2), weight_.tail(m2), eps_);

    vector_type p(trunc1.size() + trunc2.size());
    vector_type w(trunc1.size() + trunc2.size());

    p.head(trunc1.size()) = arma::real(trunc1.exponents());
    w.head(trunc1.size()) = arma::real(trunc1.weights());
    p.tail(trunc2.size()) = trunc2.exponents();
    w.tail(trunc2.size()) = trunc2.weights();

    exponent_.swap(p);
    weight_.swap(w);

    return;
}

template <typename T>
void pow_kernel<T>::optimal_discritization(real_type t_lower, real_type t_upper,
                                           real_type spacing)
{
    //
    // Halves delta (twice t_lower) to avoid the large error near delta. Scaling
    // factor is chosen empirically.
    //
    t_lower *= real_type(2);
    // ----- Upper bound of the number of terms
    const auto n0 =
        static_cast<size_type>(std::ceil((t_upper - t_lower) / spacing));
    //
    // Make log-space sampling points in [delta, 1]
    //
    const size_type nsamples = 1001;
    vector_type r(
        arma::logspace<vector_type>(std::log10(delta_), real_type(), nsamples));

    size_type n1 = 0;
    size_type n2 = n0;

    const auto scale = T(1) / std::tgamma(beta_);
    while (true)
    {
        const auto n = (n1 + n2) / 2;
        vector_type p(n);
        vector_type w(n);

        const auto h   = (t_upper - t_lower) / real_type(n - 1);
        const auto pre = h * scale;

        for (size_type i = 0; i < n; ++i)
        {
            p(i) = std::exp(t_lower + h * i);
            w(i) = pre * std::exp(beta_ * (t_lower + h * i));
        }

        bool smaller_than_thresh = true;
        for (size_type i = 0; i < r.size(); ++i)
        {
            const auto val = std::pow(r(i), beta_) * eval_at(r(i), p, w);
            if (std::abs(real_type(1) - val) > eps_)
            {
                smaller_than_thresh = false;
                break;
            }
        }

        if (smaller_than_thresh)
        {
            exponent_.swap(p);
            weight_.swap(w);
            if (n2 <= n1 + 1)
            {
                break;
            }
            n2 = n;
        }
        else
        {
            n1 = (n2 == n1 + 1) ? n1 + 1 : n;
        }
    }

    return;
}

// template <typename T>
// exponential_sum<T, T> approx_pow(T beta, T delta, T eps)
// {
//     using size_type   = arma::uword;
//     using vector_type = arma::Col<T>;
//     using result_type = exponential_sum<T, T>;

//     assert(beta > T());
//     assert(T() < delta && delta < T(1));
//     assert(T() < eps && eps < std::exp(T(-1)));

//     const T log_beta  = std::log(beta);
//     const T log_delta = std::log(delta);
//     const T log_eps   = std::log(eps);
//     //
//     // Parameters required for obtaining sub-optimal expansion
//     //
//     const T newton_tol = eps;
//     const T scale_l    = T(1) / std::tgamma(beta);
//     // Eq. (31) of [Beylkin2010]
//     auto t_lower = detail::newton_solve(
//         // Initial guess --- Eq. (33) of [Beylkin2010]
//         (log_eps + std::lgamma(T(1) + beta)) / beta,
//         // Tolerance
//         newton_tol,
//         // transformation: x = exp(t)
//         [=](T t) { return std::exp(t); },
//         // f(x)  = gamma_p(x) - eps (with x = exp(t / 2))
//         [=](T x) { return gamma_p(beta, x) - eps; },
//         // f'(x) * dx/dt = exp(-x) * pow(x, beta-1)  / tgamma(beta) * x
//         [=](T x) { return std::exp(beta * std::log(x) - x) * scale_l; });

//     // std::cout << "# t_lower = "   << t_lower << std::endl;

//     // Eq. (32) of [Beylkin2010]
//     const auto scale_u = scale_l / delta;
//     auto t_upper       = detail::newton_solve(
//         // Initial guess --- Eq. (34) of [Beylkin2010]
//         std::log(-log_eps) - log_delta + log_beta + T(0.5),
//         // Tolerance
//         newton_tol,
//         // x = delta * exp(t/2)
//         [=](T t) { return delta * std::exp(t); },
//         // f(x)
//         [=](T x) { return eps - gamma_q(beta, x); },
//         // f'(x) = df /dx * dx / dt
//         [=](T x) { return std::exp(beta * std::log(x) - x) * scale_u; });

//     // std::cout << "# t_upper = " << t_upper << std::endl;

//     // ----- Spacing of discritization (eq. (15) of [Beylkin2010])
//     const T q = std::log(T(3)) - beta * std::log(std::cos(T(1)));
//     auto h0   = T(2) * arma::Datum<T>::pi / (q - log_eps);
//     //
//     // Make sub-optimal approximation with exponential sum
//     //

//     const auto n0 = static_cast<size_type>(std::ceil((t_upper - t_lower) /
//     h0));
//     const auto pre = h0 * scale_l; // h0 / tgamma(beta);

//     vector_type a(n0);
//     vector_type w(n0);
//     for (size_type i = 0; i < n0; ++i)
//     {
//         a(i) = std::exp(t_lower + h0 * i);
//         w(i) = pre * std::exp(beta * (t_lower + h0 * i));
//     }

//     // reduce terms with exponents (0, 1);
//     size_type n1 = static_cast<size_type>(-std::floor(t_lower) / h0);
//     size_type n2 = n0 - n1;
//     vector_type a1_(a.memptr(), n1, false, true);
//     vector_type w1_(w.memptr(), n1, false, true);

//     modified_prony_truncation<T> truncation1;
//     std::cout << "*** truncation 1" << std::endl;
//     truncation1.run(a1_, w1_, eps);
//     std::cout << "*** done" << std::endl;
//     const size_type m1 = truncation1.size();
//     a.head(m1)         = arma::real(truncation1.exponents());
//     w.head(m1)         = arma::real(truncation1.weights());

//     a.subvec(m1, m1 + n2 - 1) = a.subvec(n1, n0 - 1);
//     w.subvec(m1, m1 + n2 - 1) = w.subvec(n1, n0 - 1);

//     return result_type(a.head(m1 + n2), w.head(m1 + n2));
// }

} // namespace: expsum

#endif /* EXPSUM_APPROX_POW_HPP */
