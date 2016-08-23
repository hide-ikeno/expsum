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

    static size_type get_num_intervals(real_type beta__, real_type delta__,
                                       real_type eps__);
    static size_type get_num_gauss_jacobi(real_type beta__, real_type delta__,
                                          real_type eps__);
    static size_type get_num_gauss_legendre(real_type beta__, real_type delta__,
                                            real_type eps__);

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

    // const auto log_beta  = std::log(beta_);
    // const auto log_delta = std::log(delta_);
    // const auto log_eps   = std::log(eps_);
    //
    auto J = get_num_intervals(beta__, delta__ / 2, eps__);

    return;
}

template <typename T>
typename pow_kernel<T>::size_type
pow_kernel<T>::get_num_intervals(real_type beta__, real_type delta__,
                                 real_type eps__)
{
    //
    // Find minimal J, such that
    //
    // eps / (J + 1) = Gamma(beta, delta * 2^J) / Gamma(beta)
    //
    size_type J1 = 0;
    size_type J2 = 20;

    while (true)
    {
        auto x  = delta__ * std::pow(T(2), J2);
        auto fj = gamma_q(beta__, x);

        std::cout << "J1 = " << J1 << ", J2 = " << J2 << ", Delta(J) = " << fj
                  << '\n';

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
            J2 = J2 + 10;
        }
    }

    return J2;
}
template <typename T>
typename pow_kernel<T>::size_type
pow_kernel<T>::get_num_gauss_jacobi(real_type beta__, real_type delta__,
                                    real_type eps__)
{
    //
    // Find N such that R(N, b) <= eps with
    //
    // R(N, b) = 2^(beta + 2) / Gamma(beta + 1) * exp(cosh(b)-1)
    //         * exp(-2*N*b) / (1 - exp(-2*b))
    //
    // with minimizing R(N,b) w.r.t. b > 0
    //
    size_type n1 = 0;
    size_type n2 = 8;

    auto pre = std::pow(T(2), beta__ + 2) / std::tgamma(beta__ + 1);

    while (true)
    {
        auto resid = 0;

        if (resid <= eps__)
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
            n2 = n2 + 4;
        }
    }
}

} // namespace: expsum

#endif /* EXPSUM_APPROX_POW_HPP */
