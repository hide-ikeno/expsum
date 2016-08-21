#ifndef EXPSUM_APPROX_POW_HPP
#define EXPSUM_APPROX_POW_HPP

#include <cassert>
#include <tuple>

#include <armadillo>

#include "expsum/balanced_truncation.hpp"
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

//
// Reduce terms of exponential sum by modified Prony method.
// This method is for reducing terms with small exponent.
//
template <typename T>
arma::uword modified_prony_reduction(arma::Col<T>& exponents,
                                     arma::Col<T>& weights,
                                     typename arma::get_pod_type<T>::result eps)
{
    using size_type    = arma::uword;
    using real_type    = typename arma::get_pod_type<T>::result;
    using complex_type = std::complex<real_type>;
    using vector_type  = arma::Col<T>;
    using matrix_type  = arma::Mat<T>;

    using complex_vector_type = arma::Col<complex_type>;

    vector_type h(2 * exponents.size());
    vector_type a_pow(exponents);

    h(0) = arma::sum(weights);
    h(1) = -arma::sum(weights % a_pow);

    size_type m         = 1;
    real_type factorial = real_type(1);

    for (; m < exponents.size(); ++m)
    {
        a_pow %= exponents;
        h(2 * m) = arma::sum(weights % a_pow);
        a_pow %= exponents;
        h(2 * m + 1) = -arma::sum(weights % a_pow);
        factorial *= real_type((2 * m) * (2 * m + 1));
        if (std::abs(h(2 * m + 1)) < eps * factorial)
        {
            // Taylor expansion converges with the tolerance eps.
            ++m;
            break;
        }
    }

    //
    // Construct a Hankel matrix from the sequence h, and solve the linear
    // equation, H q = b, with b = -h(m:2m-1).
    //
    matrix_type H(m, m);
    for (size_type k = 0; k < m; ++k)
    {
        H.col(k) = h.subvec(k, k + m - 1);
    }
    vector_type b(-h.tail(m));
    vector_type q(m);
    arma::solve(q, H, b);
    //
    // Find the roots of the Prony polynomial,
    //
    // q(z) = \sum_{k=0}^{m-1} q_k z^{k}.
    //
    // The roots of q(z) can be obtained as the eigenvalues of the companion
    // matrix,
    //
    //     (0  0  ...  0 -p[0]  )
    //     (1  0  ...  0 -p[1]  )
    // C = (0  1  ...  0 -p[2]  )
    //     (.. .. ...  .. ..    )
    //     (0  0  ...  1 -p[m-1])
    //
    matrix_type C(m, m, arma::fill::zeros);
    for (size_type i = 0; i < m - 1; ++i)
    {
        C(i + 1, i) = real_type(1);
    }
    C.col(m - 1) = -q;

    complex_vector_type eigvals(m);
    arma::eig_gen(eigvals, C);

    exponents.head(m) = arma::real(eigvals);
    //
    // Construct Vandermonde matrix from gamma
    //
    matrix_type V(2 * m, m);
    for (size_type i = 0; i < m; ++i)
    {
        // We assume all the eigenvalues are real here
        const auto z = exponents(i);
        V(0, i) = real_type(1);
        for (size_type j = 1; j < V.n_rows; ++j)
        {
            V(j, i) = V(j - 1, i) * z; // z[i]**j
        }
    }
    //
    // Solve overdetermined Vandermonde system,
    //
    // V(0:2m-1,0:m-1) w(0:m-1) = h(0:2m-1)
    //
    // by the least square method.
    //
    vector_type b2 = h.head(2 * m);

    arma::solve(q, V, b2);
    weights.head(m) = q;

    return m;
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
    using vector_type = arma::Col<T>;
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
    const auto pre = h0 * scale_l; // h0 / tgamma(beta);

    vector_type a(n0);
    vector_type w(n0);
    for (size_type i = 0; i < n0; ++i)
    {
        a(i) = std::exp(t_lower + h0 * i);
        w(i) = pre * std::exp(beta * (t_lower + h0 * i));
    }

    // reduce terms with exponents (0, 1);
    size_type n1 = static_cast<size_type>(-std::floor(t_lower) / h0);
    vector_type a1_(a.memptr(), n1, false, true);
    vector_type w1_(w.memptr(), n1, false, true);
    size_type m1 = detail::modified_prony_reduction(a1_, w1_, eps);
    for (size_type i = n1; i < n0; ++i)
    {
        a(m1) = a(i);
        w(m1) = w(i);
        ++m1;
    }

    return result_type(a.head(m1), w.head(m1));
}

} // namespace: expsum

#endif /* EXPSUM_APPROX_POW_HPP */
