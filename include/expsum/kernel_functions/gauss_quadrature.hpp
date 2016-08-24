#ifndef EXPSUM_KERNEL_FUNCTIONS_GAUSS_QUADRATURE_HPP
#define EXPSUM_KERNEL_FUNCTIONS_GAUSS_QUADRATURE_HPP

#include <cassert>
#include <cmath>

#include <tuple>
#include <vector>

#include "expsum/constants.hpp"

namespace expsum
{
//
// Container for holding nodes and weights of quadrature rule
//
template <typename T, typename RuleGen>
class quadrature_rule
{
public:
    using value_type     = T;
    using container_type = std::vector<value_type>;
    using size_type      = typename container_type::size_type;

    using quadrature_rule_generator = RuleGen;

private:
    container_type node_;   // nodes
    container_type weight_; // weights

public:
    quadrature_rule()                       = default;
    quadrature_rule(const quadrature_rule&) = default;
    quadrature_rule(quadrature_rule&&)      = default;
    ~quadrature_rule()                      = default;

    quadrature_rule& operator=(const quadrature_rule&) = default;
    quadrature_rule& operator=(quadrature_rule&&) = default;

    //
    // Construst `n`-points quadrature rule
    //
    // @n number of integration points
    //
    template <typename... ArgsT>
    explicit quadrature_rule(size_type n, ArgsT&&... args)
    {
        compute(n, std::forward<ArgsT>(args)...);
    }

    //
    // @return number of integration points.
    //
    size_type size() const
    {
        return node_.size();
    }
    //
    // @return const reference to the vector of integration points.
    //
    const container_type& x() const
    {
        return node_;
    }

    //
    // Get a node
    //
    // @i  index of integration point. `i < size()` required.
    // @return `i`-th integration point.
    //
    value_type x(size_type i) const
    {
        assert(i < size());
        return node_[i];
    }
    //
    // @return const reference to the vector of integration weights.
    //
    const container_type& w() const
    {
        return weight_;
    }

    //
    // Get a weight
    //
    // @i  index of integration point. `i < size()` required.
    // @return `i`-th integration weight.
    //
    value_type w(size_type i) const
    {
        assert(i < size());
        return weight_[i];
    }
    //
    // Compute nodes and weights of the @c n point Gauss-Legendre
    // quadrature rule. Nodes are located in the interval [-1,1].
    //
    // @n  number of abscissas. `n >= 2` is required.
    //
    template <typename... ArgsT>
    void compute(size_type n, ArgsT&&... args)
    {
        resize(n);
        quadrature_rule_generator::run(node_, weight_,
                                       std::forward<ArgsT>(args)...);
    }

private:
    void resize(size_type n)
    {
        node_.resize(n);
        weight_.resize(n);
    }
};

//
// Generator of Gaussian quadrature with Chebyshev polynomial of the first kind
//
template <typename T>
struct gen_gauss_chebyshev_kind_1
{
    using container_type = std::vector<T>;
    static void run(container_type& node, container_type& weight)
    {
        auto n = node.size(); // number of nodes
        if (n == 0)
        {
            return;
        }

        auto x               = std::begin(node);
        const auto pi_per_2n = constant<T>::pi / T(2 * n);
        for (decltype(n) k = n; k > 0; --k)
        {
            *x = std::cos(T(2 * k - 1) * pi_per_2n);
            ++x;
        }

        std::fill_n(std::begin(weight), n, constant<T>::pi / T(n));

        return;
    }
};

//
// Generator of Gaussian quadrature with Chebyshev polynomial of the second kind
//
template <typename T>
struct gen_gauss_chebyshev_kind_2
{
    using container_type = std::vector<T>;
    static void run(container_type& node, container_type& weight)
    {

        auto n = node.size(); // number of nodes
        if (n == 0)
        {
            return;
        }

        auto x                = std::begin(node);
        auto w                = std::begin(weight);
        const auto pi_per_np1 = constant<T>::pi / T(n + 1);
        for (decltype(n) k = n; k > 0; --k)
        {
            const auto cs = std::cos(T(k) * pi_per_np1);
            const auto sn = std::sin(T(k) * pi_per_np1);
            *x            = cs;
            *w            = pi_per_np1 * sn * sn;
            ++x;
            ++w;
        }

        return;
    }
};

//
// Generator of Gauss-Legendre rule
//
// Quadrature nodes and weights are computed using the Newton's method with the
// three-term recurrence relation of Legendre polynomials. The required
// operations to find all nodes is O(n^2).
//
// NOTE: The Hale's algorithm is more efficient (O(n)) and suitable for
// computing large number of nodes. We do not implement the method here because
// only the quadrature rule with small number of nodes is required for present
// application.
//
template <typename T>
struct gen_gauss_legendre
{
    using container_type = std::vector<T>;
    using size_type      = typename container_type::size_type;
    static void run(container_type& node, container_type& weight)
    {
        const size_type n = node.size(); // number of nodes

        if (n == 0)
        {
            return;
        }

        if (n == 1)
        {
            node[0]   = T();
            weight[0] = T(2);
            return;
        }

        //
        // Compute nodes only at x >= 0, because the nodes are symmetric in the
        // interval
        //
        const size_type mid = n / 2;
        for (size_type i = n; i > mid; --i)
        {
            std::tie(node[i - 1], weight[i - 1]) =
                root_and_weight(n, n - i + 1);
        }

        //
        // Reflect for negative nodes
        //
        for (size_type i = 0; i <= mid; ++i)
        {
            node[i]   = -node[n - i - 1];
            weight[i] = weight[n - i - 1];
        }
    }

    //
    // Evaluate Legendre polynomial P_n(x) and its derivative P_n'(x) at given
    // point
    //
    static std::tuple<T, T> eval_legendre_p(size_type n, T x)
    {
        auto pm2  = T(1); // P_0(x) = 1
        auto pm1  = x;    // P_1(x) = x
        auto ppm2 = T();  // P_0'(x) = 0
        auto ppm1 = T(1); // P_1'(x) = 1
        T p, pp;

        for (size_type k = 1; k < n; ++k)
        {
            // (k+1) P_{k+1}(x) = (2k+1) x P_{k}(x) - k P_{k-1}(x)
            p = ((2 * k + 1) * pm1 * x - k * pm2) / (k + 1);
            // (k+1) P'_{k+1}(x)
            //      = (2k+1) (x P'_{k}(x) + P_{k}(x)) - k P'_{k-1}(x)
            pp = ((2 * k + 1) * (ppm1 * x + pm1) - k * ppm2) / (k + 1);

            pm2 = pm1;
            pm1 = p;

            ppm2 = ppm1;
            ppm1 = pp;
        }

        return {p, pp};
    }
    //
    // Find the k-th root of Legendre polynomial of degree n on x > 0 with its
    // derivative.
    //
    static std::tuple<T, T> root_and_weight(size_type n, size_type k)
    {
        constexpr const size_type max_iter = 10;
        //
        // Asymptotic formula of the roots of Legendre polynomial (only valid
        // for x > 0)
        //
        const auto theta = constant<T>::pi * T(4 * k - 1) / T(4 * n + 2);
        const auto cs    = std::cos(theta);
        const auto sn    = std::sin(theta);
        const auto n2    = n * n;
        // initial guess
        auto x = cs * (T(1) - T(n - 1) / T(8 * n * n2) -
                       (T(39) - T(28) / (sn * sn)) / T(384 * n2 * n2));

        // Compute k-th positive node  by Newton's method
        size_type count = max_iter;
        T p, pp; // P_n(x) and P_n'(x), respectively

        while (--count)
        {
            std::tie(p, pp) = eval_legendre_p(n, x);
            // Newton step
            auto dx = -p / pp;
            x += dx;
        }

        // Once more, for updating derivative
        std::tie(p, pp) = eval_legendre_p(n, x);
        // Compute the corresponding weight
        const auto w = T(2) / ((T(1) - x) * (T(1) + x) * pp * pp);

        return {x, w};
    }
};

//
// Generator of Gauss-Jacobi rule
//
// Quadrature nodes and weights are computed using the Newton's method with the
// three-term recurrence relation of Jacobi polynomials. The required
// operations to find all nodes is O(n^2).
//
// NOTE: The Hale's algorithm is more efficient (O(n)) and suitable for
// computing large number of nodes. We do not implement the method here because
// only the quadrature rule with small number of nodes is required for present
// application.
//
template <typename T>
struct gen_gauss_jacobi
{
    using container_type = std::vector<T>;
    using size_type      = typename container_type::size_type;

    static void run(container_type& node, container_type& weight, T a, T b)
    {
        assert(a > T(-1) && b > T(-1));

        const size_type n = node.size(); // number of nodes

        if (n == 0)
        {
            return;
        }

        if (n == 1)
        {
            node[0] = (b - a) / (a + b + 2);
            // 2^(a+b+1) * beta(a+1, b+1);
            // Assume tgamma(a+b+2) doesn't overflow
            weight[0] = std::pow(T(2), (a + b + 1)) * std::tgamma(a + 1) *
                        std::tgamma(b + 1) / std::tgamma(a + b + 2);
            return;
        }
        //
        // Compute nodes on x < 0 and corresponding weights
        //
        const size_type mid1 = n / 2;
        for (size_type i = 0; i < mid1; ++i)
        {
            std::tie(node[i], weight[i]) = root_and_weight(n, i + 1, b, a);
            node[i] = -node[i];
        }
        //
        // Compute nodes on x > 0 and corresponding weights
        //
        for (size_type i = mid1; i < n; ++i)
        {
            std::tie(node[i], weight[i]) = root_and_weight(n, n - i, a, b);
        }
        //
        // Scaling weights
        //
        const T scale =
            std::pow(T(2), a + b + 1) *
            std::exp(std::lgamma(a + n + 1) + std::lgamma(n + b + 1) -
                     std::lgamma(a + b + n + 1) - std::lgamma(T(n + 1)));
        for (auto& w : weight)
        {
            w *= scale;
        }
    }

    // Evaluate Jacobi polynomial and its derivative using recurrence relation
    static std::tuple<T, T> eval_jacobi(size_type n, T a, T b, T x)
    {
        T p    = (a - b + (a + b + 2) * x) / T(2);
        T pm1  = T(1);
        T pp   = (a + b + 2) / T(2);
        T ppm1 = T();

        T c[4];
        for (size_type k = 1; k < n; ++k)
        {
            c[0] = 2 * (k + 1) * (a + b + k + 1) * (a + b + 2 * k);
            c[1] = (a + b + 2 * k + 1) * (a + b) * (a - b);
            c[2] = (a + b + 2 * k) * (a + b + 2 * k + 1) * (a + b + 2 * k + 2);
            c[3] = 2 * (a + k) * (b + k) * (a + b + 2 * k + 2);

            auto pa1 = ((c[1] + c[2] * x) * p - c[3] * pm1) / c[0];
            auto ppa1 =
                ((c[1] + c[2] * x) * pp + c[2] * p - c[3] * ppm1) / c[0];

            pm1  = p;
            p    = pa1;
            ppm1 = pp;
            pp   = ppa1;
        }

        return {p, pp};
    }

    static std::tuple<T, T> root_and_weight(size_type n, size_type k, T a, T b)
    {
        constexpr const size_type max_iter = 20;
        constexpr const auto eps           = std::numeric_limits<T>::epsilon();
        //
        // Asymptotic formula of the roots of Legendre polynomial (only valid
        // for x > 0)
        //
        const auto rho = a + b + 2 * n + 1;
        const auto phi = constant<T>::pi * (2 * k + a - T(0.5)) / rho;
        const auto tn  = std::tan(phi / 2);
        const auto a1  = (T(0.5) - a) * (T(0.5) + a) / tn;
        const auto b1  = (T(0.5) - b) * (T(0.5) + b) * tn;
        auto x = std::cos(phi + (a1 - b1) / (rho * rho)); // initial guess

        // Compute k-th positive node by Newton's method
        size_type count = max_iter;
        T p, pp; // P_n(x) and P_n'(x), respectively

        while (--count)
        {
            std::tie(p, pp) = eval_jacobi(n, a, b, x);
            // Newton step
            auto dx = -p / pp;
            x += dx;
            if (std::abs(dx) < eps * std::abs(x))
            {
                break;
            }
        }

        // Once more, for updating derivative
        std::tie(p, pp) = eval_jacobi(n, a, b, x);
        // Compute the corresponding weight
        const auto w = T(1) / ((T(1) - x) * (T(1) + x) * pp * pp);

        return {x, w};
    }
};

//------------------------------------------------------------------------------
// Aliases that defines quadrature rule
//------------------------------------------------------------------------------

template <typename T>
using gauss_legendre_rule = quadrature_rule<T, gen_gauss_legendre<T>>;

template <typename T>
using gauss_jacobi_rule = quadrature_rule<T, gen_gauss_jacobi<T>>;

} // namespace: expsum

#endif /* EXPSUM_KERNEL_FUNCTIONS_GAUSS_QUADRATURE_HPP */
