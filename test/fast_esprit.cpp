// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-

#include <cassert>
#include <cmath>

#include <iomanip>
#include <iostream>
#include <vector>

#include "expsum/fast_esprit.hpp"
#include <boost/math/special_functions/bessel.hpp>

using size_type = arma::uword;
// using Complex   = std::complex<double>;

//------------------------------------------------------------------------------
// Test functors
//------------------------------------------------------------------------------

struct BesselJ0
{
    double operator()(double x) const
    {
        return boost::math::cyl_bessel_j(0, x);
    }
};

struct rinv
{
    double operator()(double x) const
    {
        return 1.0 / x;
    }
};

template <typename F, typename Vec>
void make_sample(F f, double xmin, double xmax, Vec& result)
{
    auto np = result.n_elem;
    auto h  = (xmax - xmin) / (np - 1);
    for (size_type n = 0; n < np; ++n)
    {
        result(n) = f(xmin + n * h);
    }

    return;
}

template <typename F>
void test_fast_esprit(F fn, double xmin, double xmax, size_type N, size_type L,
                      size_type M, double eps)
{
    using value_type  = decltype(fn(xmin));
    using vector_type = arma::Col<value_type>;
    using esprit_type = expsum::fast_esprit<value_type>;
    using real_type   = typename esprit_type::real_type;

    vector_type exact(N);
    make_sample(fn, xmin, xmax, exact);
    auto delta = (xmax - xmin) / (N - 1);

    // ESPRIT esprit(N, std::min<size_type>(100, n / 2));
    esprit_type esprit(N, L, M);

    esprit.fit(exact, xmin, delta, eps);

    auto nterms = esprit.exponents().n_elem;
    std::cout << "# " << nterms << " terms found\n"
              << "# exponent, weight\n";

    for (size_type i = 0; i < nterms; ++i)
    {
        std::cout << esprit.exponents()(i) << '\t' << esprit.weights()(i)
                  << '\n';
    }

    std::cout << "# x, approx, exact, abserr, relerr\n";

    for (size_type i = 0; i < N; ++i)
    {
        auto x      = xmin + i * delta;
        auto approx = esprit.eval_at(x);
        auto abserr = std::abs(approx - exact(i));
        auto relerr =
            (abserr == real_type()) ? real_type() : abserr / std::abs(exact(i));

        std::cout << x << '\t' << approx << '\t' << exact(i) << '\t' << abserr
                  << '\t' << relerr << '\n';
    }
}

// void test_frequency_estimation(const numeric::ExponentialSum<Complex>& orig,
//                                size_type N, size_type L, size_type M, double
//                                eps)
// {
//     typedef arma::Col<Complex>          Vector;
//     typedef numeric::FastESPRIT<Complex> ESPRIT;
//     typedef numeric::ESPRITExpSum<Complex> ParamGenerator;
//     typedef typename ESPRIT::Real        Real;

//     Vector exact(N);
//     make_sample([&](double x) { return orig.evalAt(x); },
//                0.0, static_cast<double>(N), exact);
//     auto delta = 1.0;

//     // ESPRIT esprit(N, std::min<size_type>(100, n / 2));
//     ESPRIT esprit(N, L, M);

//     auto expsum = esprit.compute(exact, eps, ParamGenerator(0.0, delta));

//     std::cout << "# exponents and weights (original)\n";

//     for (size_type i = 0; i < orig.size(); ++i)
//     {
//         std::cout << orig.exponents(i) << '\t'
//                   << orig.weights(i) << '\n';
//     }

//     std::cout << "# exponents and weights (found)\n";
//     for (size_type i = 0; i < expsum.size(); ++i)
//     {
//         std::cout << expsum.exponents(i) << '\t'
//                   << expsum.weights(i) << '\n';
//     }
// }

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    std::cout << "# Bessel J0(x):  x in [0, 1000]" << std::endl;
    size_type N = 1024;  // # of sampling points
    size_type L = N / 2; // window length
    size_type M = 100;   // max # of terms
    double xmin = 0.0;
    double xmax = 1000.0;
    double eps  = 1.0e-10;
    test_fast_esprit(BesselJ0(), xmin, xmax, N, L, M, eps);

    std::cout << "# 1/r: r in [1, 10^{6}]" << std::endl;
    N    = (1 << 12);
    L    = N / 2;
    M    = 100;
    xmin = 1.0;
    xmax = 1.0e+6;
    eps  = 1.0e-8;
    test_fast_esprit(rinv(), xmin, xmax, N, L, M, eps);

    // std::cout << "# Exponential sum recovery test" << std::endl;
    // const auto pi = arma::datum::pi;
    // numeric::ExponentialSum<Complex> orig(5);

    // orig.exponents(0) = Complex(0.0,     0.0);
    // orig.exponents(1) = Complex(0.0,  pi / 4);
    // orig.exponents(2) = Complex(0.0, -pi / 4);
    // orig.exponents(3) = Complex(0.0,  pi / 2);
    // orig.exponents(4) = Complex(0.0, -pi / 2);

    // orig.weights(0) = Complex(34.0,  0.0);
    // orig.weights(1) = Complex(300.0, 0.0);
    // orig.weights(2) = Complex(300.0, 0.0);
    // orig.weights(3) = Complex(1.0,   0.0);
    // orig.weights(4) = Complex(1.0,   0.0);

    // N = 1024;
    // L = N / 2;
    // M = 20;
    // eps = 1.0e-10;
    // test_frequency_estimation(orig, N, L, M, eps);

    return 0;
}
