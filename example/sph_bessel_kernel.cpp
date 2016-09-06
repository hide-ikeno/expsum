#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "expsum/exponential_sum.hpp"
#include "expsum/kernel_functions/sph_bessel_kernel.hpp"
// #include "expsum/fitting/fast_esprit.hpp"
#include <boost/math/special_functions/bessel.hpp>

using size_type   = arma::uword;
using expsum_type = expsum::exponential_sum<double, std::complex<double>>;

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

// expsum_type sph_bessel_kernel(int l, double xmax, size_type N, size_type L,
//                               size_type M, double eps)
// {
//     using esprit_type = expsum::fast_esprit<double>;
//     using vector_type = arma::Col<double>;

//     vector_type exact(N);
//     make_sample([=](double x) { return boost::math::sph_bessel(l, x); }, 0.0,
//                 xmax, exact);
//     auto delta = xmax / (N - 1);

//     esprit_type esprit(N, L, M);

//     esprit.fit(exact, 0.0, delta, eps);
//     expsum_type ret(esprit.exponents(), esprit.weights());

//     ret.remove_small_terms(eps / 100.0);

//     return ret;
// }

void sph_bessel_kernel(int l, double xmax, double eps)
{
    using vector_type = arma::Col<double>;
    expsum::sph_bessel_kernel<double> body;
    body.compute(l, xmax, eps);

    return;
}

void sph_bessel_kernel_error(int l, double xmax, size_type n_samples,
                             const expsum_type& ret)
{
    using vector_type = arma::Col<double>;

    vector_type x(arma::linspace<vector_type>(0.0, xmax, n_samples));
    vector_type exact(n_samples), approx(n_samples);

    for (size_type i = 0; i < n_samples; ++i)
    {
        exact(i)  = boost::math::sph_bessel(l, x(i));
        approx(i) = ret(x(i));
    }

    vector_type abserr(arma::abs(exact - approx));

    size_type imax = abserr.index_max();

    std::cout << "\n  abs. error in interval [0," << xmax << "]\n"
              << "    maximum : " << abserr(imax) << '\n'
              << "    averaged: " << arma::sum(abserr) / n_samples << '\n';
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    const int lmax = 5;

    // size_type N = 1024 * 1024; // # of sampling points
    // size_type L = N / 2;       // window length
    // size_type M = 1000;        // max # of terms
    double xmax = 1.0e5;
    double eps  = 1.0e-12;

    expsum_type ret;

    std::cout
        << "# Approximation of spherical Bessel function by exponential sum\n";

    // for (int l = 0; l <= lmax; ++l)
    // {
    //     std::cout << "\n# --- order " << l << '\n';
    //     ret = sph_bessel_kernel(l, xmax, N, L, M, eps);
    //     ret.print(std::cout);

    //     sph_bessel_kernel_error(l, 10.0, 10001, ret);
    //     sph_bessel_kernel_error(l, 1.0e8, 1000001, ret);
    // }

    for (int l = 0; l <= lmax; ++l)
    {
        std::cout << "\n# --- order " << l << '\n';
        sph_bessel_kernel(l, xmax, eps);
    }

    return 0;
}
