#include <iomanip>
#include <iostream>

#include "expsum/approx_pow.hpp"
#include "expsum/reduction.hpp"

using size_type   = arma::uword;
using real_type   = double;
using real_vector = arma::Col<real_type>;

void run_approx_pow(real_type beta, real_type delta, real_type eps,
                    size_type n_samples = 1000)
{
    const auto ret = expsum::approx_pow(beta, delta, eps);
    std::cout << "# Approximation of r^(" << beta << ") by exponential sum\n"
              << "# no. of terms and (exponents, weights)\n"
              << ret << '\n';
    // Sampling
    std::cout << "\n\n# r, exact, approx, abs. err., rel. err.\n";
    auto grid =
        arma::logspace<real_vector>(std::log10(delta) - 1, 1.0, n_samples);
    for (auto x : grid)
    {
        const auto approx  = ret(x);
        const auto exact   = std::pow(x, -beta);
        const auto abs_err = std::abs(exact - approx);
        const auto rel_err = abs_err / exact;

        std::cout << ' ' << std::setw(24) << x       // r
                  << ' ' << std::setw(24) << exact   // r^(-beta)
                  << ' ' << std::setw(24) << approx  // exponential sum
                  << ' ' << std::setw(24) << abs_err // absolute error
                  << ' ' << std::setw(24) << rel_err // relative error
                  << '\n';
    }
    std::cout << "\n\n# After reduction\n";
    expsum::reduction_body<real_type> body;
    body.run(ret.exponent(), ret.weight(), eps);
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    const auto delta = 1.0e-9;
    const auto eps   = 1.0e-10;

    run_approx_pow(0.5, delta, eps);

    return 0;
}
