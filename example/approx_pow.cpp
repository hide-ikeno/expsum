#include <iostream>
#include <iomanip>

#include "expsum/approx_pow.hpp"

using size_type = arma::uword;
using real_type = double;
using real_vector = arma::Col<real_type>;

real_type eval_expsum(const real_vector& a, const real_vector& w, real_type x)
{
    return arma::sum(arma::exp(-x * a) % w);
}

void run_approx_pow(real_type beta, real_type delta, real_type eps,
                    size_type n_samples = 1000)
{
    const auto ret = expsum::approx_pow(beta, delta, eps);
    const auto& a = std::get<0>(ret);
    const auto& w = std::get<1>(ret);
    std::cout << "# Approximation of r^(" << beta << ") by exponential sum\n"
              << "# exponents, weights (" << a.size() << " terms)\n";

    for (size_type i = 0; i < a.size(); ++i)
    {
        std::cout << ' ' << std::setw(24) << a(i)          // exponent
                  << ' ' << std::setw(24) << w(i) << '\n'; // weight
    }

    // Sampling
    std::cout << "\n\n# r, exact, approx, abs. err., rel. err.\n";
    auto grid =
        arma::logspace<real_vector>(std::log10(delta) - 1, 1.0, n_samples);
    for (auto x : grid)
    {
        const auto approx  = eval_expsum(a, w, x);
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
