#include <iomanip>
#include <iostream>

#include "expsum/approx_pow.hpp"
#include "expsum/balanced_truncation.hpp"

using size_type     = arma::uword;
using real_type     = double;
using real_vector   = arma::Col<real_type>;
using function_type = expsum::exponential_sum<real_type, real_type>;

void print_result(real_type beta, const real_vector& grid,
                  const function_type& ret)
{
    // Sampling
    std::cout << "\n\n# r, exact, approx, abs. err., rel. err.\n";
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
}

void run_approx_pow(real_type beta, real_type delta, real_type eps,
                    size_type n_samples = 1000)
{
    expsum::pow_kernel<real_type> pow_kern;
    pow_kern.compute(beta, delta, eps);

    function_type ret(pow_kern.exponents(), pow_kern.weights());
    std::cout << "# no. of terms and (exponents, weights)\n" << ret << '\n';
    const auto grid = arma::logspace<real_vector>(std::log10(delta) - 1,
                                                  real_type(1), n_samples);

    print_result(beta, grid, ret);

    expsum::balanced_truncation<real_type> truncation;
    truncation.run(ret.exponent(), ret.weight(), eps);

    function_type ret_trunc(truncation.exponents(), truncation.weights());
    std::cout << "\n\n# After reduction\n" << ret_trunc << '\n';
    print_result(beta, grid, ret_trunc);
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    const auto delta = 1.0e-8;
    const auto eps   = 1.0e-10;

    std::cout << "#\n"
                 "# Approximation of r^(-1) by exponential sum\n"
                 "#\n"
              << std::endl;
    run_approx_pow(1.0, delta, eps);
    std::cout << "\n\n#\n"
                 "# Approximation of r^(-1/2) by exponential sum\n"
                 "#\n"
              << std::endl;
    run_approx_pow(0.5, delta, eps);
    std::cout << "\n\n#\n"
                 "# Approximation of r^(-2) by exponential sum\n"
                 "#\n"
              << std::endl;
    run_approx_pow(2.0, delta, eps);

    return 0;
}
