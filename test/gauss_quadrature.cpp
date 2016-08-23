#include <iomanip>
#include <iostream>
#include <numeric>

#include "expsum/gauss_quadrature.hpp"

void make_gauss_legendre(std::size_t n)
{
    expsum::gauss_legendre_rule<double> gauleg(n);

    auto w_sum =
        std::accumulate(std::begin(gauleg.w()), std::end(gauleg.w()), 0.0);

    std::cout << "#\n# Gauss-Legendre rule of degree " << n << "\n#\n";

    for (std::size_t i = 0; i < gauleg.size(); ++i)
    {
        std::cout << ' ' << std::setw(24) << gauleg.x(i)          // node
                  << ' ' << std::setw(24) << gauleg.w(i) << '\n'; // weight
    }
    std::cout << "# (sum of weight = " << w_sum << ')' << std::endl;
}

void make_gauss_jacobi(std::size_t n, double a, double b)
{
    expsum::gauss_jacobi_rule<double> gaujac(n, a, b);

    auto w_sum =
        std::accumulate(std::begin(gaujac.w()), std::end(gaujac.w()), 0.0);

    std::cout << "#\n# Gauss-Jacobi rule (n = " << n << ", alpha = " << a
              << ", beta = " << b << ")\n#\n";

    for (std::size_t i = 0; i < gaujac.size(); ++i)
    {
        const auto fjac =
            expsum::gen_gauss_jacobi<double>::eval_jacobi(n, a, b, gaujac.x(i));
        std::cout << ' ' << std::setw(24) << gaujac.x(i) // node
                  << ' ' << std::setw(24) << gaujac.w(i) // weight
                  << ' ' << std::setw(24) << std::get<0>(fjac) << '\n';
    }
    std::cout << "# (sum of weight = " << w_sum << ')' << std::endl;
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    for (std::size_t n = 0; n <= 20; ++n)
    {
        make_gauss_legendre(n);
    }

    for (std::size_t n = 0; n <= 20; ++n)
    {
        make_gauss_jacobi(n, 0.5, 1.5);
    }

    return 0;
}
