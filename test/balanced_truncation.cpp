#include <complex>
#include <iomanip>
#include <iostream>

#include "expsum/balanced_truncation.hpp"

using size_type = arma::uword;

//
// Test for coneig_quasi_cauchy
//
template <typename T>
void test_balanced_truncation(size_type n)
{
    using real_type   = typename arma::get_pod_type<T>::result;
    using vector_type = arma::Col<T>;

    const auto delta = std::sqrt(n) * arma::Datum<real_type>::eps;

    vector_type a(n, arma::fill::randu);
    vector_type w(n, arma::fill::randu);

    expsum::balanced_truncation<T> truncation;
    truncation.run(a, w, delta);
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    arma::arma_rng::set_seed_random();

    const size_type n       = 200;
    const size_type n_trial = 20;

    std::cout << "# real quasi-cauchy matrix of dimension " << n << '\n';

    for (size_type i = 0; i < n_trial; ++i)
    {
        std::cout << "--- trial " << i + 1 << '\n';
        test_balanced_truncation<double>(n);
    }

    std::cout << "# complex quasi-cauchy matrix of dimension " << n << '\n';
    for (size_type i = 0; i < n_trial; ++i)
    {
        std::cout << "--- trial " << i + 1 << '\n';
        test_balanced_truncation<std::complex<double>>(n);
    }

    return 0;
}
