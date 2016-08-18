#include <complex>
#include <iomanip>
#include <iostream>

#include "expsum/balanced_truncation.hpp"
#include "expsum/exponential_sum.hpp"

using size_type = arma::uword;

//
// Test balanced truncation for exponential sum
//

template <typename T>
void print_funcs(const expsum::exponential_sum<T, T>& orig,
                 const expsum::exponential_sum<T, T>& truncated)
{
    using real_type        = typename arma::get_pod_type<T>::result;
    using real_vector_type = arma::Col<real_type>;

    const size_type n = 100001;
    // const auto xmin   = real_type(-5);
    // const auto xmax   = real_type(1);
    // const auto grid   = arma::logspace<real_vector_type>(xmin, xmax, n);

    const auto xmin = real_type();
    const auto xmax = real_type(10);
    const auto grid = arma::linspace<real_vector_type>(xmin, xmax, n);
    real_vector_type abserr(n);
    real_vector_type relerr(n);

    for (size_type i = 0; i < n; ++i)
    {
        const auto x  = grid(i);
        const auto f1 = orig(x);
        const auto f2 = truncated(x);
        abserr(i)     = std::abs(f1 - f2);
        relerr(i)     = (f1 != T()) ? abserr(i) / std::abs(f1) : abserr(i);
    }

    std::cout << "    size before truncation = " << orig.size() << '\n'
              << "    size after truncation  = " << truncated.size() << '\n'
              << "    max abs. error = " << arma::max(abserr) << '\n'
              << "    max rel. error = " << arma::max(relerr) << std::endl;
}

template <typename T>
void test_balanced_truncation(size_type n)
{
    using real_type   = typename arma::get_pod_type<T>::result;
    using vector_type = arma::Col<T>;
    using result_type = expsum::exponential_sum<T, T>;

    const auto delta = std::sqrt(n) * arma::Datum<real_type>::eps;

    vector_type a(n, arma::fill::randu);
    vector_type w(n, arma::fill::randu);

    result_type orig(a, w);

    expsum::balanced_truncation<T> truncation;
    truncation.run(a, w, delta);

    result_type truncated(truncation.exponents(), truncation.weights());

    print_funcs(orig, truncated);
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    arma::arma_rng::set_seed_random();

    const size_type n       = 200;
    const size_type n_trial = 5;

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
