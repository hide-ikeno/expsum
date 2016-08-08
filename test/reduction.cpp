#include <complex>
#include <iomanip>
#include <iostream>

#include "expsum/reduction.hpp"

using size_type = arma::uword;

//
// Make random vector of given size with elemnts x[i] whose absolute value is
// less than unity.
//
template <typename T>
arma::Col<T> make_random_vector(size_type n)
{
    if (arma::is_complex<T>::value)
    {
        static const auto scale = 1 / std::sqrt(2);
        arma::Col<T> ret(2 * n);
        ret.head(n).randu();
        ret.head(n) *= scale;
        ret.tail(n) = arma::conj(ret.head(n));
        return ret;
    }
    else
    {
        return arma::Col<T>(n, arma::fill::randu);
    }
}

//
// Test for coneig_quasi_cauchy
//
template <typename T>
void test_reduction(size_type n)
{
    using real_type   = typename arma::get_pod_type<T>::result;
    using vector_type = arma::Col<T>;
    using matrix_type = arma::Mat<T>;
    using real_vector = arma::Col<real_type>;

    const auto delta = std::sqrt(n) * arma::Datum<real_type>::eps;

    // vector_type a(n, arma::fill::randu);
    // vector_type w(n, arma::fill::randu);
    vector_type a(make_random_vector<T>(n));
    vector_type w(make_random_vector<T>(n));

    std::cout << "# quasi-Cauchy matrix of dimension " << a.size() << '\n';
    expsum::reduction_body<T> reduction;
    real_vector sigma;
    matrix_type X;
    reduction.run(a, w, delta);

    // const auto n_coneigs = sigma.size();
    // vector_type work(n);
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    arma::arma_rng::set_seed_random();
    test_reduction<double>(200);
    test_reduction<std::complex<double>>(100);

    return 0;
}