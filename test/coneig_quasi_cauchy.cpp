#include <complex>
#include <iomanip>
#include <iostream>

#include "expsum/coneig_quasi_cauchy.hpp"

using size_type = arma::uword;

//
// Make random vector of given size with elemnts x[i] whose absolute value is
// less than unity.
//
template <typename T>
arma::Col<T> make_random_vector(size_type n)
{
    static const auto scale = 1 / std::sqrt(2);
    arma::Col<T> ret(n, arma::fill::randu);
    if (arma::is_complex<T>::value)
    {
        ret *= scale;
    }
    return ret;
}

//
// Test for coneig_quasi_cauchy
//
template <typename T>
void test_coneig_quasi_cauchy(size_type n)
{
    using real_type   = typename arma::get_pod_type<T>::result;
    using vector_type = arma::Col<T>;
    using matrix_type = arma::Mat<T>;
    using real_vector = arma::Col<real_type>;

    const auto delta = std::sqrt(n) * arma::Datum<real_type>::eps;

    vector_type alpha = make_random_vector<T>(n);
    vector_type gamma = make_random_vector<T>(n);

    vector_type a = arma::sqrt(alpha) / gamma;
    vector_type b = arma::sqrt(arma::conj(alpha));
    vector_type x = T(1) / gamma;
    vector_type y = arma::conj(-gamma);

    matrix_type C(n, n);
    for (size_type j = 0; j < n; ++j)
    {
        for (size_type i = 0; i < n; ++i)
        {
            C(i, j) = a(i) * b(j) / (x(i) + y(j));
        }
    }

    expsum::coneig_quasi_cauchy<T> solver;
    real_vector sigma;
    matrix_type X;
    solver.compute(a, b, x, y, delta, sigma, X);

    const auto n_coneigs = sigma.size();
    vector_type work(n);

    std::cout << "# quasi-Cauchy matrix of dimension " << n << '\n'
              << n_coneigs << " con-eigenpairs found\n"
              << "\n# lambda, |u|, |C * u - lambda * conj(u)|\n";
    for (size_type k = 0; k < n_coneigs; ++k)
    {
        work = C * X.col(k);
        work -= sigma(k) * arma::conj(X.col(k));
        const auto norm_uk = arma::norm(X.col(k));
        const auto err     = arma::norm(work, 2);
        std::cout << sigma(k) << '\t' << norm_uk << '\t' << err << '\n';
    }
    std::cout << std::endl;
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    arma::arma_rng::set_seed_random();
    // test_coneig_quasi_cauchy<double>(200);
    test_coneig_quasi_cauchy<std::complex<double>>(100);

    return 0;
}
