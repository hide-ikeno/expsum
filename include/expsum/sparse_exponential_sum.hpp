// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-
#ifndef EXPSUM_SPARSE_EXPONENTIAL_SUM_HPP
#define EXPSUM_SPARSE_EXPONENTIAL_SUM_HPP

#include <armadillo>

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace expsum
{

template <typename T>
struct sparse_exponential_sum
{
    using value_type = T;
    using real_type  = typename arma::Col<T>::pod_type;
    using size_type  = arma::uword;

    using vector_type = arma::Col<value_type>;

private:
    vector_type exponent_;
    vector_type weight_;

public:
    // Default constructor
    sparse_exponential_sum() = default;

    // Create container of size `n`
    sparse_exponential_sum(size_type n) : exponent_(n), weight_(n)
    {
    }

    // Create from the vectors of exponents and weights
    template <typename V1, typename V2>
    sparse_exponential_sum(
        const V1& exponents, const V2& weights,
        typename std::enable_if<(arma::is_arma_type<V1>::value &&
                                 arma::is_arma_type<V2>::value)>::type* = 0)
        : exponent_(exponents), weight_(weights)
    {
        assert(exponent_.size() == weight_.size());
    }

    // Default copy constructor
    sparse_exponential_sum(const sparse_exponential_sum&) = default;

    // Default move constructor
    sparse_exponential_sum(sparse_exponential_sum&&) = default;

    // Default destructor
    ~sparse_exponential_sum() = default;

    // Default copy assignment operator
    sparse_exponential_sum& operator=(const sparse_exponential_sum&) = default;

    // Default move assignment operator
    sparse_exponential_sum& operator=(sparse_exponential_sum&&) = default;

    //
    // Get the number of terms
    //
    size_type size() const
    {
        return exponent_.size();
    }
    //
    // Change the container size
    //
    void resize(size_type n)
    {
        exponent_.set_size(n);
        weight_.set_size(n);
    }
    //
    // Return true if container is empty
    //
    bool empty() const
    {
        return exponent_.empty();
    }
    //
    // Evaluate value at the given point
    //
    value_type operator()(real_type x) const
    {
        return arma::sum(arma::exp(-x * exponent_) % weight_);
    }

    // //
    // // Evaluate exponential sum times power function
    // // ``$g(x)=r^{n}\sum_{i}w_{i}\exp(-z_{i}x)$`` at given point.
    // //
    // // @n power factor
    // // @x coodinate
    // // @return valus of ``$g(x)$``
    // //
    // value_type operator()(int n, real_type x) const
    // {
    //     return std::pow(x, n) * arma::sum(arma::exp(-x * exponent_) %
    //     weight_);
    // }

    //
    // Get const reference to the vector of exponents
    //
    const vector_type& exponent() const
    {
        return exponent_;
    }
    //
    // Get const reference to the vector of weights
    //
    const vector_type& weight() const
    {
        return weight_;
    }
    //
    // Get const reference to the `i`-th exponent
    //
    const value_type& exponent(size_type i) const
    {
        return exponent_(i);
    }
    //
    // Get reference to the `i`-th exponent
    //
    value_type& exponent(size_type i)
    {
        return exponent_(i);
    }
    //
    // Get const reference to the `i`-th weight
    //
    const value_type& weight(size_type i) const
    {
        return weight_(i);
    }
    //
    // Get reference to the `i`-th weight
    //
    value_type& weight(size_type i)
    {
        return weight_(i);
    }
    //
    // Set new exponents and weights
    //
    template <typename V1, typename V2>
    typename std::enable_if<(arma::is_arma_type<V1>::value &&
                             arma::is_arma_type<V2>::value)>::type
    set(const V1& new_exponents, const V2& new_weights)
    {
        assert(new_exponents.is_vec() && new_weights.is_vec() &&
               new_exponents.n_elem == new_weights.n_elem);
        exponent_ = new_exponents;
        weight_   = new_weights;
    }
    //
    // Set new exponents and weights
    //
    void set(vector_type&& new_exponents, vector_type&& new_weights)
    {
        assert(new_exponents.n_elem == new_weights.n_elem);
        exponent_ = std::move(new_exponents);
        weight_   = std::move(new_weights);
    }
    //
    // Set new exponents and weights, but discard terms if the absolute value of
    // weights are smaller than given threshould value.
    //
    template <typename V1, typename V2>
    typename std::enable_if<(arma::is_arma_type<V1>::value &&
                             arma::is_arma_type<V2>::value)>::type
    set_without_small_weight(const V1& new_exponents, const V2& new_weights,
                             real_type tolerance)
    {
        assert(new_exponents.is_vec() && new_weights.is_vec() &&
               new_exponents.n_elem == new_weights.n_elem);
        const auto max_weight = arma::max(arma::abs(new_weights));
        const arma::uvec selected =
            arma::find(arma::abs(new_weights) > max_weight * tolerance);

        exponent_.set_size(selected.n_elem);
        weight_.set_size(selected.n_elem);
        for (size_type i = 0; i < selected.n_elem; ++i)
        {
            exponent_(i) = new_exponents(selected(i));
            weight_(i)   = new_weights(selected(i));
        }
    }

    void remove_small_terms(real_type tolerance)
    {
        const auto max_weight = arma::max(arma::abs(weight_));
        const auto selected =
            arma::find(arma::abs(weight_) > max_weight * tolerance);
        vector_type tmp_e(exponent_(selected));
        vector_type tmp_w(weight_(selected));
        exponent_ = std::move(tmp_e);
        weight_   = std::move(tmp_w);
    }
    //
    // Swap to other instance
    //
    void swap(sparse_exponential_sum& x)
    {
        exponent_.swap(x.exponent_);
        weight_.swap(x.weight_);
    }
    //
    // Output exponents and weights to a stream.
    //
    void print(std::ostream& os) const
    {
        fmt::print(os, "# exponents, weights ({} terms)\n", size());
        print_element_impl(
            os, std::integral_constant<bool, arma::is_complex<T>::value>());
    }

    bool save(std::ostream& os, const arma::file_type type = arma::arma_binary,
              const bool print_status = true) const
    {
        bool ok = exponent_.save(os, type, print_status);
        if (ok)
        {
            ok = weight_.save(os, type, print_status);
        }
        return ok;
    }

    bool load(std::istream& is, const arma::file_type type = arma::auto_detect,
              const bool print_status = true)
    {
        bool ok = exponent_.load(is, type, print_status);
        if (ok)
        {
            ok = weight_.load(is, type, print_status);
        }
        return ok;
    }

private:
    void print_element_impl(std::ostream& os, std::true_type) const
    {
        for (size_type i = 0; i < size(); ++i)
        {
            const auto& xi = exponent(i);
            const auto& wi = weight(i);
            fmt::print(os, " ({:e},{:e})  ({:e},{:e})\n", std::real(xi),
                       std::imag(xi), std::real(wi), std::imag(wi));
        }
    }

    void print_element_impl(std::ostream& os, std::false_type) const
    {
        for (size_type i = 0; i < size(); ++i)
        {
            const auto& xi = exponent(i);
            const auto& wi = weight(i);
            fmt::print(os, " {:e}  {:e}\n", xi, wi);
        }
    }
};

//
// Multiply two spaese exponential sum functions.
//
template <typename T1, typename T2>
auto multiply(const sparse_exponential_sum<T1>& x,
              const sparse_exponential_sum<T2>& y)
    -> sparse_exponential_sum<decltype(T1() + T2())>
{
    using result_type = sparse_exponential_sum<decltype(T1() + T2())>;
    using size_type   = typename result_type::size_type;
    result_type z(x.size() * y.size());

    size_type k = 0;
    for (size_type j = 0; j < y.size(); ++j)
    {
        for (size_type i = 0; i < x.size(); ++i, ++k)
        {
            z.exponent(k) = x.exponent(i) + y.exponent(j);
            z.weight(k)   = x.weight(i) * y.weight(j);
        }
    }

    return z;
}
}
//) namespace expsum

#endif /* EXPSUM_SPARSE_EXPONENTIAL_SUM_HPP */
