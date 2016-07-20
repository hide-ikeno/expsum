#ifndef EXPSUM_EXPONENTIAL_SUM_HPP
#define EXPSUM_EXPONENTIAL_SUM_HPP

#include <iosfwd>
#include <armadillo>

#include "expsum/cholesky_cauchy.hpp"

namespace expsum
{

template <typename ResultT, typename ParamT = ResultT>
class exponential_sum;

//
// Univariate function expressed as an exponential sum
//
template <typename ResultT, typename ParamT>
class exponential_sum
{
public:
    using result_type      = ResultT;
    using argument_type    = typename arma::get_pod_type<ResultT>::result;
    using parameter_type   = ParamT;
    using size_type        = arma::uword;
    using parameter_vector = arma::Col<parameter_type>;

private:
    template <bool>
    struct eval_dispacher
    {
    };

    using eval_default      = eval_dispacher<false>;
    using eval_as_real_func = eval_dispacher<true>;

    parameter_vector exponent_; // exponents
    parameter_vector weight_;   // weights

public:
    constexpr static const bool is_real_function =
        arma::is_real<result_type>::value;

    // Default constructor
    exponential_sum() = default;

    // Create `n` term exponential sum
    exponential_sum(size_type n) : exponent_(n), weight_(n)
    {
    }

    // Create from the vectors of exponents and weights
    template <typename V1, typename V2>
    exponential_sum(
        const V1& exponents, const V2& weights,
        typename std::enable_if<(arma::is_arma_type<V1>::value &&
                                 arma::is_arma_type<V2>::value)>::type* = 0)
        : exponent_(exponents), weight_(weights)
    {
        assert(exponents.is_vec() && weights.is_vec());
        assert(exponents.size() == weights.size());
    }

    // Default copy constructor
    exponential_sum(const exponential_sum&) = default;

    // Default move constructor
    exponential_sum(exponential_sum&&) = default;

    // Default destructor
    ~exponential_sum() = default;

    // Default copy assignment operator
    exponential_sum& operator=(const exponential_sum&) = default;

    // Default move assignment operator
    exponential_sum& operator=(exponential_sum&&) = default;

    //
    // Get the number of terms
    //
    size_type size() const
    {
        return exponent_.size();
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
    result_type operator()(const argument_type& x) const
    {
        return eval_impl(
            x, eval_dispacher<(arma::is_real<result_type>::value &&
                               arma::is_complex<parameter_type>::value)>());
    }
    //
    // Get const reference to the vector of exponents
    //
    const parameter_vector& exponent() const
    {
        return exponent_;
    }
    //
    // Get const reference to the `i`-th exponent
    //
    const parameter_type& exponent(size_type i) const
    {
        return exponent_(i);
    }
    //
    // Get reference to the `i`-th exponent
    //
    parameter_type& exponent(size_type i)
    {
        return exponent_(i);
    }
    //
    // Get const reference to the vector of weights
    //
    const parameter_vector& weight() const
    {
        return weight_;
    }
    //
    // Get const reference to the `i`-th weight
    //
    const parameter_type& weight(size_type i) const
    {
        return weight_(i);
    }
    //
    // Get reference to the `i`-th weight
    //
    parameter_type& weight(size_type i)
    {
        return weight_(i);
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
    // Truncation
    //
    void trucate(argument_type tolerance);
    //
    // Output to ostream
    //
    template <typename Ch, typename Tr>
    void print(std::basic_ostream<Ch, Tr>& os) const
    {
        os << size() << '\n'; // number of terms
        for (size_type i = 0; i < size(); ++i)
        {
            os << exponent(i) << '\t' << weight(i) << '\n';
        }
    }

private:
    result_type eval_impl(const argument_type& x, eval_default) const
    {
        return arma::sum(arma::exp(-x * exponent_) % weight_);
    }

    result_type eval_impl(const argument_type& x, eval_as_real_func) const
    {
        return std::real(arma::sum(arma::exp(-x * exponent_) % weight_));
    }
};

template <typename ResultT, typename ParamT>
void exponential_sum<ResultT, ParamT>::trucate(argument_type tolerance)
{
    weight_ = arma::sqrt(weight_);
    const auto n = weight_.size();
    cholesky_cauchy_rrd<parameter_type> chol(n);
    auto matL = chol.run(exponent_, weight_, tolerance);
    // A * P + P * A.t() = B * B.t()
    // A * (L * D * D * L.t()) + (L * D * D * L.t()) * A.t() = B * B.t()

    // A * (S * S.t()) + (S * S.t()) * A.t() = B * B.t()
    // A * (L * L.t()) + (L * L.t()) * A.t() = C * C.t()

    // (L.t() * L) * A * (S * S.t()) * (L * L.t())
    //  + (L.t() * L) *  (S * S.t()) * A.t() * (L * L.t())
    //  = (L.t() * L) * B * B.t() * (L * L.t())

    // B' = D * U.t() * L * B
    // C' = C * S.t() * V * D
}

// Ostream operator
template <typename Ch, typename Tr, typename ResultT, typename ParamT>
std::basic_ostream<Ch, Tr>&
operator<<(std::basic_ostream<Ch, Tr>& os,
           const exponential_sum<ResultT, ParamT>& fn)
{
    fn.print(os);
    return os;
}

//=============================================================================
// Function multiplication
//=============================================================================
//
// Multiply two exponential sum functions.
//
template <typename T1, typename P1, typename T2, typename P2>
auto multiply(const exponential_sum<T1, P1>& x,
              const exponential_sum<T2, P2>& y)
    -> exponential_sum<decltype(T1() + T2()), decltype(P1() + P2())>
{
    using result_type =
        exponential_sum<decltype(T1() + T2()), decltype(P1() + P2())>;
    using size_type = typename result_type::size_type;

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

} // namespace: expsum

#endif /* EXPSUM_EXPONENTIAL_SUM_HPP */
