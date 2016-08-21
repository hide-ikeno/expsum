#ifndef EXPSUM_FUNCTION_FITTER_HPP
#define EXPSUM_FUNCTION_FITTER_HPP

#include <type_traits>

#include "expsum/fitting/fast_esprit.hpp"

namespace expsum
{

template <typename T>
class function_fitter
{
public:
    using esprit_type  = expsum::fast_esprit<T>;
    using size_type    = typename esprit_type::size_type;
    using value_type   = typename esprit_type::value_type;
    using real_type    = typename esprit_type::real_type;
    using complex_type = typename esprit_type::complex_type;

    using vector_type      = typename esprit_type::vector_type;
    using real_vector_type = typename esprit_type::real_vector_type;

private:
    using is_complex_t =
        std::integral_constant<bool, arma::is_complex<T>::value>;

    static const real_type default_tolerance;

    esprit_type esprit_;
    real_type tolerance_;
    real_type x0_;
    real_type xstep_;
    vector_type work_;
    bool is_finished_;

public:
    function_fitter() = default;

    explicit function_fitter(size_type n_samples, size_type max_terms,
                             real_type tol = default_tolerance)
        : esprit_(n_samples, (n_samples + 1) / 2, max_terms),
          tolerance_(tol),
          x0_(),
          xstep_(real_type(1)),
          work_(n_samples),
          is_finished_(false)
    {
    }

    function_fitter(const function_fitter&) = default;
    function_fitter(function_fitter&&)      = default;
    ~function_fitter()                      = default;

    function_fitter& operator=(const function_fitter&) = default;
    function_fitter& operator=(function_fitter&&) = default;

    void set_size(size_type n_samples, size_type max_terms)
    {
        esprit_.resize(n_samples, (n_samples + 1) / 2, max_terms);
        work_.resize(n_samples);
        is_finished_ = false;
    }

    void set_tolerance(real_type tol)
    {
        tolerance_ = tol;
    }

    real_type tolerance() const noexcept
    {
        return tolerance_;
    }

    size_type num_samples() const
    {
        return esprit_.size();
    }

    size_type max_terms() const
    {
        return esprit_.ncols();
    }

    ///
    /// Fit a funciton by a sum of exponential functions at given inteval
    /// [x0,x1].
    ///
    template <typename UnaryFunction>
    void run(UnaryFunction fn, real_type x0, real_type x1)
    {
        assert(num_samples() > 1);
        assert(x1 != x0);

        is_finished_ = false;
        x0_          = x0;
        xstep_       = (x1 - x0) / (num_samples() - 1);

        // sampling function values
        auto xval = x0_;
        for (size_type i = 0; i < num_samples(); ++i)
        {
            work_(i) = fn(xval);
            xval += xstep_;
        }

        // do fit
        esprit_.fit(work_, x0_, xstep_, tolerance_);
        is_finished_ = true;
    }
    ///
    /// @return Vector view to the exponents.
    ///
    auto exponents() const -> decltype(esprit_.exponents())
    {
        return esprit_.exponents();
    }

    ///
    /// @return Vector view to the weights.
    ///
    auto weights() const -> decltype(esprit_.weights())
    {
        return esprit_.weights();
    }

    ///
    /// Calculate errors on sampling points
    ///
    const vector_type& eval_error()
    {
        assert(is_finished_ && "fitting not yet done");
        //
        // REMARK: on entry, work_ holds sampled function values
        //
        auto xval = x0_;
        for (size_type i = 0; i < num_samples(); ++i)
        {
            work_(i) = eval_at(xval, is_complex_t()) - work_(i);
            xval += xstep_;
        }
    }

    //
    // Get the maximum absolute error of fitting. You must call `run`
    // beforehand.
    //
    real_type max_abs_error() const
    {
        assert(is_finished_ && "fitting not finished");
        return arma::max(arma::abs(work_));
    }

    real_type max_error(size_type& imax) const
    {
        return arma::abs(work_).max(imax);
    }

private:
    value_type eval_at(real_type x, /*is_complex_t*/ std::true_type) const
    {
        return esprit_.eval_at(x);
    }

    value_type eval_at(real_type x, /*is_complex_t*/ std::false_type) const
    {
        return std::real(esprit_.eval_at(x));
    }
};

template <typename T>
const typename function_fitter<T>::real_type
    function_fitter<T>::default_tolerance =
        std::sqrt(std::numeric_limits<real_type>::epsilon());

} // namespace: expsum

#endif /* EXPSUM_FUNCTION_FITTER_HPP */
