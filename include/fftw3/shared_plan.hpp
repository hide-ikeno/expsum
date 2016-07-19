// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-

#ifndef FFTW3_SHARED_PLAN_HPP
#define FFTW3_SHARED_PLAN_HPP

#include <cassert>
#include <complex>
#include <memory>
#include <type_traits>

#include <fftw3.h>

namespace fftw3
{
namespace detail
{

//------------------------------------------------------------------------------
// Convert argument types compartible to FFTW interfaces.
//------------------------------------------------------------------------------
inline float* fftw_cast(float* ptr)
{
    return ptr;
}

inline double* fftw_cast(double* ptr)
{
    return ptr;
}

inline long double* fftw_cast(long double* ptr)
{
    return ptr;
}

inline fftwf_complex* fftw_cast(std::complex<float>* ptr)
{
    return reinterpret_cast<fftwf_complex*>(ptr);
}

inline fftw_complex* fftw_cast(std::complex<double>* ptr)
{
    return reinterpret_cast<fftw_complex*>(ptr);
}

inline fftwl_complex* fftw_cast(std::complex<long double>* ptr)
{
    return reinterpret_cast<fftwl_complex*>(ptr);
}
//
// ***** As fftw uses non-const pointer, const_cast is mandatry.
//
inline float* fftw_cast(const float* ptr)
{
    return const_cast<float*>(ptr);
}

inline double* fftw_cast(const double* ptr)
{
    return const_cast<double*>(ptr);
}

inline long double* fftw_cast(const long double* ptr)
{
    return const_cast<long double*>(ptr);
}

inline fftwf_complex* fftw_cast(const std::complex<float>* ptr)
{
    return reinterpret_cast<fftwf_complex*>(
        const_cast<std::complex<float>*>(ptr));
}

inline fftw_complex* fftw_cast(const std::complex<double>* ptr)
{
    return reinterpret_cast<fftw_complex*>(
        const_cast<std::complex<double>*>(ptr));
}

inline fftwl_complex* fftw_cast(const std::complex<long double>* ptr)
{
    return reinterpret_cast<fftwl_complex*>(
        const_cast<std::complex<long double>*>(ptr));
}

//==============================================================================
// Mutex for FFTW plan
//==============================================================================

template <typename T>
struct fftw_plan_mutex
{
    static std::mutex m;
};

template <typename T>
std::mutex fftw_plan_mutex<T>::m;

//==============================================================================
// 1D FFT Plans
//==============================================================================

template <typename T>
struct fftw_impl;

#define FFTW_IMPL_SPECIALIZATION_MACRO(PRECISION, PREFIX)                      \
    template <>                                                                \
    struct fftw_impl<PRECISION>                                                \
    {                                                                          \
        using real_type    = PRECISION;                                        \
        using complex_type = std::complex<real_type>;                          \
        using plan_type    = PREFIX##_plan;                                    \
        using plan_s_type  = typename std::remove_pointer<plan_type>::type;    \
        using plan_deleter = decltype(&PREFIX##_destroy_plan);                 \
        using r2r_kind     = PREFIX##_r2r_kind;                                \
                                                                               \
        static plan_type make_plan_fft(int n, int howmany,                     \
                                       const complex_type* in,                 \
                                       complex_type* out, unsigned flags)      \
        {                                                                      \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            return PREFIX##_plan_many_dft(1, &n, howmany, fftw_cast(in),       \
                                          nullptr, 1, n, fftw_cast(out),       \
                                          nullptr, 1, n, FFTW_FORWARD, flags); \
        }                                                                      \
                                                                               \
        static plan_type make_plan_ifft(int n, int howmany,                    \
                                        const complex_type* in,                \
                                        complex_type* out, unsigned flags)     \
        {                                                                      \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            return PREFIX##_plan_many_dft(                                     \
                1, &n, howmany, fftw_cast(in), nullptr, 1, n, fftw_cast(out),  \
                nullptr, 1, n, FFTW_BACKWARD, flags);                          \
        }                                                                      \
                                                                               \
        static plan_type make_plan_fft(int n, int howmany,                     \
                                       const real_type* in, complex_type* out, \
                                       unsigned flags)                         \
        {                                                                      \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            return PREFIX##_plan_many_dft_r2c(1, &n, howmany, fftw_cast(in),   \
                                              nullptr, 1, n, fftw_cast(out),   \
                                              nullptr, 1, n / 2 + 1, flags);   \
        }                                                                      \
                                                                               \
        static plan_type make_plan_ifft(int n, int howmany, complex_type* in,  \
                                        real_type* out, unsigned flags)        \
        {                                                                      \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            return PREFIX##_plan_many_dft_c2r(                                 \
                1, &n, howmany, fftw_cast(in), nullptr, 1, n / 2 + 1,          \
                fftw_cast(out), nullptr, 1, n, flags);                         \
        }                                                                      \
                                                                               \
        static plan_type make_plan_r2r(int n, int howmany,                     \
                                       const real_type* in, real_type* out,    \
                                       PREFIX##_r2r_kind kind, unsigned flags) \
        {                                                                      \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            return PREFIX##_plan_many_r2r(1, &n, howmany, fftw_cast(in),       \
                                          nullptr, 1, n, fftw_cast(out),       \
                                          nullptr, 1, n, &kind, flags);        \
        }                                                                      \
                                                                               \
        static void destroy_plan(plan_type p)                                  \
        {                                                                      \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            PREFIX##_destroy_plan(p);                                          \
        }                                                                      \
                                                                               \
        static void execute(plan_type p, const complex_type* in,               \
                            complex_type* out)                                 \
        {                                                                      \
            PREFIX##_execute_dft(p, fftw_cast(in), fftw_cast(out));            \
        }                                                                      \
                                                                               \
        static void execute(plan_type p, const real_type* in,                  \
                            complex_type* out)                                 \
        {                                                                      \
            PREFIX##_execute_dft_r2c(p, fftw_cast(in), fftw_cast(out));        \
        }                                                                      \
                                                                               \
        static void execute(plan_type p, complex_type* in, real_type* out)     \
        {                                                                      \
            PREFIX##_execute_dft_c2r(p, fftw_cast(in), fftw_cast(out));        \
        }                                                                      \
                                                                               \
        static void execute(plan_type p, const real_type* in, real_type* out)  \
        {                                                                      \
            PREFIX##_execute_r2r(p, fftw_cast(in), fftw_cast(out));            \
        }                                                                      \
                                                                               \
        static void cleanup()                                                  \
        {                                                                      \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            PREFIX##_cleanup();                                                \
        }                                                                      \
    };

FFTW_IMPL_SPECIALIZATION_MACRO(float, fftwf);
FFTW_IMPL_SPECIALIZATION_MACRO(double, fftw);
FFTW_IMPL_SPECIALIZATION_MACRO(long double, fftwl);

#undef FFTW_IMPL_SPECIALIZATION_MACRO

} // namespace: detail

//
// Fast Fourier transform (forward)
//
template <typename T>
struct fft
{
private:
    using impl_type = detail::fftw_impl<T>;

public:
    using real_type    = T;
    using complex_type = std::complex<real_type>;
    using plan_pointer = std::shared_ptr<typename impl_type::plan_s_type>;

    //--------------------------------------------------------------------------
    // complex_type-to-complex transform
    //--------------------------------------------------------------------------
    static plan_pointer make_plan(int n, int howmany, const complex_type* in,
                                  complex_type* out)
    {
        return plan_pointer(
            impl_type::make_plan_fft(n, howmany, in, out, FFTW_MEASURE),
            [](typename impl_type::plan_type p) {
                impl_type::destroy_plan(p);
            });
    }

    static void run(const plan_pointer& plan, const complex_type* in,
                    complex_type* out)
    {
        impl_type::execute(plan.get(), in, out);
    }

    //--------------------------------------------------------------------------
    // real_type-to-complex transform
    //--------------------------------------------------------------------------
    static plan_pointer make_plan(int n, int howmany, const real_type* in,
                                  complex_type* out)
    {
        return plan_pointer(
            impl_type::make_plan_fft(n, howmany, in, out, FFTW_MEASURE),
            [](typename impl_type::plan_type p) {
                impl_type::destroy_plan(p);
            });
    }

    static void run(const plan_pointer& plan, const real_type* in,
                    complex_type* out)
    {
        impl_type::execute(plan.get(), in, out);
    }
};

//
// Fast Fourier transform (inverse transform)
//
template <typename T>
struct ifft
{
private:
    using impl_type = detail::fftw_impl<T>;

public:
    using real_type    = T;
    using complex_type = std::complex<real_type>;
    using plan_pointer = std::shared_ptr<typename impl_type::plan_s_type>;

    //--------------------------------------------------------------------------
    // complex_type-to-complex transform
    //--------------------------------------------------------------------------
    static plan_pointer make_plan(int n, int howmany, const complex_type* in,
                                  complex_type* out)
    {
        return plan_pointer(
            impl_type::make_plan_ifft(n, howmany, in, out, FFTW_MEASURE),
            [](typename impl_type::plan_type p) {
                impl_type::destroy_plan(p);
            });
    }

    static void run(const plan_pointer& plan, const complex_type* in,
                    complex_type* out)
    {
        impl_type::execute(plan.get(), in, out);
    }

    //--------------------------------------------------------------------------
    // complex_type-to-real transform
    //--------------------------------------------------------------------------
    static plan_pointer make_plan(int n, int howmany, complex_type* in,
                                  real_type* out)
    {
        return plan_pointer(
            impl_type::make_plan_ifft(n, howmany, in, out, FFTW_MEASURE),
            [](typename impl_type::plan_type p) {
                impl_type::destroy_plan(p);
            });
    }

    static void run(const plan_pointer& plan, complex_type* in, real_type* out)
    {
        impl_type::execute(plan.get(), in, out);
    }
};

//==============================================================================
// real_type-to-real transform
//==============================================================================

namespace detail
{

template <typename T, typename fftw_impl<T>::r2r_kind Kind>
struct transform_r2r_impl
{
private:
    using impl_type = fftw_impl<T>;

public:
    using real_type    = T;
    using plan_pointer = std::shared_ptr<typename impl_type::plan_s_type>;

    static plan_pointer make_plan(int n, int howmany, const real_type* in,
                                  real_type* out)
    {
        return plan_pointer(
            impl_type::make_plan_r2r(n, howmany, in, out, Kind, FFTW_MEASURE),
            [](typename impl_type::plan_type p) {
                impl_type::destroy_plan(p);
            });
    }

    static void run(const plan_pointer& plan, const real_type* in,
                    real_type* out)
    {
        impl_type::execute(plan.get(), in, out);
    }
};

} // namespace detail

//
// Discrete Hartley transform
//
template <typename T>
using dht = detail::transform_r2r_impl<T, FFTW_DHT>;
//
// Discrete cosine transform
//
template <typename T>
using dct1 = detail::transform_r2r_impl<T, FFTW_REDFT00>;

template <typename T>
using dct2 = detail::transform_r2r_impl<T, FFTW_REDFT10>;

template <typename T>
using dct3 = detail::transform_r2r_impl<T, FFTW_REDFT01>;

template <typename T>
using dct4 = detail::transform_r2r_impl<T, FFTW_REDFT11>;

//
// Discrete sine transform
//
template <typename T>
using dst1 = detail::transform_r2r_impl<T, FFTW_RODFT00>;

template <typename T>
using dst2 = detail::transform_r2r_impl<T, FFTW_RODFT10>;

template <typename T>
using dst3 = detail::transform_r2r_impl<T, FFTW_RODFT01>;

template <typename T>
using dst4 = detail::transform_r2r_impl<T, FFTW_RODFT11>;

} // namespace: fftw3

#endif /* FFTW3_SHARED_PLAN_HPP */
