#ifndef EXPSUM_MODIFIED_PRONY_TRUNCATION_HPP
#define EXPSUM_MODIFIED_PRONY_TRUNCATION_HPP

#include <armadillo>

namespace expsum
{

//
// Truncation of sum-of-exponentials with small exponents.
//
// This class find the optimal sum-of-exponentials with small exponents with
// smaller number of terms using the modified Prony method proposed by Beylkin
// and Monzon (2010).
//
template <typename T>
class modified_prony_truncation
{
public:
    using size_type    = arma::uword;
    using value_type   = T;
    using real_type    = typename arma::get_pod_type<T>::result;
    using complex_type = std::complex<real_type>;

    using vector_type      = arma::Col<value_type>;
    using real_vector_type = arma::Col<real_type>;
    // using complex_vector_type = arma::Col<complex_type>;
    using complex_vector_type = arma::Col<value_type>;

    using matrix_type = arma::Mat<value_type>;
    // using complex_matrix_type = arma::Mat<complex_type>;
    using complex_matrix_type = arma::Mat<value_type>;

private:
    size_type size_;
    complex_vector_type exponent_;
    complex_vector_type weight_;

    complex_vector_type vec_work_;
    complex_matrix_type mat_work_;

public:
    template <typename VecP, typename VecW>
    typename std::enable_if<(arma::is_basevec<VecP>::value &&
                             arma::is_basevec<VecW>::value),
                            void>::type
    run(const VecP& p, const VecW& w, real_type eps);

    void resize(size_type n)
    {
        if (exponent_.size() < n)
        {
            exponent_.set_size(n);
            weight_.set_size(n);
            vec_work_.set_size(2 * n);
            mat_work_.set_size(2 * n, n);
        }
    }

    //
    // @return number of terms after truncation
    //
    size_type size() const
    {
        return size_;
    }

    //
    // @return Vector view to the exponents.
    //
    auto exponents() const -> decltype(exponent_.head(size_))
    {
        return exponent_.head(size_);
    }
    //
    // @return Vector view to the weights.
    //
    auto weights() const -> decltype(weight_.head(size_))
    {
        return weight_.head(size_);
    }
};

template <typename T>
template <typename VecP, typename VecW>
typename std::enable_if<(arma::is_basevec<VecP>::value &&
                         arma::is_basevec<VecW>::value),
                        void>::type
modified_prony_truncation<T>::run(const VecP& p, const VecW& w, real_type eps)
{
    const auto n1 = p.n_elem;
    resize(n1);

    if (n1 == size_type())
    {
        size_ = n1;
        return;
    }

    value_type* h_ptr = reinterpret_cast<value_type*>(mat_work_.memptr());
    value_type* v_ptr = h_ptr + n1 * n1;

    vector_type p_pow(h_ptr, n1, false, true);
    vector_type h_vec(v_ptr, 2 * n1, false, true);

    p_pow = p;

    h_vec(0) = arma::sum(w);
    h_vec(1) = -arma::sum(w % p_pow);

    size_type n2   = 1;
    auto factorial = real_type(1);
    while (n2 < n1)
    {
        p_pow %= p;
        h_vec(2 * n2) = arma::sum(w % p_pow);
        p_pow %= p;
        h_vec(2 * n2 + 1) = -arma::sum(w % p_pow);

        factorial *= real_type(2 * n2 * (2 * n2 + 1));

        if (std::abs(h_vec(2 * n2 + 1)) < eps * factorial)
        {
            // Taylor expansion converges with the tolerance eps.
            ++n2;
            break;
        }
        ++n2;
    }

    size_ = n2;
    //
    // Construct a Hankel matrix from the sequence h_vec, and solve the linear
    // equation, H q = b, with b = -h_vec(m:2m-1).
    //
    std::cout << "***** Hankel matrix" << std::endl;
    matrix_type H(h_ptr, n2, n2, false, true);

    for (size_type k = 0; k < n2; ++k)
    {
        H.col(k) = h_vec.subvec(k, k + n2 - 1);
    }

    value_type* q_ptr = reinterpret_cast<value_type*>(exponent_.memptr());
    vector_type q(q_ptr, n2, false, true);
    std::cout << "***** linear solve" << std::endl;
    q = arma::solve(H, h_vec.subvec(n2, 2 * n2 - 1));

    //
    // Find the roots of the Prony polynomial,
    //
    // q(z) = \sum_{k=0}^{m-1} q_k z^{k}.
    //
    // The roots of q(z) can be obtained as the eigenvalues of the companion
    // matrix,
    //
    //     (0  0  ...  0 -q[0]  )
    //     (1  0  ...  0 -q[1]  )
    // C = (0  1  ...  0 -q[2]  )
    //     (.. .. ...  .. ..    )
    //     (0  0  ...  1 -q[m-1])
    //
    std::cout << "***** find roots" << std::endl;
    matrix_type& C = H; // overwrite H
    C.zeros();
    for (size_type i = 0; i < n2 - 1; ++i)
    {
        C(i + 1, i) = real_type(1);
    }
    C.col(n2 - 1) = -q;
    // exponent_.head(n2) = arma::eig_gen(C);
    // exponent_.head(n2) = -exponent_.head(n2);
    auto eigvals       = arma::eig_gen(C);
    exponent_.head(n2) = -arma::real(eigvals);

    //
    // Solve overdetermined Vandermonde system,
    //
    // V(0:2m-1,0:m-1) w(0:m-1) = h(0:2m-1)
    //
    // by the least square method.
    //
    std::cout << "***** least squares" << std::endl;
    complex_vector_type b2(vec_work_.memptr(), 2 * n2, false, true);
    for (size_type i = 0; i < 2 * n2; ++i)
    {
        b2(i) = h_vec(i);
    }
    // Construct Vandermonde matrix from exponents
    complex_matrix_type V(mat_work_.memptr(), 2 * n2, n2, false, true);
    for (size_type i = 0; i < n2; ++i)
    {
        // We assume all the eigenvalues are real here
        const auto z = exponent_(i);
        V(0, i) = T(1);
        for (size_type j = 1; j < V.n_rows; ++j)
        {
            V(j, i) = V(j - 1, i) * z; // z[i]**j
        }
    }

    weight_.subvec(0, n2 - 1) = arma::solve(V, b2);
}

} // namespace expsum

#endif /* EXPSUM_MODIFIED_PRONY_TRUNCATION_HPP */
