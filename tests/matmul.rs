#![feature(const_generics, const_evaluatable_checked)]

use rand::{Rng, thread_rng, distributions::{Standard, Distribution}};
use simd_linalg::*;

#[test]
fn test_matmul() {
    const M: usize = 3*32;
    const N: usize = 3*32;
    const K: usize = 128;
    let rng = rand::thread_rng();
    let mut iter = Standard.sample_iter(rng);
    
    let mut a = zero_box::<Matrix<M, K>>();
    let mut a_t = zero_box::<Matrix<K, M>>();
    a.fill(&mut iter);
    a.copy_transposed(&mut a_t);

    let mut b = zero_box::<Matrix<K, N>>();
    b.fill(&mut iter);

    let mut c = zero_box::<Matrix<M, N>>();
    let mut d = zero_box::<Matrix<M, N>>();

    matmul_naive_t(&a_t, &b, &mut c);
    block::matmul_t(&a_t, &b, &mut d);
    assert!(c.max_diff(&d) < 1e-3, "block::matmul_t");

    block::matmul(&a, &b, &mut d);
    assert!(c.max_diff(&d) < 1e-3, "block::matmul");

    #[cfg(feature="mm")]
    {
        matrixmultiply::matmul(&a, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3, "matmul_matrixmultiply");
    }

    #[cfg(feature="blis")]
    {
        blis::matmul(&a, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3, "matmul_blis");

        blis::matmul_t(&a_t, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3, "matmul_blis_t");
    }

    #[cfg(feature="mkl")]
    {
        mkl::matmul(&a, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3, "matmul_mkl");

        mkl::matmul_t(&a_t, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3, "matmul_mkl_t");

        if let Some(f) = mkl::matmul_jit() {
            f(&a, &b, &mut d);
            assert!(c.max_diff(&d) < 1e-3, "matmul_mkl_jit");
        }
    }
}