#![feature(const_generics, const_evaluatable_checked)]

use rand::{Rng, thread_rng, distributions::{Standard, Distribution}};
use simd_linalg::*;

#[test]
fn test_matmul() {
    const N: usize = 3*32;
    const M: usize = 3*32;
    const O: usize = 64;
    let rng = rand::thread_rng();
    let mut iter = Standard.sample_iter(rng);
    
    let mut a = zero_box::<Matrix<N, M>>();
    let mut a_t = zero_box::<Matrix<M, N>>();
    a.fill(&mut iter);
    a.copy_transposed(&mut a_t);

    let mut b = zero_box::<Matrix<M, O>>();
    b.fill(&mut iter);

    let mut c = zero_box::<Matrix<N, O>>();
    let mut d = zero_box::<Matrix<N, O>>();

    matmul_naive_t(&a_t, &b, &mut c);
    matmul_block_t(&a_t, &b, &mut d);
    matmul_block(&a, &b, &mut d);

    assert!(c.max_diff(&d) < 1e-3);

    #[cfg(feature="mm")]
    {
        matmul_matrixmultiply(&a, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3, "matmul_matrixmultiply");
    }

    #[cfg(feature="blis")]
    {
        matmul_blis(&a, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3, "matmul_blis");

        matmul_blis_t(&a_t, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3, "matmul_blis_t");
    }

    #[cfg(feature="mkl")]
    {
        matmul_mkl(&a, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3, "matmul_mkl");

        matmul_mkl_t(&a_t, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3, "matmul_mkl_t");
    }

    #[cfg(feature="mkl_jit")]
    {
        if let Some(f) = matmul_mkl_jit() {
            f(&a, &b, &mut d);
            assert!(c.max_diff(&d) < 1e-3, "matmul_mkl_jit");
        }
    }
}