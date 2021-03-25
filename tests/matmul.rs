#![feature(const_generics, const_evaluatable_checked)]

use rand::{Rng, thread_rng, distributions::{Standard, Distribution}};
use simd_linalg::*;

#[test]
fn test_matmul() {
    const N: usize = 100;
    const M: usize = 150;
    const O: usize = 90;
    let rng = rand::thread_rng();
    let mut iter = Standard.sample_iter(rng);
    
    let mut a = zero_box::<Matrix<M, N>>();
    a.fill(&mut iter);

    let mut b = zero_box::<Matrix<M, O>>();
    b.fill(&mut iter);

    let mut c = zero_box::<Matrix<N, O>>();
    let mut d = zero_box::<Matrix<N, O>>();

    matmul_naive(&a, &b, &mut c);
    matmul_block(&a, &b, &mut d);

    assert!(c.max_diff(&d) < 1e-3);

    #[cfg(feature="mm")]
    {
        matmul_matrixmultiply(&a, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3);
    }

    #[cfg(feature="blis")]
    {
        matmul_blis(&a, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3);
    }

    #[cfg(feature="mkl")]
    {
        matmul_mkl(&a, &b, &mut d);
        assert!(c.max_diff(&d) < 1e-3);
    }

    #[cfg(feature="mkl_jit")]
    {
        if let Some(f) = matmul_mkl_jit() {
            f(&a, &b, &mut d);
            assert!(c.max_diff(&d) < 1e-3);
        }
    }
}