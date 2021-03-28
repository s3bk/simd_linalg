#![feature(test)]
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

extern crate test;
use test::{Bencher, black_box};
use simd_linalg::*;

const N: usize = 256;
const M: usize = 256;
const O: usize = 256;

#[bench]
    fn bench_matmul_naive_t(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();
    bencher.iter(|| matmul_naive_t(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[bench]
fn bench_matmul_block(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<N, M>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();
    bencher.iter(|| block::matmul(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[bench]
fn bench_matmul_block_t(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();
    bencher.iter(|| block::matmul_t(black_box(&a), black_box(&b), black_box(&mut c)));
}
#[bench]
fn bench_matmul_block_4_6_8(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<4, 6>>();
    let b = zero_box::<Matrix<6, 8>>();
    let mut c = zero_box::<Matrix<4, 8>>();
    bencher.iter(|| block::matmul_4_6_8(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[cfg(feature="mm")]
#[bench]
fn bench_matmul_matrixmultiply(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<N, M>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();

    bencher.iter(|| matrixmultiply::matmul(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[cfg(feature="mm")]
#[bench]
fn bench_matmul_matrixmultiply_t(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();

    bencher.iter(|| matrixmultiply::matmul_t(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[cfg(feature="blis")]
#[bench]
fn bench_matmul_blis_t(bencher: &mut Bencher) {
    blis_init();
    
    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();

    bencher.iter(|| blis::matmul_t(black_box(&a), black_box(&b), black_box(&mut c)));
}
#[cfg(feature="blis")]
#[bench]
fn bench_matmul_blis(bencher: &mut Bencher) {
    blis_init();

    let a = zero_box::<Matrix<N, M>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();

    bencher.iter(|| blis::matmul(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[cfg(feature="mkl")]
#[bench]
fn bench_matmul_mkl(bencher: &mut Bencher) {
    mkl_init();

    let a = zero_box::<Matrix<N, M>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();

    bencher.iter(|| mkl::matmul(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[cfg(feature="mkl")]
#[bench]
fn bench_matmul_mkl_t(bencher: &mut Bencher) {
    mkl_init();

    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();

    bencher.iter(|| mkl::matmul_t(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[cfg(feature="mkl_jit")]
#[bench]
fn bench_matmul_mkl_jit(bencher: &mut Bencher) {
    mkl_init();

    if let Some(f) = mkl::matmul_jit() {
        let a = zero_box::<Matrix<N, M>>();
        let b = zero_box::<Matrix<M, O>>();
        let mut c = zero_box::<Matrix<N, O>>();

        bencher.iter(|| f(black_box(&a), black_box(&b), black_box(&mut c)));
    }
}

#[cfg(feature="mkl_jit")]
#[bench]
fn bench_matmul_mkl_jit_t(bencher: &mut Bencher) {
    mkl_init();

    if let Some(f) = mkl::matmul_jit_t() {
        let a = zero_box::<Matrix<M, N>>();
        let b = zero_box::<Matrix<M, O>>();
        let mut c = zero_box::<Matrix<N, O>>();

        bencher.iter(|| f(black_box(&a), black_box(&b), black_box(&mut c)));
    }
}
