#![feature(test)]
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

extern crate test;
use test::{Bencher, black_box};
use simd_linalg::*;

const N: usize = 64;
const M: usize = 32;
const O: usize = 8;

#[bench]
    fn bench_matmul_naive(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();
    bencher.iter(|| matmul_naive(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[bench]
fn bench_matmul_block(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();
    bencher.iter(|| matmul_block(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[bench]
fn bench_matmul_block_t(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<N, M>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();
    bencher.iter(|| matmul_block_t(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[cfg(feature="mm")]
#[bench]
fn bench_matmul_matrixmultiply(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();

    bencher.iter(|| matmul_matrixmultiply(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[cfg(feature="blis")]
#[bench]
fn bench_matmul_blis(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();

    bencher.iter(|| matmul_blis(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[cfg(feature="mkl")]
#[bench]
fn bench_matmul_mkl(bencher: &mut Bencher) {
    mkl_init();

    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();

    bencher.iter(|| matmul_mkl(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[cfg(feature="mkl_jit")]
#[bench]
fn bench_matmul_mkl_jit(bencher: &mut Bencher) {
    mkl_init();

    if let Some(f) = matmul_mkl_jit() {
        let a = zero_box::<Matrix<M, N>>();
        let b = zero_box::<Matrix<M, O>>();
        let mut c = zero_box::<Matrix<N, O>>();

        bencher.iter(|| f(black_box(&a), black_box(&b), black_box(&mut c)));
    }
}

#[cfg(feature="mkl_jit")]
#[bench]
fn bench_matmul_mkl_jit_t(bencher: &mut Bencher) {
    mkl_init();

    if let Some(f) = matmul_mkl_jit_t() {
        let a = zero_box::<Matrix<N, M>>();
        let b = zero_box::<Matrix<M, O>>();
        let mut c = zero_box::<Matrix<N, O>>();

        bencher.iter(|| f(black_box(&a), black_box(&b), black_box(&mut c)));
    }
}
