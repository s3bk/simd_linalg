#![feature(test)]
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

extern crate test;
use test::{Bencher, black_box};
use simd_linalg::*;

const N: usize = 256;
const M: usize = 1024;

#[bench]
    fn bench_matmul_naive_1024(bencher: &mut Bencher) {
    const O: usize = 1024;
    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();
    bencher.iter(|| matmul_naive(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[bench]
fn bench_matmul_block_100(bencher: &mut Bencher) {
    const O: usize = 100;

    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();
    bencher.iter(|| matmul_block(black_box(&a), black_box(&b), black_box(&mut c)));
}
#[bench]
fn bench_matmul_block_256(bencher: &mut Bencher) {
    const O: usize = 256;

    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();
    bencher.iter(|| matmul_block(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[bench]
fn bench_matmul_block_1024(bencher: &mut Bencher) {
    const O: usize = 1024;

    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();
    bencher.iter(|| matmul_block(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[bench]
fn bench_mat_mul_vec(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<M, N>>();
    let b = zero::<Vector<M>>();
    bencher.iter(|| black_box(&*a) * black_box(&b));
}

#[bench]
fn bench_matmul_matrixmultiply_1024(bencher: &mut Bencher) {
    const O: usize = 1024;

    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();

    bencher.iter(|| matmul_matrixmultiply(black_box(&a), black_box(&b), black_box(&mut c)));
}

#[bench]
fn bench_matmul_blas_1024(bencher: &mut Bencher) {
    const O: usize = 1024;

    let a = zero_box::<Matrix<M, N>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();

    bencher.iter(|| matmul_blas(black_box(&a), black_box(&b), black_box(&mut c)));
}
