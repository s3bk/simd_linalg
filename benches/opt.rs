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
fn bench_matmul_block(bencher: &mut Bencher) {
    let a = zero_box::<Matrix<N, M>>();
    let b = zero_box::<Matrix<M, O>>();
    let mut c = zero_box::<Matrix<N, O>>();
    bencher.iter(|| block::matmul(black_box(&a), black_box(&b), black_box(&mut c)));
}
