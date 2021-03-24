#![feature(test, core_intrinsics)]
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

extern crate test;
use test::{Bencher, black_box};
use simd_linalg::*;

const K: usize = 10000;
const N: usize = 7;

#[bench]
fn bench_dot7_vector(bencher: &mut Bencher) {
    let a = vec![zero::<Vector<N>>(); K];
    let b = vec![zero::<Vector<N>>(); K];
    let mut c = [0.0; K];
    bencher.iter(|| dot_slice(
        black_box(a.as_slice()),
        black_box(b.as_slice()),
        black_box(c.as_mut())
    ));
}
#[bench]
fn bench_dot7_naive(bencher: &mut Bencher) {
    let a = vec![[0.0; N]; K];
    let b = vec![[0.0; N]; K];
    let mut c = [0.0; K];
    bencher.iter(|| dot_slice_naive(
        black_box(a.as_slice()),
        black_box(b.as_slice()),
        black_box(c.as_mut())
    ));
}

#[inline(never)]
pub fn dot_slice(a: &[Vector<N>], b: &[Vector<N>], c: &mut [f32]) {
    assert_eq3(a.len(), b.len(), c.len());
    for ((a, b), c) in a.iter().zip(b).zip(c) {
        *c = a.dot_fast(b);
    }
}

#[inline(never)]
pub fn dot_slice_naive(a: &[[f32; N]], b: &[[f32; N]], c: &mut [f32]) {
    assert_eq3(a.len(), b.len(), c.len());
    for ((a, b), c) in a.iter().zip(b).zip(c) {
        *c = a.iter().zip(b).map(|(&a, &b)| a * b).sum();
    }
}

fn assert_eq3(a: usize, b: usize, c: usize) {
    unsafe {
        std::intrinsics::assume(a == b);
        std::intrinsics::assume(a == c);
    }
}
