#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

use simd_linalg::*;

const M: usize = 128;
const K: usize = 128;
const N: usize = 128;

fn main() {
    let a = zero_box::<Matrix<M, K>>();
    let b = zero_box::<Matrix<K, N>>();
    let mut c = zero_box::<Matrix<M, N>>();

    println!("a at {:p} {} {}", a.as_ref(), simd_size(M), K);
    println!("b at {:p} {} {}", b.as_ref(), simd_size(K), N);
    println!("c at {:p} {} {}", c.as_ref(), simd_size(M), N);

    #[cfg(feature="mkl")]
    matmul_mkl(&a, &b, &mut c);
}