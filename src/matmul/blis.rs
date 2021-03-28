use crate::{Matrix, MatrixT, Vector, simd, simd_size, zero};
use packed_simd::f32x8;
use unroll::unroll_for_loops;
use itertools::{Itertools, iproduct};


pub fn matmul_t<const M: usize, const N: usize, const K: usize>
(a: &MatrixT<M, K>, b: &Matrix<K, N>, c: &mut Matrix<M, N>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(K)]: Sized
{
    extern crate blas_src;
    use blas::*;
    unsafe {
        sgemm(b'T', b'N',
            M as i32, N as i32, K as i32,
            1.0, // alpha
            a.buffer(), simd_size(K) as i32,
            b.buffer(), simd_size(K) as i32,
            0.0, // beta
            c.buffer_mut(), simd_size(M) as i32,
        );
    }
}

pub fn matmul<const M: usize, const N: usize, const K: usize>
(a: &Matrix<M, K>, b: &Matrix<K, N>, c: &mut Matrix<M, N>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(K)]: Sized
{
    extern crate blas_src;
    use blas::*;
    unsafe {
        sgemm(b'N', b'N',
            M as i32, N as i32, K as i32,
            1.0, // alpha
            a.buffer(), simd_size(M) as i32,
            b.buffer(), simd_size(K) as i32,
            0.0, // beta
            c.buffer_mut(), simd_size(M) as i32,
        );
    }
}