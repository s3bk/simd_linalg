use crate::{Matrix, MatrixT, Vector, simd, simd_size, zero};
use packed_simd::f32x8;
use unroll::unroll_for_loops;
use itertools::{Itertools, iproduct};


pub fn matmul<const N: usize, const M: usize, const O: usize>
(a: &Matrix<N, M>, b: &Matrix<M, O>, c: &mut Matrix<N, O>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(O)]: Sized
{
    unsafe {
        // m=N, k=M, n=O
        // a: m·k | N·M
        // b: k·n | M·O
        // c: m·n | N·O
        matrixmultiply::sgemm(N, M, O, 1.0,
            a as *const _ as _, 1, simd_size(N) as isize,
            b as *const _ as _, 1, simd_size(M) as isize,
            0.0,
            c as *mut _ as _, 1, simd_size(N) as isize,
        );
    }
}

pub fn matmul_t<const N: usize, const M: usize, const O: usize>
(a: &MatrixT<N, M>, b: &Matrix<M, O>, c: &mut Matrix<N, O>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(O)]: Sized
{
    unsafe {
        // m=N, k=M, n=O
        // a: m·k | N·M
        // b: k·n | M·O
        // c: m·n | N·O
        matrixmultiply::sgemm(N, M, O, 1.0,
            a as *const _ as _, simd_size(M) as isize, 1,
            b as *const _ as _, 1, simd_size(M) as isize,
            0.0,
            c as *mut _ as _, 1, simd_size(N) as isize,
        );
    }
}
