use crate::{Matrix, MatrixT, Vector, simd, simd_size, zero};
use packed_simd::f32x8;
use unroll::unroll_for_loops;
use itertools::{Itertools, iproduct};


#[cfg(feature="mm")]
pub mod matrixmultiply;

#[cfg(feature="blis")]
pub mod blis;

#[cfg(feature="mkl")]
pub mod mkl;

pub mod block;

pub fn matmul_naive_t<const N: usize, const M: usize, const O: usize>
    (a: &MatrixT<N, M>, b: &Matrix<M, O>, c: &mut Matrix<N, O>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(O)]: Sized
{
    for n in 0 .. N {
        for o in 0 .. O {
            c[o][n] = a[n].dot(&b[o]);
        }
    }
}


fn prefetch_l0<T>(t: &T) {
    use core::arch::x86_64::*;
    unsafe {
        core::arch::x86_64::_mm_prefetch(t as *const T as *const i8, _MM_HINT_T0);
    }
}

fn prefetch_l1<T>(t: &T) {
    use core::arch::x86_64::*;
    unsafe {
        core::arch::x86_64::_mm_prefetch(t as *const T as *const i8, _MM_HINT_T1);
    }
}

fn prefetch_l2<T>(t: &T) {
    use core::arch::x86_64::*;
    unsafe {
        core::arch::x86_64::_mm_prefetch(t as *const T as *const i8, _MM_HINT_T2);
    }
}




/// N: width, M: height, S: stride in counts of f32x8
pub struct Slice<const N: usize, const M: usize, const S: usize> {
    first: f32x8
}
impl<const N: usize, const M: usize, const S: usize> Slice<N, M, S>
where
    [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    pub fn row(&self, m: usize) -> &[f32x8; simd(N)] {
        assert!(m < M);
        unsafe {
            &*((&self.first as *const f32x8).offset((m * S) as isize).cast())
        }
    }
}

impl<const N: usize, const M: usize, const S: usize> Slice<N, M, S>
where
    [u8; simd(N)]: Sized, [u8; simd(M)]: Sized,
    [u8; simd(lower(N))]: Sized, [u8; upper(N)]: Sized,
    [u8; simd(lower(M))]: Sized, [u8; upper(M)]: Sized
{
    pub fn subdivide(&self) -> (
        &Slice<{lower(N)}, {lower(M)}, S>,
        &Slice<{upper(N)}, {lower(M)}, S>,
        &Slice<{lower(N)}, {upper(M)}, S>,
        &Slice<{upper(N)}, {upper(M)}, S>
    ) {
        let ptr = &self.first as *const f32x8;
        unsafe { (
            &*(ptr.cast()),
            &*(ptr.offset((N/8) as isize).cast()),
            &*(ptr.offset((M/2 * simd(N)) as isize).cast()),
            &*(ptr.offset((N/8 + M/2 * simd(N)) as isize).cast()),
        ) }
    }
}

pub const fn lower(n: usize) -> usize {
    (n / 2) & !7
}
pub const fn upper(n: usize) -> usize {
    n - lower(n)
}
