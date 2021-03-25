use core::ops::{
    AddAssign, Mul, SubAssign,
    Index, IndexMut
};
use crate::{simd, simd_size, ZeroInit, Vector};
use packed_simd::f32x8;

#[cfg(feature="std")]
use std::fmt;


/// A Matrix of `N` columns and `M` rows, each column is padded.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct Matrix<const N: usize, const M: usize>(pub [Vector<N>; M])
where [u8; simd(N)]: Sized;

pub type MatrixT<const N: usize, const M: usize> = Matrix<M, N>;

impl<const N: usize, const M: usize> Matrix<N, M>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    pub fn copy_transposed(&self, out: &mut Matrix<M, N>) {
        for n in 0 .. N {
            for m in 0 .. M {
                out[m][n] = self[n][m];
            }
        }
    }
    // stride in bytes
    pub const fn stride() -> usize {
        core::mem::size_of::<Vector<N>>()
    }

    /// Fill the matrix from the given iterator one row at a time
    pub fn fill(&mut self, mut values: impl Iterator<Item=f32>) {
        self.0.iter_mut().for_each(|v| v.fill(&mut values))
    }

    pub fn argmax1(&self, vector: &Vector<N>) -> (Vector<M>, [usize; M]) {
        let mut out_v = Vector::null();
        let mut out_i = [0; M];
        for i in 0 .. M {
            let sum = self.0[i] + vector;
            let (idx, val) = sum.max_idx();
            out_i[i] = idx;
            out_v[i] = val;
        }
        (out_v, out_i)
    }

    pub fn max_abs(&self) -> f32 {
        self.0.iter().fold(0.0, |max, v| max.max(v.max_abs()))
    }
    pub fn max_diff(&self, rhs: &Self) -> f32 {
        self.0.iter().zip(rhs.0.iter()).fold(0.0, |max, (a, b)| max.max(a.max_diff(b)))
    }

    pub fn buffer(&self) -> &[f32] {
        unsafe { core::slice::from_raw_parts(self as *const Self as *const f32, simd_size(N) * M) }
    }
    pub fn buffer_mut(&mut self) -> &mut [f32] {
        unsafe { core::slice::from_raw_parts_mut(self as *mut Self as *mut f32, simd_size(N) * M) }
    }
}

impl<'a, const N: usize, const M: usize> Mul<&'a Vector<N>> for &'a Matrix<N, M>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    type Output = Vector<M>;
    fn mul(self, rhs: &'a Vector<N>) -> Vector<M> {
        let mut out = Vector::null();
        for (i, v) in self.0.iter().enumerate() {
            out.set(i, v.dot(rhs))
        }
        out
    }
}


impl<'a, const N: usize, const M: usize> AddAssign<&'a Self> for Matrix<N, M>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    fn add_assign(&mut self, rhs: &'a Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a += b;
        }
    }
}
impl<'a, const N: usize, const M: usize> SubAssign<&'a Self> for Matrix<N, M>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    fn sub_assign(&mut self, rhs: &'a Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a -= b;
        }
    }
}

unsafe impl<const N: usize, const M: usize> ZeroInit for Matrix<N, M>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized {}

impl<'a, const N: usize, const M: usize> Index<usize> for Matrix<N, M>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    type Output = Vector<N>;
    fn index(&self, idx: usize) -> &Vector<N> {
        &self.0[idx]
    }
}
impl<'a, const N: usize, const M: usize> IndexMut<usize> for Matrix<N, M>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    fn index_mut(&mut self, idx: usize) -> &mut Vector<N> {
        &mut self.0[idx]
    }
}


#[cfg(feature="std")]
impl<const N: usize, const M: usize> fmt::Display for Matrix<N, M>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Matrix<{}, {}>(", N, M)?;
        for row in self.0.iter().take(10) {
            for n in row.iter().take(10) {
                write!(f, "{:12.3e}", n)?;
            }
            if N > 10 {
                write!(f, " ...")?;
            }
            writeln!(f)?;
        }
        if M > 10 {
            writeln!(f, "... )")
        } else {
            writeln!(f, ")")
        }
    }
}
