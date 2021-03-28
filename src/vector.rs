use packed_simd::{f32x8, m32x8};
use core::ops::{
    Add, Mul, Sub, AddAssign, MulAssign, SubAssign,
    Deref, DerefMut, Index, IndexMut
};
use crate::{simd, simd_size, ZeroInit};

#[cfg(feature="std")]
use std::fmt;

/// Array of N `f32`s in SIMD layout (padded).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vector<const N: usize>(pub [f32x8; simd(N)])
where [u8; simd(N)]: Sized;


macro_rules! vec_op {
    ($op:ident, $fn:ident; $op_assign:ident, $fn_assign:ident) => {
        impl<const N: usize> $op for Vector<N> where
            [u8; simd(N)]: Sized
        {
            type Output = Self;
            fn $fn(mut self, rhs: Self) -> Self {
                self.$fn_assign(&rhs);
                self
            }
        }
        impl<'a, const N: usize> $op<&'a Self> for Vector<N> where
            [u8; simd(N)]: Sized
        {
            type Output = Self;
            fn $fn(mut self, rhs: &'a Self) -> Self {
                self.$fn_assign(rhs);
                self
            }
        }
        impl<const N: usize> $op_assign<Self> for Vector<N> where 
            [u8; simd(N)]: Sized
        {
            fn $fn_assign(&mut self, rhs: Self) {
                self.$fn_assign(&rhs);
            }
        }
        impl<'a, const N: usize> $op_assign<&'a Self> for Vector<N> where 
            [u8; simd(N)]: Sized
        {
            fn $fn_assign(&mut self, rhs: &'a Self) {
                for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
                    *a = a.$fn(*b);
                }
            }
        }
    }
}

vec_op!(Sub, sub; SubAssign, sub_assign);
vec_op!(Mul, mul; MulAssign, mul_assign);
vec_op!(Add, add; AddAssign, add_assign);

unsafe impl<const N: usize> ZeroInit for Vector<N> where
[u8; simd(N)]: Sized {}

impl<const N: usize> Vector<N> where
    [u8; simd(N)]: Sized
{
    const fn mask() -> m32x8 {
        m32x8::new(
            N % 8 > 0,
            N % 8 > 1,
            N % 8 > 2,
            N % 8 > 3,
            N % 8 > 4,
            N % 8 > 5,
            N % 8 > 6,
            N % 8 > 7,
        )
    }

    /// create a Vector full of zeros
    pub fn null() -> Self {
        Vector([f32x8::splat(0.0); simd(N)])
    }

    /// return a new Vector with tanh of each value
    pub fn tanh(mut self) -> Self {
        self.0.iter_mut().for_each(|x| *x = x.tanh());
        self
    }

    pub fn sigmoid(mut self) -> Self {
        let one = f32x8::splat(1.0);
        self.0.iter_mut().for_each(|x| *x = one / ((-*x).exp() + one));
        self
    }

    /// set the item at the given index to the given value
    pub fn set(&mut self, idx: usize, value: f32) {
        assert!(idx < N);
        self[idx] = value;
    }

    /// dot product with another Vector
    pub fn dot_fast_v(&self, rhs: &Self) -> f32x8 {
        self.0.iter().zip(rhs.0.iter())
            .map(|(a, b)| *a * *b)
            .fold(f32x8::splat(0.0), f32x8::add)
    }
    /// dot product with another Vector
    pub fn dot_fast(&self, rhs: &Self) -> f32 {
        self.dot_fast_v(rhs).sum()
    }

    /// dot product with another Vector
    pub fn dot(&self, rhs: &Self) -> f32 {
        let mut sum = self.0.iter().take(N/8).zip(rhs.0.iter())
            .map(|(a, b)| *a * *b)
            .fold(f32x8::splat(0.0), f32x8::add)
            .sum();
        if N % 8 != 0 {
            let last = simd(N) - 1;
            let ab = self.0[last] * rhs.0[last];
            sum += (Self::mask().select(ab, f32x8::splat(0.0))).sum();
        }
        sum
    }

    /// returns the index of the highest value and the value itself
    pub fn max_idx(&self) -> (usize, f32) {
        let mut max_val = -core::f32::INFINITY;
        let mut max_idx = 0;
        for (i, &v) in self.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        (max_idx, max_val)
    }

    pub fn max_abs(&self) -> f32 {
        self.0.iter().fold(f32x8::splat(0.0), |max, s| max.max(s.abs())).max_element()
    }
    pub fn max_diff(&self, rhs: &Self) -> f32 {
        self.0.iter().zip(rhs.0.iter()).take(N-1)
            .fold(f32x8::splat(0.0), |max, (&a, &b)| max.max((a - b).abs()))
            .max(Self::mask().select(self.0[simd(N)-1] - rhs.0[simd(N)-1], f32x8::splat(0.0)).abs())
            .max_element()
    }
    /// create a Vector with every element set to the given value
    pub fn splat(value: f32) -> Self {
        let mut v = Vector::null();
        v.fill(core::iter::repeat(value));
        v
    }

    /// Fill the value from the given iterator
    pub fn fill(&mut self, values: impl Iterator<Item=f32>) {
        self.iter_mut().zip(values).for_each(|(o, i)| *o = i)
    }
    pub fn fill_val(&mut self, val: f32) {
        for slice in self.0.iter_mut() {
            *slice = f32x8::splat(val);
        }
    }

    /// Concatenate self with another vector
    /// 
    /// The resulting vector contains the values from `self`
    /// in `0..N` and `rhs` in `N..N+M`
    pub fn concat<const M: usize>(&self, rhs: &Vector<M>) -> Vector<{N + M}>
    where [u8; simd(M)]: Sized, [u8; simd(N+M)]: Sized
    {
        let mut out = Vector::null();
        let (a, b) = out.split_at_mut(N);
        a.copy_from_slice(&**self);
        b.copy_from_slice(&**rhs);
        out
    }

    /// Concatenate self with two other vectors
    /// 
    /// The resulting vector contains the values from `self`
    /// in `0..N`, `rhs` in `N..N+M` and `other` in `N+M..N+M+O`
    pub fn concat2<const M: usize, const O: usize>(&self, rhs: &Vector<M>, other: &Vector<O>) -> Vector<{N + M + O}>
    where [u8; simd(M)]: Sized, [u8; simd(O)]: Sized, [u8; simd(N+M+O)]: Sized
    {
        let mut out = Vector::null();
        let (a, bc) = out.split_at_mut(N);
        let (b, c) = bc.split_at_mut(M);
        a.copy_from_slice(&**self);
        b.copy_from_slice(&**rhs);
        c.copy_from_slice(&**other);
        out
    }

    pub fn buffer(&self) -> &[f32] {
        unsafe { core::slice::from_raw_parts(self.as_ptr().cast(), simd_size(N)) }
    }
    pub fn buffer_mut(&mut self) -> &mut [f32] {
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr().cast(), simd_size(N)) }
    }
}
impl<const N: usize> Deref for Vector<N> where
    [u8; simd(N)]: Sized
{
    type Target = [f32; N];
    #[inline(always)]
    fn deref(&self) -> &[f32; N] {
        unsafe {
            &*(self as *const _ as *const [f32; N])
        }
    }
}
impl<const N: usize> DerefMut for Vector<N> where
    [u8; simd(N)]: Sized
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [f32; N] {
        unsafe {
            &mut *(self as *mut _ as *mut [f32; N])
        }
    }
}
impl<'a, const N: usize> From<&'a [f32; N]> for Vector<N> where
    [u8; simd(N)]: Sized
{
    fn from(array: &[f32; N]) -> Self {
        let mut v = Self::null();
        v.copy_from_slice(array);
        v
    }
}

#[cfg(feature="std")]
impl<'a, const N: usize> fmt::Display for Vector<N>
where [u8; simd(N)]: Sized
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector<{}>(", N)?;
        for n in self.iter().take(10) {
            write!(f, "{:10.2e}", n)?;
        }
        if N > 10 {
            write!(f, " ...)")
        } else {
            write!(f, ")")
        }
    }
}
