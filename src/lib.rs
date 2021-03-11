#![feature(const_generics, const_evaluatable_checked)]
#![feature(new_uninit)]
#![allow(incomplete_features)]
use packed_simd::{f32x8};
use std::ops::{
    Add, Mul, Sub, AddAssign, MulAssign, SubAssign,
    Deref, DerefMut, Index, IndexMut
};
use std::fmt;

/// calculate the number of `f32x8`'s needed to hold `n` f32s
pub const fn simd(n: usize) -> usize {
    (n + 8 - 1) / 8
}

/// Array of N `f32`s in SIMD layout (padded).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vector<const N: usize>([f32x8; simd(N)])
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

impl<const N: usize> Vector<N> where
    [u8; simd(N)]: Sized
{
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
    pub fn dot(&self, rhs: &Self) -> f32 {
        self.0.iter().zip(rhs.0.iter())
            .map(|(a, b)| *a * *b)
            .fold(f32x8::splat(0.0), f32x8::add)
            .sum()
    }

    /// returns the index of the highest value and the value itself
    pub fn max_idx(&self) -> (usize, f32) {
        let mut max_val = -std::f32::INFINITY;
        let mut max_idx = 0;
        for (i, &v) in self.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        (max_idx, max_val)
    }

    /// create a Vector with every element set to the given value
    pub fn splat(value: f32) -> Self {
        let mut v = Vector::null();
        v.fill(std::iter::repeat(value));
        v
    }

    /// Fill the value from the given iterator
    pub fn fill(&mut self, values: impl Iterator<Item=f32>) {
        self.iter_mut().zip(values).for_each(|(o, i)| *o = i)
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
}
impl<const N: usize> Deref for Vector<N> where
    [u8; simd(N)]: Sized
{
    type Target = [f32; N];
    fn deref(&self) -> &[f32; N] {
        unsafe {
            &*(self as *const _ as *const [f32; N])
        }
    }
}
impl<const N: usize> DerefMut for Vector<N> where
    [u8; simd(N)]: Sized
{
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

/// A Matrix of `N` columns and `M` rows, each column is padded.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct Matrix<const N: usize, const M: usize>([Vector<N>; M])
where [u8; simd(N)]: Sized;

impl<const N: usize, const M: usize> Matrix<N, M>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    /// create a Box<Self> full of zeros
    pub fn null() -> Box<Self> {
        unsafe {
            Box::new_zeroed().assume_init()
        }
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

/// Linear projection with bias
#[derive(Clone, Debug)]
#[repr(C)]
pub struct Linear<const N: usize, const M: usize>
    where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    pub bias: Vector<M>,
    pub weight: Matrix<N, M>
}

impl<const N: usize, const M: usize> Linear<N, M>
    where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    /// create a `Box<Self>` full of zeros
    pub fn null() -> Box<Self> {
        unsafe {
            Box::new_zeroed().assume_init()
        }
    }

    /// `x -> self.weight * x + self.bias`
    pub fn transform(&self, x: &Vector<N>) -> Vector<M> {
        &self.weight * x + self.bias
    }
}
