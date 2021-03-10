#![feature(const_generics, const_evaluatable_checked)]
#![feature(new_uninit)]
#![allow(incomplete_features)]
use packed_simd::{f32x8};
use std::ops::{Add, Mul, Sub, Deref, DerefMut, Index, IndexMut};

pub const fn simd(n: usize) -> usize {
    (n + 8 - 1) / 8
}

pub const fn add(a: usize, b: usize) -> usize {
    a + b
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Vector<const N: usize>([f32x8; simd(N)])
where [u8; simd(N)]: Sized;


macro_rules! vec_op {
    ($op:ident, $fn:ident) => {
        impl<const N: usize> $op for Vector<N> where
            [u8; simd(N)]: Sized
        {
            type Output = Self;
            fn $fn(mut self, rhs: Self) -> Self {
                for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
                    *a = a.$fn(*b);
                }
                self
            }
        }
        impl<'a, const N: usize> $op<&'a Self> for Vector<N> where 
            [u8; simd(N)]: Sized
        {
            type Output = Self;
            fn $fn(mut self, rhs: &'a Self) -> Self {
                for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
                    *a = a.$fn(*b);
                }
                self
            }
        }
    }
}

vec_op!(Add, add);
vec_op!(Sub, sub);
vec_op!(Mul, mul);

impl<const N: usize> Vector<N> where
    [u8; simd(N)]: Sized
{
    pub fn null() -> Self {
        Vector([f32x8::splat(0.0); simd(N)])
    }
    pub fn tanh(mut self) -> Self {
        self.0.iter_mut().for_each(|x| *x = x.tanh());
        self
    }
    pub fn set(&mut self, idx: usize, val: f32) {
        assert!(idx < N);
        self[idx] = val;
    }
    pub fn dot(&self, rhs: &Self) -> f32 {
        self.0.iter().zip(rhs.0.iter())
            .map(|(a, b)| *a * *b)
            .fold(f32x8::splat(0.0), f32x8::add)
            .sum()
    }
    pub fn fill(&mut self, values: impl Iterator<Item=f32>) {
        self.iter_mut().zip(values).for_each(|(o, i)| *o = i)
    }
    pub fn concat<const M: usize>(&self, rhs: &Vector<M>) -> Vector<{N + M}>
    where [u8; simd(M)]: Sized, [u8; simd(N+M)]: Sized
    {
        let mut out = Vector::null();
        let (a, b) = out.split_at_mut(N);
        a.copy_from_slice(&**self);
        b.copy_from_slice(&**rhs);
        out
    }
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

#[repr(transparent)]
pub struct Matrix<const N: usize, const M: usize>([Vector<N>; M])
where [u8; simd(N)]: Sized;

impl<const N: usize, const M: usize> Matrix<N, M>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized
{
    pub fn null() -> Box<Self> {
        unsafe {
            Box::new_zeroed().assume_init()
        }
    }
    pub fn fill(&mut self, mut values: impl Iterator<Item=f32>) {
        self.0.iter_mut().for_each(|v| v.fill(&mut values))
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
    pub fn null() -> Box<Self> {
        unsafe {
            Box::new_zeroed().assume_init()
        }
    }

    pub fn transform(&self, x: &Vector<N>) -> Vector<M> {
        &self.weight * x + self.bias
    }
}
