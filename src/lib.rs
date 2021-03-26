#![feature(const_generics, const_evaluatable_checked)]
#![cfg_attr(feature="alloc", feature(new_uninit))]
#![cfg_attr(feature="mkl_jit", feature(once_cell))]
#![allow(incomplete_features)]
#![no_std]

#[cfg(feature="alloc")]
extern crate alloc;

#[cfg(feature="alloc")]
use alloc::boxed::Box;

#[cfg(feature="std")]
extern crate std;

#[cfg(feature="mkl")]
#[allow(dead_code, non_camel_case_types, non_upper_case_globals, improper_ctypes, non_snake_case)]
mod mkl;

#[cfg(feature="blis")]
mod blis;

#[cfg(feature="mkl")]
pub fn mkl_init() {
    unsafe {
        mkl::MKL_Set_Threading_Layer(mkl::MKL_THREADING_SEQUENTIAL as _);
    }
}
#[cfg(feature="blis")]
pub fn blis_init() {
    unsafe {
        blis::bli_thread_set_num_threads(1);
    }
}
use core::mem::MaybeUninit;

/// Marker trait for types that can be initialized with zeros.
pub unsafe trait ZeroInit: Sized {}
unsafe impl<T: ZeroInit, const N: usize> ZeroInit for [T; N] {}

pub fn zero<T: ZeroInit>() -> T {
    unsafe {
        MaybeUninit::zeroed().assume_init()
    }
}

#[cfg(feature="alloc")]
pub fn zero_box<T: ZeroInit>() -> Box<T> {
    unsafe {
        Box::new_zeroed().assume_init()
    }
}

/// calculate the number of `f32x8`'s needed to hold `n` f32s
pub const fn simd(n: usize) -> usize {
    (n + 8 - 1) / 8
}
pub const fn simd_size(n: usize) -> usize {
    8 * simd(n)
}

mod vector;
mod matrix;
mod matmul;

pub use vector::*;
pub use matrix::*;
pub use matmul::*;
