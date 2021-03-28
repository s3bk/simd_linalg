use crate::{Matrix, MatrixT, Vector, simd, simd_size, zero};
use packed_simd::f32x8;
use unroll::unroll_for_loops;
use itertools::{Itertools, iproduct};
use crate::ffi::mkl::*;


pub fn matmul<const M: usize, const N: usize, const K: usize>
(a: &Matrix<M, K>, b: &Matrix<K, N>, c: &mut Matrix<M, N>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(K)]: Sized
{
    let lda = simd_size(M) as _;
    let ldb = simd_size(K) as _;
    let ldc = simd_size(M) as _;
    let alpha = 1.0;
    let beta = 0.0;
    
    unsafe {
        sgemm_direct(&(b'N' as i8), &(b'N' as i8),
            &(M as _), &(N as _), &(K as _),
            &alpha,
            a.buffer().as_ptr(), &lda,
            b.buffer().as_ptr(), &ldb,
            &beta,
            c.buffer_mut().as_mut_ptr(), &ldc,
            &1
        );
    }
}

pub fn matmul_t<const M: usize, const N: usize, const K: usize>
(a: &MatrixT<M, K>, b: &Matrix<K, N>, c: &mut Matrix<M, N>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(K)]: Sized
{
    use crate::mkl::*;

    let lda = simd_size(K) as _;
    let ldb = simd_size(K) as _;
    let ldc = simd_size(M) as _;
    let alpha = 1.0;
    let beta = 0.0;
    
    unsafe {
        sgemm_direct(&(b'T' as i8), &(b'N' as i8),
            &(M as _), &(N as _), &(K as _),
            &alpha,
            a.buffer().as_ptr(), &lda,
            b.buffer().as_ptr(), &ldb,
            &beta,
            c.buffer_mut().as_mut_ptr(), &ldc,
            &1
        );
    }
}

mod jit {
    use crate::ffi::mkl::*;
    use std::ffi::c_void;
    type KernelFn = unsafe extern "C" fn(
        arg1: *mut ::std::os::raw::c_void,
        arg2: *mut f32,
        arg3: *mut f32,
        arg4: *mut f32,
    );
    pub struct Kernel {
        jit: *mut c_void,
        f: KernelFn
    }
    pub struct Config {
        pub a_transposed: bool,
        pub m: usize,
        pub n: usize,
        pub k: usize,
        pub alpha: f32,
        pub beta: f32,
        pub lda: usize,
        pub ldb: usize,
        pub ldc: usize
    }
    impl Kernel {
        pub fn create(config: Config) -> Option<Self> {
            let mut jit = core::ptr::null_mut();
            unsafe {
                match mkl_cblas_jit_create_sgemm(&mut jit,
                    MKL_LAYOUT_MKL_COL_MAJOR,
                    if config.a_transposed { MKL_TRANSPOSE_MKL_TRANS } else { MKL_TRANSPOSE_MKL_NOTRANS },
                    MKL_TRANSPOSE_MKL_NOTRANS,
                    config.m as _, config.n as _, config.k as _,
                    config.alpha,
                    config.lda as _,
                    config.ldb as _,
                    config.beta,
                    config.ldc as _,
                ) {
                    mkl_jit_status_t_MKL_JIT_SUCCESS => {
                        Some(Kernel {
                            f: mkl_jit_get_sgemm_ptr(jit).unwrap(),
                            jit
                        })
                    },
                    _ => {
                        mkl_jit_destroy(jit);
                        None
                    }
                }
            }
        }
        pub unsafe fn exec(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
            let a = a.as_ptr() as *mut f32;
            let b = b.as_ptr() as *mut f32;
            let c = c.as_mut_ptr();
            (self.f)(self.jit, a, b, c);
        }
    }
    impl Drop for Kernel {
        fn drop(&mut self) {
            unsafe {
                crate::mkl::mkl_jit_destroy(self.jit);
            }
        }
    }
    unsafe impl Sync for Kernel {}
    unsafe impl Send for Kernel {}
}


pub fn matmul_jit_t<const M: usize, const N: usize, const K: usize>
() -> Option<impl Fn(&MatrixT<M, K>, &Matrix<K, N>, &mut Matrix<M, N>)>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(K)]: Sized
{
    unsafe {
        use jit::*;

        let kernel = Kernel::create(Config {
            a_transposed: true,
            m: M, n: N, k: K,
            alpha: 1.0, beta: 0.0,
            lda: simd_size(K),
            ldb: simd_size(K),
            ldc: simd_size(M)
        })?;

        Some(move |a: &MatrixT<M, K>, b: &Matrix<K, N>, c: &mut Matrix<M, N>| {
            kernel.exec(a.buffer(), b.buffer(), c.buffer_mut());
        })
    }
}

pub fn matmul_jit<const M: usize, const N: usize, const K: usize>
() -> Option<impl Fn(&Matrix<M, K>, &Matrix<K, N>, &mut Matrix<M, N>)>
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(K)]: Sized
{
    use jit::*;

    let kernel = Kernel::create(Config {
        a_transposed: false,
        m: M, n: N, k: K,
        alpha: 1.0, beta: 0.0,
        lda: simd_size(M),
        ldb: simd_size(K),
        ldc: simd_size(M)
    })?;

    Some(move |a: &Matrix<M, K>, b: &Matrix<K, N>, c: &mut Matrix<M, N>| {
        unsafe {
            kernel.exec(a.buffer(), b.buffer(), c.buffer_mut());
        }
    })
}
