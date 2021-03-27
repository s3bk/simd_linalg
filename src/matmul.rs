use crate::{Matrix, MatrixT, Vector, simd, simd_size, zero};
use packed_simd::f32x8;
use unroll::unroll_for_loops;
use itertools::{Itertools, iproduct};

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

pub fn matmul_block_t<const N: usize, const M: usize, const O: usize>
    (a: &MatrixT<N, M>, b: &Matrix<M, O>, c: &mut Matrix<N, O>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(O)]: Sized
{
    const B: usize = 8;
    
    for block_n in 0 .. N/B {
        for block_o in 0 .. O/B {
            for idx_n in 0 .. B {
                for idx_o in 0 .. B {
                    let n = B * block_n + idx_n;
                    let o = B * block_o + idx_o;
                    c[o][n] = a[n].dot(&b[o]);
                }
            }
        }
    }
    for n in 0 .. N {
        for o in B*(O/B) .. O {
            c[o][n] = a[n].dot(&b[o]);
        }
    }
    for n in B*(N/B) .. N {
        for o in 0 .. B*(O/B) {
            c[o][n] = a[n].dot(&b[o]);
        }
    }
}

const fn div_ceil8(a: usize) -> usize {
    a.saturating_add(7) >> 3
}
const fn div_floor4(a: usize) -> usize {
    a >> 2
}
const fn div_floor2(a: usize) -> usize {
    a >> 1
}
const fn mod2(a: usize) -> usize {
    a & 1
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

const fn split(n: usize, g: usize) -> usize {
    if n <= g {
        n
    } else if n <= 2 * g {
        n/2
    } else {
        g
    }
}

pub fn matmul_block<const N: usize, const M: usize, const K: usize>
    (a: &Matrix<M, K>, b: &Matrix<K, N>, c: &mut Matrix<M, N>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(K)]: Sized
{
    const B_N: usize = 3; // fixed
    const B_M: usize = 3*8; // fixed
    const G_N: usize = 16 * B_N;
    const G_M: usize = 16 * B_M;
    const G_K: usize = 128;

    let g_m = split(M, G_M);
    let g_n = split(N, G_N);
    let g_k = split(K, G_K);

    if g_k < K {
        for n in 0 .. N {
            for m in 0 .. M/8 {
                c.0[n].0[m] = f32x8::splat(0.0);
            }
        }
    }

    let iter = iproduct!(0 .. M/g_m, 0 .. N/g_n, 0 .. K/g_k, 0 .. g_n/B_N, 0 .. g_m/B_M);

    for (group_m, group_n, group_k, block_n, block_m) in iter {
        let idx_m = group_m * (g_m/B_M) + block_m;
        let off_n = group_n * (g_n/B_M) + block_n;
        let off_k = group_k * g_k;

        let a_0 = a[off_k].0[idx_m+0];
        let a_1 = a[off_k].0[idx_m+1];
        let a_2 = a[off_k].0[idx_m+2];

        let b_0 = f32x8::splat(b[off_n+0][off_k]);
        let mut c_0_0 = a_0 * b_0;
        let mut c_1_0 = a_1 * b_0;
        let mut c_2_0 = a_2 * b_0;

        let b_1 = f32x8::splat(b[off_n+1][off_k]);
        let mut c_0_1 = a_0 * b_1;
        let mut c_1_1 = a_1 * b_1;
        let mut c_2_1 = a_2 * b_1;

        let b_2 = f32x8::splat(b[off_n+2][off_k]);
        let mut c_0_2 = a_0 * b_2;
        let mut c_1_2 = a_1 * b_2;
        let mut c_2_2 = a_2 * b_2;

        for idx_k in 1 .. g_k {
            let k = off_k+idx_k;

            let a_0 = a[k].0[idx_m+0];
            let a_1 = a[k].0[idx_m+1];
            let a_2 = a[k].0[idx_m+2];

            let b_0 = f32x8::splat(b[off_n+0][k]);
            c_0_0 = a_0.mul_add(b_0, c_0_0);
            c_1_0 = a_1.mul_add(b_0, c_1_0);
            c_2_0 = a_2.mul_add(b_0, c_2_0);

            let b_1 = f32x8::splat(b[off_n+1][k]);
            c_0_1 = a_0.mul_add(b_1, c_0_1);
            c_1_1 = a_1.mul_add(b_1, c_1_1);
            c_2_1 = a_2.mul_add(b_1, c_2_1);

            let b_2 = f32x8::splat(b[off_n+2][k]);
            c_0_2 = a_0.mul_add(b_2, c_0_2);
            c_1_2 = a_1.mul_add(b_2, c_1_2);
            c_2_2 = a_2.mul_add(b_2, c_2_2);
        }

        c[off_n+0].0[idx_m+0] += c_0_0;
        c[off_n+0].0[idx_m+1] += c_1_0;
        c[off_n+0].0[idx_m+2] += c_2_0;
        c[off_n+1].0[idx_m+0] += c_0_1;
        c[off_n+1].0[idx_m+1] += c_1_1;
        c[off_n+1].0[idx_m+2] += c_2_1;
        c[off_n+2].0[idx_m+0] += c_0_2;
        c[off_n+2].0[idx_m+1] += c_1_2;
        c[off_n+2].0[idx_m+2] += c_2_2;
    }
}

#[unroll_for_loops]
pub fn matmul_block_4_6_8(a: &Matrix<4, 6>, b: &Matrix<6, 8>, c: &mut Matrix<4, 8>) 
{
    for o in 0 .. 2 {
        let off_o = 4 * o;

        let a_block = a[0].0[0];
        let mut acc0 = a_block * f32x8::splat(b[off_o+0][0]);
        let mut acc1 = a_block * f32x8::splat(b[off_o+1][0]);
        let mut acc2 = a_block * f32x8::splat(b[off_o+2][0]);
        let mut acc3 = a_block * f32x8::splat(b[off_o+3][0]);
        let mut acc4 = a_block * f32x8::splat(b[off_o+0][1]);
        let mut acc5 = a_block * f32x8::splat(b[off_o+1][1]);
        let mut acc6 = a_block * f32x8::splat(b[off_o+2][1]);
        let mut acc7 = a_block * f32x8::splat(b[off_o+3][1]);

        for m in 1 .. 3 {
            let m1 = 2*m;
            let a_block = a[m1].0[0];
            acc0 = a_block.mul_add(f32x8::splat(b[off_o+0][m1]), acc0);
            acc1 = a_block.mul_add(f32x8::splat(b[off_o+1][m1]), acc1);
            acc2 = a_block.mul_add(f32x8::splat(b[off_o+2][m1]), acc2);
            acc3 = a_block.mul_add(f32x8::splat(b[off_o+3][m1]), acc3);
            
            let m2 = m1 + 1;
            let a_block = a[m2].0[0];
            acc4 = a_block.mul_add(f32x8::splat(b[off_o+0][m2]), acc4);
            acc5 = a_block.mul_add(f32x8::splat(b[off_o+1][m2]), acc5);
            acc6 = a_block.mul_add(f32x8::splat(b[off_o+2][m2]), acc6);
            acc7 = a_block.mul_add(f32x8::splat(b[off_o+3][m2]), acc7);
        }

        c[off_o+0].0[0] = acc0 + acc4;
        c[off_o+1].0[0] = acc1 + acc5;
        c[off_o+2].0[0] = acc2 + acc6;
        c[off_o+3].0[0] = acc3 + acc7;
    }
}

#[cfg(feature="mm")]
pub fn matmul_matrixmultiply<const N: usize, const M: usize, const O: usize>
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

#[cfg(feature="mm")]
pub fn matmul_matrixmultiply_t<const N: usize, const M: usize, const O: usize>
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

#[cfg(feature="blis")]
pub fn matmul_blis_t<const M: usize, const N: usize, const K: usize>
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
#[cfg(feature="blis")]
pub fn matmul_blis<const M: usize, const N: usize, const K: usize>
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

#[cfg(feature="mkl")]
pub fn matmul_mkl<const M: usize, const N: usize, const K: usize>
(a: &Matrix<M, K>, b: &Matrix<K, N>, c: &mut Matrix<M, N>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(K)]: Sized
{
    use crate::mkl::*;

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

#[cfg(feature="mkl")]
pub fn matmul_mkl_t<const M: usize, const N: usize, const K: usize>
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

#[cfg(feature="mkl_jit")]
mod jit {
    use crate::mkl::*;
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

#[cfg(feature="mkl_jit")]
pub fn matmul_mkl_jit_t<const M: usize, const N: usize, const K: usize>
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

#[cfg(feature="mkl_jit")]
pub fn matmul_mkl_jit<const M: usize, const N: usize, const K: usize>
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
