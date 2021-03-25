use crate::{Matrix, MatrixT, Vector, simd, simd_size, zero};
use packed_simd::f32x8;

pub fn matmul_naive<const N: usize, const M: usize, const O: usize>
    (a: &MatrixT<N, M>, b: &Matrix<M, O>, c: &mut Matrix<N, O>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(O)]: Sized
{
    for n in 0 .. N {
        for o in 0 .. O {
            c[o][n] = a[n].dot(&b[o]);
        }
    }
}

pub fn matmul_block<const N: usize, const M: usize, const O: usize>
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

pub fn matmul_block_t<const N: usize, const M: usize, const O: usize>
    (a: &Matrix<N, M>, b: &Matrix<M, O>, c: &mut Matrix<N, O>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(O)]: Sized
{
    const B: usize = 8;
    
    for block_n in 0 .. (N+B-1)/B {
        for block_o in 0 .. O/4 {
            let off_o = 4 * block_o;
            let mut acc0 = f32x8::splat(0.0);
            let mut acc1 = f32x8::splat(0.0);
            let mut acc2 = f32x8::splat(0.0);
            let mut acc3 = f32x8::splat(0.0);
            let mut acc4 = f32x8::splat(0.0);
            let mut acc5 = f32x8::splat(0.0);
            let mut acc6 = f32x8::splat(0.0);
            let mut acc7 = f32x8::splat(0.0);

            for m in 0 .. (M+1)/2 {
                let m1 = 2*m;
                let a_block = a[m1].0[block_n];
                acc0 += a_block * f32x8::splat(b[off_o+0][m1]);
                acc1 += a_block * f32x8::splat(b[off_o+1][m1]);
                acc2 += a_block * f32x8::splat(b[off_o+2][m1]);
                acc3 += a_block * f32x8::splat(b[off_o+3][m1]);
                
                let m2 = m1 + 1;
                if m2 < M {
                    let a_block = a[m2].0[block_n];
                    acc4 += a_block * f32x8::splat(b[off_o+0][m2]);
                    acc5 += a_block * f32x8::splat(b[off_o+1][m2]);
                    acc6 += a_block * f32x8::splat(b[off_o+2][m2]);
                    acc7 += a_block * f32x8::splat(b[off_o+3][m2]);
                }
            }
            c[off_o+0].0[block_n] = acc0 + acc4;
            c[off_o+1].0[block_n] = acc1 + acc5;
            c[off_o+2].0[block_n] = acc2 + acc6;
            c[off_o+3].0[block_n] = acc3 + acc7;
        }
    }
    if O % 4 > 0 {
        let off_o = 4*(O/4);
        for block_n in 0 .. (N+B-1)/B {
            let mut acc0 = f32x8::splat(0.0);
            let mut acc1 = f32x8::splat(0.0);
            let mut acc2 = f32x8::splat(0.0);

            for m in 0 .. M {
                let a_block = a[m].0[block_n];

                if O%4 > 0 {
                    acc0 += a_block * f32x8::splat(b[off_o+0][m]);
                }
                if O%4 > 1 {
                    acc1 += a_block * f32x8::splat(b[off_o+1][m]);
                }
                if O%4 > 2 {
                    acc2 += a_block * f32x8::splat(b[off_o+2][m]);
                }
            }
            if O%4 > 0 {
                c[off_o+0].0[block_n] = acc0;
            }
            if O%4 > 1 {
                c[off_o+1].0[block_n] = acc1;
            }
            if O%4 > 2 {
                c[off_o+2].0[block_n] = acc2;
            }
        }
    }
}

#[cfg(feature="mm")]
pub fn matmul_matrixmultiply<const N: usize, const M: usize, const O: usize>
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
pub fn matmul_blis<const M: usize, const N: usize, const K: usize>
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

#[cfg(feature="mkl")]
pub fn matmul_mkl<const M: usize, const N: usize, const K: usize>
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
            (self.f)
            (self.jit,
                a.as_ptr() as *mut f32,
                b.as_ptr() as *mut f32,
                c.as_mut_ptr()
            );
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
pub fn matmul_mkl_jit<const M: usize, const N: usize, const K: usize>
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
pub fn matmul_mkl_jit_t<const M: usize, const N: usize, const K: usize>
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
