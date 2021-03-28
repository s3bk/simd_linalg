use crate::{Matrix, MatrixT, Vector, simd, simd_size, zero};
use packed_simd::f32x8;
use unroll::unroll_for_loops;
use itertools::{Itertools, iproduct};

const fn split(n: usize, g: usize) -> usize {
    if n <= g {
        n
    } else if n <= 2 * g {
        n/2
    } else {
        g
    }
}

/* what I want ...
for (group_m, group_n, group_k, block_n, block_m) in iter {
    let idx_m = group_m * (g_m/8) + block_m;
    let off_n = group_n * g_n + block_n;
    let off_k = group_k * g_k;

    for!($m in 0..3 {
        let a_$m = a[off_k].0[idx_m+$m];
    });

    for $n in 0..3 {
        let b_$n = f32x8::splat(b[off_n+$n][off_k]);

        for $m in 0..3 {
            let c_$m#_$n = a_$m * b_ $n;
        }
    }

    for idx_k in 1 .. g_k {
        let k = off_k+idx_k;

        for $m in 0..3 {
            let a_$m = a[k].0[idx_m+$m];
        }

        for $n in 0..3 {
            let b_$n = f32x8::splat(b[off_n+$n][k]);
            for $m in 0..3 {
                c_$m#_$n = a_$m#.mul_add(b_$n, c_$m#_$n);
            }
        }
    }

    for $n in 0..3 {
        for $m in 0..3 {
            c[off_n+$n].0[idx_m+$m] += c_$m_$n;
        }
    }
}
*/

pub fn matmul<const N: usize, const M: usize, const K: usize>
    (a: &Matrix<M, K>, b: &Matrix<K, N>, c: &mut Matrix<M, N>) 
where [u8; simd(N)]: Sized, [u8; simd(M)]: Sized, [u8; simd(K)]: Sized
{
    const B_N: usize = 3; // fixed
    const B_M: usize = 3*8; // fixed
    const G_N: usize = 3 * 32;
    const G_M: usize = 3 * 32;
    const G_K: usize = 128;

    let g_m = M; //split(M, G_M);
    let g_n = N; //split(N, G_N);
    let g_k = K; //split(K, G_K);


    assert_eq!(M % g_m, 0);
    assert_eq!(N % g_n, 0);

    c.fill_val(0.0);

    /*
    let iter = iproduct!(0 .. M/g_m, 0 .. N/g_n, 0 .. K/g_k, 0 .. g_n/B_N, 0 .. g_m/B_M);
    for (group_m, group_n, group_k, block_n, block_m) in iter {
        let idx_m = group_m * (g_m/8) + block_m;
        let off_n = group_n * g_n + block_n;
        let off_k = group_k * g_k;
    */
    let iter = iproduct!(0..M/B_M, 0..N/B_N);
    for (block_m, block_n) in iter {
        let idx_m = block_m * B_M/8;
        let off_n = block_n * B_N;
        let off_k = 0;

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

        c[off_n+0].0[idx_m+0] = c_0_0;
        c[off_n+0].0[idx_m+1] = c_1_0;
        c[off_n+0].0[idx_m+2] = c_2_0;
        c[off_n+1].0[idx_m+0] = c_0_1;
        c[off_n+1].0[idx_m+1] = c_1_1;
        c[off_n+1].0[idx_m+2] = c_2_1;
        c[off_n+2].0[idx_m+0] = c_0_2;
        c[off_n+2].0[idx_m+1] = c_1_2;
        c[off_n+2].0[idx_m+2] = c_2_2;
    }
}

#[unroll_for_loops]
pub fn matmul_4_6_8(a: &Matrix<4, 6>, b: &Matrix<6, 8>, c: &mut Matrix<4, 8>) 
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

pub fn matmul_t<const N: usize, const M: usize, const O: usize>
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
