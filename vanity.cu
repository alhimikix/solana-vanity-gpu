// GPU Solana vanity address grinder (prefix / suffix / both, multi-pattern, multi-GPU).
//
// Pipeline per batch of NK keys:
//   k_gen   : seed = ChaCha12(run key, index) -> SHA-512 -> clamp -> fixed-base comb with
//             W-bit signed windows over a precomputed table -> projective (X, Y, Z)
//   k_inv   : Montgomery batch inversion over MC keys per thread, affine y + sign of x,
//             cheap gate (suffix low-byte bitmap / prefix top-64-bit ranges) -> candidates
//   k_check : exact base58 check of candidates against the patterns -> matches
// The host re-derives every match from its index, recomputes the public key with an
// independent reference implementation and re-checks the pattern before writing it.
//
// Build: nvcc -O3 -arch=sm_86 -o vanity_gpu vanity.cu   (see build.bat / build.sh)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

typedef int32_t  i32;
typedef int64_t  i64;
typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while(0)

#define HD __host__ __device__ __forceinline__

#ifndef GEN_THREADS
#define GEN_THREADS 256 // k_gen block size
#endif
#ifndef GEN_MINB
#define GEN_MINB 2      // __launch_bounds__ min blocks/SM for k_gen: 256 x 2 -> <= 128 regs
#endif

// ============================================================================
// Field arithmetic GF(2^255-19): ref10 radix-2^25.5, 10 signed i32 limbs.
// Shared by the GPU hot path and the host reference.
// ============================================================================

typedef i32 gf[10];

HD void fe_copy(gf o, const gf a) { for (int i = 0; i < 10; i++) o[i] = a[i]; }
HD void fe_0(gf o) { for (int i = 0; i < 10; i++) o[i] = 0; }
HD void fe_1(gf o) { fe_0(o); o[0] = 1; }
HD void fe_add(gf h, const gf f, const gf g) { for (int i = 0; i < 10; i++) h[i] = f[i] + g[i]; }
HD void fe_sub(gf h, const gf f, const gf g) { for (int i = 0; i < 10; i++) h[i] = f[i] - g[i]; }
HD void fe_neg(gf h, const gf f) { for (int i = 0; i < 10; i++) h[i] = -f[i]; }
HD void fe_d2(gf o) {   // 2*d
    o[0] = -21827239; o[1] = -5839606; o[2] = -30745221; o[3] = 13898782; o[4] = 229458;
    o[5] = 15978800; o[6] = -12551817; o[7] = -6495438; o[8] = 29715968; o[9] = 9444199;
}

// h = f*g (ref10 fe_mul)
HD void fe_mul(gf h, const gf f, const gf g) {
    i32 f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    i32 f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    i32 g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3], g4 = g[4];
    i32 g5 = g[5], g6 = g[6], g7 = g[7], g8 = g[8], g9 = g[9];
    i32 g1_19 = 19 * g1, g2_19 = 19 * g2, g3_19 = 19 * g3, g4_19 = 19 * g4;
    i32 g5_19 = 19 * g5, g6_19 = 19 * g6, g7_19 = 19 * g7, g8_19 = 19 * g8, g9_19 = 19 * g9;
    i32 f1_2 = 2 * f1, f3_2 = 2 * f3, f5_2 = 2 * f5, f7_2 = 2 * f7, f9_2 = 2 * f9;

    i64 h0 = (i64)f0*g0 + (i64)f1_2*g9_19 + (i64)f2*g8_19 + (i64)f3_2*g7_19 + (i64)f4*g6_19 + (i64)f5_2*g5_19 + (i64)f6*g4_19 + (i64)f7_2*g3_19 + (i64)f8*g2_19 + (i64)f9_2*g1_19;
    i64 h1 = (i64)f0*g1 + (i64)f1*g0    + (i64)f2*g9_19 + (i64)f3*g8_19   + (i64)f4*g7_19 + (i64)f5*g6_19   + (i64)f6*g5_19 + (i64)f7*g4_19   + (i64)f8*g3_19 + (i64)f9*g2_19;
    i64 h2 = (i64)f0*g2 + (i64)f1_2*g1  + (i64)f2*g0    + (i64)f3_2*g9_19 + (i64)f4*g8_19 + (i64)f5_2*g7_19 + (i64)f6*g6_19 + (i64)f7_2*g5_19 + (i64)f8*g4_19 + (i64)f9_2*g3_19;
    i64 h3 = (i64)f0*g3 + (i64)f1*g2    + (i64)f2*g1    + (i64)f3*g0      + (i64)f4*g9_19 + (i64)f5*g8_19   + (i64)f6*g7_19 + (i64)f7*g6_19   + (i64)f8*g5_19 + (i64)f9*g4_19;
    i64 h4 = (i64)f0*g4 + (i64)f1_2*g3  + (i64)f2*g2    + (i64)f3_2*g1    + (i64)f4*g0    + (i64)f5_2*g9_19 + (i64)f6*g8_19 + (i64)f7_2*g7_19 + (i64)f8*g6_19 + (i64)f9_2*g5_19;
    i64 h5 = (i64)f0*g5 + (i64)f1*g4    + (i64)f2*g3    + (i64)f3*g2      + (i64)f4*g1    + (i64)f5*g0      + (i64)f6*g9_19 + (i64)f7*g8_19   + (i64)f8*g7_19 + (i64)f9*g6_19;
    i64 h6 = (i64)f0*g6 + (i64)f1_2*g5  + (i64)f2*g4    + (i64)f3_2*g3    + (i64)f4*g2    + (i64)f5_2*g1    + (i64)f6*g0    + (i64)f7_2*g9_19 + (i64)f8*g8_19 + (i64)f9_2*g7_19;
    i64 h7 = (i64)f0*g7 + (i64)f1*g6    + (i64)f2*g5    + (i64)f3*g4      + (i64)f4*g3    + (i64)f5*g2      + (i64)f6*g1    + (i64)f7*g0      + (i64)f8*g9_19 + (i64)f9*g8_19;
    i64 h8 = (i64)f0*g8 + (i64)f1_2*g7  + (i64)f2*g6    + (i64)f3_2*g5    + (i64)f4*g4    + (i64)f5_2*g3    + (i64)f6*g2    + (i64)f7_2*g1    + (i64)f8*g0    + (i64)f9_2*g9_19;
    i64 h9 = (i64)f0*g9 + (i64)f1*g8    + (i64)f2*g7    + (i64)f3*g6      + (i64)f4*g5    + (i64)f5*g4      + (i64)f6*g3    + (i64)f7*g2      + (i64)f8*g1    + (i64)f9*g0;

    i64 c0 = (h0 + (1LL << 25)) >> 26; h1 += c0; h0 -= c0 << 26;
    i64 c4 = (h4 + (1LL << 25)) >> 26; h5 += c4; h4 -= c4 << 26;
    i64 c1 = (h1 + (1LL << 24)) >> 25; h2 += c1; h1 -= c1 << 25;
    i64 c5 = (h5 + (1LL << 24)) >> 25; h6 += c5; h5 -= c5 << 25;
    i64 c2 = (h2 + (1LL << 25)) >> 26; h3 += c2; h2 -= c2 << 26;
    i64 c6 = (h6 + (1LL << 25)) >> 26; h7 += c6; h6 -= c6 << 26;
    i64 c3 = (h3 + (1LL << 24)) >> 25; h4 += c3; h3 -= c3 << 25;
    i64 c7 = (h7 + (1LL << 24)) >> 25; h8 += c7; h7 -= c7 << 25;
    c4 = (h4 + (1LL << 25)) >> 26; h5 += c4; h4 -= c4 << 26;
    i64 c8 = (h8 + (1LL << 25)) >> 26; h9 += c8; h8 -= c8 << 26;
    i64 c9 = (h9 + (1LL << 24)) >> 25; h0 += c9 * 19; h9 -= c9 << 25;
    c0 = (h0 + (1LL << 25)) >> 26; h1 += c0; h0 -= c0 << 26;

    h[0] = (i32)h0; h[1] = (i32)h1; h[2] = (i32)h2; h[3] = (i32)h3; h[4] = (i32)h4;
    h[5] = (i32)h5; h[6] = (i32)h6; h[7] = (i32)h7; h[8] = (i32)h8; h[9] = (i32)h9;
}

// h = f^2 (ref10 fe_sq)
HD void fe_sq(gf h, const gf f) {
    i32 f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    i32 f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    i32 f0_2 = 2*f0, f1_2 = 2*f1, f2_2 = 2*f2, f3_2 = 2*f3, f4_2 = 2*f4;
    i32 f5_2 = 2*f5, f6_2 = 2*f6, f7_2 = 2*f7;
    i32 f5_38 = 38*f5, f6_19 = 19*f6, f7_38 = 38*f7, f8_19 = 19*f8, f9_38 = 38*f9;

    i64 h0 = (i64)f0*f0     + (i64)f1_2*f9_38 + (i64)f2_2*f8_19 + (i64)f3_2*f7_38 + (i64)f4_2*f6_19 + (i64)f5*f5_38;
    i64 h1 = (i64)f0_2*f1   + (i64)f2*f9_38   + (i64)f3_2*f8_19 + (i64)f4*f7_38   + (i64)f5_2*f6_19;
    i64 h2 = (i64)f0_2*f2   + (i64)f1_2*f1    + (i64)f3_2*f9_38 + (i64)f4_2*f8_19 + (i64)f5_2*f7_38 + (i64)f6*f6_19;
    i64 h3 = (i64)f0_2*f3   + (i64)f1_2*f2    + (i64)f4*f9_38   + (i64)f5_2*f8_19 + (i64)f6*f7_38;
    i64 h4 = (i64)f0_2*f4   + (i64)f1_2*f3_2  + (i64)f2*f2      + (i64)f5_2*f9_38 + (i64)f6_2*f8_19 + (i64)f7*f7_38;
    i64 h5 = (i64)f0_2*f5   + (i64)f1_2*f4    + (i64)f2_2*f3    + (i64)f6*f9_38   + (i64)f7_2*f8_19;
    i64 h6 = (i64)f0_2*f6   + (i64)f1_2*f5_2  + (i64)f2_2*f4    + (i64)f3_2*f3    + (i64)f7_2*f9_38 + (i64)f8*f8_19;
    i64 h7 = (i64)f0_2*f7   + (i64)f1_2*f6    + (i64)f2_2*f5    + (i64)f3_2*f4    + (i64)f8*f9_38;
    i64 h8 = (i64)f0_2*f8   + (i64)f1_2*f7_2  + (i64)f2_2*f6    + (i64)f3_2*f5_2  + (i64)f4*f4      + (i64)f9*f9_38;
    i64 h9 = (i64)f0_2*f9   + (i64)f1_2*f8    + (i64)f2_2*f7    + (i64)f3_2*f6    + (i64)f4_2*f5;

    i64 c0 = (h0 + (1LL << 25)) >> 26; h1 += c0; h0 -= c0 << 26;
    i64 c4 = (h4 + (1LL << 25)) >> 26; h5 += c4; h4 -= c4 << 26;
    i64 c1 = (h1 + (1LL << 24)) >> 25; h2 += c1; h1 -= c1 << 25;
    i64 c5 = (h5 + (1LL << 24)) >> 25; h6 += c5; h5 -= c5 << 25;
    i64 c2 = (h2 + (1LL << 25)) >> 26; h3 += c2; h2 -= c2 << 26;
    i64 c6 = (h6 + (1LL << 25)) >> 26; h7 += c6; h6 -= c6 << 26;
    i64 c3 = (h3 + (1LL << 24)) >> 25; h4 += c3; h3 -= c3 << 25;
    i64 c7 = (h7 + (1LL << 24)) >> 25; h8 += c7; h7 -= c7 << 25;
    c4 = (h4 + (1LL << 25)) >> 26; h5 += c4; h4 -= c4 << 26;
    i64 c8 = (h8 + (1LL << 25)) >> 26; h9 += c8; h8 -= c8 << 26;
    i64 c9 = (h9 + (1LL << 24)) >> 25; h0 += c9 * 19; h9 -= c9 << 25;
    c0 = (h0 + (1LL << 25)) >> 26; h1 += c0; h0 -= c0 << 26;

    h[0] = (i32)h0; h[1] = (i32)h1; h[2] = (i32)h2; h[3] = (i32)h3; h[4] = (i32)h4;
    h[5] = (i32)h5; h[6] = (i32)h6; h[7] = (i32)h7; h[8] = (i32)h8; h[9] = (i32)h9;
}

// Fully reduce to canonical limbs (0 <= value < p), each limb non-negative and in range.
HD void fe_canon(i32 h[10]) {
    i32 q = (19 * h[9] + (1 << 24)) >> 25;
    q = (h[0] + q) >> 26; q = (h[1] + q) >> 25; q = (h[2] + q) >> 26; q = (h[3] + q) >> 25;
    q = (h[4] + q) >> 26; q = (h[5] + q) >> 25; q = (h[6] + q) >> 26; q = (h[7] + q) >> 25;
    q = (h[8] + q) >> 26; q = (h[9] + q) >> 25;
    h[0] += 19 * q;
    i32 c;
    c = h[0] >> 26; h[1] += c; h[0] -= c << 26;
    c = h[1] >> 25; h[2] += c; h[1] -= c << 25;
    c = h[2] >> 26; h[3] += c; h[2] -= c << 26;
    c = h[3] >> 25; h[4] += c; h[3] -= c << 25;
    c = h[4] >> 26; h[5] += c; h[4] -= c << 26;
    c = h[5] >> 25; h[6] += c; h[5] -= c << 25;
    c = h[6] >> 26; h[7] += c; h[6] -= c << 26;
    c = h[7] >> 25; h[8] += c; h[7] -= c << 25;
    c = h[8] >> 26; h[9] += c; h[8] -= c << 26;
    c = h[9] >> 25;             h[9] -= c << 25;
}

// Canonical 32-byte little-endian encoding as 8 little-endian u32 words.
HD void fe_pack_words(u32 w[8], const gf f) {
    i32 h[10];
    for (int i = 0; i < 10; i++) h[i] = f[i];
    fe_canon(h);
    w[0] = (u32)h[0] | ((u32)h[1] << 26);
    w[1] = ((u32)h[1] >> 6)  | ((u32)h[2] << 19);
    w[2] = ((u32)h[2] >> 13) | ((u32)h[3] << 13);
    w[3] = ((u32)h[3] >> 19) | ((u32)h[4] << 6);
    w[4] = (u32)h[5] | ((u32)h[6] << 25);
    w[5] = ((u32)h[6] >> 7)  | ((u32)h[7] << 19);
    w[6] = ((u32)h[7] >> 13) | ((u32)h[8] << 12);
    w[7] = ((u32)h[8] >> 20) | ((u32)h[9] << 6);
}

// Parity of the canonical value (low bit), without a full pack.
HD u32 fe_parity(const gf h) {
    i32 q = (19 * h[9] + (1 << 24)) >> 25;
    q = (h[0] + q) >> 26; q = (h[1] + q) >> 25; q = (h[2] + q) >> 26; q = (h[3] + q) >> 25;
    q = (h[4] + q) >> 26; q = (h[5] + q) >> 25; q = (h[6] + q) >> 26; q = (h[7] + q) >> 25;
    q = (h[8] + q) >> 26; q = (h[9] + q) >> 25;
    return (u32)(h[0] + 19 * q) & 1;
}

// Unpack 8 little-endian words (canonical, bit 255 ignored) into limbs.
HD void fe_unpack_words(gf h, const u32 w[8]) {
    h[0] = (i32)(w[0] & 0x3ffffff);
    h[1] = (i32)(((w[0] >> 26) | (w[1] << 6)) & 0x1ffffff);
    h[2] = (i32)(((w[1] >> 19) | (w[2] << 13)) & 0x3ffffff);
    h[3] = (i32)(((w[2] >> 13) | (w[3] << 19)) & 0x1ffffff);
    h[4] = (i32)((w[3] >> 6) & 0x3ffffff);
    h[5] = (i32)(w[4] & 0x1ffffff);
    h[6] = (i32)(((w[4] >> 25) | (w[5] << 7)) & 0x3ffffff);
    h[7] = (i32)(((w[5] >> 19) | (w[6] << 13)) & 0x1ffffff);
    h[8] = (i32)(((w[6] >> 12) | (w[7] << 20)) & 0x3ffffff);
    h[9] = (i32)((w[7] >> 6) & 0x1ffffff);
}

HD void fe_invert(gf out, const gf z) {
    gf t0, t1, t2, t3;
    int i;
    fe_sq(t0, z);
    fe_sq(t1, t0); fe_sq(t1, t1);
    fe_mul(t1, z, t1);
    fe_mul(t0, t0, t1);
    fe_sq(t2, t0);
    fe_mul(t1, t2, t1);
    fe_sq(t2, t1); for (i = 1; i < 5; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);
    fe_sq(t2, t1); for (i = 1; i < 10; i++) fe_sq(t2, t2);
    fe_mul(t2, t2, t1);
    fe_sq(t3, t2); for (i = 1; i < 20; i++) fe_sq(t3, t3);
    fe_mul(t2, t3, t2);
    fe_sq(t2, t2); for (i = 1; i < 10; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);
    fe_sq(t2, t1); for (i = 1; i < 50; i++) fe_sq(t2, t2);
    fe_mul(t2, t2, t1);
    fe_sq(t3, t2); for (i = 1; i < 100; i++) fe_sq(t3, t3);
    fe_mul(t2, t3, t2);
    fe_sq(t2, t2); for (i = 1; i < 50; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);
    fe_sq(t1, t1); for (i = 1; i < 5; i++) fe_sq(t1, t1);
    fe_mul(out, t1, t0);
}

// ============================================================================
// Reference group operations (table generation + host check). Extended coords.
// ============================================================================

struct ge { gf X, Y, Z, T; };

HD void ge_add(ge &p, const ge &q) {   // p += q (complete twisted Edwards addition)
    gf a, b, c, d, t, e, f, g, h, d2;
    fe_sub(a, p.Y, p.X); fe_sub(t, q.Y, q.X); fe_mul(a, a, t);
    fe_add(b, p.X, p.Y); fe_add(t, q.X, q.Y); fe_mul(b, b, t);
    fe_d2(d2); fe_mul(c, p.T, q.T); fe_mul(c, c, d2);
    fe_mul(d, p.Z, q.Z); fe_add(d, d, d);
    fe_sub(e, b, a); fe_sub(f, d, c); fe_add(g, d, c); fe_add(h, b, a);
    fe_mul(p.X, e, f); fe_mul(p.Y, h, g); fe_mul(p.Z, g, f); fe_mul(p.T, e, h);
}

HD void ge_identity(ge &p) { fe_0(p.X); fe_1(p.Y); fe_1(p.Z); fe_0(p.T); }

HD void ge_cswap(ge &p, ge &q, int b) {
    i32 m = -b;
    for (int i = 0; i < 10; i++) {
        i32 t;
        t = m & (p.X[i] ^ q.X[i]); p.X[i] ^= t; q.X[i] ^= t;
        t = m & (p.Y[i] ^ q.Y[i]); p.Y[i] ^= t; q.Y[i] ^= t;
        t = m & (p.Z[i] ^ q.Z[i]); p.Z[i] ^= t; q.Z[i] ^= t;
        t = m & (p.T[i] ^ q.T[i]); p.T[i] ^= t; q.T[i] ^= t;
    }
}

// r = s * q, s little-endian bytes, nbits low bits used (TweetNaCl-style ladder).
HD void ge_scalarmult(ge &r, ge q, const u8 *s, int nbits) {
    ge_identity(r);
    for (int i = nbits - 1; i >= 0; i--) {
        int b = (s[i >> 3] >> (i & 7)) & 1;
        ge_cswap(r, q, b);
        ge_add(q, r);
        ge_add(r, r);
        ge_cswap(r, q, b);
    }
}

HD void ge_basepoint(ge &p) {
    // Canonical encodings of the ed25519 basepoint coordinates.
    const u32 bx[8] = {0x8f25d51a, 0xc9562d60, 0x9525a7b2, 0x692cc760, 0xfdd6dc5c, 0xc0a4e231, 0xcd6e53fe, 0x216936d3};
    const u32 by[8] = {0x66666658, 0x66666666, 0x66666666, 0x66666666, 0x66666666, 0x66666666, 0x66666666, 0x66666666};
    fe_unpack_words(p.X, bx);
    fe_unpack_words(p.Y, by);
    fe_1(p.Z);
    fe_mul(p.T, p.X, p.Y);
}

// ============================================================================
// ChaCha12 seed derivation: seed(i) = first 32 bytes of ChaCha12(key, counter=i, nonce).
// Seeds are independent: one revealed key says nothing about the others.
// ============================================================================

HD u32 rotl32(u32 x, int n) { return (x << n) | (x >> (32 - n)); }
#define CC_QR(a, b, c, d) \
    a += b; d ^= a; d = rotl32(d, 16); c += d; b ^= c; b = rotl32(b, 12); \
    a += b; d ^= a; d = rotl32(d, 8);  c += d; b ^= c; b = rotl32(b, 7);

HD void chacha12_seed(const u32 key[8], u64 counter, u32 nonce, u32 out[8]) {
    u32 x0 = 0x61707865, x1 = 0x3320646e, x2 = 0x79622d32, x3 = 0x6b206574;
    u32 x4 = key[0], x5 = key[1], x6 = key[2], x7 = key[3];
    u32 x8 = key[4], x9 = key[5], x10 = key[6], x11 = key[7];
    u32 x12 = (u32)counter, x13 = (u32)(counter >> 32), x14 = nonce, x15 = 0x76616e74;
    for (int r = 0; r < 6; r++) {
        CC_QR(x0, x4, x8, x12); CC_QR(x1, x5, x9, x13); CC_QR(x2, x6, x10, x14); CC_QR(x3, x7, x11, x15);
        CC_QR(x0, x5, x10, x15); CC_QR(x1, x6, x11, x12); CC_QR(x2, x7, x8, x13); CC_QR(x3, x4, x9, x14);
    }
    out[0] = x0 + 0x61707865; out[1] = x1 + 0x3320646e; out[2] = x2 + 0x79622d32; out[3] = x3 + 0x6b206574;
    out[4] = x4 + key[0]; out[5] = x5 + key[1]; out[6] = x6 + key[2]; out[7] = x7 + key[3];
}

// ============================================================================
// SHA-512
// ============================================================================

#define SHA512_K_LIST \
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL, \
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL, \
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL, \
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL, \
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL, \
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL, \
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL, \
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL, \
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL, \
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL, \
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL, \
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL, \
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL, \
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL, \
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL, \
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL, \
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL, \
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL, \
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL, \
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL

__device__ __constant__ u64 K512[80] = { SHA512_K_LIST };
static const u64 K512_HOST[80] = { SHA512_K_LIST };

__device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }
__device__ __forceinline__ u32 bswap32(u32 x) { return __byte_perm(x, 0, 0x0123); }

// SHA-512 of a 32-byte seed (8 LE words) -> clamped ed25519 scalar as 4 LE u64 limbs.
// Only the first 32 digest bytes are produced; the message block is mostly constant.
__device__ __forceinline__ void sha512_seed_scalar(const u32 sw[8], u64 s[4]) {
    u64 W[16];
    #pragma unroll
    for (int i = 0; i < 4; i++) W[i] = ((u64)bswap32(sw[2 * i]) << 32) | bswap32(sw[2 * i + 1]);
    W[4] = 0x8000000000000000ULL;
    #pragma unroll
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 256;

    u64 a = 0x6a09e667f3bcc908ULL, b = 0xbb67ae8584caa73bULL, c = 0x3c6ef372fe94f82bULL, d = 0xa54ff53a5f1d36f1ULL;
    u64 e = 0x510e527fade682d1ULL, f = 0x9b05688c2b3e6c1fULL, g = 0x1f83d9abfb41bd6bULL, h = 0x5be0cd19137e2179ULL;
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        if (i >= 16) {
            u64 w15 = W[(i - 15) & 15], w2 = W[(i - 2) & 15];
            u64 s0 = rotr64(w15, 1) ^ rotr64(w15, 8) ^ (w15 >> 7);
            u64 s1 = rotr64(w2, 19) ^ rotr64(w2, 61) ^ (w2 >> 6);
            W[i & 15] += s0 + W[(i - 7) & 15] + s1;
        }
        u64 S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        u64 ch = (e & f) ^ (~e & g);
        u64 t1 = h + S1 + ch + K512[i] + W[i & 15];
        u64 S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        u64 mj = (a & b) ^ (a & c) ^ (b & c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + S0 + mj;
    }
    u64 st[4] = { a + 0x6a09e667f3bcc908ULL, b + 0xbb67ae8584caa73bULL, c + 0x3c6ef372fe94f82bULL, d + 0xa54ff53a5f1d36f1ULL };
    #pragma unroll
    for (int k = 0; k < 4; k++)   // digest bytes are big-endian per word; scalar is little-endian
        s[k] = ((u64)bswap32((u32)st[k]) << 32) | bswap32((u32)(st[k] >> 32));
    s[0] &= ~7ULL;
    s[3] &= 0x3fffffffffffffffULL;
    s[3] |= 0x4000000000000000ULL;
}

// Plain host SHA-512 for the reference path.
static void sha512_host(const u8 *msg, size_t len, u8 out[64]) {
    u64 H[8] = { 0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
                 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL };
    std::vector<u8> m(msg, msg + len);
    m.push_back(0x80);
    while (m.size() % 128 != 112) m.push_back(0);
    for (int i = 0; i < 8; i++) m.push_back(0);
    u64 bits = (u64)len * 8;
    for (int i = 7; i >= 0; i--) m.push_back((u8)(bits >> (8 * i)));
    auto rr = [](u64 x, int n) { return (x >> n) | (x << (64 - n)); };
    for (size_t off = 0; off < m.size(); off += 128) {
        u64 W[80];
        for (int i = 0; i < 16; i++) { W[i] = 0; for (int j = 0; j < 8; j++) W[i] = (W[i] << 8) | m[off + i * 8 + j]; }
        for (int i = 16; i < 80; i++)
            W[i] = W[i-16] + (rr(W[i-15],1) ^ rr(W[i-15],8) ^ (W[i-15] >> 7)) + W[i-7] + (rr(W[i-2],19) ^ rr(W[i-2],61) ^ (W[i-2] >> 6));
        u64 a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
        for (int i = 0; i < 80; i++) {
            u64 t1 = h + (rr(e,14) ^ rr(e,18) ^ rr(e,41)) + ((e & f) ^ (~e & g)) + K512_HOST[i] + W[i];
            u64 t2 = (rr(a,28) ^ rr(a,34) ^ rr(a,39)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        H[0] += a; H[1] += b; H[2] += c; H[3] += d; H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) out[i * 8 + j] = (u8)(H[i] >> (56 - 8 * j));
}

// Host reference: seed -> public key (sha512 + ladder, no tables).
static void ref_pubkey(const u8 seed[32], u8 pk[32]) {
    u8 h[64];
    sha512_host(seed, 32, h);
    h[0] &= 248; h[31] &= 127; h[31] |= 64;
    ge B, P;
    ge_basepoint(B);
    ge_scalarmult(P, B, h, 256);
    gf zi, x, y;
    fe_invert(zi, P.Z);
    fe_mul(x, P.X, zi);
    fe_mul(y, P.Y, zi);
    u32 w[8];
    fe_pack_words(w, y);
    w[7] ^= fe_parity(x) << 31;
    for (int i = 0; i < 32; i++) pk[i] = (u8)(w[i >> 2] >> (8 * (i & 3)));
}

// ============================================================================
// Fixed-base comb table.
// clamped scalar s = 2^254 + 8*t, t < 2^251. t is split into NWIN windows of W bits,
// recoded to signed digits in [-2^(W-1), 2^(W-1)] (top window unsigned, [0, 2^top]).
// Window 0 is folded into the start point: W0[d + HALF] = 2^254*G + d*8*G for
// d in [-HALF, HALF], stored affine as packed (X, Y, XY).
// Window k >= 1: entry [W0N + (k-1)*HALF + j-1] = j * 8 * 2^(W*k) * G, stored as packed
// (Y+X, Y-X, 2dXY). Every entry is 96 bytes.
// ============================================================================

struct CombCfg {
    int w, nwin, half, topbits, w0n, entries;
};

static CombCfg comb_cfg(int w) {
    CombCfg c;
    c.w = w;
    c.nwin = (251 + w - 1) / w;
    c.half = 1 << (w - 1);
    c.topbits = 251 - w * (c.nwin - 1);
    c.w0n = 2 * c.half + 1;
    c.entries = c.w0n + (c.nwin - 2) * c.half + (1 << c.topbits);
    return c;
}

// Window bases B_k = 8 * 2^(W*k) * G (extended), computed by a ladder.
__global__ void k_window_bases(ge *bases, int w, int nwin) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > nwin) return;
    u8 s[32];
    for (int i = 0; i < 32; i++) s[i] = 0;
    int bit = (k == nwin) ? 254 : 3 + w * k;     // k == nwin: start point 2^254 * G
    s[bit >> 3] = (u8)(1 << (bit & 7));
    ge B, P;
    ge_basepoint(B);
    ge_scalarmult(P, B, s, 256);
    bases[k] = P;
}

__global__ void k_table(uint4 *table, const ge *bases, int half, int nwin, int w0n, int first, int count, int entries) {
    int idx = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= first + count || idx >= entries) return;
    int k, j;
    if (idx < w0n) { k = 0; j = idx - half; }
    else {
        int r = idx - w0n;
        k = 1 + r / half; j = r % half + 1;
        if (k > nwin - 1) { k = nwin - 1; j = r - (nwin - 2) * half + 1; }
    }
    int aj = j < 0 ? -j : j;
    u8 s[4] = { (u8)aj, (u8)(aj >> 8), (u8)(aj >> 16), (u8)(aj >> 24) };
    ge P;
    ge_scalarmult(P, bases[k], s, 32);
    if (j < 0) { fe_neg(P.X, P.X); fe_neg(P.T, P.T); }
    if (k == 0) ge_add(P, bases[nwin]);
    gf zi, x, y, xy, d2;
    fe_invert(zi, P.Z);
    fe_mul(x, P.X, zi);
    fe_mul(y, P.Y, zi);
    fe_mul(xy, x, y);
    u32 w[24];
    if (k == 0) {
        fe_pack_words(w, x);
        fe_pack_words(w + 8, y);
        fe_pack_words(w + 16, xy);
    } else {
        gf ypx, ymx, t2d;
        fe_add(ypx, y, x);
        fe_sub(ymx, y, x);
        fe_d2(d2);
        fe_mul(t2d, xy, d2);
        fe_pack_words(w, ypx);
        fe_pack_words(w + 8, ymx);
        fe_pack_words(w + 16, t2d);
    }
    uint4 *e = table + (size_t)idx * 6;
    for (int i = 0; i < 6; i++) e[i] = make_uint4(w[4*i], w[4*i+1], w[4*i+2], w[4*i+3]);
}

// ============================================================================
// Hot kernel 1: seed -> projective point
// ============================================================================

// GF(2^255-19) on 8 x 32-bit limbs, value kept lazily in [0, 2^256). 2^256 = 38 (mod p).
// ---- GENERATED by tools/gen_fe8.py: begin ----
__device__ __forceinline__ void fe8_mul(u32 r[8], const u32 a[8], const u32 b[8]) {
    asm volatile(
        "{\n\t"
        ".reg .u32 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, q1_a0, q1_a1, q1_a2, q1_a3, q1_a4, q1_b0, q1_b1, q1_b2, q1_b3, q1_b4, q1_m0, q1_m1, q1_m2, q1_m3, q1_m4, q1_m5, q1_m6, q1_m7, q1_m8, q1_k, q1_mk;\n\t"
        "mov.u32 t0, 0;\n\t"
        "mov.u32 t1, 0;\n\t"
        "mov.u32 t2, 0;\n\t"
        "mov.u32 t3, 0;\n\t"
        "mov.u32 t4, 0;\n\t"
        "mov.u32 t5, 0;\n\t"
        "mov.u32 t6, 0;\n\t"
        "mov.u32 t7, 0;\n\t"
        "mov.u32 t8, 0;\n\t"
        "mov.u32 t9, 0;\n\t"
        "mov.u32 t10, 0;\n\t"
        "mov.u32 t11, 0;\n\t"
        "mov.u32 t12, 0;\n\t"
        "mov.u32 t13, 0;\n\t"
        "mov.u32 t14, 0;\n\t"
        "mov.u32 t15, 0;\n\t"
        "mad.lo.cc.u32 t0, %8, %16, t0;\n\t"
        "madc.lo.cc.u32 t1, %8, %17, t1;\n\t"
        "madc.lo.cc.u32 t2, %8, %18, t2;\n\t"
        "madc.lo.cc.u32 t3, %8, %19, t3;\n\t"
        "addc.u32 t4, t4, 0;\n\t"
        "mad.hi.cc.u32 t1, %8, %16, t1;\n\t"
        "madc.hi.cc.u32 t2, %8, %17, t2;\n\t"
        "madc.hi.cc.u32 t3, %8, %18, t3;\n\t"
        "madc.hi.u32 t4, %8, %19, t4;\n\t"
        "mad.lo.cc.u32 t1, %9, %16, t1;\n\t"
        "madc.lo.cc.u32 t2, %9, %17, t2;\n\t"
        "madc.lo.cc.u32 t3, %9, %18, t3;\n\t"
        "madc.lo.cc.u32 t4, %9, %19, t4;\n\t"
        "addc.u32 t5, t5, 0;\n\t"
        "mad.hi.cc.u32 t2, %9, %16, t2;\n\t"
        "madc.hi.cc.u32 t3, %9, %17, t3;\n\t"
        "madc.hi.cc.u32 t4, %9, %18, t4;\n\t"
        "madc.hi.u32 t5, %9, %19, t5;\n\t"
        "mad.lo.cc.u32 t2, %10, %16, t2;\n\t"
        "madc.lo.cc.u32 t3, %10, %17, t3;\n\t"
        "madc.lo.cc.u32 t4, %10, %18, t4;\n\t"
        "madc.lo.cc.u32 t5, %10, %19, t5;\n\t"
        "addc.u32 t6, t6, 0;\n\t"
        "mad.hi.cc.u32 t3, %10, %16, t3;\n\t"
        "madc.hi.cc.u32 t4, %10, %17, t4;\n\t"
        "madc.hi.cc.u32 t5, %10, %18, t5;\n\t"
        "madc.hi.u32 t6, %10, %19, t6;\n\t"
        "mad.lo.cc.u32 t3, %11, %16, t3;\n\t"
        "madc.lo.cc.u32 t4, %11, %17, t4;\n\t"
        "madc.lo.cc.u32 t5, %11, %18, t5;\n\t"
        "madc.lo.cc.u32 t6, %11, %19, t6;\n\t"
        "addc.u32 t7, t7, 0;\n\t"
        "mad.hi.cc.u32 t4, %11, %16, t4;\n\t"
        "madc.hi.cc.u32 t5, %11, %17, t5;\n\t"
        "madc.hi.cc.u32 t6, %11, %18, t6;\n\t"
        "madc.hi.u32 t7, %11, %19, t7;\n\t"
        "mad.lo.cc.u32 t8, %12, %20, t8;\n\t"
        "madc.lo.cc.u32 t9, %12, %21, t9;\n\t"
        "madc.lo.cc.u32 t10, %12, %22, t10;\n\t"
        "madc.lo.cc.u32 t11, %12, %23, t11;\n\t"
        "addc.u32 t12, t12, 0;\n\t"
        "mad.hi.cc.u32 t9, %12, %20, t9;\n\t"
        "madc.hi.cc.u32 t10, %12, %21, t10;\n\t"
        "madc.hi.cc.u32 t11, %12, %22, t11;\n\t"
        "madc.hi.u32 t12, %12, %23, t12;\n\t"
        "mad.lo.cc.u32 t9, %13, %20, t9;\n\t"
        "madc.lo.cc.u32 t10, %13, %21, t10;\n\t"
        "madc.lo.cc.u32 t11, %13, %22, t11;\n\t"
        "madc.lo.cc.u32 t12, %13, %23, t12;\n\t"
        "addc.u32 t13, t13, 0;\n\t"
        "mad.hi.cc.u32 t10, %13, %20, t10;\n\t"
        "madc.hi.cc.u32 t11, %13, %21, t11;\n\t"
        "madc.hi.cc.u32 t12, %13, %22, t12;\n\t"
        "madc.hi.u32 t13, %13, %23, t13;\n\t"
        "mad.lo.cc.u32 t10, %14, %20, t10;\n\t"
        "madc.lo.cc.u32 t11, %14, %21, t11;\n\t"
        "madc.lo.cc.u32 t12, %14, %22, t12;\n\t"
        "madc.lo.cc.u32 t13, %14, %23, t13;\n\t"
        "addc.u32 t14, t14, 0;\n\t"
        "mad.hi.cc.u32 t11, %14, %20, t11;\n\t"
        "madc.hi.cc.u32 t12, %14, %21, t12;\n\t"
        "madc.hi.cc.u32 t13, %14, %22, t13;\n\t"
        "madc.hi.u32 t14, %14, %23, t14;\n\t"
        "mad.lo.cc.u32 t11, %15, %20, t11;\n\t"
        "madc.lo.cc.u32 t12, %15, %21, t12;\n\t"
        "madc.lo.cc.u32 t13, %15, %22, t13;\n\t"
        "madc.lo.cc.u32 t14, %15, %23, t14;\n\t"
        "addc.u32 t15, t15, 0;\n\t"
        "mad.hi.cc.u32 t12, %15, %20, t12;\n\t"
        "madc.hi.cc.u32 t13, %15, %21, t13;\n\t"
        "madc.hi.cc.u32 t14, %15, %22, t14;\n\t"
        "madc.hi.u32 t15, %15, %23, t15;\n\t"
        "mov.u32 q1_m0, 0;\n\t"
        "mov.u32 q1_m1, 0;\n\t"
        "mov.u32 q1_m2, 0;\n\t"
        "mov.u32 q1_m3, 0;\n\t"
        "mov.u32 q1_m4, 0;\n\t"
        "mov.u32 q1_m5, 0;\n\t"
        "mov.u32 q1_m6, 0;\n\t"
        "mov.u32 q1_m7, 0;\n\t"
        "mov.u32 q1_m8, 0;\n\t"
        "add.cc.u32 q1_a0, %8, %12;\n\t"
        "addc.cc.u32 q1_a1, %9, %13;\n\t"
        "addc.cc.u32 q1_a2, %10, %14;\n\t"
        "addc.cc.u32 q1_a3, %11, %15;\n\t"
        "addc.u32 q1_a4, 0, 0;\n\t"
        "add.cc.u32 q1_b0, %16, %20;\n\t"
        "addc.cc.u32 q1_b1, %17, %21;\n\t"
        "addc.cc.u32 q1_b2, %18, %22;\n\t"
        "addc.cc.u32 q1_b3, %19, %23;\n\t"
        "addc.u32 q1_b4, 0, 0;\n\t"
        "mad.lo.cc.u32 q1_m0, q1_a0, q1_b0, q1_m0;\n\t"
        "madc.lo.cc.u32 q1_m1, q1_a0, q1_b1, q1_m1;\n\t"
        "madc.lo.cc.u32 q1_m2, q1_a0, q1_b2, q1_m2;\n\t"
        "madc.lo.cc.u32 q1_m3, q1_a0, q1_b3, q1_m3;\n\t"
        "addc.u32 q1_m4, q1_m4, 0;\n\t"
        "mad.hi.cc.u32 q1_m1, q1_a0, q1_b0, q1_m1;\n\t"
        "madc.hi.cc.u32 q1_m2, q1_a0, q1_b1, q1_m2;\n\t"
        "madc.hi.cc.u32 q1_m3, q1_a0, q1_b2, q1_m3;\n\t"
        "madc.hi.u32 q1_m4, q1_a0, q1_b3, q1_m4;\n\t"
        "mad.lo.cc.u32 q1_m1, q1_a1, q1_b0, q1_m1;\n\t"
        "madc.lo.cc.u32 q1_m2, q1_a1, q1_b1, q1_m2;\n\t"
        "madc.lo.cc.u32 q1_m3, q1_a1, q1_b2, q1_m3;\n\t"
        "madc.lo.cc.u32 q1_m4, q1_a1, q1_b3, q1_m4;\n\t"
        "addc.u32 q1_m5, q1_m5, 0;\n\t"
        "mad.hi.cc.u32 q1_m2, q1_a1, q1_b0, q1_m2;\n\t"
        "madc.hi.cc.u32 q1_m3, q1_a1, q1_b1, q1_m3;\n\t"
        "madc.hi.cc.u32 q1_m4, q1_a1, q1_b2, q1_m4;\n\t"
        "madc.hi.u32 q1_m5, q1_a1, q1_b3, q1_m5;\n\t"
        "mad.lo.cc.u32 q1_m2, q1_a2, q1_b0, q1_m2;\n\t"
        "madc.lo.cc.u32 q1_m3, q1_a2, q1_b1, q1_m3;\n\t"
        "madc.lo.cc.u32 q1_m4, q1_a2, q1_b2, q1_m4;\n\t"
        "madc.lo.cc.u32 q1_m5, q1_a2, q1_b3, q1_m5;\n\t"
        "addc.u32 q1_m6, q1_m6, 0;\n\t"
        "mad.hi.cc.u32 q1_m3, q1_a2, q1_b0, q1_m3;\n\t"
        "madc.hi.cc.u32 q1_m4, q1_a2, q1_b1, q1_m4;\n\t"
        "madc.hi.cc.u32 q1_m5, q1_a2, q1_b2, q1_m5;\n\t"
        "madc.hi.u32 q1_m6, q1_a2, q1_b3, q1_m6;\n\t"
        "mad.lo.cc.u32 q1_m3, q1_a3, q1_b0, q1_m3;\n\t"
        "madc.lo.cc.u32 q1_m4, q1_a3, q1_b1, q1_m4;\n\t"
        "madc.lo.cc.u32 q1_m5, q1_a3, q1_b2, q1_m5;\n\t"
        "madc.lo.cc.u32 q1_m6, q1_a3, q1_b3, q1_m6;\n\t"
        "addc.u32 q1_m7, q1_m7, 0;\n\t"
        "mad.hi.cc.u32 q1_m4, q1_a3, q1_b0, q1_m4;\n\t"
        "madc.hi.cc.u32 q1_m5, q1_a3, q1_b1, q1_m5;\n\t"
        "madc.hi.cc.u32 q1_m6, q1_a3, q1_b2, q1_m6;\n\t"
        "madc.hi.u32 q1_m7, q1_a3, q1_b3, q1_m7;\n\t"
        "neg.s32 q1_mk, q1_a4;\n\t"
        "and.b32 q1_k, q1_b0, q1_mk;\n\t"
        "add.cc.u32 q1_m4, q1_m4, q1_k;\n\t"
        "and.b32 q1_k, q1_b1, q1_mk;\n\t"
        "addc.cc.u32 q1_m5, q1_m5, q1_k;\n\t"
        "and.b32 q1_k, q1_b2, q1_mk;\n\t"
        "addc.cc.u32 q1_m6, q1_m6, q1_k;\n\t"
        "and.b32 q1_k, q1_b3, q1_mk;\n\t"
        "addc.cc.u32 q1_m7, q1_m7, q1_k;\n\t"
        "addc.u32 q1_m8, q1_m8, 0;\n\t"
        "neg.s32 q1_mk, q1_b4;\n\t"
        "and.b32 q1_k, q1_a0, q1_mk;\n\t"
        "add.cc.u32 q1_m4, q1_m4, q1_k;\n\t"
        "and.b32 q1_k, q1_a1, q1_mk;\n\t"
        "addc.cc.u32 q1_m5, q1_m5, q1_k;\n\t"
        "and.b32 q1_k, q1_a2, q1_mk;\n\t"
        "addc.cc.u32 q1_m6, q1_m6, q1_k;\n\t"
        "and.b32 q1_k, q1_a3, q1_mk;\n\t"
        "addc.cc.u32 q1_m7, q1_m7, q1_k;\n\t"
        "addc.u32 q1_m8, q1_m8, 0;\n\t"
        "and.b32 q1_k, q1_a4, q1_b4;\n\t"
        "add.u32 q1_m8, q1_m8, q1_k;\n\t"
        "sub.cc.u32 q1_m0, q1_m0, t0;\n\t"
        "subc.cc.u32 q1_m1, q1_m1, t1;\n\t"
        "subc.cc.u32 q1_m2, q1_m2, t2;\n\t"
        "subc.cc.u32 q1_m3, q1_m3, t3;\n\t"
        "subc.cc.u32 q1_m4, q1_m4, t4;\n\t"
        "subc.cc.u32 q1_m5, q1_m5, t5;\n\t"
        "subc.cc.u32 q1_m6, q1_m6, t6;\n\t"
        "subc.cc.u32 q1_m7, q1_m7, t7;\n\t"
        "subc.u32 q1_m8, q1_m8, 0;\n\t"
        "sub.cc.u32 q1_m0, q1_m0, t8;\n\t"
        "subc.cc.u32 q1_m1, q1_m1, t9;\n\t"
        "subc.cc.u32 q1_m2, q1_m2, t10;\n\t"
        "subc.cc.u32 q1_m3, q1_m3, t11;\n\t"
        "subc.cc.u32 q1_m4, q1_m4, t12;\n\t"
        "subc.cc.u32 q1_m5, q1_m5, t13;\n\t"
        "subc.cc.u32 q1_m6, q1_m6, t14;\n\t"
        "subc.cc.u32 q1_m7, q1_m7, t15;\n\t"
        "subc.u32 q1_m8, q1_m8, 0;\n\t"
        "add.cc.u32 t4, t4, q1_m0;\n\t"
        "addc.cc.u32 t5, t5, q1_m1;\n\t"
        "addc.cc.u32 t6, t6, q1_m2;\n\t"
        "addc.cc.u32 t7, t7, q1_m3;\n\t"
        "addc.cc.u32 t8, t8, q1_m4;\n\t"
        "addc.cc.u32 t9, t9, q1_m5;\n\t"
        "addc.cc.u32 t10, t10, q1_m6;\n\t"
        "addc.cc.u32 t11, t11, q1_m7;\n\t"
        "addc.cc.u32 t12, t12, q1_m8;\n\t"
        "addc.cc.u32 t13, t13, 0;\n\t"
        "addc.cc.u32 t14, t14, 0;\n\t"
        "addc.cc.u32 t15, t15, 0;\n\t"
        "addc.u32 q1_k, 0, 0;\n\t"
        ".reg .u32 c;\n\t"
        "mad.lo.cc.u32 %0, t8, 38, t0;\n\t"
        "madc.lo.cc.u32 %1, t9, 38, t1;\n\t"
        "madc.lo.cc.u32 %2, t10, 38, t2;\n\t"
        "madc.lo.cc.u32 %3, t11, 38, t3;\n\t"
        "madc.lo.cc.u32 %4, t12, 38, t4;\n\t"
        "madc.lo.cc.u32 %5, t13, 38, t5;\n\t"
        "madc.lo.cc.u32 %6, t14, 38, t6;\n\t"
        "madc.lo.cc.u32 %7, t15, 38, t7;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.hi.cc.u32 %1, t8, 38, %1;\n\t"
        "madc.hi.cc.u32 %2, t9, 38, %2;\n\t"
        "madc.hi.cc.u32 %3, t10, 38, %3;\n\t"
        "madc.hi.cc.u32 %4, t11, 38, %4;\n\t"
        "madc.hi.cc.u32 %5, t12, 38, %5;\n\t"
        "madc.hi.cc.u32 %6, t13, 38, %6;\n\t"
        "madc.hi.cc.u32 %7, t14, 38, %7;\n\t"
        "madc.hi.u32 c, t15, 38, c;\n\t"
        "mad.lo.cc.u32 %0, c, 38, %0;\n\t"
        "addc.cc.u32 %1, %1, 0;\n\t"
        "addc.cc.u32 %2, %2, 0;\n\t"
        "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.cc.u32 %5, %5, 0;\n\t"
        "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.cc.u32 %7, %7, 0;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.lo.u32 %0, c, 38, %0;\n\t"
        "}\n\t"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
}
__device__ __forceinline__ void fe8_sq(u32 r[8], const u32 a[8]) {
    asm volatile(
        "{\n\t"
        ".reg .u32 t<17>;\n\t"
        "mov.u32 t0, 0;\n\t"
        "mov.u32 t1, 0;\n\t"
        "mov.u32 t2, 0;\n\t"
        "mov.u32 t3, 0;\n\t"
        "mov.u32 t4, 0;\n\t"
        "mov.u32 t5, 0;\n\t"
        "mov.u32 t6, 0;\n\t"
        "mov.u32 t7, 0;\n\t"
        "mov.u32 t8, 0;\n\t"
        "mov.u32 t9, 0;\n\t"
        "mov.u32 t10, 0;\n\t"
        "mov.u32 t11, 0;\n\t"
        "mov.u32 t12, 0;\n\t"
        "mov.u32 t13, 0;\n\t"
        "mov.u32 t14, 0;\n\t"
        "mov.u32 t15, 0;\n\t"
        "mov.u32 t16, 0;\n\t"
        "mad.lo.cc.u32 t1, %8, %9, t1;\n\t"
        "madc.lo.cc.u32 t2, %8, %10, t2;\n\t"
        "madc.lo.cc.u32 t3, %8, %11, t3;\n\t"
        "madc.lo.cc.u32 t4, %8, %12, t4;\n\t"
        "madc.lo.cc.u32 t5, %8, %13, t5;\n\t"
        "madc.lo.cc.u32 t6, %8, %14, t6;\n\t"
        "madc.lo.cc.u32 t7, %8, %15, t7;\n\t"
        "addc.u32 t8, t8, 0;\n\t"
        "mad.hi.cc.u32 t2, %8, %9, t2;\n\t"
        "madc.hi.cc.u32 t3, %8, %10, t3;\n\t"
        "madc.hi.cc.u32 t4, %8, %11, t4;\n\t"
        "madc.hi.cc.u32 t5, %8, %12, t5;\n\t"
        "madc.hi.cc.u32 t6, %8, %13, t6;\n\t"
        "madc.hi.cc.u32 t7, %8, %14, t7;\n\t"
        "madc.hi.u32 t8, %8, %15, t8;\n\t"
        "mad.lo.cc.u32 t3, %9, %10, t3;\n\t"
        "madc.lo.cc.u32 t4, %9, %11, t4;\n\t"
        "madc.lo.cc.u32 t5, %9, %12, t5;\n\t"
        "madc.lo.cc.u32 t6, %9, %13, t6;\n\t"
        "madc.lo.cc.u32 t7, %9, %14, t7;\n\t"
        "madc.lo.cc.u32 t8, %9, %15, t8;\n\t"
        "addc.u32 t9, t9, 0;\n\t"
        "mad.hi.cc.u32 t4, %9, %10, t4;\n\t"
        "madc.hi.cc.u32 t5, %9, %11, t5;\n\t"
        "madc.hi.cc.u32 t6, %9, %12, t6;\n\t"
        "madc.hi.cc.u32 t7, %9, %13, t7;\n\t"
        "madc.hi.cc.u32 t8, %9, %14, t8;\n\t"
        "madc.hi.u32 t9, %9, %15, t9;\n\t"
        "mad.lo.cc.u32 t5, %10, %11, t5;\n\t"
        "madc.lo.cc.u32 t6, %10, %12, t6;\n\t"
        "madc.lo.cc.u32 t7, %10, %13, t7;\n\t"
        "madc.lo.cc.u32 t8, %10, %14, t8;\n\t"
        "madc.lo.cc.u32 t9, %10, %15, t9;\n\t"
        "addc.u32 t10, t10, 0;\n\t"
        "mad.hi.cc.u32 t6, %10, %11, t6;\n\t"
        "madc.hi.cc.u32 t7, %10, %12, t7;\n\t"
        "madc.hi.cc.u32 t8, %10, %13, t8;\n\t"
        "madc.hi.cc.u32 t9, %10, %14, t9;\n\t"
        "madc.hi.u32 t10, %10, %15, t10;\n\t"
        "mad.lo.cc.u32 t7, %11, %12, t7;\n\t"
        "madc.lo.cc.u32 t8, %11, %13, t8;\n\t"
        "madc.lo.cc.u32 t9, %11, %14, t9;\n\t"
        "madc.lo.cc.u32 t10, %11, %15, t10;\n\t"
        "addc.u32 t11, t11, 0;\n\t"
        "mad.hi.cc.u32 t8, %11, %12, t8;\n\t"
        "madc.hi.cc.u32 t9, %11, %13, t9;\n\t"
        "madc.hi.cc.u32 t10, %11, %14, t10;\n\t"
        "madc.hi.u32 t11, %11, %15, t11;\n\t"
        "mad.lo.cc.u32 t9, %12, %13, t9;\n\t"
        "madc.lo.cc.u32 t10, %12, %14, t10;\n\t"
        "madc.lo.cc.u32 t11, %12, %15, t11;\n\t"
        "addc.u32 t12, t12, 0;\n\t"
        "mad.hi.cc.u32 t10, %12, %13, t10;\n\t"
        "madc.hi.cc.u32 t11, %12, %14, t11;\n\t"
        "madc.hi.u32 t12, %12, %15, t12;\n\t"
        "mad.lo.cc.u32 t11, %13, %14, t11;\n\t"
        "madc.lo.cc.u32 t12, %13, %15, t12;\n\t"
        "addc.u32 t13, t13, 0;\n\t"
        "mad.hi.cc.u32 t12, %13, %14, t12;\n\t"
        "madc.hi.u32 t13, %13, %15, t13;\n\t"
        "mad.lo.cc.u32 t13, %14, %15, t13;\n\t"
        "addc.u32 t14, t14, 0;\n\t"
        "mad.hi.cc.u32 t14, %14, %15, t14;\n\t"
        "addc.u32 t15, t15, 0;\n\t"
        "add.cc.u32 t1, t1, t1;\n\t"
        "addc.cc.u32 t2, t2, t2;\n\t"
        "addc.cc.u32 t3, t3, t3;\n\t"
        "addc.cc.u32 t4, t4, t4;\n\t"
        "addc.cc.u32 t5, t5, t5;\n\t"
        "addc.cc.u32 t6, t6, t6;\n\t"
        "addc.cc.u32 t7, t7, t7;\n\t"
        "addc.cc.u32 t8, t8, t8;\n\t"
        "addc.cc.u32 t9, t9, t9;\n\t"
        "addc.cc.u32 t10, t10, t10;\n\t"
        "addc.cc.u32 t11, t11, t11;\n\t"
        "addc.cc.u32 t12, t12, t12;\n\t"
        "addc.cc.u32 t13, t13, t13;\n\t"
        "addc.cc.u32 t14, t14, t14;\n\t"
        "addc.cc.u32 t15, t15, t15;\n\t"
        "addc.u32 t16, 0, 0;\n\t"
        "mad.lo.cc.u32 t0, %8, %8, t0;\n\t"
        "madc.hi.cc.u32 t1, %8, %8, t1;\n\t"
        "madc.lo.cc.u32 t2, %9, %9, t2;\n\t"
        "madc.hi.cc.u32 t3, %9, %9, t3;\n\t"
        "madc.lo.cc.u32 t4, %10, %10, t4;\n\t"
        "madc.hi.cc.u32 t5, %10, %10, t5;\n\t"
        "madc.lo.cc.u32 t6, %11, %11, t6;\n\t"
        "madc.hi.cc.u32 t7, %11, %11, t7;\n\t"
        "madc.lo.cc.u32 t8, %12, %12, t8;\n\t"
        "madc.hi.cc.u32 t9, %12, %12, t9;\n\t"
        "madc.lo.cc.u32 t10, %13, %13, t10;\n\t"
        "madc.hi.cc.u32 t11, %13, %13, t11;\n\t"
        "madc.lo.cc.u32 t12, %14, %14, t12;\n\t"
        "madc.hi.cc.u32 t13, %14, %14, t13;\n\t"
        "madc.lo.cc.u32 t14, %15, %15, t14;\n\t"
        "madc.hi.u32 t15, %15, %15, t15;\n\t"
        ".reg .u32 c;\n\t"
        "mad.lo.cc.u32 %0, t8, 38, t0;\n\t"
        "madc.lo.cc.u32 %1, t9, 38, t1;\n\t"
        "madc.lo.cc.u32 %2, t10, 38, t2;\n\t"
        "madc.lo.cc.u32 %3, t11, 38, t3;\n\t"
        "madc.lo.cc.u32 %4, t12, 38, t4;\n\t"
        "madc.lo.cc.u32 %5, t13, 38, t5;\n\t"
        "madc.lo.cc.u32 %6, t14, 38, t6;\n\t"
        "madc.lo.cc.u32 %7, t15, 38, t7;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.hi.cc.u32 %1, t8, 38, %1;\n\t"
        "madc.hi.cc.u32 %2, t9, 38, %2;\n\t"
        "madc.hi.cc.u32 %3, t10, 38, %3;\n\t"
        "madc.hi.cc.u32 %4, t11, 38, %4;\n\t"
        "madc.hi.cc.u32 %5, t12, 38, %5;\n\t"
        "madc.hi.cc.u32 %6, t13, 38, %6;\n\t"
        "madc.hi.cc.u32 %7, t14, 38, %7;\n\t"
        "madc.hi.u32 c, t15, 38, c;\n\t"
        "mad.lo.cc.u32 %0, c, 38, %0;\n\t"
        "addc.cc.u32 %1, %1, 0;\n\t"
        "addc.cc.u32 %2, %2, 0;\n\t"
        "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.cc.u32 %5, %5, 0;\n\t"
        "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.cc.u32 %7, %7, 0;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.lo.u32 %0, c, 38, %0;\n\t"
        "}\n\t"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
}
__device__ __forceinline__ void fe8_add(u32 r[8], const u32 a[8], const u32 b[8]) {
    asm volatile(
        "{\n\t"
        ".reg .u32 c;\n\t"
        "add.cc.u32 %0, %8, %16;\n\t"
        "addc.cc.u32 %1, %9, %17;\n\t"
        "addc.cc.u32 %2, %10, %18;\n\t"
        "addc.cc.u32 %3, %11, %19;\n\t"
        "addc.cc.u32 %4, %12, %20;\n\t"
        "addc.cc.u32 %5, %13, %21;\n\t"
        "addc.cc.u32 %6, %14, %22;\n\t"
        "addc.cc.u32 %7, %15, %23;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.lo.cc.u32 %0, c, 38, %0;\n\t"
        "addc.cc.u32 %1, %1, 0;\n\t"
        "addc.cc.u32 %2, %2, 0;\n\t"
        "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.cc.u32 %5, %5, 0;\n\t"
        "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.cc.u32 %7, %7, 0;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.lo.u32 %0, c, 38, %0;\n\t"
        "}\n\t"
        : "=&r"(r[0]), "=&r"(r[1]), "=&r"(r[2]), "=&r"(r[3]), "=&r"(r[4]), "=&r"(r[5]), "=&r"(r[6]), "=&r"(r[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
}
__device__ __forceinline__ void fe8_sub(u32 r[8], const u32 a[8], const u32 b[8]) {
    asm volatile(
        "{\n\t"
        ".reg .u32 c;\n\t"
        "sub.cc.u32 %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.cc.u32 %7, %15, %23;\n\t"
        "subc.u32 c, 0, 0;\n\t"
        "and.b32 c, c, 38;\n\t"
        "sub.cc.u32 %0, %0, c;\n\t"
        "subc.cc.u32 %1, %1, 0;\n\t"
        "subc.cc.u32 %2, %2, 0;\n\t"
        "subc.cc.u32 %3, %3, 0;\n\t"
        "subc.cc.u32 %4, %4, 0;\n\t"
        "subc.cc.u32 %5, %5, 0;\n\t"
        "subc.cc.u32 %6, %6, 0;\n\t"
        "subc.cc.u32 %7, %7, 0;\n\t"
        "subc.u32 c, 0, 0;\n\t"
        "and.b32 c, c, 38;\n\t"
        "sub.u32 %0, %0, c;\n\t"
        "}\n\t"
        : "=&r"(r[0]), "=&r"(r[1]), "=&r"(r[2]), "=&r"(r[3]), "=&r"(r[4]), "=&r"(r[5]), "=&r"(r[6]), "=&r"(r[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
}
// ---- GENERATED by tools/gen_fe8.py: end ----

typedef u32 fe8[8];

__device__ __forceinline__ void fe8_copy(fe8 o, const fe8 a) {
    #pragma unroll
    for (int i = 0; i < 8; i++) o[i] = a[i];
}

// Canonical value (< p) in place.
__device__ __forceinline__ void fe8_canon(fe8 r) {
    u32 top = r[7] >> 31;
    r[7] &= 0x7fffffff;
    u64 c = (u64)r[0] + 19 * top;
    r[0] = (u32)c;
    #pragma unroll
    for (int i = 1; i < 8; i++) { c = (c >> 32) + r[i]; r[i] = (u32)c; }
    fe8 s;                                  // s = r + 19; r >= p  <=>  bit 255 of s set
    c = (u64)r[0] + 19;
    s[0] = (u32)c;
    #pragma unroll
    for (int i = 1; i < 8; i++) { c = (c >> 32) + r[i]; s[i] = (u32)c; }
    bool ge = s[7] >> 31;
    s[7] &= 0x7fffffff;
    #pragma unroll
    for (int i = 0; i < 8; i++) r[i] = ge ? s[i] : r[i];
}

__device__ void fe8_invert(fe8 out, const fe8 z) {
    fe8 t0, t1, t2, t3;
    int i;
    fe8_sq(t0, z);
    fe8_sq(t1, t0); fe8_sq(t1, t1);
    fe8_mul(t1, z, t1);
    fe8_mul(t0, t0, t1);
    fe8_sq(t2, t0);
    fe8_mul(t1, t2, t1);
    fe8_sq(t2, t1); for (i = 1; i < 5; i++) fe8_sq(t2, t2);
    fe8_mul(t1, t2, t1);
    fe8_sq(t2, t1); for (i = 1; i < 10; i++) fe8_sq(t2, t2);
    fe8_mul(t2, t2, t1);
    fe8_sq(t3, t2); for (i = 1; i < 20; i++) fe8_sq(t3, t3);
    fe8_mul(t2, t3, t2);
    fe8_sq(t2, t2); for (i = 1; i < 10; i++) fe8_sq(t2, t2);
    fe8_mul(t1, t2, t1);
    fe8_sq(t2, t1); for (i = 1; i < 50; i++) fe8_sq(t2, t2);
    fe8_mul(t2, t2, t1);
    fe8_sq(t3, t2); for (i = 1; i < 100; i++) fe8_sq(t3, t3);
    fe8_mul(t2, t3, t2);
    fe8_sq(t2, t2); for (i = 1; i < 50; i++) fe8_sq(t2, t2);
    fe8_mul(t1, t2, t1);
    fe8_sq(t1, t1); for (i = 1; i < 5; i++) fe8_sq(t1, t1);
    fe8_mul(out, t1, t0);
}

struct GenParams {
    u32 key[8];
    u32 nonce;
    u64 base_index;
    u32 nk;
    const uint4 *table;
    u32 *X, *Y, *Z;
};

__device__ __forceinline__ void load_fe8(fe8 h, const uint4 *p) {
    uint4 a = __ldg(p), b = __ldg(p + 1);
    h[0] = a.x; h[1] = a.y; h[2] = a.z; h[3] = a.w;
    h[4] = b.x; h[5] = b.y; h[6] = b.z; h[7] = b.w;
}

// p (+/-)= table entry (Y+X, Y-X, 2dXY) with Z = 1 (ref10 ge_madd / ge_msub).
// Subtracting swaps the entry's Y+X / Y-X and negates 2dXY, which swaps f and g.
__device__ __forceinline__ void madd_entry(fe8 X, fe8 Y, fe8 Z, fe8 T, const uint4 *e, bool neg, bool needT) {
    fe8 a, b, c, t;
    fe8_sub(a, Y, X);
    load_fe8(t, e + (neg ? 0 : 2));
    fe8_mul(a, a, t);
    fe8_add(b, Y, X);
    load_fe8(t, e + (neg ? 2 : 0));
    fe8_mul(b, b, t);
    load_fe8(t, e + 4);
    fe8_mul(c, T, t);
    fe8_add(t, Z, Z);                    // t = 2Z
    fe8 ee, f, g, h;
    fe8_sub(ee, b, a); fe8_add(h, b, a);
    fe8_sub(f, t, c);  fe8_add(g, t, c);
    #pragma unroll
    for (int i = 0; i < 8; i++) { u32 x = f[i], y = g[i]; f[i] = neg ? y : x; g[i] = neg ? x : y; }
    fe8_mul(X, ee, f);
    fe8_mul(Y, h, g);
    fe8_mul(Z, g, f);
    if (needT) fe8_mul(T, ee, h);
}

template <int W>
__global__ void __launch_bounds__(GEN_THREADS, GEN_MINB) k_gen(GenParams P) {
    constexpr int NWIN = (251 + W - 1) / W;
    constexpr int HALF = 1 << (W - 1);
    constexpr int W0N = 2 * HALF + 1;
    const u32 stride = gridDim.x * blockDim.x;
    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < P.nk; i += stride) {
        u32 seed[8];
        chacha12_seed(P.key, P.base_index + i, P.nonce, seed);
        u64 s[4];
        sha512_seed_scalar(seed, s);
        // t = (s - 2^254) >> 3
        u64 t0 = (s[0] >> 3) | (s[1] << 61);
        u64 t1 = (s[1] >> 3) | (s[2] << 61);
        u64 t2 = (s[2] >> 3) | (s[3] << 61);
        u64 t3 = (s[3] >> 3) & ((1ULL << 59) - 1);

        // window 0: start point 2^254*G + d0*8*G, affine
        int d0 = (int)((u32)t0 & ((1u << W) - 1));
        t0 = (t0 >> W) | (t1 << (64 - W));
        t1 = (t1 >> W) | (t2 << (64 - W));
        t2 = (t2 >> W) | (t3 << (64 - W));
        t3 >>= W;
        u32 carry = d0 > HALF;
        if (carry) d0 -= (1 << W);
        fe8 X, Y, Z, T;
        const uint4 *e0 = P.table + (size_t)(d0 + HALF) * 6;
        load_fe8(X, e0); load_fe8(Y, e0 + 2); load_fe8(T, e0 + 4);
        #pragma unroll
        for (int l = 0; l < 8; l++) Z[l] = 0;
        Z[0] = 1;

        // Digits are produced one window ahead so the next table entry can be prefetched
        // into L2 while the current addition runs (matters once the table outgrows L2).
        auto next_digit = [&](int k) {
            int d = (int)((u32)t0 & ((1u << W) - 1)) + (int)carry;
            t0 = (t0 >> W) | (t1 << (64 - W));
            t1 = (t1 >> W) | (t2 << (64 - W));
            t2 = (t2 >> W) | (t3 << (64 - W));
            t3 >>= W;
            if (k < NWIN - 1) {
                carry = d > HALF;
                if (carry) d -= (1 << W);
            }
            return d;
        };
        auto entry = [&](int k, int d) {
            int ad = d < 0 ? -d : d;
            return P.table + (size_t)(W0N + (k - 1) * HALF + ad - 1) * 6;
        };
        auto prefetch = [&](int k, int d) {
            const uint4 *pe = entry(k, d);
            asm volatile("prefetch.global.L2 [%0];" :: "l"(pe));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(pe + 5));
        };
        int dn = next_digit(1);
        #pragma unroll 1
        for (int k = 1; k < NWIN; k++) {
            int d = dn;
            if (k + 1 < NWIN) {
                dn = next_digit(k + 1);
                prefetch(k + 1, dn);
            }
            if (d != 0) madd_entry(X, Y, Z, T, entry(k, d), d < 0, k < NWIN - 1);
        }
        #pragma unroll
        for (int l = 0; l < 8; l++) {
            __stcs(P.X + (size_t)l * P.nk + i, X[l]);
            __stcs(P.Y + (size_t)l * P.nk + i, Y[l]);
            __stcs(P.Z + (size_t)l * P.nk + i, Z[l]);
        }
    }
}

// ============================================================================
// Hot kernel 2: batch inversion + affine + gate
// ============================================================================

#define MAX_RANGES 64
__device__ __constant__ u32 c_sfx_bitmap[8];      // allowed low bytes (N mod 256) for suffix patterns
__device__ __constant__ int c_nranges;            // prefix-only patterns: top-64-bit ranges [lo, hi]
__device__ __constant__ u64 c_range_lo[MAX_RANGES];
__device__ __constant__ u64 c_range_hi[MAX_RANGES];

struct Cand { u64 idx; u32 pk[8]; };

struct InvParams {
    u32 *X, *Y, *Z, *Pf;
    u32 nk, nthreads, mc;
    Cand *cand; u32 *ncand; u32 cand_cap;
    u32 *dump;     // optional: all public keys (selftest)
};

__device__ __forceinline__ void ld_fe8(fe8 h, const u32 *buf, u32 nk, u32 i) {
    #pragma unroll
    for (int l = 0; l < 8; l++) h[l] = __ldcs(buf + (size_t)l * nk + i);
}
__device__ __forceinline__ void st_fe8(u32 *buf, u32 nk, u32 i, const fe8 h) {
    #pragma unroll
    for (int l = 0; l < 8; l++) __stcs(buf + (size_t)l * nk + i, h[l]);
}

__global__ void __launch_bounds__(256) k_inv(InvParams P) {
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= P.nthreads) return;
    const u32 T = P.nthreads, MC = P.mc, nk = P.nk;

    fe8 acc, z;
    ld_fe8(acc, P.Z, nk, t);
    for (u32 j = 1; j < MC; j++) {
        st_fe8(P.Pf, nk, t + (j - 1) * T, acc);      // prefix product P_{j-1}
        ld_fe8(z, P.Z, nk, t + j * T);
        fe8_mul(acc, acc, z);
    }
    fe8 inv;
    fe8_invert(inv, acc);

    for (int j = (int)MC - 1; j >= 0; j--) {
        u32 i = t + (u32)j * T;
        fe8 iz;
        if (j > 0) {
            fe8 pr;
            ld_fe8(pr, P.Pf, nk, i - T);
            fe8_mul(iz, inv, pr);
            ld_fe8(z, P.Z, nk, i);
            fe8_mul(inv, inv, z);
        } else {
            fe8_copy(iz, inv);
        }
        fe8 x, w;
        ld_fe8(x, P.X, nk, i);
        ld_fe8(w, P.Y, nk, i);
        fe8_mul(w, w, iz);
        fe8_mul(x, x, iz);
        fe8_canon(w);
        fe8_canon(x);
        w[7] ^= (x[0] & 1) << 31;

        if (P.dump) {
            #pragma unroll
            for (int q = 0; q < 8; q++) P.dump[(size_t)i * 8 + q] = w[q];
        }

        // N = public key bytes read big-endian (base58 integer)
        u32 low = w[7] >> 24;
        bool pass = (c_sfx_bitmap[low >> 5] >> (low & 31)) & 1;
        if (!pass && c_nranges) {
            u64 top = ((u64)__byte_perm(w[0], 0, 0x0123) << 32) | __byte_perm(w[1], 0, 0x0123);
            for (int r = 0; r < c_nranges; r++)
                if (top >= c_range_lo[r] && top <= c_range_hi[r]) { pass = true; break; }
        }
        if (pass) {
            u32 slot = atomicAdd(P.ncand, 1);
            if (slot < P.cand_cap) {
                P.cand[slot].idx = (u64)i;
                #pragma unroll
                for (int q = 0; q < 8; q++) P.cand[slot].pk[q] = w[q];
            }
        }
    }
}

// ============================================================================
// Kernel 3: exact base58 check of candidates
// ============================================================================

#define MAX_PATTERNS 64
#define MAX_PAT_LEN 44
struct Pattern { char pre[MAX_PAT_LEN + 1]; char suf[MAX_PAT_LEN + 1]; u8 plen, slen; };
__device__ __constant__ Pattern c_pats[MAX_PATTERNS];
__device__ __constant__ int c_npats;
__device__ __constant__ int c_icase;
__device__ __constant__ char c_b58[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

struct Match { u64 idx; u32 pk[8]; u32 pat; u32 pad; };

// Base58 of 32 bytes (given as 8 LE words = byte order) -> out, returns length.
HD int b58_words(const u32 w[8], char *out, const char *alpha) {
    u32 n[8];
    for (int i = 0; i < 8; i++) {       // big-endian 32-bit limbs of N
        u32 x = w[i];
        n[i] = (x >> 24) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000) | (x << 24);
    }
    int zeros = 0;
    while (zeros < 32 && ((w[zeros >> 2] >> (8 * (zeros & 3))) & 0xff) == 0) zeros++;
    char tmp[56];
    int len = 0, start = 0;
    while (start < 8 && n[start] == 0) start++;
    while (start < 8) {
        u64 rem = 0;
        for (int i = start; i < 8; i++) {
            u64 cur = (rem << 32) | n[i];
            n[i] = (u32)(cur / 656356768ULL);     // 58^5
            rem = cur % 656356768ULL;
        }
        while (start < 8 && n[start] == 0) start++;
        u32 r = (u32)rem;
        for (int q = 0; q < 5; q++) { tmp[len++] = alpha[r % 58]; r /= 58; }
    }
    while (len > 0 && tmp[len - 1] == alpha[0]) len--;   // strip padding digits of the last group
    int total = 0;
    for (int i = 0; i < zeros; i++) out[total++] = alpha[0];
    for (int i = len - 1; i >= 0; i--) out[total++] = tmp[i];
    out[total] = 0;
    return total;
}

HD char fold(char c, int icase) { return (icase && c >= 'A' && c <= 'Z') ? (char)(c + 32) : c; }

HD int pattern_match(const char *s, int len, const Pattern &p, int icase) {
    if (p.plen > len || p.slen > len) return 0;
    for (int i = 0; i < p.plen; i++) if (fold(s[i], icase) != fold(p.pre[i], icase)) return 0;
    for (int i = 0; i < p.slen; i++) if (fold(s[len - p.slen + i], icase) != fold(p.suf[i], icase)) return 0;
    return 1;
}

__global__ void k_check(const Cand *cand, const u32 *ncand, u32 cand_cap, Match *m, u32 *nm, u32 mcap) {
    u32 n = min(*ncand, cand_cap);
    for (u32 c = blockIdx.x * blockDim.x + threadIdx.x; c < n; c += gridDim.x * blockDim.x) {
        char s[48];
        int len = b58_words(cand[c].pk, s, c_b58);
        for (int p = 0; p < c_npats; p++) {
            if (pattern_match(s, len, c_pats[p], c_icase)) {
                u32 slot = atomicAdd(nm, 1);
                if (slot < mcap) {
                    m[slot].idx = cand[c].idx;
                    for (int q = 0; q < 8; q++) m[slot].pk[q] = cand[c].pk[q];
                    m[slot].pat = p;
                }
                break;
            }
        }
    }
}

// ============================================================================
// Host: patterns, gates, probabilities
// ============================================================================

static const char B58[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static int b58_index(char c) { const char *p = strchr(B58, c); return (p && c) ? (int)(p - B58) : -1; }

// Minimal unsigned big integer (little-endian 32-bit limbs).
struct Big {
    u32 w[12];
    Big() { memset(w, 0, sizeof(w)); }
    static Big pow2(int b) { Big r; r.w[b / 32] = 1u << (b % 32); return r; }
    void mul_add(u32 m, u32 a) { u64 c = a; for (int i = 0; i < 12; i++) { u64 v = (u64)w[i] * m + c; w[i] = (u32)v; c = v >> 32; } }
    int cmp(const Big &o) const { for (int i = 11; i >= 0; i--) if (w[i] != o.w[i]) return w[i] < o.w[i] ? -1 : 1; return 0; }
    double dbl() const { double r = 0; for (int i = 11; i >= 0; i--) r = r * 4294967296.0 + w[i]; return r; }
    u64 top64() const { return ((u64)w[7] << 32) | w[6]; }   // bits 192..255 (value < 2^256)
    Big sub1() const { Big r = *this; for (int i = 0; i < 12; i++) { if (r.w[i]--) break; } return r; }
};

// All case variants of a pattern string that only use base58 characters.
static void case_variants(const std::string &s, bool icase, std::vector<std::string> &out) {
    out.clear();
    out.push_back("");
    for (char ch : s) {
        std::vector<char> opts;
        if (icase && isalpha((unsigned char)ch)) {
            char lo = (char)tolower(ch), up = (char)toupper(ch);
            if (b58_index(lo) >= 0) opts.push_back(lo);
            if (b58_index(up) >= 0) opts.push_back(up);
        } else if (b58_index(ch) >= 0) opts.push_back(ch);
        std::vector<std::string> next;
        for (auto &o : out) for (char c : opts) next.push_back(o + c);
        out.swap(next);
        if (out.size() > 65536) { fprintf(stderr, "error: too many case variants\n"); exit(1); }
    }
}

struct Range { u64 lo, hi; };

// Integer intervals [lo, hi) of N (base58 integer of the key) whose address starts with p.
static double prefix_ranges(const std::string &p, std::vector<Range> &ranges) {
    int z = 0;
    while (z < (int)p.size() && p[z] == '1') z++;
    if (z > 30) return 0;
    std::string r = p.substr(z);
    Big lim_hi = Big::pow2(256 - 8 * z), lim_lo;
    if (!r.empty()) lim_lo = Big::pow2(248 - 8 * z);
    double total = 0;
    auto add = [&](Big lo, Big hi) {
        if (lo.cmp(lim_lo) < 0) lo = lim_lo;
        if (hi.cmp(lim_hi) > 0) hi = lim_hi;
        if (lo.cmp(hi) >= 0) return;
        total += hi.dbl() - lo.dbl();
        ranges.push_back({ lo.top64(), hi.sub1().top64() });
    };
    if (r.empty()) { add(lim_lo, lim_hi); return total; }
    Big v;
    for (char c : r) v.mul_add(58, (u32)b58_index(c));
    Big lo = v, hi = v; hi.mul_add(1, 1);
    for (int m = 0; m < 50 && lo.cmp(lim_hi) < 0; m++) {
        add(lo, hi);
        lo.mul_add(58, 0); hi.mul_add(58, 0);
    }
    return total;
}

struct PatternSet {
    std::vector<Pattern> pats;
    std::vector<std::string> labels;
    bool icase = false;
    u32 bitmap[8] = {0};
    std::vector<Range> ranges;
    double prob = 0;
};

static bool build_patterns(PatternSet &ps, const std::vector<std::pair<std::string, std::string>> &in) {
    for (auto &pr : in) {
        const std::string &pre = pr.first, &suf = pr.second;
        if (pre.empty() && suf.empty()) { fprintf(stderr, "error: empty pattern\n"); return false; }
        if (pre.size() + suf.size() > MAX_PAT_LEN) { fprintf(stderr, "error: pattern too long\n"); return false; }
        for (const std::string *s : { &pre, &suf }) {
            std::vector<std::string> v;
            case_variants(*s, ps.icase, v);
            if (v.empty()) {
                fprintf(stderr, "error: '%s' has characters outside base58 (0, O, I, l are not allowed%s)\n",
                        s->c_str(), ps.icase ? "" : "; try -i");
                return false;
            }
        }
        if ((int)ps.pats.size() >= MAX_PATTERNS) { fprintf(stderr, "error: at most %d patterns\n", MAX_PATTERNS); return false; }
        Pattern p;
        memset(&p, 0, sizeof(p));
        memcpy(p.pre, pre.data(), pre.size()); p.plen = (u8)pre.size();
        memcpy(p.suf, suf.data(), suf.size()); p.slen = (u8)suf.size();
        ps.pats.push_back(p);
        ps.labels.push_back(pre + "*" + suf);

        std::vector<std::string> pv, sv;
        case_variants(pre, ps.icase, pv);
        case_variants(suf, ps.icase, sv);
        double ppre = 1.0;
        if (!pre.empty()) {
            std::vector<Range> rr;
            double tot = 0;
            for (auto &x : pv) tot += prefix_ranges(x, rr);
            ppre = tot / ldexp(1.0, 256);
            if (suf.empty()) ps.ranges.insert(ps.ranges.end(), rr.begin(), rr.end());
        }
        double psuf = 1.0;
        if (!suf.empty()) {
            int K = (int)suf.size();
            psuf = (double)sv.size() * pow(58.0, -K);
            for (auto &x : sv) {
                Big v;
                for (char c : x) v.mul_add(58, (u32)b58_index(c));
                u32 low = v.w[0] & 0xff;
                int kb = K < 8 ? K : 8;
                u32 m = (1u << kb) - 1;
                for (u32 b = 0; b < 256; b++) if ((b & m) == (low & m)) ps.bitmap[b >> 5] |= 1u << (b & 31);
            }
        }
        ps.prob += ppre * psuf;
    }
    // Merge prefix ranges; shrink to MAX_RANGES by joining the closest neighbours (superset).
    std::sort(ps.ranges.begin(), ps.ranges.end(), [](const Range &a, const Range &b) { return a.lo < b.lo; });
    std::vector<Range> merged;
    for (auto &r : ps.ranges) {
        if (!merged.empty() && r.lo <= merged.back().hi + 1) merged.back().hi = std::max(merged.back().hi, r.hi);
        else merged.push_back(r);
    }
    while (merged.size() > MAX_RANGES) {
        size_t best = 0; u64 gap = ~0ULL;
        for (size_t i = 0; i + 1 < merged.size(); i++)
            if (merged[i + 1].lo - merged[i].hi < gap) { gap = merged[i + 1].lo - merged[i].hi; best = i; }
        merged[best].hi = merged[best + 1].hi;
        merged.erase(merged.begin() + best + 1);
    }
    ps.ranges = merged;
    return true;
}

static bool host_match(const PatternSet &ps, const char *s, int len) {
    for (auto &p : ps.pats) if (pattern_match(s, len, p, ps.icase)) return true;
    return false;
}

// ============================================================================
// Host: randomness, encoding
// ============================================================================

static void os_random(void *buf, size_t n) {
#ifdef _WIN32
    if (BCryptGenRandom(NULL, (PUCHAR)buf, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
        fprintf(stderr, "error: BCryptGenRandom failed\n"); exit(1);
    }
#else
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f || fread(buf, 1, n, f) != n) { fprintf(stderr, "error: cannot read /dev/urandom\n"); exit(1); }
    fclose(f);
#endif
}

static std::string b58_bytes(const u8 *in, int n) {
    std::vector<u8> buf(in, in + n);
    int zeros = 0;
    while (zeros < n && buf[zeros] == 0) zeros++;
    std::string tmp;
    int start = zeros;
    while (start < n) {
        int rem = 0;
        for (int i = start; i < n; i++) { int v = (rem << 8) | buf[i]; buf[i] = (u8)(v / 58); rem = v % 58; }
        tmp.push_back(B58[rem]);
        while (start < n && buf[start] == 0) start++;
    }
    std::string out(zeros, '1');
    out.append(tmp.rbegin(), tmp.rend());
    return out;
}

static void words_to_bytes(const u32 w[8], u8 b[32]) { for (int i = 0; i < 32; i++) b[i] = (u8)(w[i >> 2] >> (8 * (i & 3))); }

// ============================================================================
// Host: device context
// ============================================================================

struct Slot {
    Cand *d_cand; u32 *d_ncand;
    Match *d_match; u32 *d_nmatch;
    u32 *h_counts;      // pinned: [ncand, nmatch]
    Match *h_match;     // pinned
    u32 *X, *Y, *Z, *Pf;   // projective points and prefix products of this batch
    cudaEvent_t gen_done, done;
    u64 base_index;
    bool busy;
};

struct Dev {
    int id;
    cudaDeviceProp prop;
    CombCfg cfg;
    uint4 *d_table;
    u32 nk, mc, inv_threads;
    u32 cand_cap, match_cap;
    int gen_blocks;
    cudaStream_t stream, inv_stream, copy;   // k_gen (low priority) / k_inv + k_check (high)
    Slot slot[2];
    u32 key[8];
    u64 next_index;
    int cur;
    u32 *d_dump;
};

typedef void (*GenLaunch)(int, int, cudaStream_t, const GenParams &);
template <int W> static void launch_gen(int blocks, int threads, cudaStream_t s, const GenParams &p) {
    k_gen<W><<<blocks, threads, 0, s>>>(p);
}
static GenLaunch gen_for(int w) {
    switch (w) {
    case 8: return launch_gen<8>;   case 9: return launch_gen<9>;   case 10: return launch_gen<10>;
    case 11: return launch_gen<11>; case 12: return launch_gen<12>; case 13: return launch_gen<13>;
    case 14: return launch_gen<14>; case 15: return launch_gen<15>; case 16: return launch_gen<16>;
    case 17: return launch_gen<17>; case 18: return launch_gen<18>;
    }
    return nullptr;
}
static const void *gen_func(int w) {
    switch (w) {
    case 8: return (const void *)k_gen<8>;   case 9: return (const void *)k_gen<9>;   case 10: return (const void *)k_gen<10>;
    case 11: return (const void *)k_gen<11>; case 12: return (const void *)k_gen<12>; case 13: return (const void *)k_gen<13>;
    case 14: return (const void *)k_gen<14>; case 15: return (const void *)k_gen<15>; case 16: return (const void *)k_gen<16>;
    case 17: return (const void *)k_gen<17>; case 18: return (const void *)k_gen<18>;
    }
    return nullptr;
}

struct Options {
    std::vector<std::pair<std::string, std::string>> pats;
    bool icase = false;
    long long count = -1;
    std::string out;
    int window = 0;               // 0 = auto
    std::vector<int> devices;     // empty = device 0
    bool all_devices = false;
    double bench = 0;
    bool selftest = false;
    bool tune = false;
    int batch_log2 = 22;
    int mc = 128;
    bool l2persist = true;
    bool quiet_matches = false;
};

static int auto_window(const cudaDeviceProp &p) {
    // Largest window whose table stays within ~80% of L2 (at least 12: the next entry is
    // prefetched, so a table partly in DRAM still beats many more additions). --tune measures.
    for (int w = 17; w > 12; w--) {
        CombCfg c = comb_cfg(w);
        if ((double)c.entries * 96.0 <= 0.8 * p.l2CacheSize) return w;
    }
    return 12;
}

static void dev_init(Dev &d, int id, const Options &o, const PatternSet &ps) {
    d.id = id;
    CUDA_CHECK(cudaSetDevice(id));
    CUDA_CHECK(cudaGetDeviceProperties(&d.prop, id));
    int w = o.window ? o.window : auto_window(d.prop);
    d.cfg = comb_cfg(w);
    d.nk = 1u << o.batch_log2;
    d.mc = (u32)o.mc;
    d.inv_threads = d.nk / d.mc;
    d.cand_cap = d.nk;
    d.match_cap = 1u << 16;
    int prio_lo = 0, prio_hi = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&d.stream, cudaStreamNonBlocking, prio_lo));
    CUDA_CHECK(cudaStreamCreateWithPriority(&d.inv_stream, cudaStreamNonBlocking, prio_hi));
    CUDA_CHECK(cudaStreamCreateWithFlags(&d.copy, cudaStreamNonBlocking));

    // Table
    size_t tbytes = (size_t)d.cfg.entries * 96;
    CUDA_CHECK(cudaMalloc(&d.d_table, tbytes));
    ge *d_bases;
    CUDA_CHECK(cudaMalloc(&d_bases, sizeof(ge) * (d.cfg.nwin + 1)));
    auto t0 = std::chrono::steady_clock::now();
    k_window_bases<<<1, 64>>>(d_bases, d.cfg.w, d.cfg.nwin);
    CUDA_CHECK(cudaGetLastError());
    for (int first = 0; first < d.cfg.entries; first += 65536) {
        int cnt = std::min(65536, d.cfg.entries - first);
        k_table<<<(cnt + 127) / 128, 128>>>(d.d_table, d_bases, d.cfg.half, d.cfg.nwin, d.cfg.w0n, first, cnt, d.cfg.entries);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(d_bases));
    double tsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (o.l2persist && d.prop.persistingL2CacheMaxSize > 0) {
        size_t lim = std::min((size_t)d.prop.persistingL2CacheMaxSize, tbytes);
        if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, lim) == cudaSuccess) {
            cudaStreamAttrValue a = {};
            a.accessPolicyWindow.base_ptr = d.d_table;
            a.accessPolicyWindow.num_bytes = std::min(tbytes, (size_t)d.prop.accessPolicyMaxWindowSize);
            a.accessPolicyWindow.hitRatio = std::min(1.0f, (float)lim / (float)a.accessPolicyWindow.num_bytes);
            a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(d.stream, cudaStreamAttributeAccessPolicyWindow, &a);
        }
        cudaGetLastError();
    }

    size_t fe_bytes = (size_t)d.nk * 32;
    for (int s = 0; s < 2; s++) {
        Slot &sl = d.slot[s];
        CUDA_CHECK(cudaMalloc(&sl.X, fe_bytes));
        CUDA_CHECK(cudaMalloc(&sl.Y, fe_bytes));
        CUDA_CHECK(cudaMalloc(&sl.Z, fe_bytes));
        CUDA_CHECK(cudaMalloc(&sl.Pf, fe_bytes));
        CUDA_CHECK(cudaEventCreateWithFlags(&sl.gen_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(&sl.d_cand, sizeof(Cand) * d.cand_cap));
        CUDA_CHECK(cudaMalloc(&sl.d_ncand, 4));
        CUDA_CHECK(cudaMalloc(&sl.d_match, sizeof(Match) * d.match_cap));
        CUDA_CHECK(cudaMalloc(&sl.d_nmatch, 4));
        CUDA_CHECK(cudaMallocHost(&sl.h_counts, 8));
        CUDA_CHECK(cudaMallocHost(&sl.h_match, sizeof(Match) * d.match_cap));
        CUDA_CHECK(cudaEventCreateWithFlags(&sl.done, cudaEventDisableTiming));
        sl.busy = false;
    }
    d.d_dump = nullptr;

    // Patterns / gates
    CUDA_CHECK(cudaMemcpyToSymbol(c_sfx_bitmap, ps.bitmap, sizeof(ps.bitmap)));
    int nr = (int)ps.ranges.size();
    std::vector<u64> lo(MAX_RANGES, 0), hi(MAX_RANGES, 0);
    for (int i = 0; i < nr; i++) { lo[i] = ps.ranges[i].lo; hi[i] = ps.ranges[i].hi; }
    CUDA_CHECK(cudaMemcpyToSymbol(c_nranges, &nr, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_range_lo, lo.data(), sizeof(u64) * MAX_RANGES));
    CUDA_CHECK(cudaMemcpyToSymbol(c_range_hi, hi.data(), sizeof(u64) * MAX_RANGES));
    int np = (int)ps.pats.size(), ic = ps.icase ? 1 : 0;
    CUDA_CHECK(cudaMemcpyToSymbol(c_pats, ps.pats.data(), sizeof(Pattern) * np));
    CUDA_CHECK(cudaMemcpyToSymbol(c_npats, &np, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_icase, &ic, sizeof(int)));

    os_random(d.key, sizeof(d.key));
    d.next_index = 0;
    d.cur = 0;

    int per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gen_func(d.cfg.w), GEN_THREADS, 0));
    // Many short blocks, so k_inv of the previous batch (high-priority stream) can slot in
    // between k_gen blocks: memory-bound and compute-bound work then overlap.
    d.gen_blocks = std::max(d.prop.multiProcessorCount * std::max(per_sm, 1), (int)(d.nk / (GEN_THREADS * 16)));
    cudaFuncAttributes fa;
    CUDA_CHECK(cudaFuncGetAttributes(&fa, gen_func(d.cfg.w)));

    printf("GPU %d:   %s (SM %d.%d, %d SMs, L2 %d MB)\n", id, d.prop.name, d.prop.major, d.prop.minor,
           d.prop.multiProcessorCount, d.prop.l2CacheSize >> 20);
    printf("         comb W=%d: %d windows (first folded into the start point), table %.1f MB built in %.2fs; gen %d regs, %d blocks/SM; batch %u keys\n",
           d.cfg.w, d.cfg.nwin, tbytes / 1048576.0, tsec, fa.numRegs, per_sm, d.nk);
}

static cudaEvent_t g_prof_ev[4];
static bool g_prof = false;

static void dev_enqueue(Dev &d) {
    Slot &sl = d.slot[d.cur];
    CUDA_CHECK(cudaSetDevice(d.id));
    sl.base_index = d.next_index;
    d.next_index += d.nk;
    CUDA_CHECK(cudaMemsetAsync(sl.d_ncand, 0, 4, d.inv_stream));
    CUDA_CHECK(cudaMemsetAsync(sl.d_nmatch, 0, 4, d.inv_stream));
    GenParams gp;
    memcpy(gp.key, d.key, sizeof(gp.key));
    gp.nonce = (u32)d.id;
    gp.base_index = sl.base_index;
    gp.nk = d.nk;
    gp.table = d.d_table;
    gp.X = sl.X; gp.Y = sl.Y; gp.Z = sl.Z;
    if (g_prof) CUDA_CHECK(cudaEventRecord(g_prof_ev[0], d.stream));
    gen_for(d.cfg.w)(d.gen_blocks, GEN_THREADS, d.stream, gp);
    if (g_prof) CUDA_CHECK(cudaEventRecord(g_prof_ev[1], d.stream));
    CUDA_CHECK(cudaEventRecord(sl.gen_done, d.stream));
    CUDA_CHECK(cudaStreamWaitEvent(d.inv_stream, sl.gen_done, 0));
    InvParams ip;
    ip.X = sl.X; ip.Y = sl.Y; ip.Z = sl.Z; ip.Pf = sl.Pf;
    ip.nk = d.nk; ip.nthreads = d.inv_threads; ip.mc = d.mc;
    ip.cand = sl.d_cand; ip.ncand = sl.d_ncand; ip.cand_cap = d.cand_cap;
    ip.dump = d.d_dump;
    k_inv<<<(d.inv_threads + 255) / 256, 256, 0, d.inv_stream>>>(ip);
    if (g_prof) CUDA_CHECK(cudaEventRecord(g_prof_ev[2], d.inv_stream));
    k_check<<<d.prop.multiProcessorCount * 4, 256, 0, d.inv_stream>>>(sl.d_cand, sl.d_ncand, d.cand_cap,
                                                                      sl.d_match, sl.d_nmatch, d.match_cap);
    if (g_prof) CUDA_CHECK(cudaEventRecord(g_prof_ev[3], d.inv_stream));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(sl.h_counts, sl.d_ncand, 4, cudaMemcpyDeviceToHost, d.inv_stream));
    CUDA_CHECK(cudaMemcpyAsync(sl.h_counts + 1, sl.d_nmatch, 4, cudaMemcpyDeviceToHost, d.inv_stream));
    CUDA_CHECK(cudaEventRecord(sl.done, d.inv_stream));
    sl.busy = true;
    d.cur ^= 1;
}

struct Stats { u64 found = 0, mismatched = 0, dropped = 0, keys = 0; };

// Host check of one GPU match: re-derive the seed, recompute the key with the reference
// implementation and re-check the pattern. Thread-safe.
struct Checked { bool ok; std::string pub, priv; };

static Checked check_match(const Dev &d, const PatternSet &ps, const Match &m, u64 base) {
    Checked c;
    c.ok = false;
    u64 index = base + m.idx;
    u32 sw[8];
    chacha12_seed(d.key, index, (u32)d.id, sw);
    u8 seed[32], pk[32], gpk[32];
    words_to_bytes(sw, seed);
    ref_pubkey(seed, pk);
    words_to_bytes(m.pk, gpk);
    if (memcmp(pk, gpk, 32) != 0) return c;
    c.pub = b58_bytes(pk, 32);
    if (!host_match(ps, c.pub.c_str(), (int)c.pub.size())) return c;
    u8 full[64];
    memcpy(full, seed, 32);
    memcpy(full + 32, pk, 32);
    c.priv = b58_bytes(full, 64);
    c.ok = true;
    return c;
}

// Waits for the oldest busy slot of a device and processes it.
static void dev_drain_one(Dev &d, const PatternSet &ps, FILE *csv, Stats &st, long long limit, bool quiet) {
    Slot &sl = d.slot[d.cur];        // the older of the two (cur points at the next to fill)
    if (!sl.busy) return;
    CUDA_CHECK(cudaSetDevice(d.id));
    CUDA_CHECK(cudaEventSynchronize(sl.done));
    sl.busy = false;
    st.keys += d.nk;
    u32 nc = sl.h_counts[0], nm = sl.h_counts[1];
    if (nc > d.cand_cap) st.dropped += nc - d.cand_cap;
    if (nm > d.match_cap) { st.dropped += nm - d.match_cap; nm = d.match_cap; }
    if (!nm) return;
    if (limit > 0) nm = (u32)std::min<long long>(nm, std::max<long long>(0, limit - (long long)st.found));
    CUDA_CHECK(cudaMemcpyAsync(sl.h_match, sl.d_match, sizeof(Match) * nm, cudaMemcpyDeviceToHost, d.copy));
    CUDA_CHECK(cudaStreamSynchronize(d.copy));
    std::vector<Checked> out(nm);
    unsigned nt = std::min<unsigned>(std::max(1u, std::thread::hardware_concurrency()), (nm + 63) / 64);
    if (nt <= 1) {
        for (u32 i = 0; i < nm; i++) out[i] = check_match(d, ps, sl.h_match[i], sl.base_index);
    } else {
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; t++)
            th.emplace_back([&, t]() { for (u32 i = t; i < nm; i += nt) out[i] = check_match(d, ps, sl.h_match[i], sl.base_index); });
        for (auto &x : th) x.join();
    }
    for (auto &c : out) {
        if (!c.ok) { st.mismatched++; continue; }
        if (csv) fprintf(csv, "%s,%s\n", c.pub.c_str(), c.priv.c_str());
        st.found++;
        if (!quiet) printf("Found #%llu: %s\n", (unsigned long long)st.found, c.pub.c_str());
    }
    if (csv) fflush(csv);
    fflush(stdout);     // stdout is fully buffered when redirected
}

// ============================================================================
// Self-test
// ============================================================================

// Rate of one device for a few seconds with window w (no output written).
static double measure_window(Options o, int w, double seconds) {
    o.window = w;
    PatternSet ps;
    build_patterns(ps, { {"", "zzzzzzzzzz"} });
    Dev d;
    dev_init(d, o.devices.empty() ? 0 : o.devices[0], o, ps);
    Stats st;
    dev_enqueue(d); dev_enqueue(d);
    for (int i = 0; i < 4; i++) { dev_drain_one(d, ps, nullptr, st, -1, true); dev_enqueue(d); }   // warm-up
    u64 k0 = st.keys;
    auto t0 = std::chrono::steady_clock::now();
    double el = 0;
    while ((el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()) < seconds) {
        dev_drain_one(d, ps, nullptr, st, -1, true);
        dev_enqueue(d);
    }
    double rate = (st.keys - k0) / el;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceReset());
    return rate;
}

static int tune(Options o) {
    double best = 0;
    int bw = 0;
    for (int w = 11; w <= 18; w++) {
        double r = measure_window(o, w, 4.0);
        printf("  W=%-2d  %8.2fM keys/sec\n\n", w, r / 1e6);
        if (r > best) { best = r; bw = w; }
    }
    printf("Fastest: -w %d (%.2fM keys/sec)\n", bw, best / 1e6);
    return 0;
}

static int selftest(Options o) {
    int fails = 0;
    // 1. Host reference against RFC 8032 test 1.
    {
        const char *sh = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";
        const char *ph = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
        u8 seed[32], pk[32], want[32];
        for (int i = 0; i < 32; i++) { unsigned v; sscanf(sh + 2 * i, "%2x", &v); seed[i] = (u8)v; sscanf(ph + 2 * i, "%2x", &v); want[i] = (u8)v; }
        ref_pubkey(seed, pk);
        bool ok = memcmp(pk, want, 32) == 0;
        printf("[%s] host reference vs RFC 8032 test 1\n", ok ? " ok " : "FAIL");
        fails += !ok;
    }
    // 2. GPU keys vs host reference, and GPU matcher vs host matcher, for each window size.
    std::vector<int> windows;
    if (o.window) windows.push_back(o.window);
    else for (int w = 8; w <= 18; w++) windows.push_back(w);
    o.batch_log2 = std::min(o.batch_log2, 20);
    for (int w : windows) {
        o.window = w;
        PatternSet ps;
        ps.icase = true;
        build_patterns(ps, { {"", "Ab"}, {"Bc", ""}, {"9", "z"}, {"11", ""}, {"Zz", ""}, {"", "ppp"}, {"x", ""} });
        Dev d;
        dev_init(d, o.devices.empty() ? 0 : o.devices[0], o, ps);
        CUDA_CHECK(cudaMalloc(&d.d_dump, (size_t)d.nk * 32));
        dev_enqueue(d);
        Slot &sl = d.slot[0];
        CUDA_CHECK(cudaEventSynchronize(sl.done));
        u32 nm = std::min(sl.h_counts[1], d.match_cap);
        CUDA_CHECK(cudaMemcpy(sl.h_match, sl.d_match, sizeof(Match) * nm, cudaMemcpyDeviceToHost));
        std::vector<u32> dump((size_t)d.nk * 8);
        CUDA_CHECK(cudaMemcpy(dump.data(), d.d_dump, dump.size() * 4, cudaMemcpyDeviceToHost));

        // keys: first 64 + 512 spread out, recomputed on the host
        int bad = 0, checked = 0;
        for (u32 j = 0; j < 576; j++) {
            u32 i = j < 64 ? j : (u32)(((u64)(j - 64) * 2654435761u) % d.nk);
            u32 sw[8]; u8 seed[32], pk[32], g[32];
            chacha12_seed(d.key, i, (u32)d.id, sw);
            words_to_bytes(sw, seed);
            ref_pubkey(seed, pk);
            words_to_bytes(&dump[(size_t)i * 8], g);
            bad += memcmp(pk, g, 32) != 0;
            checked++;
        }
        // matcher: host scans every key of the batch
        std::vector<u64> want, got;
        for (u32 i = 0; i < d.nk; i++) {
            char s[48];
            int len = b58_words(&dump[(size_t)i * 8], s, B58);
            if (host_match(ps, s, len)) want.push_back(i);
        }
        for (u32 i = 0; i < nm; i++) got.push_back(sl.h_match[i].idx);
        std::sort(got.begin(), got.end());
        bool keys_ok = bad == 0, match_ok = want == got && sl.h_counts[1] <= d.match_cap;
        printf("[%s] W=%-2d GPU keys vs host reference (%d checked, %d bad)\n", keys_ok ? " ok " : "FAIL", w, checked, bad);
        printf("[%s] W=%-2d GPU matcher vs host scan of %u keys (%zu expected, %zu found)\n",
               match_ok ? " ok " : "FAIL", w, d.nk, want.size(), got.size());
        fails += !keys_ok + !match_ok;
        CUDA_CHECK(cudaDeviceReset());
    }
    printf(fails ? "SELFTEST FAILED (%d)\n" : "SELFTEST PASSED\n", fails);
    return fails ? 1 : 0;
}

// ============================================================================
// main
// ============================================================================

static void usage() {
    printf(
        "usage: vanity_gpu [options] [SUFFIX [COUNT]]\n"
        "  -s, --suffix S      address ends with S (repeatable)\n"
        "  -p, --prefix P      address starts with P (repeatable)\n"
        "  -m, --match P*S     starts with P and ends with S ('abc*', '*xyz', 'ab*yz'; repeatable)\n"
        "  -i, --ignore-case   case-insensitive matching\n"
        "  -n, --count N       stop after N matches (default: run forever)\n"
        "  -o, --out FILE      CSV output (default <suffix>_results.csv or vanity_results.csv)\n"
        "  -d, --device LIST   GPU ids, e.g. 0 or 0,1 or all (default 0)\n"
        "  -w, --window W      comb window bits 8..18 (default: by L2 size)\n"
        "      --batch-log2 N  keys per batch = 2^N (default 22)\n"
        "      --bench SEC     run SEC seconds and print the average rate\n"
        "      --no-l2-persist do not pin the table in L2\n"
        "      --selftest      check GPU keys and matcher against the host reference\n"
        "      --tune          measure window sizes on this GPU and print the fastest\n"
        "Patterns are OR-ed. Keys are written as public_key,private_key (Phantom/Solflare format).\n");
}

int main(int argc, char **argv) {
    Options o;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if (a == "-s" || a == "--suffix") o.pats.push_back({"", next()});
        else if (a == "-p" || a == "--prefix") o.pats.push_back({next(), ""});
        else if (a == "-m" || a == "--match") {
            std::string m = next();
            size_t star = m.find('*');
            if (star == std::string::npos) { fprintf(stderr, "error: --match needs '*'\n"); return 1; }
            o.pats.push_back({m.substr(0, star), m.substr(star + 1)});
        }
        else if (a == "-i" || a == "--ignore-case") o.icase = true;
        else if (a == "-n" || a == "--count") o.count = atoll(next().c_str());
        else if (a == "-o" || a == "--out") o.out = next();
        else if (a == "-w" || a == "--window") o.window = atoi(next().c_str());
        else if (a == "-d" || a == "--device") {
            std::string l = next();
            if (l == "all") o.all_devices = true;
            else for (size_t p = 0; p < l.size();) {
                size_t c = l.find(',', p);
                o.devices.push_back(atoi(l.substr(p, c - p).c_str()));
                p = c == std::string::npos ? l.size() : c + 1;
            }
        }
        else if (a == "--batch-log2") o.batch_log2 = atoi(next().c_str());
        else if (a == "--mc") o.mc = atoi(next().c_str());
        else if (a == "--bench") o.bench = atof(next().c_str());
        else if (a == "--no-l2-persist") o.l2persist = false;
        else if (a == "--selftest") o.selftest = true;
        else if (a == "--tune") o.tune = true;
        else if (a == "--profile") g_prof = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (!a.empty() && a[0] == '-') { fprintf(stderr, "error: unknown option %s\n", a.c_str()); usage(); return 1; }
        else pos.push_back(a);
    }
    if (pos.size() >= 1) o.pats.push_back({"", pos[0]});
    if (pos.size() >= 2) o.count = atoll(pos[1].c_str());
    if (o.window && !gen_for(o.window)) { fprintf(stderr, "error: window must be 8..18\n"); return 1; }
    if (o.batch_log2 < 16 || o.batch_log2 > 25 || o.mc < 1 || ((1u << o.batch_log2) % o.mc)) {
        fprintf(stderr, "error: bad --batch-log2 / --mc\n"); return 1;
    }
    if (o.selftest) return selftest(o);
    if (o.tune) return tune(o);
    if (o.pats.empty()) o.pats.push_back({"", "pump"});

    PatternSet ps;
    ps.icase = o.icase;
    if (!build_patterns(ps, o.pats)) return 1;

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (o.all_devices) { o.devices.clear(); for (int i = 0; i < ndev; i++) o.devices.push_back(i); }
    if (o.devices.empty()) o.devices.push_back(0);
    for (int id : o.devices) if (id < 0 || id >= ndev) { fprintf(stderr, "error: no GPU %d\n", id); return 1; }

    if (o.out.empty()) {
        if (o.pats.size() == 1 && o.pats[0].first.empty()) o.out = o.pats[0].second + "_results.csv";
        else o.out = "vanity_results.csv";
    }
    printf("Patterns:");
    for (auto &l : ps.labels) printf(" %s", l.c_str());
    printf("%s\n", ps.icase ? "  (case-insensitive)" : "");
    printf("Odds:    1 in %.4g keys per match\n", 1.0 / ps.prob);
    printf("Output:  %s\n", o.out.c_str());
    if (o.count > 0) printf("Target:  %lld keypairs\n", o.count);
    else if (o.bench <= 0) printf("Target:  unlimited (Ctrl+C to stop)\n");

    bool exists = false;
    if (FILE *f = fopen(o.out.c_str(), "r")) { exists = true; fclose(f); }
    FILE *csv = fopen(o.out.c_str(), "a");
    if (!csv) { fprintf(stderr, "cannot open %s\n", o.out.c_str()); return 1; }
    if (!exists) fprintf(csv, "public_key,private_key\n");
    fflush(csv);

    std::vector<Dev> devs(o.devices.size());
    for (size_t k = 0; k < devs.size(); k++) dev_init(devs[k], o.devices[k], o, ps);
    printf("\n");
    fflush(stdout);

    if (g_prof) {   // per-kernel time on the first device, serial batches
        Dev &d = devs[0];
        CUDA_CHECK(cudaSetDevice(d.id));
        for (int k = 0; k < 4; k++) CUDA_CHECK(cudaEventCreate(&g_prof_ev[k]));
        double t[3] = {0, 0, 0};
        const int reps = 20;
        for (int r = 0; r < reps + 2; r++) {
            dev_enqueue(d);
            Slot &sl = d.slot[d.cur ^ 1];
            CUDA_CHECK(cudaEventSynchronize(sl.done));
            sl.busy = false;
            if (r < 2) continue;
            for (int k = 0; k < 3; k++) { float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, g_prof_ev[k], g_prof_ev[k + 1])); t[k] += ms; }
        }
        const char *nm[3] = { "k_gen", "k_inv", "k_check" };
        double tot = t[0] + t[1] + t[2];
        for (int k = 0; k < 3; k++)
            printf("  %-8s %8.3f ms/batch  %6.2f ns/key  %5.1f%%\n", nm[k], t[k] / reps, t[k] / reps * 1e6 / d.nk, 100 * t[k] / tot);
        printf("  total    %8.3f ms/batch  -> %.1fM keys/sec\n", tot / reps, d.nk / (tot / reps) / 1e3);
        return 0;
    }

    Stats st;
    auto t_start = std::chrono::steady_clock::now(), t_last = t_start;
    u64 keys_last = 0;
    for (auto &d : devs) { dev_enqueue(d); dev_enqueue(d); }
    bool stop = false;
    while (!stop) {
        for (auto &d : devs) {
            dev_drain_one(d, ps, csv, st, o.count, o.quiet_matches);
            if (o.count > 0 && (long long)st.found >= o.count) stop = true;
            if (!stop) dev_enqueue(d);
        }
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - t_last).count();
        if (dt >= 2.0) {
            double rate = (double)(st.keys - keys_last) / dt;
            fprintf(stderr, "  tried ~%14llu | %12.0f keys/sec | found: %llu | ETA/match %.3gs\n",
                    (unsigned long long)st.keys, rate, (unsigned long long)st.found, 1.0 / (ps.prob * rate));
            t_last = now; keys_last = st.keys;
        }
        if (o.bench > 0 && std::chrono::duration<double>(now - t_start).count() >= o.bench) stop = true;
    }
    for (auto &d : devs) for (int k = 0; k < 2; k++) {
        Slot &sl = d.slot[d.cur];
        if (sl.busy) { CUDA_CHECK(cudaSetDevice(d.id)); CUDA_CHECK(cudaEventSynchronize(sl.done)); sl.busy = false; }   // results discarded
        d.cur ^= 1;
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("\nDone. %llu keypairs in %.2fs (%.2fM keys/sec average). Saved to %s\n",
           (unsigned long long)st.found, elapsed, st.keys / elapsed / 1e6, o.out.c_str());
    if (st.mismatched) printf("WARNING: %llu GPU matches failed the host check and were discarded\n", (unsigned long long)st.mismatched);
    if (st.dropped) printf("WARNING: %llu candidates/matches dropped (pattern too short for the buffers)\n", (unsigned long long)st.dropped);
    fclose(csv);
    return st.mismatched ? 2 : 0;
}
