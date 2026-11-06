// WGSL kernels for the browser grinder. Same pipeline as the CUDA version:
//   gen: seed = ChaCha12(key, index) -> SHA-512 -> clamp -> comb over a W-bit window table -> (X, Y, Z)
//   inv: Montgomery batch inversion (MC keys per thread) -> public key -> gate -> exact base58 check
import { FE13_WGSL } from "./fe13.wgsl.js";
import { FOLD } from "./ref.js";

const COMMON = /* wgsl */ `
${FE13_WGSL}

fn fe_zero() -> Fe { var r: Fe; return r; }
fn fe_one() -> Fe { var r: Fe; r[0] = 1u; return r; }

fn fe_invert(z: Fe) -> Fe {
    var t0 = fe_sq(z);
    var t1 = fe_sq(fe_sq(t0));
    t1 = fe_mul(z, t1);
    t0 = fe_mul(t0, t1);
    var t2 = fe_sq(t0);
    t1 = fe_mul(t2, t1);
    t2 = fe_sq(t1); for (var i = 1; i < 5; i++) { t2 = fe_sq(t2); }
    t1 = fe_mul(t2, t1);
    t2 = fe_sq(t1); for (var i = 1; i < 10; i++) { t2 = fe_sq(t2); }
    t2 = fe_mul(t2, t1);
    var t3 = fe_sq(t2); for (var i = 1; i < 20; i++) { t3 = fe_sq(t3); }
    t2 = fe_mul(t3, t2);
    t2 = fe_sq(t2); for (var i = 1; i < 10; i++) { t2 = fe_sq(t2); }
    t1 = fe_mul(t2, t1);
    t2 = fe_sq(t1); for (var i = 1; i < 50; i++) { t2 = fe_sq(t2); }
    t2 = fe_mul(t2, t1);
    t3 = fe_sq(t2); for (var i = 1; i < 100; i++) { t3 = fe_sq(t3); }
    t2 = fe_mul(t3, t2);
    t2 = fe_sq(t2); for (var i = 1; i < 50; i++) { t2 = fe_sq(t2); }
    t1 = fe_mul(t2, t1);
    t1 = fe_sq(t1); for (var i = 1; i < 5; i++) { t1 = fe_sq(t1); }
    return fe_mul(t1, t0);
}

// Fully reduced value (< p) as limbs.
fn fe_canon(a: Fe) -> Fe {
    var r = fe_carry(a);
    for (var rep = 0; rep < 2; rep++) {
        let c0 = r[19] >> 8u;
        r[19] &= 0xffu;
        r[0] += 19u * c0;
        var c: u32 = 0u;
        for (var i = 0; i < 19; i++) { r[i] += c; c = r[i] >> 13u; r[i] &= 0x1fffu; }
        r[19] += c;
    }
    var s = r;
    s[0] += 19u;
    var c: u32 = 0u;
    for (var i = 0; i < 19; i++) { s[i] += c; c = s[i] >> 13u; s[i] &= 0x1fffu; }
    s[19] += c;
    if ((s[19] >> 8u) != 0u) { s[19] &= 0xffu; return s; }
    return r;
}

struct Pt { x: Fe, y: Fe, z: Fe, t: Fe }

fn bswap(x: u32) -> u32 {
    return ((x & 0xffu) << 24u) | ((x & 0xff00u) << 8u) | ((x >> 8u) & 0xff00u) | (x >> 24u);
}
`;

// ---------------------------------------------------------------------------------------------
// Table generation
// ---------------------------------------------------------------------------------------------
export function tableShader(cfg) {
    return /* wgsl */ `
${COMMON}

struct TP { first: u32, count: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> tp: TP;
@group(0) @binding(1) var<storage, read> bases: array<u32>;        // (nwin + 1) affine points: x[8], y[8]
@group(0) @binding(2) var<storage, read_write> table: array<u32>;  // entries of 24 words

const HALF: i32 = ${cfg.half};
const NWIN: u32 = ${cfg.nwin}u;
const W0N: u32 = ${cfg.w0n}u;
const ENTRIES: u32 = ${cfg.entries}u;

fn ge_add(p: Pt, q: Pt) -> Pt {
    let a = fe_mul(fe_sub(p.y, p.x), fe_sub(q.y, q.x));
    let b = fe_mul(fe_add(p.y, p.x), fe_add(q.y, q.x));
    let c = fe_mul(fe_mul(p.t, q.t), FE_D2);
    let zz = fe_mul(p.z, q.z);
    let d = fe_add(zz, zz);
    let e = fe_sub(b, a); let f = fe_sub(d, c); let g = fe_add(d, c); let h = fe_add(b, a);
    return Pt(fe_mul(e, f), fe_mul(h, g), fe_mul(g, f), fe_mul(e, h));
}

fn load_base(k: u32) -> Pt {
    var wx: array<u32, 8>; var wy: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { wx[i] = bases[k * 16u + i]; wy[i] = bases[k * 16u + 8u + i]; }
    let x = fe_unpack(wx); let y = fe_unpack(wy);
    return Pt(x, y, fe_one(), fe_mul(x, y));
}

@compute @workgroup_size(64)
fn table_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= tp.count) { return; }
    let idx = tp.first + gid.x;
    if (idx >= ENTRIES) { return; }
    var k: u32; var j: i32;
    if (idx < W0N) { k = 0u; j = i32(idx) - HALF; }
    else {
        let r = idx - W0N;
        k = 1u + r / u32(HALF); j = i32(r % u32(HALF)) + 1;
        if (k > NWIN - 1u) { k = NWIN - 1u; j = i32(r - (NWIN - 2u) * u32(HALF)) + 1; }
    }
    let aj = u32(abs(j));
    let b = load_base(k);
    var p = Pt(fe_zero(), fe_one(), fe_one(), fe_zero());
    for (var bit = 31; bit >= 0; bit--) {
        p = ge_add(p, p);
        if (((aj >> u32(bit)) & 1u) != 0u) { p = ge_add(p, b); }
    }
    if (j < 0) { p.x = fe_sub(fe_zero(), p.x); p.t = fe_sub(fe_zero(), p.t); }
    if (k == 0u) { p = ge_add(p, load_base(NWIN)); }
    let zi = fe_invert(p.z);
    let x = fe_mul(p.x, zi);
    let y = fe_mul(p.y, zi);
    let xy = fe_mul(x, y);
    var w0: array<u32, 8>; var w1: array<u32, 8>; var w2: array<u32, 8>;
    if (k == 0u) {
        w0 = fe_pack(fe_canon(x)); w1 = fe_pack(fe_canon(y)); w2 = fe_pack(fe_canon(xy));
    } else {
        w0 = fe_pack(fe_canon(fe_add(y, x))); w1 = fe_pack(fe_canon(fe_sub(y, x))); w2 = fe_pack(fe_canon(fe_mul(xy, FE_D2)));
    }
    for (var i = 0u; i < 8u; i++) {
        table[idx * 24u + i] = w0[i];
        table[idx * 24u + 8u + i] = w1[i];
        table[idx * 24u + 16u + i] = w2[i];
    }
}
`;
}

// ---------------------------------------------------------------------------------------------
// Grinding kernels
// ---------------------------------------------------------------------------------------------
export function mainShader(cfg) {
    const fold = FOLD.map((v) => `${v}u`).join(", ");
    return /* wgsl */ `
${COMMON}

struct Params {
    key: array<vec4<u32>, 2>,
    bitmap: array<vec4<u32>, 2>,
    base_lo: u32, base_hi: u32, nonce: u32, nk: u32,
    nthreads: u32, mc: u32, npat: u32, icase: u32,
    nranges: u32, match_cap: u32, dump: u32, pad: u32,
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read_write> BX: array<u32>;
@group(0) @binding(3) var<storage, read_write> BY: array<u32>;
@group(0) @binding(4) var<storage, read_write> BZ: array<u32>;
@group(0) @binding(5) var<storage, read_write> BP: array<u32>;
@group(0) @binding(6) var<storage, read> pats: array<u32>;      // per pattern: plen, slen, pre[44], suf[44]
@group(0) @binding(7) var<storage, read> ranges: array<u32>;    // per range: lo_hi, lo_lo, hi_hi, hi_lo
@group(0) @binding(8) var<storage, read_write> outb: array<atomic<u32>>;  // [count, pad, pad, pad, records...]
@group(0) @binding(9) var<storage, read_write> dumpb: array<u32>;

const W: u32 = ${cfg.w}u;
const NWIN: u32 = ${cfg.nwin}u;
const HALF: u32 = ${cfg.half}u;
const W0N: u32 = ${cfg.w0n}u;
const FOLD = array<u32, 58>(${fold});

// ---------------- ChaCha12 ----------------
fn rotl(x: u32, n: u32) -> u32 { return (x << n) | (x >> (32u - n)); }
fn chacha12_seed(ctr_lo: u32, ctr_hi: u32) -> array<u32, 8> {
    var x = array<u32, 16>(0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u,
        P.key[0].x, P.key[0].y, P.key[0].z, P.key[0].w, P.key[1].x, P.key[1].y, P.key[1].z, P.key[1].w,
        ctr_lo, ctr_hi, P.nonce, 0x76616e74u);
    let j = x;
    for (var r = 0; r < 6; r++) {
        for (var q = 0; q < 2; q++) {
            // column round then diagonal round
            var ia: array<u32, 4>; var ib: array<u32, 4>; var ic: array<u32, 4>; var id: array<u32, 4>;
            if (q == 0) { ia = array<u32,4>(0u,1u,2u,3u); ib = array<u32,4>(4u,5u,6u,7u); ic = array<u32,4>(8u,9u,10u,11u); id = array<u32,4>(12u,13u,14u,15u); }
            else { ia = array<u32,4>(0u,1u,2u,3u); ib = array<u32,4>(5u,6u,7u,4u); ic = array<u32,4>(10u,11u,8u,9u); id = array<u32,4>(15u,12u,13u,14u); }
            for (var n = 0; n < 4; n++) {
                let a = ia[n]; let b = ib[n]; let c = ic[n]; let d = id[n];
                x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 16u);
                x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 12u);
                x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 8u);
                x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 7u);
            }
        }
    }
    var o: array<u32, 8>;
    for (var i = 0; i < 8; i++) { o[i] = x[i] + j[i]; }
    return o;
}

// ---------------- SHA-512 on (hi, lo) pairs ----------------
const K512 = array<vec2<u32>, 80>(
${K512_LIST}
);
fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.y + b.y;
    return vec2<u32>(a.x + b.x + select(0u, 1u, lo < a.y), lo);
}
fn rotr_lt32(a: vec2<u32>, n: u32) -> vec2<u32> {
    return vec2<u32>((a.x >> n) | (a.y << (32u - n)), (a.y >> n) | (a.x << (32u - n)));
}
fn rotr_ge32(a: vec2<u32>, n: u32) -> vec2<u32> { return rotr_lt32(vec2<u32>(a.y, a.x), n - 32u); }
fn shr_lt32(a: vec2<u32>, n: u32) -> vec2<u32> { return vec2<u32>(a.x >> n, (a.y >> n) | (a.x << (32u - n))); }

// SHA-512 of a 32-byte seed -> clamped scalar, 8 little-endian words.
fn sha512_scalar(sw: array<u32, 8>) -> array<u32, 8> {
    var w: array<vec2<u32>, 16>;
    for (var i = 0; i < 4; i++) { w[i] = vec2<u32>(bswap(sw[2 * i]), bswap(sw[2 * i + 1])); }
    w[4] = vec2<u32>(0x80000000u, 0u);
    w[15] = vec2<u32>(0u, 256u);
    var a = vec2<u32>(0x6a09e667u, 0xf3bcc908u); var b = vec2<u32>(0xbb67ae85u, 0x84caa73bu);
    var c = vec2<u32>(0x3c6ef372u, 0xfe94f82bu); var d = vec2<u32>(0xa54ff53au, 0x5f1d36f1u);
    var e = vec2<u32>(0x510e527fu, 0xade682d1u); var f = vec2<u32>(0x9b05688cu, 0x2b3e6c1fu);
    var g = vec2<u32>(0x1f83d9abu, 0xfb41bd6bu); var h = vec2<u32>(0x5be0cd19u, 0x137e2179u);
    for (var i = 0u; i < 80u; i++) {
        if (i >= 16u) {
            let w15 = w[(i - 15u) & 15u]; let w2 = w[(i - 2u) & 15u];
            let s0 = rotr_lt32(w15, 1u) ^ rotr_lt32(w15, 8u) ^ shr_lt32(w15, 7u);
            let s1 = rotr_lt32(w2, 19u) ^ rotr_ge32(w2, 61u) ^ shr_lt32(w2, 6u);
            w[i & 15u] = add64(add64(w[i & 15u], s0), add64(w[(i - 7u) & 15u], s1));
        }
        let S1 = rotr_lt32(e, 14u) ^ rotr_lt32(e, 18u) ^ rotr_ge32(e, 41u);
        let ch = (e & f) ^ (~e & g);
        let t1 = add64(add64(add64(h, S1), add64(ch, K512[i])), w[i & 15u]);
        let S0 = rotr_lt32(a, 28u) ^ rotr_ge32(a, 34u) ^ rotr_ge32(a, 39u);
        let mj = (a & b) ^ (a & c) ^ (b & c);
        h = g; g = f; f = e; e = add64(d, t1);
        d = c; c = b; b = a; a = add64(t1, add64(S0, mj));
    }
    let st = array<vec2<u32>, 4>(add64(a, vec2<u32>(0x6a09e667u, 0xf3bcc908u)), add64(b, vec2<u32>(0xbb67ae85u, 0x84caa73bu)),
                                 add64(c, vec2<u32>(0x3c6ef372u, 0xfe94f82bu)), add64(d, vec2<u32>(0xa54ff53au, 0x5f1d36f1u)));
    var s: array<u32, 8>;
    for (var k = 0; k < 4; k++) { s[2 * k] = bswap(st[k].x); s[2 * k + 1] = bswap(st[k].y); }
    s[0] &= 0xfffffff8u;
    s[7] &= 0x7fffffffu;
    s[7] |= 0x40000000u;
    return s;
}

// ---------------- comb ----------------
fn load_fe(e: u32, slot: u32) -> Fe {
    var w: array<u32, 8>;
    let base = e * 24u + slot * 8u;
    for (var i = 0u; i < 8u; i++) { w[i] = table[base + i]; }
    return fe_unpack(w);
}

@compute @workgroup_size(64)
fn gen_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.nk) { return; }
    let lo = P.base_lo + i;
    let hi = P.base_hi + select(0u, 1u, lo < P.base_lo);
    let s = sha512_scalar(chacha12_seed(lo, hi));
    // t = (s - 2^254) >> 3
    var t: array<u32, 8>;
    for (var k = 0; k < 7; k++) { t[k] = (s[k] >> 3u) | (s[k + 1] << 29u); }
    t[7] = (s[7] >> 3u) & 0x07ffffffu;

    var pos = 0u;
    var carry = 0u;
    // window 0: start point table (affine X, Y, XY)
    var d = i32(extract_bits(t, 0u)) ;
    carry = select(0u, 1u, d > i32(HALF));
    if (carry != 0u) { d -= i32(1u << W); }
    let e0 = u32(d + i32(HALF));
    var X = load_fe(e0, 0u); var Y = load_fe(e0, 1u); var T = load_fe(e0, 2u); var Z = fe_one();
    for (var k = 1u; k < NWIN; k++) {
        var dk = i32(extract_bits(t, k * W)) + i32(carry);
        if (k < NWIN - 1u) {
            carry = select(0u, 1u, dk > i32(HALF));
            if (carry != 0u) { dk -= i32(1u << W); }
        }
        if (dk != 0) {
            let neg = dk < 0;
            let e = W0N + (k - 1u) * HALF + u32(abs(dk)) - 1u;
            let a = fe_mul(fe_sub(Y, X), load_fe(e, select(1u, 0u, neg)));
            let b = fe_mul(fe_add(Y, X), load_fe(e, select(0u, 1u, neg)));
            let c = fe_mul(T, load_fe(e, 2u));
            let dd = fe_add(Z, Z);
            let ee = fe_sub(b, a); let h = fe_add(b, a);
            var f = fe_sub(dd, c); var g = fe_add(dd, c);
            if (neg) { let tmp = f; f = g; g = tmp; }
            X = fe_mul(ee, f); Y = fe_mul(h, g); Z = fe_mul(g, f);
            if (k < NWIN - 1u) { T = fe_mul(ee, h); }
        }
    }
    let wx = fe_pack(X); let wy = fe_pack(Y); let wz = fe_pack(Z);
    for (var l = 0u; l < 8u; l++) {
        BX[l * P.nk + i] = wx[l]; BY[l * P.nk + i] = wy[l]; BZ[l * P.nk + i] = wz[l];
    }
}

fn extract_bits(t: array<u32, 8>, pos: u32) -> u32 {
    let wi = pos >> 5u; let off = pos & 31u;
    var v = t[wi] >> off;
    if (off + W > 32u && wi < 7u) { v |= t[wi + 1u] << (32u - off); }
    return v & ((1u << W) - 1u);
}

// ---------------- inversion + match ----------------
fn ld(buf: u32, i: u32) -> Fe {
    var w: array<u32, 8>;
    for (var l = 0u; l < 8u; l++) {
        if (buf == 0u) { w[l] = BX[l * P.nk + i]; }
        else if (buf == 1u) { w[l] = BY[l * P.nk + i]; }
        else if (buf == 2u) { w[l] = BZ[l * P.nk + i]; }
        else { w[l] = BP[l * P.nk + i]; }
    }
    return fe_unpack(w);
}
fn st_p(i: u32, a: Fe) {
    let w = fe_pack(a);
    for (var l = 0u; l < 8u; l++) { BP[l * P.nk + i] = w[l]; }
}

fn check_key(i: u32, w: array<u32, 8>) {
    let low = w[7] >> 24u;
    let bm = P.bitmap[low >> 7u][(low >> 5u) & 3u];
    var hit = ((bm >> (low & 31u)) & 1u) != 0u;
    if (!hit && P.nranges > 0u) {
        let thi = bswap(w[0]); let tlo = bswap(w[1]);
        for (var r = 0u; r < P.nranges; r++) {
            let lh = ranges[r * 4u]; let ll = ranges[r * 4u + 1u]; let hh = ranges[r * 4u + 2u]; let hl = ranges[r * 4u + 3u];
            let ge_lo = thi > lh || (thi == lh && tlo >= ll);
            let le_hi = thi < hh || (thi == hh && tlo <= hl);
            if (ge_lo && le_hi) { hit = true; break; }
        }
    }
    if (!hit) { return; }
    // exact base58: N = key bytes big-endian, as 16-bit limbs (most significant first)
    var n: array<u32, 16>;
    var zeros = 0u; var counting = true;
    for (var b = 0u; b < 32u; b++) {
        let bv = (w[b >> 2u] >> (8u * (b & 3u))) & 0xffu;
        if (counting && bv == 0u) { zeros++; } else { counting = false; }
        if ((b & 1u) == 0u) { n[b >> 1u] = bv << 8u; } else { n[b >> 1u] |= bv; }
    }
    var dig: array<u32, 48>;
    var nd = 0u;
    var start = 0u;
    loop {
        while (start < 16u && n[start] == 0u) { start++; }
        if (start >= 16u) { break; }
        var rem = 0u;
        for (var j = start; j < 16u; j++) {
            let cur = (rem << 16u) | n[j];
            n[j] = cur / 58u; rem = cur % 58u;
        }
        dig[nd] = rem; nd++;
    }
    let tot = zeros + nd;
    for (var p = 0u; p < P.npat; p++) {
        let base = p * 90u;
        let plen = pats[base]; let slen = pats[base + 1u];
        if (plen > tot || slen > tot) { continue; }
        var ok = true;
        for (var q = 0u; q < plen && ok; q++) {
            var c = 0u;
            if (q >= zeros) { c = dig[nd - 1u - (q - zeros)]; }
            if (P.icase != 0u) { c = FOLD[c]; }
            ok = c == pats[base + 2u + q];
        }
        for (var q = 0u; q < slen && ok; q++) {
            let pos = tot - slen + q;
            var c = 0u;
            if (pos >= zeros) { c = dig[nd - 1u - (pos - zeros)]; }
            if (P.icase != 0u) { c = FOLD[c]; }
            ok = c == pats[base + 46u + q];
        }
        if (ok) {
            let slot = atomicAdd(&outb[0], 1u);
            if (slot < P.match_cap) {
                let o = 4u + slot * 12u;
                atomicStore(&outb[o], i);
                atomicStore(&outb[o + 1u], p);
                for (var q = 0u; q < 8u; q++) { atomicStore(&outb[o + 2u + q], w[q]); }
            }
            return;
        }
    }
}

@compute @workgroup_size(64)
fn inv_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= P.nthreads) { return; }
    let T = P.nthreads; let MC = P.mc;
    var acc = ld(2u, t);
    for (var j = 1u; j < MC; j++) {
        st_p(t + (j - 1u) * T, acc);
        acc = fe_mul(acc, ld(2u, t + j * T));
    }
    var inv = fe_invert(acc);
    for (var jj = i32(MC) - 1; jj >= 0; jj--) {
        let i = t + u32(jj) * T;
        var iz: Fe;
        if (jj > 0) {
            iz = fe_mul(inv, ld(3u, i - T));
            inv = fe_mul(inv, ld(2u, i));
        } else { iz = inv; }
        let y = fe_canon(fe_mul(ld(1u, i), iz));
        let x = fe_canon(fe_mul(ld(0u, i), iz));
        var w = fe_pack(y);
        w[7] ^= (x[0] & 1u) << 31u;
        if (P.dump != 0u) { for (var q = 0u; q < 8u; q++) { dumpb[i * 8u + q] = w[q]; } }
        check_key(i, w);
    }
}
`.replace("${K512_LIST}", K512_LIST);
}

const K512_HEX = [
    "428a2f98d728ae22", "7137449123ef65cd", "b5c0fbcfec4d3b2f", "e9b5dba58189dbbc", "3956c25bf348b538", "59f111f1b605d019",
    "923f82a4af194f9b", "ab1c5ed5da6d8118", "d807aa98a3030242", "12835b0145706fbe", "243185be4ee4b28c", "550c7dc3d5ffb4e2",
    "72be5d74f27b896f", "80deb1fe3b1696b1", "9bdc06a725c71235", "c19bf174cf692694", "e49b69c19ef14ad2", "efbe4786384f25e3",
    "0fc19dc68b8cd5b5", "240ca1cc77ac9c65", "2de92c6f592b0275", "4a7484aa6ea6e483", "5cb0a9dcbd41fbd4", "76f988da831153b5",
    "983e5152ee66dfab", "a831c66d2db43210", "b00327c898fb213f", "bf597fc7beef0ee4", "c6e00bf33da88fc2", "d5a79147930aa725",
    "06ca6351e003826f", "142929670a0e6e70", "27b70a8546d22ffc", "2e1b21385c26c926", "4d2c6dfc5ac42aed", "53380d139d95b3df",
    "650a73548baf63de", "766a0abb3c77b2a8", "81c2c92e47edaee6", "92722c851482353b", "a2bfe8a14cf10364", "a81a664bbc423001",
    "c24b8b70d0f89791", "c76c51a30654be30", "d192e819d6ef5218", "d69906245565a910", "f40e35855771202a", "106aa07032bbd1b8",
    "19a4c116b8d2d0c8", "1e376c085141ab53", "2748774cdf8eeb99", "34b0bcb5e19b48a8", "391c0cb3c5c95a63", "4ed8aa4ae3418acb",
    "5b9cca4f7763e373", "682e6ff3d6b2b8a3", "748f82ee5defb2fc", "78a5636f43172f60", "84c87814a1f0ab72", "8cc702081a6439ec",
    "90befffa23631e28", "a4506cebde82bde9", "bef9a3f7b2c67915", "c67178f2e372532b", "ca273eceea26619c", "d186b8c721c0c207",
    "eada7dd6cde0eb1e", "f57d4f7fee6ed178", "06f067aa72176fba", "0a637dc5a2c898a6", "113f9804bef90dae", "1b710b35131c471b",
    "28db77f523047d84", "32caab7b40c72493", "3c9ebe0a15c9bebc", "431d67c49c100d4c", "4cc5d4becb3e42b6", "597f299cfc657e2a",
    "5fcb6fab3ad6faec", "6c44198c4a475817"];
const K512_LIST = K512_HEX.map((h) => `    vec2<u32>(0x${h.slice(0, 8)}u, 0x${h.slice(8)}u)`).join(",\n");

export function combCfg(w) {
    const nwin = Math.ceil(251 / w), half = 1 << (w - 1);
    const topbits = 251 - w * (nwin - 1);
    const w0n = 2 * half + 1;
    return { w, nwin, half, topbits, w0n, entries: w0n + (nwin - 2) * half + (1 << topbits) };
}
