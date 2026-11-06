// Reference ed25519 (BigInt), ChaCha12 seeds, base58 and pattern maths.
// Used to precompute curve points for the GPU and to re-check every GPU match.

export const P = (1n << 255n) - 19n;
export const L_ORDER = (1n << 252n) + 27742317777372353535851937790883648493n;
const mod = (a) => { a %= P; return a < 0n ? a + P : a; };
export function powmod(b, e) {
    let r = 1n; b = mod(b);
    while (e > 0n) { if (e & 1n) r = r * b % P; b = b * b % P; e >>= 1n; }
    return r;
}
export const inv = (a) => powmod(a, P - 2n);
export const D = mod(-121665n * inv(121666n));
export const D2 = mod(2n * D);
const GX = 15112221349535400772501151409588531511454012693041857206046113283949847762202n;
const GY = 46316835694926478169428394003475163141307993866256225615783033603165251855960n;

// Extended coordinates (X, Y, Z, T), twisted Edwards a = -1.
export function ptAdd(p, q) {
    const a = mod((p[1] - p[0]) * (q[1] - q[0]));
    const b = mod((p[1] + p[0]) * (q[1] + q[0]));
    const c = mod(p[3] * D2 % P * q[3]);
    const d = mod(2n * p[2] * q[2]);
    const e = b - a, f = d - c, g = d + c, h = b + a;
    return [mod(e * f), mod(h * g), mod(g * f), mod(e * h)];
}
export const G = [GX, GY, 1n, mod(GX * GY)];
export const ID = [0n, 1n, 1n, 0n];
export function ptMul(p, k) {
    let r = ID, q = p;
    while (k > 0n) { if (k & 1n) r = ptAdd(r, q); q = ptAdd(q, q); k >>= 1n; }
    return r;
}
export function affine(p) { const zi = inv(p[2]); return [mod(p[0] * zi), mod(p[1] * zi)]; }

export function bigToWords(x) {       // 8 little-endian u32
    const w = new Uint32Array(8);
    for (let i = 0; i < 8; i++) { w[i] = Number(x & 0xffffffffn); x >>= 32n; }
    return w;
}
export function bytesToBigLE(b) { let x = 0n; for (let i = b.length - 1; i >= 0; i--) x = (x << 8n) | BigInt(b[i]); return x; }

export function encodePoint(p) {      // 32-byte public key from extended point
    const [x, y] = affine(p);
    const out = new Uint8Array(32);
    let v = y;
    for (let i = 0; i < 32; i++) { out[i] = Number(v & 0xffn); v >>= 8n; }
    out[31] |= Number(x & 1n) << 7;
    return out;
}

async function sha512(bytes) { return new Uint8Array(await crypto.subtle.digest("SHA-512", bytes)); }

// seed -> public key, plain RFC 8032 (no tables, no shortcuts)
export async function pubkeyFromSeed(seed) {
    const h = await sha512(seed);
    const a = h.slice(0, 32);
    a[0] &= 248; a[31] &= 127; a[31] |= 64;
    return encodePoint(ptMul(G, bytesToBigLE(a)));
}

// ---- ChaCha12: seed(i) = first 32 bytes of ChaCha12(key, counter = i, nonce) ----
const rotl = (x, n) => ((x << n) | (x >>> (32 - n))) >>> 0;
export function chacha12Seed(key, counter, nonce) {   // key: Uint32Array(8), counter: BigInt
    const c = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574];
    const x = new Uint32Array(16);
    x.set(c, 0); x.set(key, 4);
    x[12] = Number(counter & 0xffffffffn); x[13] = Number((counter >> 32n) & 0xffffffffn);
    x[14] = nonce >>> 0; x[15] = 0x76616e74;
    const j = x.slice();
    const qr = (a, b, cc, d) => {
        x[a] = (x[a] + x[b]) >>> 0; x[d] = rotl(x[d] ^ x[a], 16);
        x[cc] = (x[cc] + x[d]) >>> 0; x[b] = rotl(x[b] ^ x[cc], 12);
        x[a] = (x[a] + x[b]) >>> 0; x[d] = rotl(x[d] ^ x[a], 8);
        x[cc] = (x[cc] + x[d]) >>> 0; x[b] = rotl(x[b] ^ x[cc], 7);
    };
    for (let r = 0; r < 6; r++) {
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15);
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14);
    }
    const out = new Uint8Array(32);
    for (let i = 0; i < 8; i++) {
        const v = (x[i] + j[i]) >>> 0;
        out[4 * i] = v & 255; out[4 * i + 1] = (v >>> 8) & 255; out[4 * i + 2] = (v >>> 16) & 255; out[4 * i + 3] = v >>> 24;
    }
    return out;
}

// ---- base58 ----
export const B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
export function b58encode(bytes) {
    let n = 0n;
    for (const b of bytes) n = (n << 8n) | BigInt(b);
    let s = "";
    while (n > 0n) { s = B58[Number(n % 58n)] + s; n /= 58n; }
    for (const b of bytes) { if (b !== 0) break; s = "1" + s; }
    return s;
}

// ---- patterns ----
// Letter-case classes for case-insensitive matching (digit index -> class id).
export const FOLD = Array.from(B58, (ch) => {
    const lo = ch.toLowerCase(), up = ch.toUpperCase();
    if (lo !== up && B58.includes(lo) && B58.includes(up)) return B58.indexOf(lo);
    return B58.indexOf(ch);
});

function variants(s, icase) {
    let out = [""];
    for (const ch of s) {
        let opts;
        if (icase && /[a-z]/i.test(ch)) opts = [ch.toLowerCase(), ch.toUpperCase()].filter((c, i, a) => B58.includes(c) && a.indexOf(c) === i);
        else opts = B58.includes(ch) ? [ch] : [];
        const next = [];
        for (const o of out) for (const c of opts) next.push(o + c);
        out = next;
        if (out.length > 65536) throw new Error("too many case variants");
    }
    return out;
}
const numOf = (s) => { let v = 0n; for (const c of s) v = v * 58n + BigInt(B58.indexOf(c)); return v; };

// Integer ranges [lo, hi) of N (key bytes read big-endian) whose address starts with p.
function prefixRanges(p) {
    let z = 0;
    while (z < p.length && p[z] === "1") z++;
    const r = p.slice(z), out = [];
    if (z > 30) return out;
    const limHi = 1n << BigInt(256 - 8 * z), limLo = r ? 1n << BigInt(248 - 8 * z) : 0n;
    const add = (lo, hi) => { if (lo < limLo) lo = limLo; if (hi > limHi) hi = limHi; if (lo < hi) out.push([lo, hi]); };
    if (!r) { add(limLo, limHi); return out; }
    let lo = numOf(r), hi = lo + 1n;
    for (let m = 0; m < 50 && lo < limHi; m++) { add(lo, hi); lo *= 58n; hi *= 58n; }
    return out;
}

// patterns: [{prefix, suffix}], returns everything the GPU and the UI need
export function buildPatterns(list, icase) {
    const bitmap = new Uint32Array(8);
    let ranges = [], prob = 0;
    for (const { prefix = "", suffix = "" } of list) {
        if (!prefix && !suffix) throw new Error("empty pattern");
        if (prefix.length + suffix.length > 44) throw new Error("pattern too long");
        for (const s of [prefix, suffix]) {
            if (s && variants(s, icase).length === 0)
                throw new Error(`"${s}" has characters outside base58 (0, O, I, l are not used)`);
        }
        let ppre = 1;
        if (prefix) {
            let tot = 0n;
            const rr = [];
            for (const v of variants(prefix, icase)) for (const [lo, hi] of prefixRanges(v)) { tot += hi - lo; rr.push([lo, hi]); }
            ppre = Number(tot >> 128n) / 2 ** 128;
            if (!suffix) ranges = ranges.concat(rr);
        }
        let psuf = 1;
        if (suffix) {
            const vs = variants(suffix, icase), K = suffix.length;
            psuf = vs.length / 58 ** K;
            const kb = Math.min(K, 8), m = (1 << kb) - 1;
            for (const v of vs) {
                const low = Number(numOf(v) & 0xffn);
                for (let b = 0; b < 256; b++) if ((b & m) === (low & m)) bitmap[b >> 5] |= 1 << (b & 31);
            }
        }
        prob += ppre * psuf;
    }
    // top-64-bit gate ranges, merged; shrunk to 64 by joining neighbours (superset)
    let r64 = ranges.map(([lo, hi]) => [lo >> 192n, (hi - 1n) >> 192n]).sort((a, b) => (a[0] < b[0] ? -1 : 1));
    const merged = [];
    for (const r of r64) {
        if (merged.length && r[0] <= merged[merged.length - 1][1] + 1n) {
            if (r[1] > merged[merged.length - 1][1]) merged[merged.length - 1][1] = r[1];
        } else merged.push([r[0], r[1]]);
    }
    while (merged.length > 64) {
        let best = 0, gap = null;
        for (let i = 0; i + 1 < merged.length; i++) {
            const g = merged[i + 1][0] - merged[i][1];
            if (gap === null || g < gap) { gap = g; best = i; }
        }
        merged[best][1] = merged[best + 1][1];
        merged.splice(best + 1, 1);
    }
    return { list, icase, bitmap, ranges: merged, prob };
}

export function matches(addr, pats) {
    const f = pats.icase ? (s) => s.toLowerCase() : (s) => s;
    const a = f(addr);
    return pats.list.some(({ prefix = "", suffix = "" }) => a.startsWith(f(prefix)) && a.endsWith(f(suffix)));
}

// Window bases for the comb: B_k = 8 * 2^(W k) * G for k < nwin, and S = 2^254 * G.
export function combBases(W, nwin) {
    const pts = [];
    for (let k = 0; k < nwin; k++) pts.push(affine(ptMul(G, 1n << BigInt(3 + W * k))));
    pts.push(affine(ptMul(G, 1n << 254n)));
    return pts;
}

export function keypairBase58(seed, pk) {
    const full = new Uint8Array(64);
    full.set(seed, 0); full.set(pk, 32);
    return b58encode(full);
}
