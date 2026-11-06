# Emits PTX-asm CUDA functions for GF(2^255-19) on 8x32-bit limbs (lazy, value < 2^256).
# fe8_mul: one level of Karatsuba on 128-bit halves (48 products), row-wise carry chains below.
# Usage: python tools/gen_fe8.py > out.h, then paste between the GENERATED markers in vanity.cu.
def mul():
    L=[]; a=lambda i:f"%{8+i}"; b=lambda j:f"%{16+j}"; t=lambda k:f"t{k}"
    L.append(".reg .u32 t<17>;")
    for k in range(17): L.append(f"mov.u32 t{k}, 0;")
    for i in range(8):
        for j in range(8):
            op = "mad.lo.cc.u32" if j == 0 else "madc.lo.cc.u32"
            L.append(f"{op} {t(i+j)}, {a(i)}, {b(j)}, {t(i+j)};")
        L.append(f"addc.u32 {t(i+8)}, {t(i+8)}, 0;")
        for j in range(8):
            op = "mad.hi.cc.u32" if j == 0 else ("madc.hi.cc.u32" if j < 7 else "madc.hi.u32")
            L.append(f"{op} {t(i+j+1)}, {a(i)}, {b(j)}, {t(i+j+1)};")
    L += red("t")
    return L
def sqr():
    L=[]; a=lambda i:f"%{8+i}"; t=lambda k:f"t{k}"
    L.append(".reg .u32 t<17>;")
    for k in range(17): L.append(f"mov.u32 t{k}, 0;")
    # off-diagonal a_i*a_j, i<j
    for i in range(7):
        js=list(range(i+1,8))
        for n,j in enumerate(js):
            op = "mad.lo.cc.u32" if n == 0 else "madc.lo.cc.u32"
            L.append(f"{op} {t(i+j)}, {a(i)}, {a(j)}, {t(i+j)};")
        L.append(f"addc.u32 {t(i+8)}, {t(i+8)}, 0;")
        for n,j in enumerate(js):
            last = n == len(js)-1
            op = "mad.hi.cc.u32" if n == 0 else "madc.hi.cc.u32"
            if last: op = op.replace(".cc","") if n>0 else "mad.hi.cc.u32"
            L.append(f"{op} {t(i+j+1)}, {a(i)}, {a(j)}, {t(i+j+1)};")
            if last and n == 0: L.append(f"addc.u32 {t(i+j+2)}, {t(i+j+2)}, 0;")
    # double
    L.append("add.cc.u32 t1, t1, t1;")
    for k in range(2,16): L.append(f"addc.cc.u32 t{k}, t{k}, t{k};")
    L.append("addc.u32 t16, 0, 0;")
    # diagonals
    for i in range(8):
        op_lo = "mad.lo.cc.u32" if i == 0 else "madc.lo.cc.u32"
        L.append(f"{op_lo} {t(2*i)}, {a(i)}, {a(i)}, {t(2*i)};")
        op_hi = "madc.hi.cc.u32" if i < 7 else "madc.hi.u32"
        L.append(f"{op_hi} {t(2*i+1)}, {a(i)}, {a(i)}, {t(2*i+1)};")
    L += red("t")
    return L
def red(t):
    # r = t[0..7] + 38*t[8..15]; then fold the small top carry twice.
    L=[]; r=lambda k:f"%{k}"
    L.append(".reg .u32 c;")
    for k in range(8):
        op = "mad.lo.cc.u32" if k == 0 else "madc.lo.cc.u32"
        L.append(f"{op} {r(k)}, t{8+k}, 38, t{k};")
    L.append("addc.u32 c, 0, 0;")
    for k in range(8):
        op = "mad.hi.cc.u32" if k == 0 else "madc.hi.cc.u32"
        dst = r(k+1) if k < 7 else "c"
        if k == 7: op = "madc.hi.u32"
        L.append(f"{op} {dst}, t{8+k}, 38, {dst};")
    L.append(f"mad.lo.cc.u32 {r(0)}, c, 38, {r(0)};")
    for k in range(1,8): L.append(f"addc.cc.u32 {r(k)}, {r(k)}, 0;")
    L.append("addc.u32 c, 0, 0;")
    L.append(f"mad.lo.u32 {r(0)}, c, 38, {r(0)};")   # cannot carry: r0 < 38*... after wrap
    return L

def add():
    L=[".reg .u32 c;"]; a=lambda i:f"%{8+i}"; b=lambda j:f"%{16+j}"; r=lambda k:f"%{k}"
    for k in range(8):
        op = "add.cc.u32" if k == 0 else "addc.cc.u32"
        L.append(f"{op} {r(k)}, {a(k)}, {b(k)};")
    L.append("addc.u32 c, 0, 0;")
    L.append(f"mad.lo.cc.u32 {r(0)}, c, 38, {r(0)};")
    for k in range(1,8): L.append(f"addc.cc.u32 {r(k)}, {r(k)}, 0;")
    L.append("addc.u32 c, 0, 0;")
    L.append(f"mad.lo.u32 {r(0)}, c, 38, {r(0)};")
    return L
def sub():
    L=[".reg .u32 c;"]; a=lambda i:f"%{8+i}"; b=lambda j:f"%{16+j}"; r=lambda k:f"%{k}"
    for k in range(8):
        op = "sub.cc.u32" if k == 0 else "subc.cc.u32"
        L.append(f"{op} {r(k)}, {a(k)}, {b(k)};")
    L.append("subc.u32 c, 0, 0;")
    L.append("and.b32 c, c, 38;")
    L.append(f"sub.cc.u32 {r(0)}, {r(0)}, c;")
    for k in range(1,8): L.append(f"subc.cc.u32 {r(k)}, {r(k)}, 0;")
    L.append("subc.u32 c, 0, 0;")
    L.append("and.b32 c, c, 38;")
    L.append(f"sub.u32 {r(0)}, {r(0)}, c;")
    return L
def emit(name, body, nin, clob=False):
    outs = ", ".join(('"=&r"' if clob else '"=r"') + f'(r[{k}])' for k in range(8))
    ins = ", ".join([f'"r"(a[{k}])' for k in range(8)] + ([f'"r"(b[{k}])' for k in range(8)] if nin==2 else [f'"r"(a[{k}])' for k in range(8)]))
    sig = "u32 r[8], const u32 a[8], const u32 b[8]" if nin==2 else "u32 r[8], const u32 a[8]"
    BS = chr(92); NL = chr(10)
    asm = NL.join('        "' + l + BS + "n" + BS + 't"' for l in ["{"] + body + ["}"])
    return f"__device__ __forceinline__ void {name}({sig}) {{\n    asm volatile(\n{asm}\n        : {outs}\n        : {ins});\n}}\n"


def prod(L, dst, A, B):
    # dst (2n regs, zero-initialised by caller) = A * B, n-limb row-wise with carry chains
    n = len(A)
    for i in range(n):
        for j in range(n):
            op = "mad.lo.cc.u32" if j == 0 else "madc.lo.cc.u32"
            L.append(f"{op} {dst[i+j]}, {A[i]}, {B[j]}, {dst[i+j]};")
        L.append(f"addc.u32 {dst[i+n]}, {dst[i+n]}, 0;")
        for j in range(n):
            op = "mad.hi.cc.u32" if j == 0 else ("madc.hi.cc.u32" if j < n-1 else "madc.hi.u32")
            L.append(f"{op} {dst[i+j+1]}, {A[i]}, {B[j]}, {dst[i+j+1]};")

_ctr = [0]
def kara(L, decl, dst, A, B, depth):
    n = len(A)
    if depth == 0 or n <= 2:
        prod(L, dst, A, B); return
    h = n // 2
    kara(L, decl, dst[0:n], A[:h], B[:h], depth - 1)
    kara(L, decl, dst[n:2*n], A[h:], B[h:], depth - 1)
    _ctr[0] += 1; q = f"q{_ctr[0]}_"
    sa = [f"{q}a{i}" for i in range(h+1)]; sb = [f"{q}b{i}" for i in range(h+1)]
    m = [f"{q}m{i}" for i in range(n+1)]; k = f"{q}k"; mk = f"{q}mk"
    decl += sa + sb + m + [k, mk]
    for r in m: L.append(f"mov.u32 {r}, 0;")
    for X, S in ((A, sa), (B, sb)):
        L.append(f"add.cc.u32 {S[0]}, {X[0]}, {X[h]};")
        for i in range(1, h): L.append(f"addc.cc.u32 {S[i]}, {X[i]}, {X[i+h]};")
        L.append(f"addc.u32 {S[h]}, 0, 0;")
    kara(L, decl, m[0:n], sa[:h], sb[:h], depth - 1)
    for c, o in ((sa[h], sb), (sb[h], sa)):
        L.append(f"neg.s32 {mk}, {c};")
        for i in range(h):
            L.append(f"and.b32 {k}, {o[i]}, {mk};")
            L.append(f"{'add.cc.u32' if i == 0 else 'addc.cc.u32'} {m[h+i]}, {m[h+i]}, {k};")
        L.append(f"addc.u32 {m[n]}, {m[n]}, 0;")
    L.append(f"and.b32 {k}, {sa[h]}, {sb[h]};")
    L.append(f"add.u32 {m[n]}, {m[n]}, {k};")
    for part in (dst[0:n], dst[n:2*n]):
        for i in range(n):
            L.append(f"{'sub.cc.u32' if i == 0 else 'subc.cc.u32'} {m[i]}, {m[i]}, {part[i]};")
        L.append(f"subc.u32 {m[n]}, {m[n]}, 0;")
    tail = dst[h:]
    for i in range(len(tail)):
        if i <= n:
            L.append(f"{'add.cc.u32' if i == 0 else 'addc.cc.u32'} {tail[i]}, {tail[i]}, {m[i]};")
        else:
            L.append(f"addc.cc.u32 {tail[i]}, {tail[i]}, 0;")
    L.append(f"addc.u32 {k}, 0, 0;")   # final carry is always 0 (product fits)

def mul_kd(depth):
    decl = [f"t{i}" for i in range(16)]
    L = []
    a = [f"%{8+i}" for i in range(8)]; b = [f"%{16+i}" for i in range(8)]
    t = [f"t{i}" for i in range(16)]
    for r in t: L.append(f"mov.u32 {r}, 0;")
    kara(L, decl, t, a, b, depth)
    L += red("t")
    return [".reg .u32 " + ", ".join(decl) + ";"] + L
print("// Generated by tools/gen_fe8.py" + chr(10) + emit("fe8_mul", mul_kd(1), 2) + emit("fe8_sq", sqr(), 1) + emit("fe8_add", add(), 2, True) + emit("fe8_sub", sub(), 2, True))
