# solana-vanity-gpu

GPU grinder for Solana vanity addresses. Brute-forces ed25519 keypairs on CUDA until the base58 public key
starts and/or ends with what you want. Several patterns can be searched at once, case-insensitive if needed.

**~410M keys/sec on an RTX 5080** (the v1.0.0 kernel does 67–70M on the same card). Keys are standard
32-byte seeds, so they import into Phantom, Solflare and `solana-keygen` as usual.

## Requirements

- **NVIDIA GPU** with Compute Capability ≥ 7.5 (RTX 20xx and newer).
- **CUDA Toolkit** ≥ 12.0 (`nvcc`); ≥ 12.8 for RTX 50xx (`sm_120`). Tested with 13.4.
- **MSVC** (Visual Studio 2019/2022, or 2025 with `-allow-unsupported-compiler`) on Windows. GCC on Linux.

## Installation

### Pre-built binaries

Download from [Releases](../../releases):

- `solana-vanity-gpu-vX.Y.Z-windows-x64.zip` — Windows
- `solana-vanity-gpu-vX.Y.Z-linux-x64.tar.gz` — Linux

Each archive contains one binary per GPU generation:
- `vanity_gpu_sm75` — RTX 20xx, GTX 16xx (Turing)
- `vanity_gpu_sm86` — RTX 30xx, A100 (Ampere)
- `vanity_gpu_sm89` — RTX 40xx, L40 (Ada)
- `vanity_gpu_sm120` — RTX 50xx (Blackwell)

### Build from source

**Windows:**
```bash
.\build.bat                       # builds for the GPU in this machine
set SM_ARCH=sm_89 & .\build.bat   # or for a given architecture
```

**Linux:**
```bash
./build.sh                        # builds for the GPU in this machine
SM_ARCH=sm_120 ./build.sh         # or for a given architecture
```

Produces `vanity_gpu.exe` (or `vanity_gpu`). After building, run `vanity_gpu --selftest` once.

### Local release packaging

```bash
scripts\package-release.bat v2.0.0     # Windows: sm_75/86/89/120 + zip
scripts/package-release.sh v2.0.0      # Linux:   sm_75/86/89/120 + tar.gz
```

## Usage

```bash
vanity_gpu pump                 # suffix "pump", run forever (old form still works)
vanity_gpu pump 100             # 100 matches, then exit

vanity_gpu -s pump -n 10        # ends with pump
vanity_gpu -p Sol -n 10         # starts with Sol
vanity_gpu -m "So*pump" -n 1    # starts with So AND ends with pump
vanity_gpu -s pump -s bonk -i   # ends with pump OR bonk, any letter case
```

| Option | Meaning |
|---|---|
| `-s, --suffix S` | address ends with `S` (repeatable) |
| `-p, --prefix P` | address starts with `P` (repeatable) |
| `-m, --match P*S` | starts with `P` and ends with `S`: `abc*`, `*xyz`, `ab*yz` (repeatable) |
| `-i, --ignore-case` | case-insensitive matching |
| `-n, --count N` | stop after `N` matches |
| `-o, --out FILE` | CSV file (default `<suffix>_results.csv` for one suffix, else `vanity_results.csv`) |
| `-d, --device LIST` | GPUs to use: `0`, `0,1`, `all` |
| `-w, --window W` | comb window, 8..18 bits (default chosen from the L2 cache size) |
| `--tune` | measure all window sizes on this GPU and print the fastest |
| `--selftest` | check GPU keys and the matcher against the host reference |
| `--bench SEC` | run `SEC` seconds and print the average rate |
| `--profile` | time each kernel |

All patterns are OR-ed. At start the program prints the odds (`1 in N keys`) and, while running, the
expected time per match at the current speed.

Base58 has no `0`, `O`, `I`, `l`. With `-i` a letter only needs one usable case (`l` matches `L`).

CSV format: `public_key,private_key` (both base58). The file is created with a header and appended to on
later runs.

### Prefixes

A Solana address is 43 or 44 characters. 44-character addresses can only start with `2`…`H` (or `1`),
so a prefix whose first character is later in the alphabet (`J`…`z`) only appears in 43-character
addresses — roughly 17 times rarer than a suffix of the same length. The odds line accounts for this.

## Performance

RTX 5080 (driver 617, CUDA 13.4, default settings):

| Pattern length | Combinations | Average wait at 410M k/s |
|---|---|---|
| 4 (`pump`) | 11.3 M | 0.03 seconds |
| 5 | 656 M | ~1.6 seconds |
| 6 | 38 G | ~1.5 minutes |
| 7 | 2.2 T | ~1.5 hours |
| 8 | 128 T | ~3.6 days |

Prefixes starting with `J`…`z` take about 17 times longer. The wait is geometric — you may get lucky
in seconds, or it may take 2–3x the average.

Optimisation steps, same RTX 5080:

| Step | k/s |
|---|---|
| v1.0.0 kernel (built for sm_120) | 67–70M |
| 16-bit comb windows, batch inversion in its own kernel, no register spills | 255M |
| 32-bit-limb field arithmetic in PTX instead of radix-2^25.5 | 359M |
| Karatsuba multiply, first window folded into the start point | 386M |
| Inversion kernel overlapped with the next batch on a high-priority stream | 408M |
| Prefetch of the next table entry into L2 | 413–419M |

## How it works

Per key: `seed = ChaCha12(run key, index)` → SHA-512 → clamp → fixed-base scalar multiplication →
batched inversion → `y` and the sign of `x` → pattern check.

1. **Comb with wide signed windows.** The clamped scalar is `2^254 + 8t`, `t < 2^251`. `t` is cut into
   W-bit windows recoded to signed digits in `[-2^(W-1), 2^(W-1)]`, one table of `j · 8 · 2^(Wk) · G`
   per window. With W=16 that is 15 point additions per key instead of 32. Window 0 is folded into the
   start point: its table holds `2^254·G + d·8·G` directly, so the first addition disappears. Entries are
   96 bytes (Y+X, Y−X, 2dXY); the W=16 table is 48 MB and lives in L2 on cards with a big L2. The entry for
   the next window is prefetched into L2 while the current addition runs, so wider windows (17, 18) whose
   tables spill to DRAM still work. `--tune` picks the best W for a given card.

2. **Field arithmetic on 8 × 32-bit limbs.** GF(2^255−19) values are kept lazily in `[0, 2^256)`
   (`2^256 ≡ 38`). Multiplication is one level of Karatsuba over 128-bit halves (48 partial products
   instead of 64) with PTX carry chains, generated by `tools/gen_fe8.py`. On sm_120 this is 1.9x faster
   than the radix-2^25.5 ref10 multiply; the hot loop is bound by the integer multiply pipe.

3. **Batched inversion in a separate kernel.** `k_gen` writes projective `(X, Y, Z)` for a batch of 4M
   keys; `k_inv` inverts 128 Zs per thread with Montgomery's trick (one inversion per 128 keys), computes
   the public key and applies a cheap gate. `k_inv` is memory-bound and `k_gen` compute-bound, so `k_inv`
   of batch n runs on a high-priority stream next to `k_gen` of batch n+1.

4. **Matching.** The address is the base58 form of the 256-bit integer `N` (key bytes, big-endian).
   A suffix of length K fixes `N mod 58^K`, hence `N mod 2^K`: a 256-bit bitmap of allowed low bytes
   rejects most keys. A prefix is a set of integer ranges of `N` (for 44- and 43-character addresses),
   checked on the top 64 bits. Survivors go to `k_check`, which does the exact base58 compare.

5. **Host check.** Every match is re-derived from its index on the CPU, the public key is recomputed with an
   independent reference implementation (plain SHA-512 + Montgomery ladder, no tables) and the pattern
   is re-checked before the key is written. A mismatch would be reported and the key dropped.

## Security

- Seeds come from ChaCha12 keyed with 32 bytes from the OS RNG (`BCryptGenRandom` / `/dev/urandom`),
  one key per run and GPU. Seeds are independent: giving away one generated key reveals nothing about the
  others from the same run. (v1.0.0 seeded from `rand()` and `time()`; keys made with it should be
  treated as weak and not hold funds.)
- No network calls. No telemetry. All private keys stay on your machine.
- `--selftest` checks GPU keys against the host reference (which itself is checked against RFC 8032)
  and the GPU matcher against a full host scan, for every window size.
- The results CSV contains private keys in plaintext. **Never commit it.** Consider encrypting it after the
  run. The default `.gitignore` excludes `*_results.csv`.

## License

Apache License 2.0 — see `LICENSE` and `NOTICE`. You can use, modify and sell it, including in closed
products. If you redistribute it or something based on it, keep the `NOTICE` file (credit to the author)
and mark the files you changed. Versions up to v1.0.0 were released under MIT.

## Credits

- **ed25519 ref10** by Daniel J. Bernstein, Niels Duif, Tanja Lange, Peter Schwabe, Bo-Yin Yang — public
  domain (via SUPERCOP). Field constants and the inversion addition chain come from ref10.
- Signed-digit fixed-base comb: standard construction; see Hisil–Wong–Carter–Dawson "Twisted Edwards
  Curves Revisited" (2008) and the ed25519 paper.
- ChaCha by Daniel J. Bernstein.
