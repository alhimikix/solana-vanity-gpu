# Changelog

## v2.0.0 - 2026-11-06

### Performance
- **~410M keys/sec on RTX 5080** (v1.0.0: 67-70M on the same card).
- Fixed-base comb with W-bit signed windows (W = 8..18, default from L2 size, `--tune` to measure);
  window 0 folded into the start point.
- GF(2^255-19) on 8 x 32-bit limbs with PTX carry chains, Karatsuba multiply (`tools/gen_fe8.py`).
- Batch inversion moved to its own kernel (128 keys per inversion), overlapped with key generation on a
  high-priority stream.
- L2 prefetch of the next table entry.

### Features
- Prefix (`-p`), suffix (`-s`) and prefix+suffix (`-m P*S`) patterns, several at once.
- Case-insensitive matching (`-i`).
- Exact odds and expected time per match.
- Multi-GPU (`-d 0,1` / `-d all`).
- `--selftest`, `--tune`, `--bench`, `--profile`.
- The old `vanity_gpu <suffix> [count]` form still works.

### Security
- Seeds from ChaCha12 keyed with the OS RNG instead of `rand()`/`time()`. Keys made by v1.0.0 should
  be considered weak.
- Every match is recomputed on the CPU with an independent reference implementation before it is written.

### Fixes
- Suffixes longer than 10 characters no longer overflow silently.
- Speed is measured with a wall clock on every platform.
- Matches beyond the per-batch buffer are counted and reported instead of being lost silently.

### Build
- `build.bat` / `build.sh` build for the local GPU by default (`SM_ARCH` overrides).
- Release packages add `sm_120` (RTX 50xx); CI uses CUDA 12.9.

## v1.0.0 - 2026-04-29

Initial release.

### Performance
- **44M keys/sec on RTX 3090** (97x faster than CPU baseline).

### Features
- 8-bit signed-digit comb fixed-base scalar multiplication.
- ref10-style fe10 (radix-2^25.5) field arithmetic.
- Within-thread Montgomery batched inversion (N=4).
- Suffix mod-58^K prefilter to skip base58 encoding for 99.99% of non-matching candidates.
- CSV output (`<suffix>_results.csv`) with `public_key,private_key` columns in base58.
- Sustained throughput reporter every 2 seconds.
- Configurable target match count, or unlimited mode.

### Build
- Windows build script (`build.bat`) with auto-detection of Visual Studio 2019/2022/2025.
- Linux/macOS build script (`build.sh`) with configurable SM architecture.
- CUDA Toolkit ≥ 13.1 required.
