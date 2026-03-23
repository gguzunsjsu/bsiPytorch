# BSI Word Parameterization & R_SWEEP — Implementation Log

## 1. Problem Statement

BSI bitmaps were hardcoded to `uint64_t` (64-bit words) throughout the CUDA/C++ codebase.
This prevented experimentation with different word widths for inference on large models.
Additionally, the BMMA dot kernel's R_SWEEP (number of R-tiles processed per thread block)
was limited to 2 or 4, leaving potential performance on the table for large-R workloads.

**Goals:**
- Parameterize word size via `BSI_WORD_BITS` env var (8, 16, 32, 64, 128, 256)
- Expand R_SWEEP options (2, 4, 6, 8, 10, 12) for runtime dot kernel tuning

---

## 2. Architecture: Word Size Parameterization

### 2.1 Design Decision: Compile-Time via Build Env Var

```bash
BSI_WORD_BITS=32 pip install -e .   # or 8, 16, 64, 128, 256
```

`setup.py` reads the env var, validates it, and passes `-DBSI_WORD_BITS=N` to both
the C++ compiler (g++) and CUDA compiler (nvcc) via `define_macros`.

**Why compile-time, not runtime?**
- Kernel shared memory layouts, register allocation, and `__ballot_sync` packing
  strategies differ fundamentally per word size
- Template instantiation for `BsiVector<bsi_word_t>` requires the type at compile time
- The hybridbitmap library (`runninglengthword.h`) uses `sizeof(uword)` for RLW bit
  field layout — must match at compile time

### 2.2 Central Config Header: `bsi_word_config.h`

Single source of truth. Every CUDA/C++ file includes this header.

**Type system:**

| BSI_WORD_BITS | `bsi_word_t` | `bsi_sword_t` | Torch Dtype | Parts |
|---------------|-------------|---------------|-------------|-------|
| 8 | `uint8_t` | `int8_t` | `kUInt8` | 1 |
| 16 | `uint16_t` | `int16_t` | `kInt16` | 1 |
| 32 | `uint32_t` | `int32_t` | `kInt32` | 1 |
| 64 | `uint64_t` | `int64_t` | `kInt64` | 1 |
| 128 | `struct{uint64_t[2]}` | — | `kInt64` (2x) | 2 |
| 256 | `struct{uint64_t[4]}` | — | `kInt64` (4x) | 4 |

**Key constants defined:**
- `kBsiWordBits`, `kBsiWordBytes`, `kBsiWordParts`
- `kBsiBallotsPerWord` — how many `__ballot_sync` calls to fill one word
  (e.g., 1 for 32-bit, 2 for 64-bit, 4 for 128-bit)
- `kBsiWordsPerBallot` — how many words one ballot produces for <32-bit
  (e.g., 4 for 8-bit, 2 for 16-bit)
- `kBsiRlwRunBits`, `kBsiRlwLitShift` — EWAH run-length word encoding constants
  derived from `sizeof(bsi_word_t) * 4`, matching hybridbitmap's `runninglengthword.h`

**Device helpers:**
- `bsi_popc(bsi_word_t)` — popcount that dispatches to `__popc`/`__popcll`/loop
- `bsi_rlw_encode()`, `bsi_rlw_run_bit()`, `bsi_rlw_run_len()`, `bsi_rlw_lit_count()`
- Bitwise operators (`&`, `|`, `^`, `~`) for composite 128/256-bit structs
- `bsi_word_zero()`, `bsi_word_ones()` for composite types

**Host helpers:**
- `bsi_words_per_slice(d)` — `ceil(d / kBsiWordBits)`
- `bsi_torch_word_dim(W)` — tensor dimension accounting for composite parts
- `bsi_host_rlw_encode<U>()` — CPU-side RLW encoding matching GPU layout

### 2.3 Pack Kernels: Ballot-to-Word Assembly

The core challenge: `__ballot_sync()` always returns a 32-bit result. Different word
sizes require different strategies to assemble bitmap words from ballot results.

**For BSI_WORD_BITS >= 32** (one warp → one word):
```
Each warp handles one word_idx.
kBsiBallotsPerWord = kBsiWordBits / 32 ballots needed per word.

32-bit:  1 ballot → word = (uint32_t)ballot
64-bit:  2 ballots → word = lo | (hi << 32)           [original behavior]
128-bit: 4 ballots → word.parts[0] = lo0|(hi0<<32), word.parts[1] = lo1|(hi1<<32)
256-bit: 8 ballots → word.parts[0..3] filled similarly
```

Grid: one warp per word → `grid.x = ceil(words_per_slice / warps_per_block)`

**For BSI_WORD_BITS < 32** (one ballot → multiple words):
```
Each warp handles one group of 32 rows → produces kBsiWordsPerBallot words.
One ballot result is split into sub-words via bit extraction.

8-bit:  1 ballot → 4 words, extracted as (ballot >> (sub*8)) & 0xFF
16-bit: 1 ballot → 2 words, extracted as (ballot >> (sub*16)) & 0xFFFF
```

Grid: one warp per ballot group → `grid.x = ceil(groups / warps_per_block)`
where `groups = ceil(words_per_slice / kBsiWordsPerBallot)`

This same pattern is used in both:
- `bsi_cuda_kernels_pack.cuh` — pre-quantized int64 → bitmap packing
- `bsi_cuda_kernels_quant.cuh` — fused quantize + shift + pack (float → bitmap)

### 2.4 EWAH Kernels: Word-Size-Aware Compression

The EWAH (Enhanced Word-Aligned Hybrid) compression uses Run-Length Words (RLWs)
whose bit field layout depends on the word size:

```
RLW bit layout (matches hybridbitmap runninglengthword.h):
  Bit 0:                     running bit (0 or 1)
  Bits [1 .. rlw_run_bits]:  run length (number of identical words)
  Bits [rlw_lit_shift ..]:   literal word count

  rlw_run_bits = sizeof(uword) * 4
  rlw_lit_shift = 1 + rlw_run_bits

Example for different word sizes:
  64-bit: run_bits=32, lit_shift=33  (original)
  32-bit: run_bits=16, lit_shift=17
  16-bit: run_bits=8,  lit_shift=9
   8-bit: run_bits=4,  lit_shift=5
```

All EWAH kernels (`ewah_decompress`, `ewah_size`, `ewah_emit`, `slice_popcount_sum`,
`compress_flags_from_density`) use `bsi_word_t` and the config header's RLW helpers.

### 2.5 Dot Kernel: Reinterpret-Cast Strategy

The BMMA tensor core kernels are the most complex (~3000 lines) and always operate
on uint32 fragments via `mma.sync.aligned.m16n8k256`. Rather than rewriting them
for every word size, the launcher does:

```cpp
const auto* A = reinterpret_cast<const unsigned long long*>(A_raw);
const auto* B = reinterpret_cast<const unsigned long long*>(B_raw);
const int W = W_words * kBsiWordBits / 64;  // convert to uint64 word count
```

This works because:
- Bitmap memory is contiguous bits regardless of word container size
- The BMMA kernel loads contiguous 16-byte chunks via `cp.async` — it doesn't
  care about word boundaries, only about bit alignment (which is guaranteed)
- `W_words * kBsiWordBits` = total bits = constant for a given dimension `d`

### 2.6 Tensor Storage

Word tensors use the appropriate PyTorch dtype:
- 8-bit → `torch::kUInt8`, 16-bit → `kInt16`, 32-bit → `kInt32`, 64-bit → `kInt64`
- 128/256-bit → `kInt64` with last dimension multiplied by `kBsiWordParts`

This means tensor shapes differ across word sizes for the same `d`:
```
d=4096, BSI_WORD_BITS=64:  words tensor shape [..., 64]  dtype=int64
d=4096, BSI_WORD_BITS=32:  words tensor shape [..., 128] dtype=int32
d=4096, BSI_WORD_BITS=128: words tensor shape [..., 64]  dtype=int64  (32 words × 2 parts)
```

### 2.7 Chunk Scale Alignment

The BMMA kernel processes 256 bits per chunk. The number of chunks is:
```
chunks = d / 256 = W_words * kBsiWordBits / 256
```

Previously hardcoded as `W / 4` (assuming 64-bit words where 4 × 64 = 256).
Now generalized to `W * kBsiWordBits / 256`.

The BMMA alignment check (W must be a multiple of K_WORDS64=4 in uint64 terms):
```cpp
const int W_u64 = words_per_slice * kBsiWordBits / 64;
if ((W_u64 % 4) == 0) { /* can use BMMA path */ }
```

---

## 3. R_SWEEP Expansion

### 3.1 What is R_SWEEP?

R_SWEEP controls how many R-tiles (groups of 32 keys) a single BMMA thread block
processes before writing results. Higher R_SWEEP = fewer kernel launches, more data
reuse of A (query) data in shared memory, but more shared memory consumed.

```
Thread block processes: R_SWEEP × TN keys (TN=32)
Grid.x = R / (TN × R_SWEEP)    ← fewer blocks with higher R_SWEEP
```

### 3.2 Shared Memory Budget

Each R_SWEEP value requires different shared memory:

**Non-TMA path** (with K_STRIDE32=12 padding for bank conflict avoidance):
```
shared = 21,520 + R_SWEEP × 22,528 bytes

R_SWEEP=2:   ~66KB
R_SWEEP=4:  ~112KB
R_SWEEP=6:  ~157KB
R_SWEEP=8:  ~202KB   ← near H100 limit (228KB)
R_SWEEP=10: ~247KB   ← exceeds H100 limit
```

**TMA path** (K_STRIDE32_TMA=8, no padding needed):
```
shared = 14,464 + R_SWEEP × 16,384 bytes

R_SWEEP=2:   ~47KB
R_SWEEP=4:   ~80KB
R_SWEEP=6:  ~113KB
R_SWEEP=8:  ~146KB
R_SWEEP=10: ~178KB
R_SWEEP=12: ~211KB   ← near H100 limit
```

### 3.3 Available Kernel Instantiations

| R_SWEEP | Non-TMA | TMA |
|---------|---------|-----|
| 2 | yes | yes |
| 4 | yes | yes |
| 6 | yes | yes |
| 8 | yes | yes |
| 10 | yes* | yes |
| 12 | yes* | yes |

*Non-TMA for 10/12 are compiled but will fail the runtime shared memory check on
H100 (228KB). They would work on future GPUs with more shared memory.

### 3.4 Runtime Configuration

```bash
BSI_TC_R_SWEEP=8 python benchmark.py ...   # set R_SWEEP (even, 2-12)
BSI_TC_TMA=1     python benchmark.py ...   # enable TMA staging (H100+)
BSI_DOT_DEBUG=1  python benchmark.py ...   # print [BSI_DOT] config line
```

The runtime check `if (shared_bytes <= max_shared)` silently falls back to the
non-sweep tail kernel if a requested R_SWEEP doesn't fit. The `[BSI_DOT]` debug
line shows the actual `rsweep=N` value used.

### 3.5 Implementation Details

R_SWEEP is a **compile-time template parameter** for performance (loop unrolling,
register allocation). Each value is a separate compiled kernel. The runtime env var
selects which pre-compiled kernel to launch.

The dispatch uses a macro to avoid repetitive if/else chains:
```cpp
#define BSI_DISPATCH_RSWEEP(N) \
    do { \
        // handle cudaFuncSetAttribute for shared memory opt-in \
        // launch TMA or non-TMA kernel variant for R_SWEEP=N \
    } while (0)

if (r_sweep == 2)       { BSI_DISPATCH_RSWEEP(2); }
else if (r_sweep == 4)  { BSI_DISPATCH_RSWEEP(4); }
else if (r_sweep == 6)  { BSI_DISPATCH_RSWEEP(6); }
// ... etc
```

---

## 4. Bugs Found & Fixed During Testing

### Bug 1: Template Deduction Failure (compile error)
**File:** `bsi_cuda.cpp:929`
**Symptom:** `no matching function for call to bsi_flatten_words_gpu_helper(BsiVector<unsigned int>&, std::vector<long unsigned int>&, ...)`
**Root cause:** `std::vector<u64>` (always uint64) passed to a function templated on the BsiVector's word type (now uint32 with BSI_WORD_BITS=32). Template couldn't deduce `T` with conflicting types.
**Fix:** `std::vector<u64>` → `std::vector<bsi_word_t>`

### Bug 2: Hardcoded Word Count Assertion (runtime error)
**File:** `bsi_cuda.cpp:877`, `bsi_vector_cuda.cpp:1017`
**Symptom:** `RuntimeError: CUDA fixed-bit key build word count mismatch`
**Root cause:** `(d + 63) / 64` assumes 64-bit words. With 32-bit words, `words_per_slice` is `(d + 31) / 32` = 2× larger, failing the equality check.
**Fix:** → `bsi_words_per_slice(d)` which uses `kBsiWordBits`

### Bug 3: Chunk Scale Stride Mismatch (runtime error)
**File:** `bsi_cuda.cpp:556, 700`
**Symptom:** `RuntimeError: Chunk scale stride mismatch: got 16 expected 32`
**Root cause:** `keys->W / 4` computes chunks assuming W is in 64-bit words (4 × 64 = 256 bits per chunk). With 32-bit words, W=128 but chunks should still be 16 (4096/256), not 32 (128/4).
**Fix:** → `keys->W * kBsiWordBits / 256`

---

## 5. Verification

### Correctness: Word sizes produce identical accuracy
```
Model: facebook/opt-6.7b, dataset: lambada, 200 samples, dec=2

BSI_WORD_BITS=32: top1=0.7450, top5=0.9200, dot_ms=66.6
BSI_WORD_BITS=64: top1=0.7450, top5=0.9200, dot_ms=66.6

Debug output confirms parameterization:
  32-bit: W_words=128 word_bits=32 W64=64
  64-bit: W_words=64  word_bits=64 W64=64
```

W64 is identical because it represents total bits / 64, which depends on `d` not
the word container size. The BMMA kernel sees identical data in both cases.

### Performance: Dot time identical, build time varies
- `dot_ms` identical across word sizes (BMMA kernel processes same bits)
- `build_p_ms` (pack time) slightly higher for 32-bit (2× more words to pack)
- Accuracy bitwise identical (same quantization, same bits, different containers)
