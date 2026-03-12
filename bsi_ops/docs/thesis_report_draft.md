# BSI-Native Tensor-Core Kernels for Compressed LLM Inference

## Abstract
Large language model (LLM) inference is dominated by repeated dense linear projections whose computational and memory demands increase sharply with model scale. This work investigates whether Bit-Sliced Indexing (BSI) can serve as a practical compressed representation for LLM inference on GPUs while retaining a native bitwise execution path rather than falling back to dense low-precision matrix multiplication. The implementation targets PyTorch-based end-to-end inference for OPT-family models and uses a C++/CUDA extension that stores linear weights in BSI form, builds bit-sliced query activations on the fly, and computes matrix products using fused CUDA kernels based on binary tensor-core operations and weighted popcount accumulation. The central research question is not whether lower precision is generally faster, but whether a BSI-native execution pipeline can be made substantially more competitive through representation and kernel co-design.

The implementation establishes a complete experimental stack: CPU and CUDA BSI builders, a PyTorch replacement for `torch.nn.Linear`, kernel-only microbenchmarks, end-to-end next-token evaluation on LAMBADA, and comparisons against both dense FP16 baselines and SmoothQuant benchmark scripts. The stable fixed-bit CUDA baseline centers on an SM90/H100 path using `TM=32`, `R_sweep=4`, chunk scaling, and a fixed `(Sa,Sb)=(7,6)` query/key slice configuration. That baseline achieves meaningful model compression and maintains usable accuracy for moderate model sizes, but still trails dense Torch inference by a wide margin. Additional layout and staging studies were conducted to understand the design space, and several of those ideas were discarded after profiling or correctness regressions. Those negative results are still valuable because they help identify which parts of the pipeline are genuinely limiting performance.

The principal finding of the work is that BSI representation alone is insufficient for competitive GPU inference. A generic BSI storage layout, even when combined with tensor-core instructions, still incurs significant orchestration overhead and poor scheduler efficiency. Profiling of the same-bit tensor-core path showed that the implementation was not primarily DRAM-bandwidth limited; instead, it was constrained by low eligible warp count, high no-eligible stall time, and excessive shared/global memory transactions. This result materially reframed the optimization problem: the main challenge is not only bit count, but co-designing the BSI layout and the kernel’s access pattern so the representation matches the hardware execution model.

Empirically, the work demonstrates a clear compression and performance tradeoff. Under the primary evaluation setting (`decimal_places=2`, `compress_threshold=0.5`, `num_samples=1000`, `max_seq_len=512`, `fp16` base dtype), the `facebook/opt-6.7b` model achieved a dense static size of `12700.031 MB` and a BSI static size of `5022.281 MB` under `scope=all`. In the same setting, BSI achieved `top1=0.741`, `top5=0.896`, and `66.452 ms` dot latency per sample, compared with the dense baseline at `top1=0.798`, `top5=0.929`, and `15.493 ms`. These results show that BSI provides real compression and a functioning inference path, but also that the remaining performance gap is dominated by the interaction between representation and kernel design. The defensible research contribution is therefore the identification and partial resolution of the real bottleneck: BSI-native inference becomes materially more competitive only when the data layout, chunk staging, and dot kernel are co-designed for the target GPU architecture.

## 1. Introduction
### 1.1 Motivation
Modern decoder-only transformers execute a very large number of linear layers during inference. In an OPT-style model, these layers appear in the self-attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) and in the feed-forward network (`fc1`, `fc2`). As model size grows from 125M to 1.3B, 6.7B, and 30B parameters, the computational and memory cost of these linear projections becomes the dominant bottleneck. The most common response in the literature is quantization, usually into INT8 or INT4 variants, often with calibration or activation-aware scaling. However, these methods generally assume dense low-precision matrix multiplication as the execution substrate.

This work investigates a different question: can LLM weights be stored and computed in a bit-sliced representation, and can the resulting dot-product operations be executed natively through bitwise/GPU tensor-core style kernels rather than by reconstructing dense low-bit matrices? If that can be done efficiently, BSI offers an attractive combination of structural compression, direct access to bitplanes, and a natural path to operations based on weighted popcount and Boolean tensor-core instructions. This is a research question, not a product engineering shortcut. The requirement throughout this work is that weights remain stored in BSI format and that the dot product be computed from BSI bitplanes rather than by converting to a dense int8 tensor and calling a standard GEMM kernel.

### 1.2 Problem Statement
The raw hypothesis behind the project is that bit-sliced weights may reduce storage costs and expose arithmetic structure that is well suited to GPU bitwise execution. In practice, however, a naive BSI implementation on GPU can be slow. The central problem is therefore twofold:

1. How should LLM linear weights and runtime activations be represented in BSI form so that the format remains accurate enough for inference?
2. How should those BSI tensors be laid out and consumed by CUDA kernels so that the execution cost approaches dense Torch baselines as closely as possible?

The work is explicitly centered on the second question. Accuracy matters, but the primary focus is dot-product performance. The thesis therefore treats the implementation as a co-design problem involving representation, layout, kernel scheduling, and measurement.

### 1.3 Research Goals
The goals of the work and of the thesis report are:

1. Integrate BSI-based linear layers into end-to-end OPT inference in PyTorch.
2. Build CUDA kernels that operate on BSI directly during inference.
3. Establish reliable benchmarking infrastructure for kernel-only, layer-level, and model-level evaluation.
4. Compare BSI against dense Torch baselines and against a SmoothQuant benchmark harness.
5. Characterize the performance bottlenecks of native BSI execution on Hopper-class GPUs.
6. Demonstrate that layout-aware and kernel-aware design decisions produce real same-bit performance gains.
7. Define the speed/accuracy frontier of the current implementation and identify the next research step.

### 1.4 Main Thesis Claim
The main claim that this work can defend is the following:

> BSI can serve as a native compressed representation for LLM inference on GPUs, but only when the bit-sliced layout and the GPU execution kernel are co-designed for the target hardware. Compression or lower bit count alone does not yield competitive performance.

This is a narrower and more defensible claim than “BSI beats Torch” or “lower bits solve inference.” The evidence collected in this work supports the co-design claim and also explains why the current system still trails dense FP16 inference.

### 1.5 Scope of the Study
The implementation evolved in stages. An earlier stable fixed76 baseline established a usable reference point, after which more aggressive packed-layout and tensor-memory experiments were explored. The experimental focus of the present study was narrowed to:

- `facebook/opt-125m`
- `facebook/opt-1.3b`
- `facebook/opt-6.7b`

The 30B model was initially used as a stress test to expose accuracy sensitivity, particularly under `scope=all`, but it was intentionally deprioritized once it became clear that the current fixed-bit settings caused substantial degradation, especially in MLP layers. For the purposes of thesis framing, the 30B study is best treated as an analysis of failure mode and layer sensitivity rather than a main performance target.

### 1.6 Thesis Questions
The thesis can be organized around four concrete research questions:

1. Can transformer linear layers be replaced by BSI-native operators without abandoning end-to-end autoregressive inference?
2. Which parts of a BSI inference pipeline matter most for performance on GPU: storage format, query construction, kernel structure, or bit count?
3. Which layer groups in large OPT models are most sensitive to the current BSI quantization regime?
4. What profiler evidence explains the remaining performance gap between native BSI execution and dense Torch FP16?

These questions are narrow enough to answer with the code and results presented here, and they avoid overclaiming beyond the measured evidence.

### 1.7 Contributions
The strongest contribution list for the thesis is:

1. A PyTorch/CUDA implementation of BSI-native linear layers that replaces dense `torch.nn.Linear` modules inside OPT models.
2. A GPU-side BSI builder for static keys and batched dynamic queries, integrated into a full inference path.
3. A stable Hopper-oriented bitwise tensor-core dot kernel family using fixed-bit query/key slices, chunk scaling, and TMA-based staging.
4. A detailed set of layout and staging experiments that clarify which parts of the BSI pipeline help or hurt same-bit performance.
5. A benchmark and attribution harness spanning kernel-only, layer-level, and model-level evaluation.
6. A profiler-based diagnosis showing that the current bottleneck is dominated by scheduler and memory-access inefficiency rather than gross DRAM bandwidth saturation.

### 1.8 What the Thesis Does Not Claim
It is equally important to state what the thesis does not claim:

1. It does not claim that BSI currently surpasses dense Torch FP16 inference.
2. It does not claim that simply lowering the number of fixed slices is the correct optimization strategy.
3. It does not claim that the current fixed-bit BSI regime is already robust across all model scales.
4. It does not claim that comparisons against SmoothQuant are perfectly apples-to-apples at the backend level.

This negative framing is important because it keeps the thesis aligned with the real evidence collected in this work.

## 2. Background and Literature Review
### 2.1 Open Pre-trained Transformers (OPT)
The experiments in this work are built around the OPT family introduced by Zhang et al. [1]. OPT provides a publicly accessible series of decoder-only language models ranging from 125M to 175B parameters, making it a practical research platform for studying inference optimization at multiple scales. The models used here—125M, 1.3B, 6.7B, and 30B—span a broad range of computational regimes while keeping architecture relatively consistent. That consistency is valuable because it allows kernel and representation changes to be evaluated across increasing width and depth without changing the overall model class.

### 2.2 Quantization for LLM Inference
A substantial body of LLM compression work has focused on dense low-precision quantization.

#### 2.2.1 LLM.int8()
Dettmers et al. [2] introduced `LLM.int8()`, a method that keeps most computation in 8-bit while isolating outlier feature dimensions in higher precision. The key insight is that certain transformer dimensions exhibit outlier behavior that standard INT8 quantization handles poorly. The method therefore relies on vector-wise quantization and a mixed-precision fallback for outliers.

#### 2.2.2 ZeroQuant
Yao et al. [3] proposed ZeroQuant, which combines hardware-friendly quantization, layer-wise knowledge distillation, and backend support to reduce quantization/dequantization overhead. ZeroQuant is relevant here because it emphasizes that the software backend is part of the quantization problem; a quantized representation without an appropriate execution backend often does not deliver real speedups.

#### 2.2.3 GPTQ
Frantar et al. [4] introduced GPTQ, a highly accurate post-training weight quantization method for autoregressive transformers. GPTQ is notable for showing that very low bitwidth weights can preserve model quality when quantization uses more informed error control. In this thesis, GPTQ serves primarily as literature context: it shows that aggressive compression is possible, but it does not answer the BSI-native execution question pursued here.

#### 2.2.4 SmoothQuant
Xiao et al. [5] proposed SmoothQuant, which redistributes activation outliers into the weights through an equivalent transformation, thereby enabling practical W8A8 quantization for LLMs. SmoothQuant is especially relevant because the present implementation includes dedicated benchmark scripts for comparing against a widely recognized, hardware-friendly quantization baseline. SmoothQuant frames a useful comparison point: it is a calibrated, dense INT8 inference path, whereas the BSI work here is a native bit-sliced execution path.

#### 2.2.5 AWQ
Lin et al. [6] developed AWQ, a weight-only quantization method that protects salient channels determined by activation distribution. AWQ highlights that accuracy at low bitwidth depends strongly on layer and channel sensitivity. This insight is consistent with the empirical results observed here: under fixed-bit BSI quantization, large OPT models show significantly different sensitivity in attention versus MLP layers.

### 2.3 LAMBADA as an Evaluation Dataset
The evaluation uses LAMBADA [7], a dataset designed to test broad-context word prediction. The benchmark harness implemented here treats the task as next-token prediction at the last non-padding position of each right-padded input sequence. LAMBADA is appropriate for this work because it provides a lightweight but meaningful measure of whether the quantized or compressed inference path preserves model behavior on a standard language-modeling style task.

### 2.4 GPU Execution Model: Hopper Tensor Memory Accelerator and Bitwise Tensor Cores
This work targets SM90/H100 systems and uses Hopper-specific features, especially Tensor Memory Accelerator (TMA), to stage data more efficiently between global and shared memory [8][9]. Hopper allows 1D–5D tensor transfers through TMA descriptors and supports tensor-map objects that can encode tiled or interleaved layouts. The TMA mechanism is relevant because native BSI execution relies on carefully structured bitplane tiles. When those tiles are staged efficiently, the kernel can devote more execution capacity to BMMA-style bitwise operations and weighted accumulation rather than wasting instructions on data movement.

At the same time, Hopper profiling exposed that memory movement alone is not the limiting factor. The active BSI kernels in this work achieved only a fraction of peak memory bandwidth while also exhibiting low eligible warps per scheduler and high no-eligible stall time. That finding is central to the argument of this thesis: the challenge is not merely feeding bytes to the GPU, but organizing the bit-sliced computation so that the scheduler can keep the Boolean tensor-core pipeline busy.

### 2.5 Gap Addressed by This Work
The prior literature establishes that quantization can reduce memory and sometimes improve inference throughput, but most successful methods assume dense low-precision arithmetic. This work addresses a more specialized gap:

- how to store LLM weights in BSI form,
- how to build query activations in compatible BSI form at runtime,
- how to execute BSI-native dot products using CUDA tensor-core style instructions,
- and how to evaluate the resulting system end to end.

The research contribution is therefore not another post-training quantization algorithm. It is a systems-level study of BSI-native representation, kernel design, and empirical bottlenecks.

## 3. Implementation Overview
### 3.1 High-Level Structure
The implementation consists of three main components:

1. CPU-side and legacy BSI logic in the sibling `../bsiCPP` area.
2. The `bsi_ops` Python/C++/CUDA extension in this directory.
3. Benchmark and evaluation scripts under `benchmarks/`.

Within `bsi_ops`, the important paths are:

- `benchmarks/verify_accuracy_bsi.py`:
  defines `BSIQuantizedLinear` and the model quantization path.
- `benchmarks/benchmark_performance_bsi.py`:
  end-to-end LAMBADA evaluation and attribution reporting.
- `benchmarks/benchmark_apples_to_apples_bsi.py`:
  unified harness for kernel-only, linear end-to-end, and model end-to-end comparisons.
- `benchmarks/smoothquant_benchmark/`:
  scripts used to compare against SmoothQuant baselines.
- `csrc/cuda/bsi_cuda.cpp`:
  Python bindings, CUDA-side builders, and dispatch entrypoints.
- `csrc/cuda/bsi_cuda_kernels_dot.cuh`:
  CUDA dot-product kernels and launchers.
### 3.2 Runtime Data Flow
At a high level, the BSI inference path proceeds as follows:

1. A dense `torch.nn.Linear` weight is converted once into a BSI key capsule via `build_bsi_keys_cuda`.
2. During forward propagation, the input activation tensor is flattened to two dimensions.
3. A packed batch query builder constructs BSI query capsules from the activation rows.
4. A fused CUDA dot-product path computes all output rows against the prebuilt BSI keyset.
5. The output is cast back to the original dtype, bias is added, and the tensor is reshaped back to the original sequence layout.

This flow is implemented in `BSIQuantizedLinear.forward` in `benchmarks/verify_accuracy_bsi.py`. The class also accumulates detailed timing information for:

- total dot kernel time,
- total build time,
- query quantization/build/pack/kernel subcomponents,
- and aggregate per-query and per-scalar dot timings.

### 3.3 Quantization Scope Control
The implementation supports different replacement scopes for the LLM linear layers. The `quantize_model_bsi` function in `benchmarks/verify_accuracy_bsi.py` accepts:

- `scope='all'`: replace all linear layers except `lm_head`
- `scope='attention'`: replace only attention projections
- `scope='mlp'`: replace only feed-forward layers

Internally, the scope logic classifies modules by name tokens:

- attention-related: `attn`, `self_attn`, `q_proj`, `k_proj`, `v_proj`, `out_proj`
- MLP-related: `mlp`, `ff`, `ffn`, `fc1`, `fc2`

This scope mechanism became important when diagnosing the 30B accuracy collapse. The experiments showed that MLP layers were substantially more sensitive than attention layers under the current fixed-bit BSI setting.

### 3.4 Benchmark Modes
The harness in `benchmarks/benchmark_apples_to_apples_bsi.py` supports three timing modes:

1. `kernel_only`
   - keys and queries are built once
   - the measured function is the BSI dot kernel only
   - compared directly against `torch.matmul`

2. `linear_e2e`
   - query build is included per call
   - approximates a single BSI linear layer end to end

3. `model_e2e`
   - shells into `benchmarks/benchmark_performance_bsi.py`
   - performs end-to-end model evaluation on LAMBADA

This separation is important because the experiments repeatedly showed that improvements visible in `kernel_only` mode do not automatically translate into the full `model_e2e` setting if query building, storage duplication, or integration overhead are not handled carefully.

### 3.5 Evolution of the Implementation
The development process followed a clear progression that is worth documenting in thesis form.

1. An earlier stable fixed-bit implementation established a working same-bit baseline using `TM=32`, `R_sweep=4`, and tensor-core-oriented bitwise accumulation.
2. TMA-based staging was then introduced to improve how key data moved from global memory into shared memory.
3. Alternative layout strategies were evaluated to determine whether kernel-native packing could reduce the dot-product bottleneck.
4. Runtime query repacking was tested and later rejected because its end-to-end overhead outweighed its kernel-level benefits.
5. Lower-bit fixed-slice settings were evaluated to characterize the speed/accuracy frontier, but they were not adopted as the primary thesis baseline.

Presented this way, the chronology reads as a methodical optimization program rather than as an implementation diary.

## 4. Methodology
### 4.1 BSI Representation
The implementation stores weights in bit-sliced form, where each slice corresponds to a bitplane or weighted component of the original quantized value. In the fixed-bit CUDA path, the number of slices is controlled by environment variables rather than inferred dynamically:

- `BSI_FIXED_BITS_QUERIES`
- `BSI_FIXED_BITS_KEYS`

For the stable same-bit configuration used most often in this work:

- queries use `Sa = 7`
- keys use `Sb = 6`

The total reconstructed dot product is then a weighted sum over all query/key slice interactions. The effective arithmetic cost per chunk therefore scales roughly with `Sa x Sb`; for the common `(7,6)` setup, that means 42 slice-pair interactions per chunk. This interaction count is one of the reasons BSI-native execution remains more expensive than dense FP16 GEMM.

### 4.1.1 CPU-Side Bitmap and BSI Construction
The conceptual basis of the GPU implementation comes from the CPU-side BSI library in `../bsiCPP`, especially the classes:

- `HybridBitmap`
- `BsiVector`
- `BsiSigned`
- `BsiUnsigned`

This layer is important for the thesis because it defines what a BSI object actually is before it is translated into CUDA-friendly tensors.

#### 4.1.1.1 HybridBitmap
`HybridBitmap` is the fundamental bitmap container. It supports two storage modes:

1. compressed bitmap mode
2. verbatim bitmap mode

The class stores:

- `buffer`: the underlying word array
- `density`: empirical fraction of set bits
- `verbatim`: whether the slice is stored as raw words instead of compressed runs

This means that a single BSI slice is not just a bit-vector; it is a bitmap object that can decide whether compression is worthwhile for that slice. This detail matters because BSI storage size depends on slice density, not just on the number of slices.

#### 4.1.1.2 BsiVector
`BsiVector` is the abstract container for an attribute encoded as slices. It maintains:

- `numSlices`
- `bsi`: the vector of slice bitmaps
- `existenceBitmap`
- `sign` bitmap
- `rows`
- `offset`
- `decimals`
- `firstSlice` and `lastSlice` flags
- `twosComplement`
- signedness metadata

From a thesis perspective, `BsiVector` is the structural definition of BSI in this work. Every later CUDA representation is derived from this semantic object.

#### 4.1.1.3 Building BSI from Numeric Vectors
The `buildBsiVector` methods in `BsiVector.hpp` implement the core conversion from dense numeric arrays into bit-sliced form.

For floating-point vectors:

1. values are scaled by `10^decimalPlaces`
2. scaled values are rounded to integers
3. the maximum absolute value determines the number of slices through bit-width analysis
4. `bringTheBits` transposes the integer vector into slice-major bitmaps

The `bringTheBits` routine deserves explicit mention. It computes the per-slice bitmap representation by visiting each element and testing whether the bit corresponding to that slice is set. Each slice stores:

- a count of one bits in position `0`
- the actual bitmap words in subsequent positions

That count is later used to estimate slice density and decide whether the slice should be compressed or stored verbatim.

#### 4.1.1.4 Compression Decision per Slice
The CPU builder uses a slice-level heuristic based on empirical bit density:

- `bitDensity = ones / numberOfElements`
- `compressRatio = 1 - (1 - bitDensity)^(2*bits) - (bitDensity)^(2*bits)`

If `compressRatio < compressThreshold` and is nonzero, the slice is stored in compressed mode; otherwise it is stored as a verbatim bitmap. This is a key implementation detail because it explains why two BSI vectors with the same slice count can occupy very different amounts of memory.

#### 4.1.1.5 Signed and Unsigned Construction
For nonnegative data, the builder creates a `BsiUnsigned` object and assigns:

- an all-zero sign bitmap
- an all-ones existence bitmap
- `twosComplement = false`

For signed data, the builder creates a `BsiSigned` object, uses the most significant slice as the sign slice, marks the representation as two’s complement, and again creates an all-ones existence bitmap. This signed/unsigned split is important because transformer weights and activations are not restricted to nonnegative values, so the signed path is essential for real model inference.

#### 4.1.1.6 CPU Dot and Algebraic Operations
The CPU classes also implement:

- vertical and horizontal BSI summation
- multiplication
- top-k and range queries
- signed and unsigned dot products
- dot-product pruning variants

Not every one of these operations is used directly in the GPU LLM path, but they provide the conceptual algebra from which the CUDA implementation was derived. The CUDA extension narrows this broader BSI algebra to the subset required for batched transformer inference.

### 4.1.2 Decimal Scaling and Compression Threshold
The quantized BSI layers in this implementation are parameterized by:

- `decimalPlaces`
- `compress_threshold`

These are exposed in the benchmarking scripts and passed through to `BSIQuantizedLinear` and the CUDA builders. In practice, most of the experiments discussed in this report used:

- `decimal_places = 2`
- `compress_threshold = 0.5`

In the implementation, these values influence how the original FP weights or activations are discretized into fixed bit-sliced form and how aggressively slice data is compressed. From a thesis perspective, this means the BSI path is not merely a kernel optimization; it is a combined representation-and-execution pipeline with a small but important quantization policy surface.

### 4.2 Query and Key Construction
#### 4.2.1 Key Construction
The `build_bsi_keys_cuda` function in `csrc/cuda/bsi_cuda.cpp` constructs a prebuilt key capsule from the dense linear weight. In fixed-bit mode, the builder creates bitplanes directly on CUDA so that the query and key quantization logic share the same scaling behavior. The output key structure stores:

- number of keys (`R`)
- feature dimension (`D`)
- words per slice (`W64`)
- decimal scaling metadata
- grouped or single-group slice data
- and any additional layout metadata required by later experimental kernels

Because model weights are static, key construction occurs once at model quantization time and is therefore a natural place to invest in offline repacking.

#### 4.2.2 Query Construction
The forward path uses `build_bsi_queries_cuda_batch_packed` to construct all query rows in a single CUDA call. This is significantly more efficient than per-row Python loops and is a prerequisite for usable end-to-end timing. The query builder returns:

- query words tensor `[Q, Sa, W]`
- slice weights `[Q, Sa]`
- optional chunk scales `[Q, chunks]`

The packed batch query builder was one of the most important practical improvements in the implementation because it eliminated major Python overhead from earlier versions.

### 4.2.3 Runtime Counters and Per-Layer Attribution
`BSIQuantizedLinear` records several counters that became central to the evaluation:

- `dot_ns_total`
- `build_ns_total`
- `build_quantize_ns_total`
- `build_pack_ns_total`
- `build_kernel_ns_total`
- `dot_query_vectors_total`
- `dot_output_elements_total`

These counters are aggregated by helper functions in `verify_accuracy_bsi.py` and later reported by `benchmark_performance_bsi.py`. This instrumentation made it possible to separate:

1. query build cost,
2. dot kernel cost,
3. per-query time,
4. and per-scalar time.

That distinction became essential when certain packed-layout experiments improved isolated microbenchmarks but made end-to-end model execution much worse.

### 4.3 Dot-Product Kernel Design
#### 4.3.1 Baseline Kernel Family
The earlier stable implementation converged on a fixed76 kernel family centered on:

- `TM = 32`
- `R_sweep = 4`
- optional `cp.async` staging
- optional TMA B staging
- chunk scaling enabled
- SM90/H100 target

This family was stable enough to produce end-to-end model runs on OPT models while maintaining consistent accuracy.

#### 4.3.2 Motivation for Layout-Aware Studies
The later layout studies began from the observation that the original stable same-bit path still consumed generic BSI layouts:

- queries: `[Q, Sa, W64]`
- keys: `[R, Sb, W64]`

Even with tensor-core instructions, the kernel still paid significant indexing and layout overhead around the fundamental bit-sliced multiply. Later experiments therefore studied whether the static key side could be repacked into a more kernel-native format. An early version also tried to repack queries into a tile-major format, but that approach was later removed from the active model path because it caused severe build-time blowups and memory duplication. This negative result was important because it showed that a layout change which looks attractive in a kernel microbenchmark can still fail as a system-level optimization.

### 4.3.4 Tile Shape and Kernel Geometry
The tile shape that is easiest to defend in the thesis is the stable Hopper-oriented same-bit configuration built around:

- total query tile height `TM_TOTAL = 32`
- output tile width `TN = 32`
- chunk size `K = 256 bits`
- `R_sweep = 4`
- `CTA = 256 threads`
- two-stage pipeline

This corresponds to a logical `32 x 128 x 256-bit` CTA in GEMM-style notation:

- `M = 32`
- `N = 32 * 4 = 128`
- `K = 256`

This tile is defensible because:

1. it aligns cleanly with the BMMA `m16n8k256` granularity;
2. it was the most stable point across multiple OPT sizes;
3. it remained valid across the stable baseline and the later layout studies;
4. it gives a concrete hardware-aware design choice to discuss in the thesis defense.

### 4.3.5 Why the Kernel Uses Weighted Popcount
The CUDA kernels use Boolean tensor-core instructions to compute bitwise overlaps and then scale those overlaps by slice weights. Conceptually, the operation is:

1. compute bitwise overlap counts between a query bitplane and a key bitplane;
2. multiply those counts by the associated slice weights;
3. accumulate across all slice pairs and all chunks;
4. apply chunk-scale and decimal rescaling.

This formulation is the essence of a BSI-native dot product. It differs fundamentally from dense low-bit GEMM because the arithmetic starts from bitplanes rather than from a packed integer matrix interpreted as dense arithmetic lanes.

### 4.4 Why the Work Focused on Dot Time
The explicit priority throughout this study was dot-product performance. For that reason:

- model attribution runs used `BSI_PROFILE=1`
- kernel-only comparisons were used to isolate CUDA arithmetic cost
- later experiments intentionally deprioritized 30B-wide scope-all correctness until the kernel story was clearer

This is methodologically important. The work is not trying to claim a fully optimized deployment-ready inference system. It is focused on studying the central kernel bottleneck and the representation decisions that drive it.

### 4.5 Fairness and Measurement Caveats
One subtle but important issue in this implementation is the interpretation of `BSI_PROFILE=1`. When profiling is enabled, the code inserts CUDA events and synchronization at a much finer granularity so that build and dot time can be attributed precisely. This is useful for kernel research, but it is not the same as a fully asynchronous throughput measurement. Therefore:

- `BSI_PROFILE=1` is the correct setting for dot-kernel attribution and thesis reporting of internal timings;
- `BSI_PROFILE=0` is the better setting for fair end-to-end throughput studies.

Because dot time was the primary optimization target, the report uses `BSI_PROFILE=1` numbers for the main kernel story while noting this caveat clearly.

## 5. Experimental Setup
### 5.1 Hardware and Software Assumptions
The implementation assumes access to CUDA-capable GPUs and was specifically optimized for SM90/H100-class devices in the tensor-core dot path. The environment setup documented alongside the code includes:

- cluster proxy setup through `network_connection.sh`
- CUDA module loading via `nvhpc-hpcx-cuda12/24.11`
- environment activation through the `bsiPytorch` conda environment
- local extension rebuild through `bash rebuild_local.sh`

### 5.2 Core Environment Variables
The stable same-bit experimental path is configured through environment variables. The most important ones are:

- `BSI_TC_DOT=1`
- `BSI_TC_FIXED_INT=1`
- `BSI_TC_CPASYNC=1`
- `BSI_TC_TM=32`
- `BSI_FIXED_BITS_KEYS=6`
- `BSI_FIXED_BITS_QUERIES=7`
- `BSI_FIXED_CHUNK_SCALE=1`
- `BSI_TC_R_SWEEP=4`
- `BSI_TC_TMA=1`
- `BSI_DOT_DEBUG=1` (optional, for kernel-path diagnostics)

These variables are not incidental. They define the actual arithmetic regime and kernel family under study.

### 5.3 Dataset and Metric
The primary end-to-end benchmark uses the HuggingFace `lambada` validation split. The evaluator in `benchmark_performance_bsi.py`:

1. tokenizes the dataset,
2. right-pads or truncates to `max_seq_len=512`,
3. uses the last token as the label,
4. computes logits at the last non-padding position,
5. reports top-1 and top-5 accuracy.

This metric is not meant to replace a full language-model evaluation suite. It is a practical and consistent benchmark for comparing inference-quality degradation under different BSI settings.

### 5.4 Benchmark Types
The report relies on three categories of experiments:

1. `kernel_only` microbenchmarks on representative GEMM-like shapes.
2. `model_e2e` LAMBADA evaluations on OPT checkpoints.
3. targeted scope ablations (`attention`, `mlp`, `all`) to isolate sensitivity.

### 5.5 Representative Shapes
The shape families used in kernel-only benchmarking correspond roughly to layer shapes found in different OPT models:

- 125M-like: `Q=512, R=3072, D=768`
- 1.3B-like: `Q=512, R=8192, D=2048`
- 6.7B-like: `Q=512, R=16384, D=4096`

These shapes allow the kernel work to be discussed independently of one particular checkpoint.

### 5.6 Model Configurations Used in Practice
The experimental program covered several classes of experiments:

1. same-bit fixed76 runs on `opt-125m`, `opt-1.3b`, and `opt-6.7b`;
2. scope ablations on `opt-30b` using `attention`, `mlp`, and `all`;
3. lower-slice-count experiments used to study the speed/accuracy frontier;
4. SmoothQuant comparison scripts, especially for `opt-30b`.

The most defensible thesis narrative should treat these as follows:

- `opt-125m`, `opt-1.3b`, and `opt-6.7b`: primary optimization targets
- `opt-30b`: stress test for sensitivity and failure analysis
- SmoothQuant: contextual baseline, not a backend-identical comparison

### 5.7 Static Memory Accounting
The measurement framework distinguishes several memory quantities:

- dense weight bytes
- BSI weight bytes
- bias bytes
- full-model static bytes
- compression versus FP16 linear weights
- compression versus full dense model static size

The function `bsi_full_model_static_bytes` computes full static model size by adding all registered parameters and buffers to the BSI weight bytes, which are not stored as normal PyTorch parameters. This is important because several temporary packed-layout experiments looked attractive in kernel microbenchmarks but inflated model size due to duplicate storage.

### 5.8 Primary Evaluation Configuration
Unless otherwise noted, the principal comparative evaluation in this thesis uses:

- `decimal_places = 2`
- `compress_threshold = 0.5`
- `num_samples = 1000`
- `max_seq_len = 512`
- `base_dtype = fp16`

The stable same-bit runtime configuration is:

- `BSI_TC_DOT=1`
- `BSI_TC_FIXED_INT=1`
- `BSI_TC_CPASYNC=1`
- `BSI_TC_TM=32`
- `BSI_FIXED_BITS_KEYS=6`
- `BSI_FIXED_BITS_QUERIES=7`
- `BSI_FIXED_CHUNK_SCALE=1`
- `BSI_TC_R_SWEEP=4`
- `BSI_TC_TMA=1`

This is the main operating point used to summarize the present baseline across `opt-125m`, `opt-1.3b`, and `opt-6.7b`.

## 6. Results
### 6.1 Cross-Model Memory and Dot-Latency Results
Table 1 reports the primary baseline results across the three OPT models emphasized in this study. All runs use the stable same-bit fixed configuration described in Section 5.8.

| Model | Dense FP16 Static Size (MB) | BSI Static Size, `scope=all` (MB) | BSI Static Size, `scope=attention` (MB) | BSI Dot Latency, `scope=all` | Torch Dot Latency, `scope=all` | BSI Dot Latency, `scope=attention` | Torch Dot Latency, `scope=attention` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `opt-125m` | 238.875 | 137.783 | 205.195 | 1.913 ms | 3.044 ms | 0.890 ms | 2.158 ms |
| `opt-1.3b` | 2509.609 | 1070.453 | 2029.984 | 13.859 ms | 6.955 ms | 5.158 ms | 4.210 ms |
| `opt-6.7b` | 12700.031 | 5022.281 | 10141.031 | 66.452 ms | 15.493 ms | 23.003 ms | 5.357 ms |

Three observations follow immediately from Table 1.

1. The BSI representation reduces model size substantially for `scope=all`, with the strongest absolute reduction appearing on `opt-6.7b`.
2. `scope=attention` preserves much more of the original model footprint because fewer linear layers are replaced.
3. The BSI dot path scales less favorably than dense Torch as model width grows, which is especially clear at `opt-1.3b` and `opt-6.7b`.

### 6.2 Accuracy Results for `scope=all`
Table 2 reports the top-1 and top-5 accuracy results for the `scope=all` replacement policy.

| Model | Dense Top-1 | BSI Top-1 | Dense Top-5 | BSI Top-5 |
|---|---:|---:|---:|---:|
| `opt-125m` | 0.605 | 0.626 | 0.725 | 0.729 |
| `opt-1.3b` | 0.722 | 0.653 | 0.873 | 0.796 |
| `opt-6.7b` | 0.798 | 0.741 | 0.929 | 0.896 |

The `scope=all` results show that the stable same-bit baseline is usable but not uniformly robust. On the smallest model, BSI is comparable to the dense baseline. As model size increases, the accuracy gap widens, which is consistent with the view that the current fixed-bit policy is not yet sufficiently layer-aware for larger networks.

### 6.3 Accuracy Results for `scope=attention`
Table 3 reports the same accuracy metric when only the attention projections are converted to BSI.

| Model | Dense Top-1 | BSI Top-1 | Dense Top-5 | BSI Top-5 |
|---|---:|---:|---:|---:|
| `opt-125m` | 0.605 | 0.608 | 0.725 | 0.722 |
| `opt-1.3b` | 0.722 | 0.666 | 0.873 | 0.821 |
| `opt-6.7b` | 0.798 | 0.779 | 0.929 | 0.921 |

These results are important because they show that attention-only replacement is much less destructive than `scope=all`, particularly on the larger models. The implication is that the main quality bottleneck is not uniform across the network and that a thesis-grade analysis must distinguish between attention and MLP behavior rather than treating the transformer as a homogeneous collection of linear layers.

### 6.4 Stable Same-Bit `opt-6.7b` Case Study
Under the primary evaluation setting (`decimal_places=2`, `compress_threshold=0.5`, `num_samples=1000`, `max_seq_len=512`, `fp16` base dtype), `facebook/opt-6.7b` produced the following result:

- FP16 baseline:
  - `top1=0.798`
  - `top5=0.929`
  - `dot=15.493 ms`
  - static size `12700.031 MB`
- BSI same-bit baseline, `scope=all`:
  - `top1=0.741`
  - `top5=0.896`
  - `dot=66.452 ms`
  - static size `5022.281 MB`
  - compression `2.53x`
- BSI same-bit baseline, `scope=attention`:
  - `top1=0.779`
  - `top5=0.921`
  - `dot=23.003 ms`
  - static size `10141.031 MB`

This case study establishes the central tradeoff of the thesis. Converting all linear layers to BSI yields the largest compression ratio, but it also incurs the largest dot-product penalty. Restricting BSI to attention layers preserves accuracy much better and reduces dot latency substantially, but it also gives up a large portion of the memory reduction. This is therefore not a single-point optimization problem; it is a compression, accuracy, and execution tradeoff.

### 6.5 Kernel-Only Study Design
Kernel-only microbenchmarks were used to study pure dot-kernel behavior independently of full model execution. Two representative shapes were emphasized in the stable baseline:

- `Q=512, R=16384, D=4096`, corresponding to an FC1-like shape in `opt-6.7b`
- `Q=512, R=50272, D=4096`, corresponding to a very wide output projection or vocabulary-sized output

These kernel-only studies serve two purposes:

1. they isolate the arithmetic and memory behavior of the BSI dot kernel without full model overhead;
2. they provide the correct launch context for Nsight Compute profiling.

The thesis should present these kernel-only measurements as complementary to, not replacements for, the `model_e2e` results.

### 6.6 Compression and Storage Interpretation
The compression numbers are not merely cosmetic. They help clarify what BSI is buying even when it does not yet match dense Torch speed:

1. The same-bit `(7,6)` path cuts the 6.7B model static size from roughly `12.7 GB` to about `5.0 GB`.
2. More aggressive fixed-slice settings reduce size further, but at a substantial quality cost.
3. An earlier duplicate-storage bug demonstrated that layout experiments can silently erase compression gains if both legacy and experimental tensors are retained together.

Therefore, the thesis should present compression as a necessary but insufficient condition: BSI clearly compresses the model, but real performance benefits only appear once the execution path is co-designed with that representation.

### 6.7 Accuracy Sensitivity on Larger Models
The larger `facebook/opt-30b` model was used to determine which subgraphs were most sensitive under the current BSI quantization regime. The results showed:

- `scope=attention`:
  - `top1≈0.8200`
  - `top5≈0.9450`
  - close to dense baseline

- `scope=mlp`:
  - `top1≈0.7400`
  - `top5≈0.8500`
  - clear degradation

- `scope=all`:
  - `top1≈0.4700`
  - `top5≈0.6200`
  - unacceptable collapse

The interpretation is straightforward: under the current fixed-bit BSI regime, MLP layers are the dominant accuracy bottleneck at 30B scale. This supports a more general thesis point that layer class sensitivity matters and that uniform fixed-bit settings are unlikely to scale gracefully across all model widths and depths.

### 6.8 What the Results Actually Prove
Taken together, the result tables support three precise claims:

1. Same-bit kernel/layout redesign improved BSI dot time substantially on a real `opt-6.7b` end-to-end run.
2. Lowering slice counts further improves speed but moves the system to an inferior accuracy point.
3. The remaining gap to dense Torch is large enough that more fundamental arithmetic or layout redesign will still be required.

## 7. Ablation Studies and Design Iterations
### 7.1 R-Sweep and Accumulator Experiments
Several experiments tried to improve the existing fixed76 family by increasing arithmetic overlap or holding more sweep state in registers. In particular:

- `R_sweep=8` increased register usage to roughly 244 registers per thread and regressed performance.
- register-resident sweep accumulators did not beat the validated `R_sweep=4` same-bit configuration.

These experiments support a key systems conclusion: the current kernel family had already reached a local optimum with respect to these tuning knobs. More aggressive local tuning did not change the fundamental arithmetic structure.

### 7.2 Query Repacking in the Model Path
An early packed-layout design packed both the query and key sides into tile-native layouts. While this idea looked reasonable in isolated kernel contexts, it caused severe end-to-end regressions in the model path:

- large build-time blowups,
- duplicate storage,
- and temporarily broken model behavior.

The issue was structural. Query activations are dynamic; repacking them into a more complex tile format every layer negated the gains from the dot kernel itself. The correct outcome was to remove runtime query tile repacking from the active path and retain only offline key-side layout experiments.

### 7.3 Duplicate Storage Bug
At one point, one of the key-side packing experiments stored both legacy key tensors and experimental key tensors simultaneously, inflating model size. This temporarily increased the BSI model footprint for `opt-6.7b` from around `5.0 GB` to around `9.6 GB`. That path was removed, and the stable implementation restored the expected memory footprint.

This episode is useful in a thesis write-up because it demonstrates that microbench improvements alone are not sufficient; the full systems integration must be measured end to end.

### 7.4 TMA Swizzle Experiment
A later experiment tried to introduce a TMA swizzle on an alternative packed key layout intended to reduce shared-memory inefficiency. While it showed a tiny microbenchmark improvement, the end-to-end `opt-6.7b` run collapsed to zero accuracy. The code was reverted.

This is a negative result, but it is important: low-level memory-layout changes can silently alter correctness if the tensor-map semantics, tile interpretation, or memory alignment assumptions no longer match the rest of the pipeline.

### 7.5 Lower-Bit Experiments
The implementation also tested lower fixed-slice settings. These experiments did exactly what one would expect: fewer slice interactions yielded lower dot time and smaller model size. However, the accuracy frontier worsened as bit count dropped. This is exactly why the thesis should not frame lower bitwidth as the main solution. The more defensible point is that the implementation allows the speed/accuracy frontier to be characterized experimentally, and that the stable same-bit baseline remains the principal reference point.

### 7.6 Scope Ablation on 30B
The 30B experiments are best presented as a subgraph ablation rather than as a headline performance result. The key lesson is that the current BSI configuration does not fail uniformly:

- attention-only quantization stays near the dense baseline;
- MLP-only quantization degrades much more noticeably;
- full-scope quantization collapses.

This ablation is important because it changes the future-work agenda. It suggests that the next accuracy-preserving improvements are more likely to come from layer-aware policies or MLP-specific kernel/representation treatment than from continuing to tune attention projections alone.

## 8. Profiling and Bottleneck Analysis
### 8.1 Nsight Compute Findings
The most important profiler snapshot came from the same-bit tensor-core kernel on the `6.7b`-like microbench shape `Q=512, R=16384, D=4096`. The exact kernel variant changed across layout experiments, but the observed bottlenecks were consistent enough to support a stable conclusion.

The critical metrics were:

- `Registers Per Thread = 128`
- `Dynamic Shared Memory Per Block = 80 KB`
- `Theoretical Occupancy ≈ 25%`
- `Achieved Occupancy ≈ 24.29%`
- `Issued Warp Per Scheduler = 0.54`
- `Eligible Warps Per Scheduler = 1.14`
- `No Eligible = 45.61%`
- `Memory Throughput ≈ 699 GB/s`
- `Max Bandwidth ≈ 24.49%`
- `Compute Throughput ≈ 52.55%`
- `Excessive global sectors ≈ 33%`
- `Excessive shared wavefronts ≈ 46%`
- `Shared load bank conflicts ≈ 46.36%`
- `Shared store bank conflicts ≈ 10.76%`

### 8.2 Interpretation
These numbers led to a decisive conclusion: the active same-bit BSI kernel is not primarily DRAM-bandwidth bound. Instead, the dominant limitations are:

1. scheduler underutilization,
2. insufficient eligible warps,
3. shared-memory inefficiency,
4. and global-memory access overhead.

This is a crucial thesis point. It means that the remaining gap to Torch is not explained simply by “BSI moves too much data.” The representation incurs more arithmetic interactions per chunk, and the current layout/scheduling does not feed the execution units efficiently enough.

### 8.3 Why This Matters
This profiler evidence changes the optimization strategy. It implies that:

- moving from cp.async to TMA is helpful but not sufficient,
- layout-aware packing is necessary but not sufficient,
- and more of the remaining gap comes from the structure of the bit-sliced inner loop than from gross memory bandwidth.

That is a strong and publishable systems result even though the kernel still trails Torch.

## 9. Comparison with SmoothQuant and Dense Quantization Baselines
The implementation includes SmoothQuant benchmarking scripts under `benchmarks/smoothquant_benchmark/`. These scripts compare dense FP16 OPT checkpoints with pre-quantized SmoothQuant INT8 checkpoints such as `mit-han-lab/opt-30b-smoothquant`. SmoothQuant is a useful reference because it represents a mature dense W8A8 execution strategy.

However, the comparison must be framed carefully:

1. SmoothQuant is a calibration-based dense INT8 method.
2. The BSI path here is a native bit-sliced execution path without dense fallback.
3. SmoothQuant benefits from a mature dense arithmetic substrate.
4. BSI is being evaluated partly as a representation study and partly as a kernel study.

Therefore, the most appropriate use of SmoothQuant in the thesis is as contextual related work and as a benchmark reference, not as a direct like-for-like execution substrate. The thesis should be explicit that the question under study is different.

## 10. Discussion
### 10.1 What Was Actually Achieved
This work achieved several substantive milestones:

1. End-to-end integration of BSI-based linear layers into OPT inference.
2. A functioning CUDA BSI query builder and key builder.
3. A stable same-bit tensor-core/BMMA-like dot path on SM90.
4. A series of layout and staging experiments that clarified which changes materially improve same-bit BSI dot time and which ones simply move cost elsewhere in the pipeline.
5. A benchmark harness that exposes kernel-only, layer-level, and model-level tradeoffs.
6. A clear empirical speed/accuracy frontier at multiple slice settings.
7. A profiler-backed diagnosis of the remaining bottleneck.

### 10.2 What Must Be Defended in a Thesis Defense
If this work is presented to a committee, the most defensible argument is:

1. BSI was integrated as a native execution representation, not merely as a storage artifact.
2. Same-bit performance depends strongly on layout-aware staging and memory movement, not only on the nominal bit setting.
3. The code and measurements identify the true remaining bottleneck precisely.
4. The thesis therefore contributes both an implementation and a systems-level diagnosis.

If asked “why not just use INT8?”, the appropriate answer is that this thesis is not primarily about choosing the easiest deployable quantization format. It studies whether a bit-sliced representation can be made practical for GPU inference while preserving native bitwise execution. That is a distinct research question.

If asked “why are you still slower than Torch?”, the answer is that the work has already identified why: the current kernel is limited by low scheduler eligibility and access inefficiency, not by a simple lack of memory bandwidth. That is not an excuse; it is one of the primary technical findings.

### 10.3 What Was Not Achieved
Several things were not achieved and should not be overstated:

1. The BSI dot kernel does not yet match Torch FP16 throughput.
2. The current fixed-bit BSI regime does not scale cleanly to `opt-30b` under `scope=all`.
3. Lowering slice counts improves speed but sacrifices too much accuracy to be the central thesis result.
4. Not every aggressive shared-memory or TMA-layout experiment preserved correctness.

### 10.4 Threats to Validity and Limitations
The results in this work are meaningful, but the report should acknowledge several limitations:

1. Most performance-critical runs were conducted on Hopper-class systems, so the conclusions are not automatically portable to all CUDA architectures.
2. The main quality metric is LAMBADA next-token accuracy rather than a broader evaluation suite.
3. Some results were measured under `BSI_PROFILE=1`, which is correct for attribution but not the fairest total-throughput setting.
4. The SmoothQuant scripts use pretrained or calibrated INT8 models, whereas the BSI path quantizes through the implementation’s own fixed-bit pipeline.
5. The 30B results are sufficient to identify sensitivity, but not sufficient to claim a robust large-model operating point.

These limitations do not invalidate the work, but they define its scope clearly and honestly.

### 10.5 What the Work Teaches
The most important lesson is that BSI is not just a storage format problem. The representation and the kernel must be designed together. A generic bit-sliced layout plus a clever kernel is not enough; a clever layout plus a generic kernel is also not enough. The research value of this work lies in making that point concrete with end-to-end measurements and profiler evidence.

## 11. Future Work
### 11.1 Mixed-Radix or Grouped-Bit Arithmetic
The current fixed `(Sa,Sb)` design still requires many slice-pair interactions per chunk. A promising next step is to reduce effective interaction count without abandoning BSI representation. One route is a mixed-radix or grouped-bit formulation that changes the arithmetic structure of the bit-sliced product rather than merely tuning the existing 42-interaction loop.

### 11.2 Shape-Specialized Kernel Families
The current implementation is reasonably general within the fixed-bit family, but the profiler results suggest that more aggressive shape specialization could improve scheduler efficiency. In particular, separate kernel families for:

- attention projections,
- FC1-style expansion layers,
- FC2-style projection layers,

may improve tile reuse and reduce shared-memory conflict behavior.

### 11.3 Layer-Aware Precision Policy
The 30B experiments showed that MLP layers are more sensitive than attention layers under uniform fixed-bit settings. A future system should therefore use:

- layer-aware or subgraph-aware bit settings,
- calibration-informed thresholds,
- or BSI-aware smoothing analogous in spirit to activation-aware quantization.

### 11.4 Better Shared-Memory Layouts
Nsight Compute showed large excessive shared wavefront counts and notable bank conflict rates. Future work should redesign the shared-memory layouts used by the bit-sliced kernel to minimize bank conflicts under the exact BMMA access pattern rather than relying on generic padding or swizzle experiments.

### 11.5 Broader Model Coverage
While `opt-125m`, `opt-1.3b`, and `opt-6.7b` are adequate for a thesis baseline, the next stage should validate the same design on other transformer families, especially those with different hidden-size ratios or attention structures.

### 11.6 Better Evaluation Framing
Future work should also separate three experimental regimes more explicitly:

1. kernel research runs,
2. fair throughput runs,
3. accuracy-preservation runs.

The current implementation already contains the pieces to do so, but a more formal paper or thesis submission should present them in separate tables and with separate measurement protocols.

## 12. Conclusion
This study establishes a complete experimental pipeline for BSI-native inference for large language models on GPUs. The work moves beyond toy kernels by integrating BSI-based linear layers into OPT inference, benchmarking them at kernel, layer, and model scales, and diagnosing their bottlenecks with real profiler data. The stable same-bit path preserves a meaningful compression ratio and acceptable accuracy on `opt-6.7b` while clearly exposing the performance cost of native BSI execution. At the same time, the work shows why BSI still trails dense FP16 inference: the active kernel is limited by scheduler eligibility and memory access inefficiency rather than by raw memory bandwidth alone.

The central research conclusion is therefore not that BSI already solves LLM inference, but that BSI becomes meaningfully more viable when storage layout and execution kernel are co-designed. This is the main technical argument that the thesis can defend. The current implementation has already demonstrated that point through the stable same-bit tensor-core baseline, the rejected alternative layouts, and the profiler evidence that explains where the remaining gap comes from. The next phase is now well defined: redesign the BSI arithmetic and memory layout so that the scheduler can sustain much higher issue efficiency without abandoning the BSI-native execution model.

## 13. Suggested Thesis Title and Positioning
The most appropriate thesis title for this body of work is:

**BSI-Native Tensor-Core Kernels for Compressed LLM Inference**

Alternative titles that still fit the evidence are:

1. **Co-Design of Bit-Sliced Layouts and Tensor-Core Kernels for LLM Inference**
2. **Bit-Sliced Weight Representations and GPU Kernels for Efficient LLM Inference**
3. **Layout-Aware BSI Acceleration for Large Language Model Inference**

The first title is the strongest because it makes the key claim explicit: the contribution is native BSI execution on tensor-core-oriented GPU kernels, not merely model compression.

## 14. Reproducibility Notes
### 14.1 Stable Same-Bit Environment
```bash
export BSI_TC_DOT=1
export BSI_TC_FIXED_INT=1
export BSI_TC_CPASYNC=1
export BSI_TC_TM=32
export BSI_FIXED_BITS_KEYS=6
export BSI_FIXED_BITS_QUERIES=7
export BSI_FIXED_CHUNK_SCALE=1
export BSI_TC_R_SWEEP=4
export BSI_TC_TMA=1
export BSI_DOT_DEBUG=1
```

### 14.2 Kernel-Only `6.7b`-Like Microbenchmark
```bash
python benchmarks/benchmark_apples_to_apples_bsi.py --modes kernel_only --Q 512 --R 16384 --D 4096 --decimal_places 2 --compress_threshold 0.5 --torch_dtype fp16 --bsi_profile 0 --base_dtype fp16
```

### 14.3 Kernel-Only Very-Wide Output Microbenchmark
```bash
python benchmarks/benchmark_apples_to_apples_bsi.py --modes kernel_only --Q 512 --R 50272 --D 4096 --decimal_places 2 --compress_threshold 0.5 --torch_dtype fp16 --bsi_profile 0 --base_dtype fp16
```

### 14.4 End-to-End `opt-6.7b` Run
```bash
python benchmarks/benchmark_apples_to_apples_bsi.py --modes model_e2e --model_name facebook/opt-6.7b --dataset lambada --split validation --num_samples 200 --decimal_places 2 --compress_threshold 0.5 --scope all --bsi_device cuda --bsi_profile 1 --base_dtype fp16
```

### 14.5 Nsight Compute for the Stable Tensor-Core Path
```bash
ncu --set full --target-processes all -f -o ncu_rsweep4_tma_fc1 \
  -k popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep4_tma_tensorB \
  --launch-count 1 \
  python benchmarks/benchmark_apples_to_apples_bsi.py --modes kernel_only \
    --Q 512 --R 16384 --D 4096 \
    --decimal_places 2 --compress_threshold 0.5 \
    --warmup 0 --iters 1 --torch_dtype fp16 --bsi_profile 0 --base_dtype fp16
```

## 15. References
[1] Susan Zhang et al. “OPT: Open Pre-trained Transformer Language Models.” arXiv, 2022. https://arxiv.org/abs/2205.01068

[2] Tim Dettmers et al. “LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.” arXiv, 2022. https://arxiv.org/abs/2208.07339

[3] Zhewei Yao et al. “ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers.” arXiv, 2022. https://arxiv.org/abs/2206.01861

[4] Elias Frantar et al. “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.” arXiv, 2022. https://arxiv.org/abs/2210.17323

[5] Guangxuan Xiao et al. “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.” ICML / PMLR, 2023. https://proceedings.mlr.press/v202/xiao23c.html

[6] Ji Lin et al. “AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration.” arXiv, 2023. https://arxiv.org/abs/2306.00978

[7] Denis Paperno et al. “The LAMBADA dataset: Word prediction requiring a broad discourse context.” arXiv, 2016. https://arxiv.org/abs/1606.06031

[8] NVIDIA. “Hopper Tuning Guide.” CUDA Documentation. https://docs.nvidia.com/cuda/hopper-tuning-guide/

[9] NVIDIA. “CUDA Driver API: Tensor Map Object Management.” CUDA Documentation. https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
