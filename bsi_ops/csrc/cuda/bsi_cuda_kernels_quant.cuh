#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

template <typename T>
__device__ __forceinline__ float bsi_cast_to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float bsi_cast_to_float<half>(half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float bsi_cast_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ long long bsi_round_half_away_to_ll(double x) {
    if (x >= 0.0) {
        return static_cast<long long>(floor(x + 0.5));
    }
    return -static_cast<long long>(floor(-x + 0.5));
}

__device__ __forceinline__ unsigned long long bsi_uabs_ll(long long v) {
    return static_cast<unsigned long long>(v >= 0 ? v : -v);
}

__device__ __forceinline__ unsigned long long bsi_warp_reduce_max_u64(unsigned long long v) {
    for (int off = 16; off > 0; off >>= 1) {
        unsigned long long o = __shfl_down_sync(0xffffffff, v, off);
        if (o > v) v = o;
    }
    return v;
}

__device__ __forceinline__ double bsi_warp_reduce_sum_f64(double v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ void bsi_block_reduce_max_sum(
    unsigned long long local_max,
    double local_sum,
    unsigned long long* out_max,
    double* out_sum) {
    constexpr int WARPS = BLOCK_SIZE / 32;
    __shared__ unsigned long long smax[WARPS];
    __shared__ double ssum[WARPS];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    local_max = bsi_warp_reduce_max_u64(local_max);
    local_sum = bsi_warp_reduce_sum_f64(local_sum);
    if (lane == 0) {
        smax[warp] = local_max;
        ssum[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        unsigned long long v_max = (lane < WARPS) ? smax[lane] : 0ULL;
        double v_sum = (lane < WARPS) ? ssum[lane] : 0.0;
        v_max = bsi_warp_reduce_max_u64(v_max);
        v_sum = bsi_warp_reduce_sum_f64(v_sum);
        if (lane == 0) {
            *out_max = v_max;
            *out_sum = v_sum;
        }
    }
}

__device__ __forceinline__ int bsi_compute_shift(
    unsigned long long max_abs,
    double sum_abs,
    int count,
    int fixed_bits,
    float clip_k) {
    if (fixed_bits <= 0) return 0;
    int bits = 0;
    if (clip_k > 0.0f && count > 0) {
        double effective = static_cast<double>(max_abs);
        const double mean_abs = sum_abs / static_cast<double>(count);
        const double clip_max = mean_abs * static_cast<double>(clip_k);
        if (clip_max < effective) effective = clip_max;
        if (effective > 0.0) {
            int exp2 = 0;
            (void)frexp(effective, &exp2);
            bits = exp2;
        }
    } else {
        bits = (max_abs > 0ULL) ? (64 - __clzll(max_abs)) : 0;
    }

    int shift = bits - (fixed_bits - 1);
    if (shift < 0) shift = 0;
    if (shift > 62) shift = 62;
    return shift;
}

template <typename T>
__global__ void quantize_round_to_int64_batch_kernel(
    const T* __restrict__ input,
    int64_t total,
    double dec_scale,
    long long* __restrict__ out_ints) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    const float f = bsi_cast_to_float(input[idx]);
    const double scaled = static_cast<double>(f) * dec_scale;
    out_ints[idx] = bsi_round_half_away_to_ll(scaled);
}

template <int BLOCK_SIZE>
__global__ void compute_row_shift_scale_kernel(
    const long long* __restrict__ ints,
    int64_t Q,
    int64_t d,
    int fixed_bits,
    float clip_k,
    int* __restrict__ shifts,
    float* __restrict__ scales) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= Q) return;

    unsigned long long local_max = 0ULL;
    double local_sum = 0.0;
    const int64_t base = row * d;
    for (int64_t c = threadIdx.x; c < d; c += BLOCK_SIZE) {
        const unsigned long long av = bsi_uabs_ll(ints[base + c]);
        if (av > local_max) local_max = av;
        local_sum += static_cast<double>(av);
    }

    __shared__ unsigned long long block_max;
    __shared__ double block_sum;
    bsi_block_reduce_max_sum<BLOCK_SIZE>(local_max, local_sum, &block_max, &block_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        const int shift = bsi_compute_shift(block_max, block_sum, static_cast<int>(d), fixed_bits, clip_k);
        shifts[row] = shift;
        scales[row] = static_cast<float>(static_cast<unsigned long long>(1ULL) << shift);
    }
}

template <int BLOCK_SIZE>
__global__ void compute_chunk_shift_scale_kernel(
    const long long* __restrict__ ints,
    int64_t Q,
    int64_t d,
    int chunks,
    int fixed_bits,
    float clip_k,
    int* __restrict__ shifts,
    float* __restrict__ scales) {
    const int chunk = blockIdx.x;
    const int64_t row = static_cast<int64_t>(blockIdx.y);
    if (row >= Q || chunk >= chunks) return;

    const int64_t start = static_cast<int64_t>(chunk) * 256LL;
    int64_t rem = d - start;
    if (rem < 0) rem = 0;
    const int64_t valid = (rem < 256LL) ? rem : 256LL;
    if (valid <= 0) {
        if (threadIdx.x == 0) {
            const int idx = static_cast<int>(row) * chunks + chunk;
            shifts[idx] = 0;
            scales[idx] = 1.0f;
        }
        return;
    }

    unsigned long long local_max = 0ULL;
    double local_sum = 0.0;
    const int64_t base = row * d + start;
    for (int64_t i = threadIdx.x; i < valid; i += BLOCK_SIZE) {
        const unsigned long long av = bsi_uabs_ll(ints[base + i]);
        if (av > local_max) local_max = av;
        local_sum += static_cast<double>(av);
    }

    __shared__ unsigned long long block_max;
    __shared__ double block_sum;
    bsi_block_reduce_max_sum<BLOCK_SIZE>(local_max, local_sum, &block_max, &block_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        const int shift = bsi_compute_shift(block_max, block_sum, static_cast<int>(valid), fixed_bits, clip_k);
        const int idx = static_cast<int>(row) * chunks + chunk;
        shifts[idx] = shift;
        scales[idx] = static_cast<float>(static_cast<unsigned long long>(1ULL) << shift);
    }
}

__global__ void apply_row_shift_clamp_kernel(
    long long* __restrict__ ints,
    int64_t total,
    int64_t d,
    const int* __restrict__ shifts,
    long long qmin,
    long long qmax) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;

    const int64_t row = idx / d;
    const int shift = shifts[row];
    const long long v = ints[idx];
    const unsigned long long av = bsi_uabs_ll(v);
    const unsigned long long round_add = (shift > 0) ? (1ULL << (shift - 1)) : 0ULL;
    const unsigned long long q = (av + round_add) >> shift;

    long long out = (v < 0) ? -static_cast<long long>(q) : static_cast<long long>(q);
    if (out < qmin) out = qmin;
    if (out > qmax) out = qmax;
    ints[idx] = out;
}

__global__ void apply_chunk_shift_clamp_kernel(
    long long* __restrict__ ints,
    int64_t total,
    int64_t d,
    int chunks,
    const int* __restrict__ shifts,
    long long qmin,
    long long qmax) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;

    const int64_t row = idx / d;
    const int64_t col = idx - row * d;
    int chunk = static_cast<int>(col >> 8); // /256
    if (chunk >= chunks) chunk = chunks - 1;
    const int shift = shifts[static_cast<int>(row) * chunks + chunk];

    const long long v = ints[idx];
    const unsigned long long av = bsi_uabs_ll(v);
    const unsigned long long round_add = (shift > 0) ? (1ULL << (shift - 1)) : 0ULL;
    const unsigned long long q = (av + round_add) >> shift;
    long long out = (v < 0) ? -static_cast<long long>(q) : static_cast<long long>(q);
    if (out < qmin) out = qmin;
    if (out > qmax) out = qmax;
    ints[idx] = out;
}

extern "C" void launch_quantize_round_to_int64_batch(
    const void* input,
    int input_dtype,
    int64_t Q,
    int64_t d,
    double dec_scale,
    long long* out_ints,
    cudaStream_t stream) {
    const int64_t total = Q * d;
    if (total <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    switch (input_dtype) {
        case 0:
            quantize_round_to_int64_batch_kernel<float><<<grid, BLOCK, 0, stream>>>(
                reinterpret_cast<const float*>(input), total, dec_scale, out_ints);
            break;
        case 1:
            quantize_round_to_int64_batch_kernel<half><<<grid, BLOCK, 0, stream>>>(
                reinterpret_cast<const half*>(input), total, dec_scale, out_ints);
            break;
        case 2:
            quantize_round_to_int64_batch_kernel<__nv_bfloat16><<<grid, BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input), total, dec_scale, out_ints);
            break;
        default:
            quantize_round_to_int64_batch_kernel<float><<<grid, BLOCK, 0, stream>>>(
                reinterpret_cast<const float*>(input), total, dec_scale, out_ints);
            break;
    }
}

extern "C" void launch_compute_row_shift_scale(
    const long long* ints,
    int64_t Q,
    int64_t d,
    int fixed_bits,
    float clip_k,
    int* shifts,
    float* scales,
    cudaStream_t stream) {
    if (Q <= 0 || d <= 0) return;
    constexpr int BLOCK = 256;
    compute_row_shift_scale_kernel<BLOCK><<<static_cast<int>(Q), BLOCK, 0, stream>>>(
        ints, Q, d, fixed_bits, clip_k, shifts, scales);
}

extern "C" void launch_compute_chunk_shift_scale(
    const long long* ints,
    int64_t Q,
    int64_t d,
    int chunks,
    int fixed_bits,
    float clip_k,
    int* shifts,
    float* scales,
    cudaStream_t stream) {
    if (Q <= 0 || d <= 0 || chunks <= 0) return;
    constexpr int BLOCK = 256;
    dim3 grid(static_cast<unsigned>(chunks), static_cast<unsigned>(Q), 1);
    compute_chunk_shift_scale_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
        ints, Q, d, chunks, fixed_bits, clip_k, shifts, scales);
}

extern "C" void launch_apply_row_shift_clamp(
    long long* ints,
    int64_t Q,
    int64_t d,
    const int* shifts,
    int fixed_bits,
    cudaStream_t stream) {
    const int64_t total = Q * d;
    if (total <= 0) return;
    const long long qmax = (static_cast<long long>(1) << (fixed_bits - 1)) - 1LL;
    const long long qmin = -(static_cast<long long>(1) << (fixed_bits - 1));
    constexpr int BLOCK = 256;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    apply_row_shift_clamp_kernel<<<grid, BLOCK, 0, stream>>>(ints, total, d, shifts, qmin, qmax);
}

extern "C" void launch_apply_chunk_shift_clamp(
    long long* ints,
    int64_t Q,
    int64_t d,
    int chunks,
    const int* shifts,
    int fixed_bits,
    cudaStream_t stream) {
    const int64_t total = Q * d;
    if (total <= 0) return;
    const long long qmax = (static_cast<long long>(1) << (fixed_bits - 1)) - 1LL;
    const long long qmin = -(static_cast<long long>(1) << (fixed_bits - 1));
    constexpr int BLOCK = 256;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    apply_chunk_shift_clamp_kernel<<<grid, BLOCK, 0, stream>>>(
        ints, total, d, chunks, shifts, qmin, qmax);
}
