#pragma once

__inline__ __device__ unsigned long long warp_reduce_sum_ull(unsigned long long v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float warp_reduce_sum_float(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ void cp_async_16(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned int smem = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(src));
#else
    unsigned long long* d = reinterpret_cast<unsigned long long*>(dst);
    const unsigned long long* s = reinterpret_cast<const unsigned long long*>(src);
    d[0] = s[0];
    d[1] = s[1];
#endif
}

__inline__ __device__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

__inline__ __device__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

__inline__ __device__ void cp_async_copy_ull(unsigned long long* dst,
                                             const unsigned long long* src,
                                             int count) {
    int pairs = count >> 1;
    for (int idx = threadIdx.x; idx < pairs; idx += blockDim.x) {
        cp_async_16(dst + idx * 2, src + idx * 2);
    }
}

__inline__ __device__ void cp_async_copy_float(float* dst,
                                               const float* src,
                                               int count) {
    int quads = count >> 2;
    for (int idx = threadIdx.x; idx < quads; idx += blockDim.x) {
        cp_async_16(dst + idx * 4, src + idx * 4);
    }
}

__inline__ __device__ void cp_async_tail_ull(unsigned long long* dst,
                                             const unsigned long long* src,
                                             int count) {
    if ((count & 1) != 0) {
        if (threadIdx.x == 0) {
            dst[count - 1] = __ldg(src + count - 1);
        }
    }
}

__inline__ __device__ void cp_async_tail_float(float* dst,
                                               const float* src,
                                               int count) {
    int rem = count & 3;
    int base = count & ~3;
    for (int idx = threadIdx.x; idx < rem; idx += blockDim.x) {
        dst[base + idx] = __ldg(src + base + idx);
    }
}
