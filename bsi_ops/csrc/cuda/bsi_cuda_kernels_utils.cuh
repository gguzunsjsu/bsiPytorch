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

__inline__ __device__ void cp_async_4(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned int smem = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 4;\n" :: "r"(smem), "l"(src));
#else
    *reinterpret_cast<int*>(dst) = *reinterpret_cast<const int*>(src);
#endif
}

__inline__ __device__ void cp_async_8(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned int smem = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 8;\n" :: "r"(smem), "l"(src));
#else
    *reinterpret_cast<unsigned long long*>(dst) = *reinterpret_cast<const unsigned long long*>(src);
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
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        cp_async_8(dst + idx, src + idx);
    }
}

__inline__ __device__ void cp_async_copy_float(float* dst,
                                               const float* src,
                                               int count) {
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        cp_async_4(dst + idx, src + idx);
    }
}
