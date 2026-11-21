#pragma once

#include "ggml-backend.h"
#include "common.cuh"

bool ggml_cuda_get_tensor_n(int device,
        const ggml_tensor * const * tensors, void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors);

bool ggml_cuda_set_tensor_n(int device,
        ggml_tensor * const * tensors, const void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors);

#if defined(GGML_USE_HIP)
void ggml_hip_register_host_alloc(void * host_ptr, size_t size);
void ggml_hip_unregister_host_alloc(void * host_ptr);
bool ggml_hip_copy_pinned_async(int device, void * dst, const void * src, size_t size, cudaMemcpyKind kind, cudaStream_t stream);
#endif
