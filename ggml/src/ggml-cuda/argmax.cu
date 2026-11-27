#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
        const int   col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
                const int   col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

template <bool has_add>
static __global__ void argmax_f32_stage1(
        const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ out_val, int32_t * __restrict__ out_idx,
        const int64_t ncols, const int64_t sx1, const int64_t sy1, const int n_blocks) {
    const int64_t row = blockIdx.y;
    const int     bx  = blockIdx.x;
    const float * rowx = x + row * sx1;
    const float * rowy = has_add ? (y + row * sy1) : nullptr;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const int64_t stride = (int64_t) n_blocks * blockDim.x;
    for (int64_t col = (int64_t) bx * blockDim.x + threadIdx.x; col < ncols; col += stride) {
        const float val = has_add ? (rowx[col] + rowy[col]) : rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = (int) col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
        const int   col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
        if (val > maxval || (val == maxval && col >= 0 && (argmax < 0 || col < argmax))) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }
        __syncthreads();
        if (warp_id == 0) {
            maxval = lane_id < n_warps ? shared_maxval[lane_id] : -FLT_MAX;
            argmax = lane_id < n_warps ? shared_argmax[lane_id] : -1;
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
                const int   col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
                if (val > maxval || (val == maxval && col >= 0 && (argmax < 0 || col < argmax))) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        out_val[row * n_blocks + bx] = maxval;
        out_idx[row * n_blocks + bx] = argmax;
    }
}

static __global__ void argmax_f32_stage2(
        const float * __restrict__ in_val, const int32_t * __restrict__ in_idx,
        int32_t * __restrict__ dst, const int n_blocks) {
    const int64_t row = blockIdx.x;
    const float   * vals = in_val + row * n_blocks;
    const int32_t * idxs = in_idx + row * n_blocks;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        const float val = vals[i];
        const int   col = idxs[i];
        if (val > maxval || (val == maxval && col >= 0 && (argmax < 0 || col < argmax))) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
        const int   col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
        if (val > maxval || (val == maxval && col >= 0 && (argmax < 0 || col < argmax))) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }
        __syncthreads();
        if (warp_id == 0) {
            maxval = lane_id < n_warps ? shared_maxval[lane_id] : -FLT_MAX;
            argmax = lane_id < n_warps ? shared_argmax[lane_id] : -1;
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
                const int   col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
                if (val > maxval || (val == maxval && col >= 0 && (argmax < 0 || col < argmax))) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    // split wide rows across CUs when nrows is small (e.g. vocab argmax in speculative drafting)
    if (nrows <= 16 && ne00 >= 16384) {
        const int n_blocks = (int) std::min<int64_t>(128, (ne00 + 1023) / 1024);
        ggml_cuda_pool_alloc<float>   tmp_val(ctx.pool(), (size_t) nrows * n_blocks);
        ggml_cuda_pool_alloc<int32_t> tmp_idx(ctx.pool(), (size_t) nrows * n_blocks);
        argmax_f32_stage1<false><<<dim3(n_blocks, nrows), 256, 0, stream>>>(src0_d, nullptr, tmp_val.get(), tmp_idx.get(), ne00, ne00, 0, n_blocks);
        const int t2 = std::min<int>(256, ((n_blocks + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);
        argmax_f32_stage2<<<dim3(nrows), t2, 0, stream>>>(tmp_val.get(), tmp_idx.get(), dst_d, n_blocks);
        return;
    }

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}

void ggml_cuda_argmax_add(ggml_backend_cuda_context & ctx, const ggml_tensor * add_node, ggml_tensor * dst) {
    const ggml_tensor * a = add_node->src[0];
    const ggml_tensor * b = add_node->src[1];

    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(a, b) && a->ne[2] == 1 && a->ne[3] == 1);
    GGML_ASSERT(a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float));

    const int64_t ne00  = a->ne[0];
    const int64_t nrows = a->ne[1];
    const int64_t sa1   = a->nb[1] / sizeof(float);
    const int64_t sb1   = b->nb[1] / sizeof(float);

    const float * a_d   = (const float *) a->data;
    const float * b_d   = (const float *) b->data;
    int32_t     * dst_d = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();
    const int n_blocks = (int) std::min<int64_t>(128, (ne00 + 1023) / 1024);
    ggml_cuda_pool_alloc<float>   tmp_val(ctx.pool(), (size_t) nrows * n_blocks);
    ggml_cuda_pool_alloc<int32_t> tmp_idx(ctx.pool(), (size_t) nrows * n_blocks);
    argmax_f32_stage1<true><<<dim3(n_blocks, nrows), 256, 0, stream>>>(a_d, b_d, tmp_val.get(), tmp_idx.get(), ne00, sa1, sb1, n_blocks);
    const int t2 = std::min<int>(256, ((n_blocks + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);
    argmax_f32_stage2<<<dim3(nrows), t2, 0, stream>>>(tmp_val.get(), tmp_idx.get(), dst_d, n_blocks);
}
