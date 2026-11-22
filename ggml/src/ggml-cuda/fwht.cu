#include "common.cuh"
#include "convert.cuh"
#include "fwht.cuh"

// wide FWHT blocks use one row per thread block with this many threads
#define GGML_CUDA_FWHT_BLOCK_NT 256

template <int N, typename T>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * warp_size + lane]) * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// Wide blocks: one row per thread block instead of per warp, so each thread keeps N/NT
// values rather than N/32. Stages below the warp width still shuffle, those up to the
// block width go through shared memory, and the rest stay in registers.
template <int N, int NT, typename T>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_sync();

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * NT + tid]) * scale;
    }

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <typename T>
static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const T *     src_d = (const T *) src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 128:
            ggml_cuda_kernel_launch(fwht_cuda<128, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 256:
            ggml_cuda_kernel_launch(fwht_cuda<256, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        default:
            break;
    }

    // wide blocks: one row per thread block
    {
        constexpr int nt = GGML_CUDA_FWHT_BLOCK_NT;

        dim3 grid_dims_w(rows, 1, 1);
        dim3 block_dims_w(nt, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params_w =
            ggml_cuda_kernel_launch_params(grid_dims_w, block_dims_w, 0, stream);

        switch (n) {
            case 1024:
                ggml_cuda_kernel_launch(fwht_cuda_block<1024, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
            case 2048:
                ggml_cuda_kernel_launch(fwht_cuda_block<2048, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
            case 4096:
                ggml_cuda_kernel_launch(fwht_cuda_block<4096, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
#if !defined(GGML_USE_MUSA)
            // 32 KB of shared memory, above the MUSA limit; falls back there
            case 8192:
                ggml_cuda_kernel_launch(fwht_cuda_block<8192, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
#endif // !defined(GGML_USE_MUSA)
            default:
                return false;
        }
    }
}

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];

    return op->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           (b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16) && ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(b, op);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    switch (src->type) {
        case GGML_TYPE_F32:
            return ggml_cuda_op_fwht_impl<float>(ctx, src, dst);
        case GGML_TYPE_F16:
            return ggml_cuda_op_fwht_impl<half>(ctx, src, dst);
        default:
            return false;
    }
}

// fused warp-level FWHT + Q8_0 quantization + row scatter for quantized KV cache writes
template <int N, typename T>
static __device__ __forceinline__ void fwht_set_rows_q8_0_row(
        const T * __restrict__ src, const int64_t * __restrict__ idxs, char * __restrict__ dst,
        const int64_t n_rows, const int rows_per_tok, const int64_t s_idx, const size_t nb1, const float scale, const int bx) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static constexpr int el_w = N / warp_size;

    const int64_t r = (int64_t) bx * blockDim.y + threadIdx.y;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    const int lane = threadIdx.x;
    float reg[el_w];

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * warp_size + lane]) * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    const int64_t tok     = r / rows_per_tok;
    const int     head    = (int) (r % rows_per_tok);
    const int64_t dst_row = idxs[tok * s_idx];
    block_q8_0 *  row_ptr = (block_q8_0 *) (dst + dst_row * nb1);
    const int     sub     = lane & 31;

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const float v = reg[i];
        float amax = fabsf(v);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, ggml_cuda_shfl_xor(amax, mask, 32));
        }
        const float  d  = amax / 127.0f;
        const float  id = d ? 1.0f/d : 0.0f;
        const int8_t q  = roundf(v*id);

        const int blk_idx = (head * N + i * warp_size + (lane & ~31)) / QK8_0;
        block_q8_0 * blk = row_ptr + blk_idx;
        if (sub == 0) {
            blk->d = d;
        }
        blk->qs[sub] = q;
    }
}

template <int N, typename T>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void fwht_set_rows_q8_0_cuda(
        const T * __restrict__ src, const int64_t * __restrict__ idxs, char * __restrict__ dst,
        const int64_t n_rows, const int rows_per_tok, const int64_t s_idx, const size_t nb1, const float scale) {
    ggml_cuda_pdl_sync();
    fwht_set_rows_q8_0_row<N, T>(src, idxs, dst, n_rows, rows_per_tok, s_idx, nb1, scale, blockIdx.x);
}

struct fwht_set_rows_args {
    const void    * src;
    const int64_t * idxs;
    char          * dst;
    int64_t         n_rows;
    int             rows_per_tok;
    int64_t         s_idx;
    size_t          nb1;
    int             n_blocks;
};

// two fused FWHT + set_rows (K and V) in one launch: blocks [0, a.n_blocks) take a, the rest take b
template <int NA, int NB, typename T>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void fwht_set_rows2_q8_0_cuda(const fwht_set_rows_args a, const fwht_set_rows_args b) {
    ggml_cuda_pdl_sync();
    if ((int) blockIdx.x < a.n_blocks) {
        fwht_set_rows_q8_0_row<NA, T>((const T *) a.src, a.idxs, a.dst, a.n_rows, a.rows_per_tok, a.s_idx, a.nb1, 1.0f/sqrtf((float) NA), blockIdx.x);
    } else {
        fwht_set_rows_q8_0_row<NB, T>((const T *) b.src, b.idxs, b.dst, b.n_rows, b.rows_per_tok, b.s_idx, b.nb1, 1.0f/sqrtf((float) NB), blockIdx.x - a.n_blocks);
    }
}

bool ggml_cuda_op_fwht_set_rows_supported(const ggml_tensor * mm, const ggml_tensor * reshape, const ggml_tensor * view, const ggml_tensor * set_rows) {
    if (!ggml_cuda_op_mul_mat_use_fwht(mm)) {
        return false;
    }
    const int64_t n = mm->ne[0];
    if (n != 64 && n != 128 && n != 256 && n != 512) {
        return false;
    }
    if (reshape->op != GGML_OP_RESHAPE || reshape->src[0] != mm) {
        return false;
    }
    if (view->op != GGML_OP_VIEW || view->src[0] != reshape || !ggml_is_contiguous(view) || view->view_offs != 0) {
        return false;
    }
    if (view->ne[0] % n != 0 || view->ne[2] != 1 || view->ne[3] != 1 || ggml_nelements(view) != ggml_nelements(mm)) {
        return false;
    }
    if (set_rows->op != GGML_OP_SET_ROWS || set_rows->src[0] != view || set_rows->type != GGML_TYPE_Q8_0) {
        return false;
    }
    const ggml_tensor * idxs = set_rows->src[1];
    return idxs->type == GGML_TYPE_I64 && set_rows->ne[0] == view->ne[0] &&
        set_rows->ne[2] == 1 && set_rows->ne[3] == 1 && idxs->ne[0] == view->ne[1] &&
        idxs->ne[1] == 1 && idxs->ne[2] == 1 && idxs->ne[3] == 1 && set_rows->nb[0] == sizeof(block_q8_0);
}

template <typename T>
static void ggml_cuda_op_fwht_set_rows_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * set_rows) {
    const int     n            = (int) src->ne[0];
    const int64_t rows         = ggml_nrows(src);
    const int     rows_per_tok = (int) (set_rows->ne[0] / n);
    const int64_t s_idx        = set_rows->src[1]->nb[0] / sizeof(int64_t);
    const size_t  nb1          = set_rows->nb[1];
    const float   scale        = 1.0f / sqrtf((float) n);

    const int warp_size      = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;
    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    dim3 grid_dims(num_blocks, 1, 1);
    dim3 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, ctx.stream());

    const T *       src_d  = (const T *) src->data;
    const int64_t * idxs_d = (const int64_t *) set_rows->src[1]->data;
    char *          dst_d  = (char *) set_rows->data;

    switch (n) {
        case  64: ggml_cuda_kernel_launch(fwht_set_rows_q8_0_cuda< 64, T>, launch_params, src_d, idxs_d, dst_d, rows, rows_per_tok, s_idx, nb1, scale); break;
        case 128: ggml_cuda_kernel_launch(fwht_set_rows_q8_0_cuda<128, T>, launch_params, src_d, idxs_d, dst_d, rows, rows_per_tok, s_idx, nb1, scale); break;
        case 256: ggml_cuda_kernel_launch(fwht_set_rows_q8_0_cuda<256, T>, launch_params, src_d, idxs_d, dst_d, rows, rows_per_tok, s_idx, nb1, scale); break;
        case 512: ggml_cuda_kernel_launch(fwht_set_rows_q8_0_cuda<512, T>, launch_params, src_d, idxs_d, dst_d, rows, rows_per_tok, s_idx, nb1, scale); break;
        default: GGML_ABORT("unsupported fwht_set_rows n=%d", n);
    }
}

void ggml_cuda_op_fwht_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * set_rows) {
    const ggml_tensor * src = mm->src[1];
    switch (src->type) {
        case GGML_TYPE_F32: ggml_cuda_op_fwht_set_rows_impl<float>(ctx, src, set_rows); break;
        case GGML_TYPE_F16: ggml_cuda_op_fwht_set_rows_impl<half>(ctx, src, set_rows);  break;
        default: GGML_ABORT("unsupported fwht_set_rows src type %s", ggml_type_name(src->type));
    }
}

static fwht_set_rows_args fwht_set_rows_make_args(const ggml_tensor * mm, const ggml_tensor * set_rows) {
    const ggml_tensor * src = mm->src[1];
    fwht_set_rows_args a = {};
    a.src          = src->data;
    a.idxs         = (const int64_t *) set_rows->src[1]->data;
    a.dst          = (char *) set_rows->data;
    a.n_rows       = ggml_nrows(src);
    a.rows_per_tok = (int) (set_rows->ne[0] / src->ne[0]);
    a.s_idx        = set_rows->src[1]->nb[0] / sizeof(int64_t);
    a.nb1          = set_rows->nb[1];
    a.n_blocks     = (int) ((a.n_rows + 3) / 4);
    return a;
}

bool ggml_cuda_op_fwht_set_rows2(ggml_backend_cuda_context & ctx, const ggml_tensor * mm0, ggml_tensor * set_rows0,
        const ggml_tensor * mm1, ggml_tensor * set_rows1) {
    const ggml_tensor * s0 = mm0->src[1];
    const ggml_tensor * s1 = mm1->src[1];
    const int na = (int) s0->ne[0];
    const int nb = (int) s1->ne[0];
    if (s0->type != s1->type || (s0->type != GGML_TYPE_F32 && s0->type != GGML_TYPE_F16) ||
        (na != 64 && na != 128) || (nb != 64 && nb != 128)) {
        return false;
    }
    // the second store must not read what the first one writes
    if (s1->data >= set_rows0->data && (const char *) s1->data < (const char *) set_rows0->data + ggml_nbytes(set_rows0)) {
        return false;
    }
    const fwht_set_rows_args a = fwht_set_rows_make_args(mm0, set_rows0);
    const fwht_set_rows_args b = fwht_set_rows_make_args(mm1, set_rows1);

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const dim3 grid(a.n_blocks + b.n_blocks, 1, 1);
    const dim3 block(warp_size, 4, 1);

    auto go = [&](auto tag) {
        using T = decltype(tag);
        if (na == 128) {
            nb == 128 ? fwht_set_rows2_q8_0_cuda<128, 128, T><<<grid, block, 0, ctx.stream()>>>(a, b)
                      : fwht_set_rows2_q8_0_cuda<128,  64, T><<<grid, block, 0, ctx.stream()>>>(a, b);
        } else {
            nb == 128 ? fwht_set_rows2_q8_0_cuda< 64, 128, T><<<grid, block, 0, ctx.stream()>>>(a, b)
                      : fwht_set_rows2_q8_0_cuda< 64,  64, T><<<grid, block, 0, ctx.stream()>>>(a, b);
        }
    };
    s0->type == GGML_TYPE_F32 ? go(float()) : go(half());
    return true;
}
